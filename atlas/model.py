"""CPU-friendly text-path mechanisms, deliberately without an inference KV cache."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .spec import ModelSpec


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x: Tensor, base: float) -> Tensor:
    """Rotate adjacent coordinate pairs in a [B,T,H,D] tensor."""
    batch, tokens, heads, width = x.shape
    if width % 2:
        raise ValueError("RoPE head width must be even")
    half = width // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angle = torch.arange(tokens, device=x.device, dtype=torch.float32)[:, None] * freq
    cosine = angle.cos()[None, :, None, :]
    sine = angle.sin()[None, :, None, :]
    pairs = x.reshape(batch, tokens, heads, half, 2)
    left, right = pairs.unbind(dim=-1)
    return torch.stack((left * cosine - right * sine,
                        left * sine + right * cosine), dim=-1).flatten(-2)


def causal_window_mask(tokens: int, window: int | None, device: torch.device | None = None) -> Tensor:
    """Mask[q,k] is true only when a query may read that key."""
    if tokens < 1 or (window is not None and window < 1):
        raise ValueError("tokens/window must be positive")
    positions = torch.arange(tokens, device=device)
    distance = positions[:, None] - positions[None, :]
    return (distance >= 0) & ((distance < window) if window is not None else True)


class GroupedAttention(nn.Module):
    def __init__(self, spec: ModelSpec, kind: str):
        super().__init__()
        self.kind = kind
        self.query_heads = spec.query_heads
        self.kv_heads = spec.kv_heads
        self.head_width = spec.width // spec.query_heads
        self.window = spec.local_window if kind == "local" else None
        self.rope_base = spec.rope_base_local if kind == "local" else spec.rope_base_global
        self.q_proj = nn.Linear(spec.width, spec.query_heads * self.head_width, bias=False)
        self.k_proj = nn.Linear(spec.width, spec.kv_heads * self.head_width, bias=False)
        self.v_proj = nn.Linear(spec.width, spec.kv_heads * self.head_width, bias=False)
        self.out_proj = nn.Linear(spec.width, spec.width, bias=False)
        self.q_norm = RMSNorm(self.head_width)
        self.k_norm = RMSNorm(self.head_width)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, width = x.shape
        dh = self.head_width
        q = self.q_norm(self.q_proj(x).reshape(batch, tokens, self.query_heads, dh))
        k = self.k_norm(self.k_proj(x).reshape(batch, tokens, self.kv_heads, dh))
        v = self.v_proj(x).reshape(batch, tokens, self.kv_heads, dh)
        q = apply_rope(q, self.rope_base).transpose(1, 2)
        k = apply_rope(k, self.rope_base).transpose(1, 2)
        v = v.transpose(1, 2)
        repeat = self.query_heads // self.kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(dh)
        visible = causal_window_mask(tokens, self.window, x.device)
        scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        output = (weights @ v).transpose(1, 2).contiguous().reshape(batch, tokens, width)
        return self.out_proj(output)


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    def __init__(self, spec: ModelSpec, kind: str):
        super().__init__()
        self.style = spec.block_style
        self.attention = GroupedAttention(spec, kind)
        self.ffn = SwiGLU(spec.width, spec.ff_hidden)
        self.attention_input_norm = RMSNorm(spec.width) if self.style == "pre_and_post_norm" else nn.Identity()
        self.ffn_input_norm = RMSNorm(spec.width) if self.style == "pre_and_post_norm" else nn.Identity()
        self.attention_output_norm = RMSNorm(spec.width)
        self.ffn_output_norm = RMSNorm(spec.width)

    def forward(self, x: Tensor) -> Tensor:
        h = x + self.attention_output_norm(self.attention(self.attention_input_norm(x)))
        return h + self.ffn_output_norm(self.ffn(self.ffn_input_norm(h)))


class TinyLanguageModel(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        spec.validate()
        self.spec = spec
        self.embedding = nn.Embedding(spec.vocab_size, spec.width)
        self.blocks = nn.ModuleList(DecoderBlock(spec, kind) for kind in spec.attention_schedule)
        self.final_norm = RMSNorm(spec.width)
        self.lm_head = nn.Linear(spec.width, spec.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # teaching simplification, not checkpoint-compatible

    def forward(self, ids: Tensor) -> Tensor:
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.spec.max_context:
            raise ValueError("ids must have shape [batch, 1..max_context]")
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


def cache_bytes(spec: ModelSpec, batch: int, tokens: int, bytes_per_element: int) -> int:
    """Analytical raw KV capacity for a hypothetical per-layer rolling cache.

    The forward implementation above is full-sequence and creates no KV cache.
    """
    spec.validate()
    if min(batch, tokens, bytes_per_element) <= 0:
        raise ValueError("batch, tokens, bytes must be positive")
    dh = spec.width // spec.query_heads
    return sum(2 * batch * min(tokens, spec.local_window if kind == "local" else tokens)
               * spec.kv_heads * dh * bytes_per_element for kind in spec.attention_schedule)
