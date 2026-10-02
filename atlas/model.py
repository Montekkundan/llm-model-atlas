"""CPU-friendly text-path mechanisms, deliberately without an inference KV cache."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .spec import ModelSpec


# initializer_range of the OLMo 2, Gemma 3, Mistral Small 3.1, Qwen3 and DeepSeek-V3 configs.
# PyTorch's defaults (N(0,1) embeddings) would give a first-step loss far above ln(vocab).
INIT_STD = 0.02


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x: Tensor, base: float, layout: Literal["adjacent", "half"] = "adjacent") -> Tensor:
    """Rotate adjacent or split-half coordinate pairs in a [B,T,H,D] tensor."""
    batch, tokens, heads, width = x.shape
    if width % 2:
        raise ValueError("RoPE head width must be even")
    half = width // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angle = torch.arange(tokens, device=x.device, dtype=torch.float32)[:, None] * freq
    cosine = angle.cos()[None, :, None, :]
    sine = angle.sin()[None, :, None, :]
    if layout == "adjacent":
        pairs = x.reshape(batch, tokens, heads, half, 2)
        left, right = pairs.unbind(dim=-1)
        return torch.stack((left * cosine - right * sine,
                            left * sine + right * cosine), dim=-1).flatten(-2).to(x.dtype)
    if layout == "half":
        left, right = x.chunk(2, dim=-1)
        return torch.cat((left * cosine - right * sine,
                          left * sine + right * cosine), dim=-1).to(x.dtype)
    raise ValueError("unknown RoPE layout")


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
        self.head_width = spec.head_dim
        self.full_projection_qk_norm = spec.qk_norm_axis == "projection"
        self.rope_layout = spec.rope_layout
        self.window = spec.local_window if kind == "local" else None
        self.rope_base = spec.rope_base_local if kind == "local" else spec.rope_base_global
        self.q_proj = nn.Linear(spec.width, spec.query_heads * self.head_width, bias=False)
        self.k_proj = nn.Linear(spec.width, spec.kv_heads * self.head_width, bias=False)
        self.v_proj = nn.Linear(spec.width, spec.kv_heads * self.head_width, bias=False)
        self.out_proj = nn.Linear(spec.query_heads * self.head_width, spec.width, bias=False)
        q_norm_width = spec.query_heads * self.head_width if self.full_projection_qk_norm else self.head_width
        k_norm_width = spec.kv_heads * self.head_width if self.full_projection_qk_norm else self.head_width
        self.q_norm = RMSNorm(q_norm_width, spec.norm_eps) if spec.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(k_norm_width, spec.norm_eps) if spec.qk_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        dh = self.head_width
        q = self.q_proj(x)
        k = self.k_proj(x)
        if self.full_projection_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q = q.reshape(batch, tokens, self.query_heads, dh)
        k = k.reshape(batch, tokens, self.kv_heads, dh)
        if not self.full_projection_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        v = self.v_proj(x).reshape(batch, tokens, self.kv_heads, dh)
        q = apply_rope(q, self.rope_base, self.rope_layout).transpose(1, 2)
        k = apply_rope(k, self.rope_base, self.rope_layout).transpose(1, 2)
        v = v.transpose(1, 2)
        repeat = self.query_heads // self.kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(dh)
        visible = causal_window_mask(tokens, self.window, x.device)
        scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        output = (weights @ v).transpose(1, 2).contiguous().reshape(batch, tokens, self.query_heads * dh)
        return self.out_proj(output)


class GatedMLP(nn.Module):
    def __init__(self, width: int, hidden: int, gate_activation: Literal["silu", "gelu"]):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.gate_activation = gate_activation

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate(x)
        activated = (
            F.gelu(gate, approximate="tanh") if self.gate_activation == "gelu" else F.silu(gate)
        )
        return self.down(activated * self.up(x))


class LatentAttention(nn.Module):
    """Full-sequence MLA teaching path; no inference cache or weight absorption."""

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.heads = spec.query_heads
        self.content_dim = spec.head_dim
        self.rope_dim = int(spec.mla_rope_dim)
        self.kv_rank = int(spec.mla_kv_rank)
        self.rope_base = spec.rope_base_global
        self.q_down = nn.Linear(spec.width, int(spec.mla_q_rank), bias=False)
        self.q_norm = RMSNorm(int(spec.mla_q_rank), spec.norm_eps)
        self.q_up = nn.Linear(int(spec.mla_q_rank), self.heads * (self.content_dim + self.rope_dim), bias=False)
        self.kv_down = nn.Linear(spec.width, self.kv_rank + self.rope_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_rank, spec.norm_eps)
        self.kv_up = nn.Linear(self.kv_rank, self.heads * (2 * self.content_dim), bias=False)
        self.out_proj = nn.Linear(self.heads * self.content_dim, spec.width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        q = self.q_up(self.q_norm(self.q_down(x)))
        q = q.reshape(batch, tokens, self.heads, self.content_dim + self.rope_dim)
        q_content, q_rope = q.split((self.content_dim, self.rope_dim), dim=-1)
        latent, k_rope = self.kv_down(x).split((self.kv_rank, self.rope_dim), dim=-1)
        kv = self.kv_up(self.kv_norm(latent))
        kv = kv.reshape(batch, tokens, self.heads, 2 * self.content_dim)
        k_content, v = kv.split(self.content_dim, dim=-1)
        q_rope = apply_rope(q_rope, self.rope_base)
        k_rope = apply_rope(k_rope.unsqueeze(2), self.rope_base)
        q = torch.cat((q_content, q_rope), dim=-1).transpose(1, 2)
        k = torch.cat((k_content, k_rope.expand(-1, -1, self.heads, -1)), dim=-1).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.content_dim + self.rope_dim)
        scores = scores.masked_fill(~causal_window_mask(tokens, None, x.device), torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        attended = (weights @ v.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(attended.reshape(batch, tokens, self.heads * self.content_dim))


class RoutedMoE(nn.Module):
    """Sigmoid top-k routed experts plus always-on shared experts."""

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.top_k = int(spec.moe_top_k)
        self.router = nn.Linear(spec.width, int(spec.moe_experts), bias=False)
        self.routed_experts = nn.ModuleList(
            GatedMLP(spec.width, spec.ff_hidden, "silu") for _ in range(int(spec.moe_experts))
        )
        self.shared_experts = nn.ModuleList(
            GatedMLP(spec.width, spec.ff_hidden, "silu") for _ in range(int(spec.moe_shared_experts))
        )

    def route(self, x: Tensor) -> tuple[Tensor, Tensor]:
        affinities = self.router(x).sigmoid()
        selected, indices = affinities.topk(self.top_k, dim=-1)
        return selected / selected.sum(dim=-1, keepdim=True), indices

    def forward(self, x: Tensor) -> Tensor:
        flat = x.reshape(-1, x.shape[-1])
        weights, indices = self.route(flat)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.routed_experts):
            row, slot = torch.where(indices == expert_id)
            if row.numel():
                contribution = expert(flat[row]) * weights[row, slot, None]
                output = output.index_add(0, row, contribution)
        for expert in self.shared_experts:
            output = output + expert(flat)
        return output.reshape_as(x)


class DecoderBlock(nn.Module):
    def __init__(self, spec: ModelSpec, kind: str, layer_index: int):
        super().__init__()
        self.style = spec.block_style
        deepseek_style = spec.family == "deepseek_v3_style_text_tiny"
        self.attention = LatentAttention(spec) if deepseek_style else GroupedAttention(spec, kind)
        self.ffn = RoutedMoE(spec) if deepseek_style and layer_index >= 3 else GatedMLP(spec.width, spec.ff_hidden, spec.gate_activation)
        self.attention_input_norm = RMSNorm(spec.width, spec.norm_eps) if self.style in {"pre_and_post_norm", "pre_norm"} else nn.Identity()
        self.ffn_input_norm = RMSNorm(spec.width, spec.norm_eps) if self.style in {"pre_and_post_norm", "pre_norm"} else nn.Identity()
        self.attention_output_norm = RMSNorm(spec.width, spec.norm_eps) if self.style != "pre_norm" else nn.Identity()
        self.ffn_output_norm = RMSNorm(spec.width, spec.norm_eps) if self.style != "pre_norm" else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = x + self.attention_output_norm(self.attention(self.attention_input_norm(x)))
        return h + self.ffn_output_norm(self.ffn(self.ffn_input_norm(h)))


class TinyLanguageModel(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        spec.validate()
        self.spec = spec
        self.embedding = nn.Embedding(spec.vocab_size, spec.width)
        self.blocks = nn.ModuleList(DecoderBlock(spec, kind, index)
                                    for index, kind in enumerate(spec.attention_schedule))
        self.final_norm = RMSNorm(spec.width, spec.norm_eps)
        self.lm_head = nn.Linear(spec.width, spec.vocab_size, bias=False)
        if spec.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        for parameter in self.parameters():
            if parameter.ndim >= 2:  # matrices and embeddings; norm gains stay at one
                nn.init.normal_(parameter, mean=0.0, std=INIT_STD)

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
    if spec.family == "deepseek_v3_style_text_tiny":
        return spec.layers * batch * tokens * (int(spec.mla_kv_rank) + int(spec.mla_rope_dim)) * bytes_per_element
    dh = spec.head_dim
    return sum(2 * batch * min(tokens, spec.local_window if kind == "local" else tokens)
               * spec.kv_heads * dh * bytes_per_element for kind in spec.attention_schedule)
