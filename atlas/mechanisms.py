"""Differentiable CPU references for case-study operators, not checkpoint ports."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .model import apply_rope, causal_window_mask


def positional_schedule(layers: int, interval: int) -> tuple[str, ...]:
    if layers < 1 or interval < 1:
        raise ValueError("layers and interval must be positive")
    return tuple("none" if (i + 1) % interval == 0 else "rope" for i in range(layers))


def position_queries_keys(q: Tensor, k: Tensor, mode: str, base: float) -> tuple[Tensor, Tensor]:
    """Inputs/outputs are [B,T,H,D]; NoPE omits rotation, never causality."""
    if mode == "none":
        return q, k
    if mode != "rope":
        raise ValueError("position mode must be none or rope")
    return apply_rope(q, base), apply_rope(k, base)


def biased_topk_route(logits: Tensor, top_k: int, bias: Tensor | None = None,
                      gate: str = "sigmoid", normalize: bool = True) -> tuple[Tensor, Tensor]:
    """Selection may use bias; combination weights always use original scores.

    normalize=True divides the selected scores by their sum, so top_k=1 gives a gate of
    exactly one and the router receives no task gradient (DeepSeek-V3 normalises its
    sigmoid affinities). normalize=False keeps the selected scores as the gates: the
    sigmoid of the selected logit (Llama 4) or the full-softmax probability (Grok-1 code).
    """
    if logits.ndim != 2 or not 1 <= top_k <= logits.shape[-1]:
        raise ValueError("expected [tokens,experts] logits and a valid top_k")
    if gate == "sigmoid":
        scores = logits.sigmoid()
    elif gate == "softmax":
        scores = logits.softmax(dim=-1)
    else:
        raise ValueError("gate must be sigmoid or softmax")
    if bias is not None and bias.shape != (logits.shape[-1],):
        raise ValueError("bias must have one value per expert")
    selection = scores if bias is None else scores + bias
    indices = torch.argsort(selection, dim=-1, descending=True, stable=True)[..., :top_k]
    selected = scores.gather(-1, indices)
    if not normalize:
        return selected, indices
    weights = selected / selected.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)
    return weights, indices


def chunked_causal_mask(tokens: int, chunk: int, device: torch.device | None = None) -> Tensor:
    """Llama 4 chunked attention: causal inside the chunk position // chunk, blind to earlier chunks.

    This is not a sliding window: a query at the start of a chunk sees only itself.
    """
    if tokens < 1 or chunk < 1:
        raise ValueError("tokens/chunk must be positive")
    positions = torch.arange(tokens, device=device)
    same_chunk = positions[:, None] // chunk == positions[None, :] // chunk
    return same_chunk & (positions[:, None] >= positions[None, :])


def sparse_causal_attention(q: Tensor, k: Tensor, v: Tensor, index_scores: Tensor,
                            top_k: int) -> tuple[Tensor, Tensor]:
    """Reference gather on [B,H,T,D], with stable ties and causal selection."""
    if q.ndim != 4 or q.shape != k.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("q/k must share [B,H,T,D]; v shares B,H,T")
    batch, heads, tokens, dim = q.shape
    if index_scores.shape != (batch, heads, tokens, tokens) or not 1 <= top_k <= tokens:
        raise ValueError("index scores or top_k do not match sequence")
    logits = q @ k.transpose(-2, -1) / math.sqrt(dim)
    visible = causal_window_mask(tokens, None, q.device)
    order = torch.argsort(index_scores.masked_fill(~visible, -torch.inf),
                          dim=-1, descending=True, stable=True)[..., :top_k]
    legal = order <= torch.arange(tokens, device=q.device)[None, None, :, None]
    selected_logits = logits.gather(-1, order).masked_fill(~legal, -torch.inf)
    weights = selected_logits.softmax(-1)
    values = v[:, :, None].expand(-1, -1, tokens, -1, -1)
    selected_values = values.gather(-2, order[..., None].expand(-1, -1, -1, -1, v.shape[-1]))
    return (weights[..., None] * selected_values).sum(-2), order.masked_fill(~legal, -1)


def sink_attention(q: Tensor, k: Tensor, v: Tensor, sink: Tensor,
                    window: int | None = None) -> tuple[Tensor, Tensor]:
    """An extra zero-value sink absorbs probability mass without entering KV."""
    if q.ndim != 4 or q.shape != k.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("q/k must share [B,H,T,D]; v shares B,H,T")
    if sink.shape != (q.shape[1],):
        raise ValueError("sink must contain one logit per head")
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    visible = causal_window_mask(q.shape[2], window, q.device)
    scores = scores.masked_fill(~visible, -torch.inf)
    sink_scores = sink[None, :, None, None].expand(q.shape[0], -1, q.shape[2], -1)
    token_weights = torch.cat((scores, sink_scores), dim=-1).softmax(-1)[..., :-1]
    return token_weights @ v, token_weights.sum(-1)


def key_as_value_attention(q: Tensor, shared_kv: Tensor) -> Tensor:
    """Full-prefix causal reference: keys and values are the same tensor."""
    if q.shape != shared_kv.shape or q.ndim != 4:
        raise ValueError("q/shared_kv must share [B,H,T,D]")
    scores = q @ shared_kv.transpose(-2, -1) / math.sqrt(q.shape[-1])
    visible = causal_window_mask(q.shape[2], None, q.device)
    return scores.masked_fill(~visible, -torch.inf).softmax(-1) @ shared_kv


def delta_step(state: Tensor, k: Tensor, v: Tensor, alpha: Tensor,
               beta: Tensor) -> Tensor:
    """KDA-order reference: S_hat=D(alpha)S; then beta*k*(v-k^T S_hat)."""
    if state.shape != (*k.shape, v.shape[-1]) or alpha.shape != k.shape or beta.shape != k.shape[:-1]:
        raise ValueError("state [...,Dk,Dv], k/alpha [...,Dk], v [...,Dv], beta [...] required")
    decayed = alpha[..., :, None] * state
    prediction = (k[..., :, None] * decayed).sum(-2)
    return decayed + beta[..., None, None] * k[..., :, None] * (v - prediction)[..., None, :]


def delta_scan(k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor,
               initial: Tensor) -> tuple[Tensor, Tensor]:
    """Sequence axis is 1; outputs are the complete recurrent state trajectory."""
    if k.ndim < 3 or k.shape[1] < 1 or v.shape[:2] != k.shape[:2]:
        raise ValueError("expected nonempty [B,T,...,D] token arrays")
    state = initial
    states = []
    for t in range(k.shape[1]):
        state = delta_step(state, k[:, t], v[:, t], alpha[:, t], beta[:, t])
        states.append(state)
    return torch.stack(states, dim=1), state


def selective_ssm_step(state: Tensor, x: Tensor, decay: Tensor,
                       input_vector: Tensor, read_vector: Tensor) -> tuple[Tensor, Tensor]:
    """Per-head scalar-decay SSM reference, not Mamba's fused/discretization kernel.

    state=[B,H,N,P], x=[B,H,P], decay=[B,H], B/C=[B,H,N].
    Caller supplies already discretized decay and input-vector coefficients.
    """
    if state.ndim != 4 or x.shape != (state.shape[0], state.shape[1], state.shape[3]):
        raise ValueError("state [B,H,N,P] and x [B,H,P] required")
    if decay.shape != state.shape[:2] or input_vector.shape != state.shape[:3] or read_vector.shape != state.shape[:3]:
        raise ValueError("decay [B,H] and input/read vectors [B,H,N] required")
    next_state = decay[..., None, None] * state + input_vector[..., :, None] * x[..., None, :]
    return (read_vector[..., :, None] * next_state).sum(-2), next_state


@dataclass(frozen=True)
class HybridState:
    """State container; forks and snapshots own their cloned tensor storage."""

    recurrent: Tensor
    keys: Tensor
    values: Tensor
    tokens: int

    def fork(self) -> "HybridState":
        return HybridState(self.recurrent.clone(), self.keys.clone(), self.values.clone(), self.tokens)

    def reorder(self, order: Tensor) -> "HybridState":
        return HybridState(self.recurrent.index_select(0, order), self.keys.index_select(0, order),
                           self.values.index_select(0, order), self.tokens)

    def snapshot(self) -> dict:
        return {"schema_version": 1, "tokens": self.tokens, "recurrent": self.recurrent.clone(),
                "keys": self.keys.clone(), "values": self.values.clone()}

    @classmethod
    def restore(cls, saved: dict) -> "HybridState":
        if saved.get("schema_version") != 1 or saved["tokens"] < 0:
            raise ValueError("unsupported state snapshot")
        if saved["keys"].shape != saved["values"].shape or saved["keys"].shape[2] != saved["tokens"]:
            raise ValueError("snapshot KV length and token counter differ")
        return cls(saved["recurrent"].clone(), saved["keys"].clone(), saved["values"].clone(), saved["tokens"])
