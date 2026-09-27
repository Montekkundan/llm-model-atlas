"""Versioned, validated teaching presets; these are not published-model configs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PRESET_DIR = Path(__file__).resolve().parent.parent / "presets"


@dataclass(frozen=True)
class ModelSpec:
    schema_version: int
    family: str
    source_url: str
    source_revision: str
    fidelity: str
    vocab_size: int
    width: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    ff_hidden: int
    max_context: int
    attention_schedule: tuple[str, ...]
    local_window: int | None
    rope_base_global: float
    rope_base_local: float
    block_style: str
    qk_norm: bool
    qk_norm_axis: str
    gate_activation: str
    rope_layout: str
    norm_eps: float
    tie_embeddings: bool
    mla_q_rank: int | None
    mla_kv_rank: int | None
    mla_rope_dim: int | None
    moe_experts: int | None
    moe_top_k: int | None
    moe_shared_experts: int | None

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelSpec":
        allowed = set(cls.__dataclass_fields__)
        if set(raw) != allowed:
            raise ValueError(f"spec fields differ: missing={allowed - set(raw)}, extra={set(raw) - allowed}")
        spec = cls(**{**raw, "attention_schedule": tuple(raw["attention_schedule"])})
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.schema_version != 4:
            raise ValueError("unknown spec schema")
        if self.family not in {"olmo2_text_tiny", "gemma3_text_tiny", "mistral_small31_text_tiny", "qwen3_dense_text_tiny", "deepseek_v3_style_text_tiny"}:
            raise ValueError("unsupported family")
        sources = {
            "olmo2_text_tiny": "https://arxiv.org/abs/2501.00656",
            "gemma3_text_tiny": "https://arxiv.org/abs/2503.19786",
            "mistral_small31_text_tiny": "https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/config.json",
            "qwen3_dense_text_tiny": "https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json",
            "deepseek_v3_style_text_tiny": "https://arxiv.org/abs/2412.19437",
        }
        if self.source_url != sources[self.family] or not self.source_revision.strip():
            raise ValueError("family source reference is missing or mismatched")
        if self.fidelity != "tiny_text_path_not_checkpoint_compatible":
            raise ValueError("unsupported fidelity claim")
        if min(self.vocab_size, self.width, self.layers, self.query_heads,
               self.kv_heads, self.head_dim, self.ff_hidden, self.max_context) <= 0:
            raise ValueError("all dimensions must be positive")
        if self.head_dim % 2:
            raise ValueError("head width must be even for RoPE")
        if self.query_heads % self.kv_heads:
            raise ValueError("query heads must divide evenly into KV groups")
        if len(self.attention_schedule) != self.layers:
            raise ValueError("schedule length must equal layer count")
        if any(kind not in {"global", "local"} for kind in self.attention_schedule):
            raise ValueError("unknown attention kind")
        if self.local_window is not None and not 1 <= self.local_window <= self.max_context:
            raise ValueError("invalid local window")
        if "local" in self.attention_schedule and self.local_window is None:
            raise ValueError("local attention needs a window")
        if min(self.rope_base_global, self.rope_base_local) <= 1:
            raise ValueError("RoPE bases must exceed one")
        if self.rope_layout not in {"adjacent", "half"}:
            raise ValueError("unknown RoPE layout")
        if self.norm_eps <= 0:
            raise ValueError("normalization epsilon must be positive")
        if self.qk_norm_axis not in {"none", "projection", "head"}:
            raise ValueError("unknown QK norm axis")
        if self.qk_norm != (self.qk_norm_axis != "none"):
            raise ValueError("QK norm flag and axis disagree")
        if self.gate_activation not in {"silu", "gelu"}:
            raise ValueError("unknown gate activation")
        mla_moe = (self.mla_q_rank, self.mla_kv_rank, self.mla_rope_dim,
                   self.moe_experts, self.moe_top_k, self.moe_shared_experts)
        if self.family == "deepseek_v3_style_text_tiny":
            if self.layers < 4:
                raise ValueError("DeepSeek-style path needs three dense layers before a routed layer")
            if any(value is None or value <= 0 for value in mla_moe):
                raise ValueError("DeepSeek-style path requires positive MLA and MoE dimensions")
            if self.mla_rope_dim % 2 or self.mla_kv_rank >= self.query_heads * self.head_dim:
                raise ValueError("MLA rotary dimension must be even and KV rank must compress keys/values")
            if self.moe_top_k > self.moe_experts:
                raise ValueError("MoE top-k must not exceed routed expert count")
            if self.block_style != "pre_norm" or set(self.attention_schedule) != {"global"}:
                raise ValueError("DeepSeek-style text path requires pre-norm and global attention")
            if self.query_heads != self.kv_heads or self.qk_norm or self.gate_activation != "silu":
                raise ValueError("DeepSeek-style text path requires MLA heads without GQA or QK norm, and SwiGLU")
            if self.rope_layout != "adjacent" or self.tie_embeddings:
                raise ValueError("DeepSeek-style text path requires adjacent RoPE and untied embeddings")
        elif any(value is not None for value in mla_moe):
            raise ValueError("MLA and MoE dimensions only belong to the DeepSeek-style path")
        if self.family == "olmo2_text_tiny":
            if self.block_style != "reordered_output_norm" or set(self.attention_schedule) != {"global"}:
                raise ValueError("OLMo 2 teaching path requires output norms and global attention")
            if self.query_heads != self.kv_heads:
                raise ValueError("this OLMo 2 teaching preset models the MHA variants")
            if self.qk_norm_axis != "projection":
                raise ValueError("OLMo 2 requires full-projection QK norm")
            if self.gate_activation != "silu":
                raise ValueError("OLMo 2 requires SiLU gate activation")
            if self.rope_layout != "adjacent":
                raise ValueError("OLMo 2 preset uses adjacent-pair RoPE")
        if self.family == "gemma3_text_tiny":
            if self.block_style != "pre_and_post_norm" or self.attention_schedule != (
                "local", "local", "local", "local", "local", "global"
            ):
                raise ValueError("Gemma 3 teaching path requires a 5:1 local/global cycle")
            if not self.kv_heads < self.query_heads:
                raise ValueError("this Gemma 3 teaching preset requires GQA")
            if self.qk_norm_axis != "head":
                raise ValueError("Gemma 3 requires headwise QK norm")
            if self.gate_activation != "gelu":
                raise ValueError("Gemma 3 requires GELU gate activation")
            if self.rope_layout != "adjacent":
                raise ValueError("Gemma 3 preset uses adjacent-pair RoPE")
        if self.family == "mistral_small31_text_tiny":
            if self.block_style != "pre_norm" or set(self.attention_schedule) != {"global"}:
                raise ValueError("Mistral Small 3.1 text path requires pre-norm and global attention")
            if self.kv_heads >= self.query_heads or self.qk_norm or self.gate_activation != "silu":
                raise ValueError("Mistral Small 3.1 text path requires GQA without QK norm and SwiGLU")
            if self.rope_layout != "adjacent" or self.tie_embeddings:
                raise ValueError("Mistral Small 3.1 preset uses adjacent RoPE and untied embeddings")
        if self.family == "qwen3_dense_text_tiny":
            if self.block_style != "pre_norm" or set(self.attention_schedule) != {"global"}:
                raise ValueError("Qwen3 dense text path requires pre-norm and global attention")
            if self.kv_heads >= self.query_heads or self.qk_norm_axis != "head" or self.gate_activation != "silu":
                raise ValueError("Qwen3 dense text path requires GQA, headwise QK norm, and SwiGLU")
            if self.rope_layout != "half" or not self.tie_embeddings:
                raise ValueError("Qwen3 dense preset uses split-half RoPE and tied embeddings")


def load_preset(name: str) -> ModelSpec:
    if name not in {"olmo2", "gemma3", "mistral_small31", "qwen3_dense", "deepseek_v3_style"}:
        raise ValueError("choose olmo2, gemma3, mistral_small31, qwen3_dense, or deepseek_v3_style")
    raw = json.loads((PRESET_DIR / f"{name}.json").read_text())
    return ModelSpec.from_dict(raw)
