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

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelSpec":
        allowed = set(cls.__dataclass_fields__)
        if set(raw) != allowed:
            raise ValueError(f"spec fields differ: missing={allowed - set(raw)}, extra={set(raw) - allowed}")
        spec = cls(**{**raw, "attention_schedule": tuple(raw["attention_schedule"])})
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unknown spec schema")
        if self.family not in {"olmo2_text_tiny", "gemma3_text_tiny"}:
            raise ValueError("unsupported family")
        sources = {
            "olmo2_text_tiny": "https://arxiv.org/abs/2501.00656",
            "gemma3_text_tiny": "https://arxiv.org/abs/2503.19786",
        }
        if self.source_url != sources[self.family] or not self.source_revision.strip():
            raise ValueError("family source reference is missing or mismatched")
        if self.fidelity != "tiny_text_path_not_checkpoint_compatible":
            raise ValueError("unsupported fidelity claim")
        if min(self.vocab_size, self.width, self.layers, self.query_heads,
               self.kv_heads, self.ff_hidden, self.max_context) <= 0:
            raise ValueError("all dimensions must be positive")
        if self.width % self.query_heads or (self.width // self.query_heads) % 2:
            raise ValueError("head width must divide width and be even for RoPE")
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
        if not self.qk_norm:
            raise ValueError("both supported text paths require QK norm")
        if self.qk_norm_axis not in {"projection", "head"}:
            raise ValueError("unknown QK norm axis")
        if self.gate_activation not in {"silu", "gelu"}:
            raise ValueError("unknown gate activation")
        if self.family == "olmo2_text_tiny":
            if self.block_style != "reordered_output_norm" or set(self.attention_schedule) != {"global"}:
                raise ValueError("OLMo 2 teaching path requires output norms and global attention")
            if self.query_heads != self.kv_heads:
                raise ValueError("this OLMo 2 teaching preset models the MHA variants")
            if self.qk_norm_axis != "projection":
                raise ValueError("OLMo 2 requires full-projection QK norm")
            if self.gate_activation != "silu":
                raise ValueError("OLMo 2 requires SiLU gate activation")
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


def load_preset(name: str) -> ModelSpec:
    if name not in {"olmo2", "gemma3"}:
        raise ValueError("choose olmo2 or gemma3")
    raw = json.loads((PRESET_DIR / f"{name}.json").read_text())
    return ModelSpec.from_dict(raw)
