"""Run the constituent operators taught by lessons 51–73 on small CPU tensors."""

from __future__ import annotations

import argparse
import json

import torch

from .mechanisms import (HybridState, biased_topk_route, delta_scan, delta_step,
                         key_as_value_attention, position_queries_keys,
                         positional_schedule, selective_ssm_step,
                         sink_attention, sparse_causal_attention)
from .model import LatentAttention, causal_window_mask
from .spec import load_preset


CASE_OPERATORS = {
    51: ("mla", "bias_route"), 52: ("norm",), 53: ("window",),
    54: ("gqa",), 55: ("bias_route", "position"), 56: ("gqa", "softmax_route"),
    57: ("position",), 58: ("mla", "bias_route"), 59: ("window", "softmax_route", "sink"),
    60: ("gqa", "bias_route"), 61: ("gqa", "bias_route"),
    62: ("delta", "state"), 63: ("window",), 64: ("kda", "mla", "state"),
    65: ("window", "norm"), 66: ("sparse", "mla"), 67: ("mla", "bias_route"),
    68: ("ssm", "state", "latent_dispatch"), 69: ("window", "sink"),
    70: ("bias_route", "window"), 71: ("sparse", "mla"),
    72: ("registry", "delta"), 73: ("key_as_value", "window"),
}


def run_operator(name: str, family: str = "qwen3_dense") -> dict:
    torch.manual_seed(73)
    if name == "position":
        modes = positional_schedule(10, 3)
        q, k = torch.randn(1, 5, 2, 4), torch.randn(1, 5, 2, 4)
        rope_q, rope_k = position_queries_keys(q, k, "rope", 10000.0)
        none_q, none_k = position_queries_keys(q, k, "none", 10000.0)
        assert torch.equal(none_q, q) and torch.equal(none_k, k)
        assert not torch.allclose(rope_q, q) and not torch.allclose(rope_k, k)
        torch.testing.assert_close(rope_q.norm(dim=-1), q.norm(dim=-1))
        return {"nope_zero_based_layers": [i for i, mode in enumerate(modes) if mode == "none"]}
    if name in {"bias_route", "softmax_route"}:
        logits = torch.logit(torch.tensor([[0.6, 0.5, 0.4]], requires_grad=True))
        bias = torch.tensor([0.0, 0.2, 0.0]) if name == "bias_route" else None
        weights, selected = biased_topk_route(logits, 2, bias,
                                              "sigmoid" if bias is not None else "softmax")
        torch.testing.assert_close(weights.sum(-1), torch.ones(1))
        if bias is not None:
            assert selected.tolist() == [[1, 0]]
            torch.testing.assert_close(weights, torch.tensor([[5 / 11, 6 / 11]]))
        return {"selected": selected.tolist(), "gates": [[round(x, 6) for x in weights[0].tolist()]]}
    if name == "window":
        mask = causal_window_mask(7, 4)
        assert mask[5].nonzero().flatten().tolist() == [2, 3, 4, 5]
        assert not mask[0, 1]
        return {"local_keys_at_5": [2, 3, 4, 5], "global_keys_at_5": list(range(6))}
    if name == "gqa":
        from .model import GroupedAttention
        spec = load_preset(family)
        op = GroupedAttention(spec, "global")
        x = torch.randn(2, 5, spec.width, requires_grad=True)
        out = op(x)
        out.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        return {"residual_shape": list(out.shape), "query_projection_width": op.q_proj.out_features,
                "kv_projection_width": op.k_proj.out_features}
    if name == "norm":
        from .model import TinyLanguageModel
        model = TinyLanguageModel(load_preset("olmo2"))
        block = model.blocks[0]
        x = torch.randn(2, 5, 32)
        h = x + block.attention_output_norm(block.attention(x))
        reference = h + block.ffn_output_norm(block.ffn(h))
        torch.testing.assert_close(block(x), reference, rtol=0, atol=0)
        return {"branch_graph": "x+N(A(x)); h+N(M(h))", "projection_qk_norm_width": 32}
    if name == "mla":
        op = LatentAttention(load_preset("deepseek_v3_style"))
        x = torch.randn(2, 5, 32, requires_grad=True)
        out = op(x)
        out.square().mean().backward()
        assert x.grad is not None and op.kv_down.weight.grad.abs().sum() > 0
        return {"output_shape": list(out.shape), "latent_width": 12, "rotary_key_width": 4}
    if name in {"delta", "kda"}:
        k = torch.nn.functional.normalize(torch.randn(2, 5, 3), dim=-1)
        v = torch.randn(2, 5, 4, requires_grad=True)
        alpha = torch.sigmoid(torch.randn(2, 5, 3))
        if name == "delta":
            alpha = alpha[..., :1].expand_as(alpha)
        beta = torch.sigmoid(torch.randn(2, 5))
        initial = torch.zeros(2, 3, 4)
        full, final = delta_scan(k, v, alpha, beta, initial)
        prefix, saved = delta_scan(k[:, :2], v[:, :2], alpha[:, :2], beta[:, :2], initial)
        suffix, resumed = delta_scan(k[:, 2:], v[:, 2:], alpha[:, 2:], beta[:, 2:], saved)
        error = (full - torch.cat((prefix, suffix), 1)).abs().max().item()
        torch.testing.assert_close(final, resumed, rtol=0, atol=0)
        final.square().mean().backward()
        assert torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0
        return {"state_shape": list(final.shape), "resume_max_error": error,
                "gate_axis": "key_dimension" if name == "kda" else "shared_scalar"}
    if name == "ssm":
        state = torch.zeros(2, 2, 3, 4)
        x = torch.randn(2, 5, 2, 4, requires_grad=True)
        b, c = torch.randn(2, 5, 2, 3), torch.randn(2, 5, 2, 3)
        decays = torch.sigmoid(torch.randn(2, 5, 2))
        outputs = []
        for t in range(5):
            y, state = selective_ssm_step(state, x[:, t], decays[:, t], b[:, t], c[:, t])
            outputs.append(y)
        checkpoint = torch.zeros_like(state)
        for t in range(2):
            _, checkpoint = selective_ssm_step(checkpoint, x[:, t], decays[:, t], b[:, t], c[:, t])
        resumed = checkpoint.clone()
        for t in range(2, 5):
            _, resumed = selective_ssm_step(resumed, x[:, t], decays[:, t], b[:, t], c[:, t])
        torch.testing.assert_close(state, resumed, rtol=0, atol=0)
        torch.stack(outputs).square().mean().backward()
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        return {"state_shape": list(state.shape), "resume_max_error": 0.0}
    if name == "state":
        state = HybridState(torch.randn(2, 3, 4), torch.randn(2, 2, 5, 4), torch.randn(2, 2, 5, 4), 5)
        restored = HybridState.restore(state.snapshot())
        torch.testing.assert_close(state.recurrent, restored.recurrent)
        fork = state.fork()
        fork.recurrent.add_(1)
        assert not torch.equal(fork.recurrent, state.recurrent)
        reorder = state.reorder(torch.tensor([1, 0]))
        torch.testing.assert_close(reorder.keys[0], state.keys[1])
        return {"snapshot_version": 1, "tokens": 5, "fork_alias": False}
    if name == "sparse":
        q, k, v = (torch.randn(1, 2, 5, 4, requires_grad=True) for _ in range(3))
        index = torch.randn(1, 2, 5, 5)
        output, selected = sparse_causal_attention(q, k, v, index, 5)
        scores = q @ k.transpose(-2, -1) / 2
        dense = scores.masked_fill(~causal_window_mask(5, None), -torch.inf).softmax(-1) @ v
        error = (output - dense).abs().max().item()
        torch.testing.assert_close(output, dense, rtol=1e-5, atol=1e-6)
        assert (selected <= torch.arange(5)[None, None, :, None]).all()
        output.square().mean().backward()
        assert torch.isfinite(q.grad).all()
        return {"all_keys_dense_max_error": round(error, 8), "future_key_selected": False}
    if name == "sink":
        q, k = torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)
        v = torch.full((1, 1, 2, 1), 4.0)
        output, mass = sink_attention(q, k, v, torch.zeros(1))
        assert torch.all(mass < 1)
        torch.testing.assert_close(output[0, 0, 0], torch.tensor([2.0]))
        return {"first_token_mass": float(mass[0, 0, 0]), "first_output": float(output[0, 0, 0, 0])}
    if name == "key_as_value":
        q = torch.randn(1, 2, 12, 4)
        shared = torch.randn_like(q)
        reference = key_as_value_attention(q, shared)
        cache = []
        steps = []
        for t in range(12):
            cache.append(shared[:, :, t:t + 1])
            keys = torch.cat(cache, dim=2)
            weights = (q[:, :, t:t + 1] @ keys.transpose(-2, -1) / 2).softmax(-1)
            steps.append(weights @ keys)
        error = (reference - torch.cat(steps, dim=2)).abs().max().item()
        torch.testing.assert_close(reference, torch.cat(steps, dim=2), rtol=1e-5, atol=1e-6)
        return {"prefix_step_max_error": round(error, 8), "kv_tensors_stored": 1}
    if name == "latent_dispatch":
        x = torch.randn(2, 16, requires_grad=True)
        down, up = torch.randn(16, 4), torch.randn(4, 16)
        reconstructed = (x @ down) @ up
        reconstructed.square().mean().backward()
        assert torch.isfinite(x.grad).all()
        return {"standard_dispatch_bytes": 64, "latent_dispatch_bytes": 16,
                "down_up_parameters": 128, "lossless_identity_claim": False}
    if name == "registry":
        from dataclasses import replace
        try:
            replace(load_preset("olmo2"), family="linear_attention").validate()
        except ValueError as error:
            assert "unsupported family" in str(error)
        else:
            raise AssertionError("unsupported family accepted")
        return {"unsupported_family_rejected": True, "operators_available": sorted(set(sum(CASE_OPERATORS.values(), ())))}
    raise ValueError(f"unsupported reference operator: {name}")


def run_case(lesson: int) -> dict:
    if lesson not in CASE_OPERATORS:
        raise ValueError("choose a lesson number from 51 through 73")
    return {"lesson": lesson, "scope": "constituent_operator_checks_not_full_family_port",
            "checks": {name: run_operator(name, "mistral_small31" if lesson == 54 else "qwen3_dense")
                       for name in CASE_OPERATORS[lesson]}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lesson", type=int, choices=range(51, 74))
    args = parser.parse_args()
    print(json.dumps(run_case(args.lesson), sort_keys=True))


if __name__ == "__main__":
    main()
