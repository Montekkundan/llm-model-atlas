import unittest
from dataclasses import replace

import torch
from torch.nn import functional as F

from atlas import ModelSpec, TinyLanguageModel, cache_bytes, causal_window_mask, load_preset
from atlas.model import GroupedAttention, RMSNorm, apply_rope


class AtlasTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def test_two_distinct_source_backed_specs(self):
        olmo = load_preset("olmo2")
        gemma = load_preset("gemma3")
        self.assertEqual(olmo.schema_version, 2)
        self.assertEqual(gemma.schema_version, 2)
        self.assertEqual(olmo.block_style, "reordered_output_norm")
        self.assertEqual(gemma.block_style, "pre_and_post_norm")
        self.assertEqual(olmo.attention_schedule, ("global", "global"))
        self.assertEqual(gemma.attention_schedule, ("local",) * 5 + ("global",))
        self.assertEqual((olmo.query_heads, olmo.kv_heads), (4, 4))
        self.assertEqual((gemma.query_heads, gemma.kv_heads), (4, 2))
        self.assertEqual((olmo.qk_norm_axis, olmo.gate_activation), ("projection", "silu"))
        self.assertEqual((gemma.qk_norm_axis, gemma.gate_activation), ("head", "gelu"))

    def test_invalid_presets_fail_before_construction(self):
        olmo = load_preset("olmo2")
        with self.assertRaisesRegex(ValueError, "schema"):
            replace(olmo, schema_version=1).validate()
        with self.assertRaisesRegex(ValueError, "head width"):
            replace(olmo, width=33).validate()
        with self.assertRaisesRegex(ValueError, "schedule"):
            replace(olmo, attention_schedule=("global",)).validate()
        with self.assertRaisesRegex(ValueError, "MHA"):
            replace(olmo, kv_heads=2).validate()
        with self.assertRaisesRegex(ValueError, "source"):
            replace(olmo, source_url="https://example.org").validate()
        with self.assertRaisesRegex(ValueError, "requires GQA"):
            replace(load_preset("gemma3"), kv_heads=4).validate()
        with self.assertRaisesRegex(ValueError, "QK norm"):
            replace(olmo, qk_norm_axis="head").validate()
        with self.assertRaisesRegex(ValueError, "gate activation"):
            replace(load_preset("gemma3"), gate_activation="silu").validate()
        with self.assertRaisesRegex(ValueError, "QK norm"):
            replace(olmo, qk_norm_axis="invalid").validate()
        with self.assertRaises(ValueError):
            load_preset("deepseek_v3")
        raw = dict(vars(olmo), unexpected_field=True)
        with self.assertRaisesRegex(ValueError, "spec fields differ"):
            ModelSpec.from_dict(raw)

    def test_causal_and_sliding_masks(self):
        full = causal_window_mask(6, None)
        local = causal_window_mask(6, 3)
        self.assertTrue(full[5, 0].item())
        self.assertFalse(full[0, 1].item())
        self.assertFalse(local[5, 2].item())
        self.assertTrue(local[5, 3].item())
        self.assertTrue(local[5, 5].item())
        self.assertEqual(local.sum(dim=-1).tolist(), [1, 2, 3, 3, 3, 3])

    def test_rope_preserves_head_norm(self):
        x = torch.randn(2, 7, 4, 8)
        rotated = apply_rope(x, 500000.0)
        torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5)

    def test_family_specific_qk_normalization_axes(self):
        olmo = GroupedAttention(load_preset("olmo2"), "global")
        gemma = GroupedAttention(load_preset("gemma3"), "global")
        self.assertEqual(tuple(olmo.q_norm.weight.shape), (olmo.query_heads * olmo.head_width,))
        self.assertEqual(tuple(olmo.k_norm.weight.shape), (olmo.kv_heads * olmo.head_width,))
        self.assertEqual(tuple(gemma.q_norm.weight.shape), (gemma.head_width,))
        self.assertEqual(tuple(gemma.k_norm.weight.shape), (gemma.head_width,))

        projected = torch.randn(2, 3, olmo.query_heads * olmo.head_width)
        normalized = olmo.q_norm(projected)
        expected = projected * torch.rsqrt(projected.square().mean(-1, keepdim=True) + olmo.q_norm.eps)
        torch.testing.assert_close(normalized, expected)

    def test_family_specific_gated_mlp_activation(self):
        for family, gate_activation in (("olmo2", lambda x: F.silu(x)),
                                        ("gemma3", lambda x: F.gelu(x, approximate="tanh"))):
            with self.subTest(family=family):
                ffn = TinyLanguageModel(load_preset(family)).blocks[0].ffn
                x = torch.randn(2, 3, ffn.gate.in_features)
                expected = ffn.down(gate_activation(ffn.gate(x)) * ffn.up(x))
                torch.testing.assert_close(ffn(x), expected)

    def test_local_attention_changes_the_actual_forward(self):
        spec = load_preset("gemma3")
        local = GroupedAttention(spec, "local")
        global_attention = GroupedAttention(spec, "global")
        global_attention.load_state_dict(local.state_dict())
        # Keep the RoPE base equal so the mask is the only difference.
        global_attention.rope_base = local.rope_base
        x = torch.randn(1, 8, spec.width)
        local_last = local(x)[:, -1]
        global_last = global_attention(x)[:, -1]
        self.assertGreater((local_last - global_last).abs().max().item(), 1e-6)

    def test_both_tiny_models_forward_backward_and_causality(self):
        for name in ("olmo2", "gemma3"):
            with self.subTest(family=name):
                spec = load_preset(name)
                model = TinyLanguageModel(spec)
                if name == "olmo2":
                    self.assertIsInstance(model.blocks[0].attention_input_norm, torch.nn.Identity)
                else:
                    self.assertIsInstance(model.blocks[0].attention_input_norm, RMSNorm)
                ids = torch.randint(spec.vocab_size, (2, 6))
                logits = model(ids)
                self.assertEqual(tuple(logits.shape), (2, 6, spec.vocab_size))
                self.assertTrue(torch.isfinite(logits).all().item())
                targets = torch.randint(spec.vocab_size, (2, 6))
                loss = F.cross_entropy(logits.reshape(-1, spec.vocab_size), targets.reshape(-1))
                loss.backward()
                for projection in (model.blocks[0].attention.q_proj,
                                   model.blocks[0].attention.k_proj,
                                   model.blocks[0].attention.v_proj):
                    self.assertIsNotNone(projection.weight.grad)
                    self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
                model.eval()
                altered = ids.clone()
                altered[:, -1] = (altered[:, -1] + 1) % spec.vocab_size
                with torch.no_grad():
                    earlier = model(ids)[:, :-1]
                    changed = model(altered)[:, :-1]
                torch.testing.assert_close(earlier, changed, rtol=0, atol=0)

    def test_cache_ledger_is_explicitly_analytical(self):
        olmo, gemma = load_preset("olmo2"), load_preset("gemma3")
        self.assertEqual(cache_bytes(olmo, 1, 32, 4), 16384)
        self.assertEqual(cache_bytes(gemma, 1, 32, 4), 6656)
        self.assertEqual(cache_bytes(gemma, 1, 2, 4), 1536)
        self.assertLess(cache_bytes(gemma, 1, 32, 4), cache_bytes(olmo, 1, 32, 4))
        with self.assertRaises(ValueError):
            cache_bytes(gemma, 1, 0, 4)


if __name__ == "__main__":
    unittest.main()
