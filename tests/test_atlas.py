import math
import unittest
from dataclasses import replace

import torch
from torch.nn import functional as F

from atlas import ModelSpec, TinyLanguageModel, cache_bytes, causal_window_mask, load_preset
from atlas.model import GroupedAttention, LatentAttention, RMSNorm, RoutedMoE, apply_rope


class AtlasTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def test_four_distinct_source_backed_specs(self):
        olmo = load_preset("olmo2")
        gemma = load_preset("gemma3")
        mistral = load_preset("mistral_small31")
        qwen = load_preset("qwen3_dense")
        self.assertEqual({spec.schema_version for spec in (olmo, gemma, mistral, qwen)}, {4})
        self.assertEqual(olmo.block_style, "reordered_output_norm")
        self.assertEqual(gemma.block_style, "pre_and_post_norm")
        self.assertEqual(olmo.attention_schedule, ("global", "global"))
        self.assertEqual(gemma.attention_schedule, ("local",) * 5 + ("global",))
        self.assertEqual((olmo.query_heads, olmo.kv_heads), (4, 4))
        self.assertEqual((gemma.query_heads, gemma.kv_heads), (4, 2))
        self.assertEqual((olmo.qk_norm_axis, olmo.gate_activation), ("projection", "silu"))
        self.assertEqual((gemma.qk_norm_axis, gemma.gate_activation), ("head", "gelu"))
        self.assertEqual((mistral.block_style, mistral.qk_norm_axis, mistral.rope_layout), ("pre_norm", "none", "adjacent"))
        self.assertEqual((qwen.block_style, qwen.qk_norm_axis, qwen.rope_layout), ("pre_norm", "head", "half"))
        self.assertEqual(mistral.query_heads * mistral.head_dim, 24)
        self.assertEqual(qwen.query_heads * qwen.head_dim, 48)
        self.assertFalse(mistral.tie_embeddings)
        self.assertTrue(qwen.tie_embeddings)
        # OLMo 2 1B/7B/32B are untied; Gemma 3 ties. OLMo 2, Gemma 3 and Qwen3 reference code is split-half.
        self.assertFalse(olmo.tie_embeddings)
        self.assertTrue(gemma.tie_embeddings)
        self.assertEqual((olmo.rope_layout, gemma.rope_layout), ("half", "half"))
        self.assertEqual({load_preset(name).rope_layout for name in ("mistral_small31", "deepseek_v3_style")}, {"adjacent"})

    def test_invalid_presets_fail_before_construction(self):
        olmo = load_preset("olmo2")
        with self.assertRaisesRegex(ValueError, "schema"):
            replace(olmo, schema_version=1).validate()
        with self.assertRaisesRegex(ValueError, "head width"):
            replace(olmo, head_dim=7).validate()
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
        with self.assertRaisesRegex(ValueError, "QK norm flag"):
            replace(load_preset("mistral_small31"), qk_norm=True).validate()
        with self.assertRaisesRegex(ValueError, "RoPE"):
            replace(load_preset("qwen3_dense"), rope_layout="adjacent").validate()
        with self.assertRaisesRegex(ValueError, "split-half"):
            replace(olmo, rope_layout="adjacent").validate()
        with self.assertRaisesRegex(ValueError, "split-half"):
            replace(load_preset("gemma3"), rope_layout="adjacent").validate()
        with self.assertRaisesRegex(ValueError, "untied"):
            replace(olmo, tie_embeddings=True).validate()
        with self.assertRaisesRegex(ValueError, "positive"):
            replace(load_preset("mistral_small31"), norm_eps=0).validate()
        with self.assertRaises(ValueError):
            load_preset("unknown_family")
        deepseek = load_preset("deepseek_v3_style")
        with self.assertRaisesRegex(ValueError, "three dense layers"):
            replace(deepseek, layers=3, attention_schedule=("global",) * 3).validate()
        with self.assertRaisesRegex(ValueError, "MoE top-k"):
            replace(deepseek, moe_top_k=5).validate()
        with self.assertRaisesRegex(ValueError, "only belong"):
            replace(olmo, moe_experts=4).validate()
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
        for layout in ("adjacent", "half"):
            rotated = apply_rope(x, 500000.0, layout)
            torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5)
        self.assertFalse(torch.allclose(apply_rope(x, 500000.0, "adjacent"),
                                        apply_rope(x, 500000.0, "half")))
        for layout in ("adjacent", "half"):
            self.assertEqual(apply_rope(x.to(torch.bfloat16), 500000.0, layout).dtype, torch.bfloat16)

        basis = torch.zeros(1, 2, 1, 4)
        basis[0, 1, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        angle = torch.tensor(1.0)
        adjacent = apply_rope(basis, 10000.0, "adjacent")[0, 1, 0]
        half = apply_rope(basis, 10000.0, "half")[0, 1, 0]
        torch.testing.assert_close(adjacent, torch.tensor([angle.cos(), angle.sin(), 0.0, 0.0]))
        torch.testing.assert_close(half, torch.tensor([angle.cos(), 0.0, angle.sin(), 0.0]))

    def test_rope_layouts_are_one_rotation_on_permuted_channels(self):
        width = 8
        x = torch.randn(2, 7, 3, width)
        y = torch.randn(2, 7, 3, width)
        # adjacent pair i is channels (2i, 2i+1); split-half pair i is channels (i, i + width/2)
        perm = torch.cat((torch.arange(0, width, 2), torch.arange(1, width, 2)))
        torch.testing.assert_close(apply_rope(x[..., perm], 10000.0, "half"),
                                   apply_rope(x, 10000.0, "adjacent")[..., perm])
        scores_adjacent = torch.einsum("bthd,bshd->bhts", apply_rope(x, 10000.0, "adjacent"), apply_rope(y, 10000.0, "adjacent"))
        scores_half = torch.einsum("bthd,bshd->bhts", apply_rope(x[..., perm], 10000.0, "half"),
                                   apply_rope(y[..., perm], 10000.0, "half"))
        torch.testing.assert_close(scores_adjacent, scores_half, rtol=1e-5, atol=1e-5)
        # the same weights read under the other layout are a different model
        self.assertGreater((apply_rope(x, 10000.0, "half") - apply_rope(x, 10000.0, "adjacent")).abs().max().item(), 1e-3)

    def test_checkpoint_port_needs_the_matching_rope_layout(self):
        for family in ("olmo2", "gemma3"):
            with self.subTest(family=family):
                half_spec = load_preset(family)
                adjacent_spec = replace(half_spec, rope_layout="adjacent")
                adjacent = GroupedAttention(adjacent_spec, "global")
                half = GroupedAttention(half_spec, "global")
                with torch.no_grad():
                    for parameter in adjacent.parameters():
                        parameter.copy_(torch.randn_like(parameter) * 0.3 + (1.0 if parameter.ndim == 1 else 0.0))
                head = half_spec.head_dim
                perm = torch.cat((torch.arange(0, head, 2), torch.arange(1, head, 2)))

                def channels(heads, head=head, perm=perm):  # per-head channel permutation over a [heads * head] axis
                    return (torch.arange(heads)[:, None] * head + perm[None, :]).flatten()

                x = torch.randn(2, 6, half_spec.width)
                naive = GroupedAttention(half_spec, "global")
                naive.load_state_dict(adjacent.state_dict())  # weights trained for the other layout
                converted = half
                converted.load_state_dict(adjacent.state_dict())
                with torch.no_grad():
                    q_axis, k_axis = channels(half_spec.query_heads), channels(half_spec.kv_heads)
                    converted.q_proj.weight.copy_(adjacent.q_proj.weight[q_axis])
                    converted.k_proj.weight.copy_(adjacent.k_proj.weight[k_axis])
                    if half_spec.qk_norm_axis == "projection":
                        converted.q_norm.weight.copy_(adjacent.q_norm.weight[q_axis])
                        converted.k_norm.weight.copy_(adjacent.k_norm.weight[k_axis])
                    else:
                        converted.q_norm.weight.copy_(adjacent.q_norm.weight[perm])
                        converted.k_norm.weight.copy_(adjacent.k_norm.weight[perm])
                    expected = adjacent(x)
                    torch.testing.assert_close(converted(x), expected, rtol=1e-5, atol=1e-5)
                    self.assertGreater((naive(x) - expected).abs().max().item(), 1e-3)

    def test_matrices_start_at_the_configs_initializer_range(self):
        for name in ("olmo2", "gemma3", "mistral_small31", "qwen3_dense", "deepseek_v3_style"):
            with self.subTest(family=name):
                spec = load_preset(name)
                model = TinyLanguageModel(spec)
                for parameter_name, parameter in model.named_parameters():
                    if parameter.ndim >= 2:
                        self.assertAlmostEqual(parameter.std().item(), 0.02, delta=0.006, msg=parameter_name)
                    else:
                        self.assertTrue(torch.equal(parameter, torch.ones_like(parameter)), parameter_name)
                ids = torch.randint(spec.vocab_size, (2, 8))
                targets = torch.randint(spec.vocab_size, (2, 8))  # not the inputs, which a tied head could favour
                loss = F.cross_entropy(model(ids).reshape(-1, spec.vocab_size), targets.reshape(-1))
                self.assertAlmostEqual(loss.item(), math.log(spec.vocab_size), delta=0.3)

    def test_grouped_attention_matches_per_head_loop_and_torch_sdpa(self):
        # Query head h reads KV head h // groups (repeat_interleave); the scale is 1/sqrt(head_dim).
        for family, kind in (("mistral_small31", "global"), ("qwen3_dense", "global"), ("gemma3", "local")):
            with self.subTest(family=family, kind=kind):
                spec = load_preset(family)
                module = GroupedAttention(spec, kind).double()
                x = torch.randn(2, 7, spec.width, dtype=torch.double)
                batch, tokens, _ = x.shape
                heads, kv_heads, width = module.query_heads, module.kv_heads, module.head_width
                q = module.q_norm(module.q_proj(x).reshape(batch, tokens, heads, width))
                k = module.k_norm(module.k_proj(x).reshape(batch, tokens, kv_heads, width))
                v = module.v_proj(x).reshape(batch, tokens, kv_heads, width).transpose(1, 2)
                q = apply_rope(q, module.rope_base, module.rope_layout).transpose(1, 2)
                k = apply_rope(k, module.rope_base, module.rope_layout).transpose(1, 2)
                allowed = causal_window_mask(tokens, module.window)
                kv_of_head = [head // (heads // kv_heads) for head in range(heads)]
                per_head = []
                for head, kv in enumerate(kv_of_head):
                    scores = q[:, head] @ k[:, kv].transpose(-2, -1) / math.sqrt(width)
                    per_head.append(scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ v[:, kv])
                loop = module.out_proj(torch.stack(per_head, 1).transpose(1, 2).reshape(batch, tokens, -1))
                sdpa = F.scaled_dot_product_attention(q, k[:, kv_of_head], v[:, kv_of_head], attn_mask=allowed)
                sdpa = module.out_proj(sdpa.transpose(1, 2).reshape(batch, tokens, -1))
                torch.testing.assert_close(module(x), loop, rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(module(x), sdpa, rtol=1e-10, atol=1e-10)

    def test_latent_attention_matches_naive_concatenation_with_torch_sdpa(self):
        spec = load_preset("deepseek_v3_style")
        module = LatentAttention(spec).double()
        x = torch.randn(2, 7, spec.width, dtype=torch.double)
        batch, tokens, _ = x.shape
        heads, c, r = module.heads, module.content_dim, module.rope_dim
        q = module.q_up(module.q_norm(module.q_down(x))).reshape(batch, tokens, heads, c + r)
        q_content, q_rope = q.split((c, r), dim=-1)
        latent, k_rope = module.kv_down(x).split((module.kv_rank, r), dim=-1)
        k_content, v = module.kv_up(module.kv_norm(latent)).reshape(batch, tokens, heads, 2 * c).split(c, dim=-1)
        q_rope = apply_rope(q_rope, module.rope_base)
        k_rope = apply_rope(k_rope.unsqueeze(2), module.rope_base).expand(-1, -1, heads, -1)
        query = torch.cat((q_content, q_rope), dim=-1).transpose(1, 2)
        key = torch.cat((k_content, k_rope), dim=-1).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, v.transpose(1, 2), is_causal=True)
        expected = module.out_proj(attended.transpose(1, 2).reshape(batch, tokens, heads * c))
        torch.testing.assert_close(module(x), expected, rtol=1e-10, atol=1e-10)

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
        mistral = GroupedAttention(load_preset("mistral_small31"), "global")
        self.assertIsInstance(mistral.q_norm, torch.nn.Identity)
        self.assertIsInstance(mistral.k_norm, torch.nn.Identity)

    def test_family_specific_gated_mlp_activation(self):
        for family, gate_activation in (("olmo2", lambda x: F.silu(x)),
                                        ("gemma3", lambda x: F.gelu(x, approximate="tanh")),
                                        ("mistral_small31", lambda x: F.silu(x)),
                                        ("qwen3_dense", lambda x: F.silu(x))):
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

    def test_tiny_models_forward_backward_and_causality(self):
        for name in ("olmo2", "gemma3", "mistral_small31", "qwen3_dense"):
            with self.subTest(family=name):
                spec = load_preset(name)
                model = TinyLanguageModel(spec)
                if name == "olmo2":
                    self.assertIsInstance(model.blocks[0].attention_input_norm, torch.nn.Identity)
                else:
                    self.assertIsInstance(model.blocks[0].attention_input_norm, RMSNorm)
                if name in {"mistral_small31", "qwen3_dense"}:
                    self.assertIsInstance(model.blocks[0].attention_output_norm, torch.nn.Identity)
                if name == "mistral_small31":
                    self.assertIsNot(model.embedding.weight, model.lm_head.weight)
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

    def test_deepseek_style_mla_moe_shapes_gradients_and_causality(self):
        spec = load_preset("deepseek_v3_style")
        model = TinyLanguageModel(spec)
        self.assertTrue(all(isinstance(block.attention, LatentAttention) for block in model.blocks))
        self.assertTrue(all(not isinstance(block.ffn, RoutedMoE) for block in model.blocks[:3]))
        self.assertIsInstance(model.blocks[3].ffn, RoutedMoE)
        ids = torch.randint(spec.vocab_size, (2, 6))
        logits = model(ids)
        self.assertEqual(tuple(logits.shape), (2, 6, spec.vocab_size))
        self.assertTrue(torch.isfinite(logits).all().item())
        loss = F.cross_entropy(logits.reshape(-1, spec.vocab_size), ids.reshape(-1))
        loss.backward()
        attention = model.blocks[0].attention
        moe = model.blocks[3].ffn
        for projection in (attention.q_down, attention.q_up, attention.kv_down,
                           attention.kv_up, moe.router):
            self.assertIsNotNone(projection.weight.grad)
            self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(sum(expert.gate.weight.grad is not None
                               for expert in moe.routed_experts), 0)
        self.assertIsNotNone(moe.shared_experts[0].gate.weight.grad)
        weights, indices = moe.route(torch.randn(7, spec.width))
        self.assertEqual(tuple(weights.shape), (7, spec.moe_top_k))
        self.assertEqual(tuple(indices.shape), (7, spec.moe_top_k))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(7))
        altered = ids.clone()
        altered[:, -1] = (altered[:, -1] + 1) % spec.vocab_size
        with torch.no_grad():
            torch.testing.assert_close(model(ids)[:, :-1], model(altered)[:, :-1], rtol=0, atol=1e-6)
        self.assertEqual(cache_bytes(spec, 1, 32, 4), 4 * 32 * (12 + 4) * 4)

    def test_deepseek_style_routes_top_k_and_runs_optimizer_step(self):
        spec = load_preset("deepseek_v3_style")
        model = TinyLanguageModel(spec)
        ids = torch.randint(spec.vocab_size, (2, 7))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.blocks[3].ffn.router.weight.detach().clone()
        logits = model(ids[:, :-1])
        F.cross_entropy(logits.reshape(-1, spec.vocab_size), ids[:, 1:].reshape(-1)).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.blocks[3].ffn.router.weight))


if __name__ == "__main__":
    unittest.main()
