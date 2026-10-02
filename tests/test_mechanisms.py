import unittest

import torch

from atlas.case_study import CASE_OPERATORS, run_case, run_operator
from atlas.mechanisms import (HybridState, biased_topk_route, chunked_causal_mask, delta_step,
                              key_as_value_attention, selective_ssm_step,
                              sink_attention, sparse_causal_attention)
from atlas.model import causal_window_mask


class MechanismTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_selection_bias_never_enters_combination_weights(self):
        logits = torch.logit(torch.tensor([[0.6, 0.5, 0.4]]))
        w, selected = biased_topk_route(logits, 2, torch.tensor([0.0, 0.2, 0.0]))
        self.assertEqual(selected.tolist(), [[1, 0]])
        torch.testing.assert_close(w, torch.tensor([[5 / 11, 6 / 11]]))
        w2, _ = biased_topk_route(logits, 2, torch.tensor([0.0, 0.2, -1.0]))
        torch.testing.assert_close(w, w2)
        _, ties = biased_topk_route(torch.zeros(2, 4), 2)
        self.assertEqual(ties.tolist(), [[0, 1], [0, 1]])

    def test_unnormalized_gates_keep_the_selected_score_and_train_a_top1_router(self):
        # Llama 4: sigmoid of the selected logit is the gate. Grok-1 code: full-softmax probability.
        logits = torch.randn(5, 16, requires_grad=True)
        weights, _ = biased_topk_route(logits, 1, None, "sigmoid", normalize=False)
        torch.testing.assert_close(weights.squeeze(-1), logits.max(-1).values.sigmoid())
        weights.sum().backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        normalized = torch.randn(5, 16, requires_grad=True)
        unit, _ = biased_topk_route(normalized, 1)
        self.assertTrue(torch.equal(unit, torch.ones_like(unit)))
        unit.sum().backward()
        self.assertLess(normalized.grad.abs().max().item(), 1e-6)  # DeepSeek-style top-1 gate carries no signal
        scores = torch.randn(4, 8)
        gates, picks = biased_topk_route(scores, 2, None, "softmax", normalize=False)
        torch.testing.assert_close(gates, scores.softmax(-1).gather(-1, picks))
        self.assertTrue((gates.sum(-1) < 1).all())
        renormalized, _ = biased_topk_route(scores, 2, None, "softmax")
        torch.testing.assert_close(renormalized.sum(-1), torch.ones(4))

    def test_chunked_mask_is_block_local_and_not_a_sliding_window(self):
        mask = chunked_causal_mask(10, 4)
        self.assertEqual([mask[row].nonzero().flatten().tolist() for row in (0, 4, 5, 7, 9)],
                         [[0], [4], [4, 5], [4, 5, 6, 7], [8, 9]])
        self.assertEqual(causal_window_mask(10, 4)[4].nonzero().flatten().tolist(), [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            chunked_causal_mask(10, 0)

    def test_published_nope_schedules_replace_the_toy_for_llama4_and_smollm3(self):
        smollm3 = run_case(57)["checks"]["position"]
        self.assertEqual(smollm3, {"layers": 36, "interval": 4, "nope_zero_based_layers": list(range(3, 36, 4))})
        self.assertEqual(len(smollm3["nope_zero_based_layers"]), 9)
        llama4 = run_case(55)["checks"]["position"]
        self.assertEqual((llama4["layers"], llama4["interval"]), (48, 4))
        self.assertEqual(llama4["nope_zero_based_layers"], list(range(3, 48, 4)))
        self.assertEqual(run_operator("position"),
                         {"layers": 10, "interval": 3, "nope_zero_based_layers": [2, 5, 8]})  # toy default
        self.assertEqual(CASE_OPERATORS[55], ("top1_scaled_route", "position", "chunk"))
        self.assertEqual(CASE_OPERATORS[60], ("gqa", "softmax_all_route"))

    def test_sparse_all_keys_equals_dense_with_gradients_and_causality(self):
        q, k, v = (torch.randn(2, 2, 7, 4, requires_grad=True) for _ in range(3))
        index = torch.randn(2, 2, 7, 7)
        out, selected = sparse_causal_attention(q, k, v, index, 7)
        scores = q @ k.transpose(-2, -1) / 2
        dense = scores.masked_fill(~causal_window_mask(7, None), -torch.inf).softmax(-1) @ v
        torch.testing.assert_close(out, dense, atol=1e-6, rtol=1e-5)
        self.assertTrue((selected <= torch.arange(7)[None, None, :, None]).all())
        out.square().mean().backward()
        for tensor in (q, k, v):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        changed = v.detach().clone()
        changed[:, :, -1] += 100
        perturbed, _ = sparse_causal_attention(q, k, changed, index, 2)
        baseline, _ = sparse_causal_attention(q, k, v.detach(), index, 2)
        torch.testing.assert_close(perturbed[:, :, :-1], baseline[:, :, :-1], atol=0, rtol=0)

    def test_kda_decay_order_and_independent_correction_gate(self):
        state = torch.tensor([[[2.0], [4.0]]])
        k, v = torch.tensor([[1.0, 1.0]]), torch.tensor([[3.0]])
        alpha, beta = torch.tensor([[0.5, 0.25]]), torch.tensor([0.5])
        torch.testing.assert_close(delta_step(state, k, v, alpha, beta), torch.tensor([[[1.5], [1.5]]]))
        no_correction = delta_step(state, k, v, alpha, torch.zeros(1))
        torch.testing.assert_close(no_correction, torch.tensor([[[1.0], [1.0]]]))
        wrong = alpha[..., :, None] * (state + beta[..., None, None] * k[..., :, None] * (v - (k[..., :, None] * state).sum(-2))[..., None, :])
        self.assertFalse(torch.allclose(wrong, delta_step(state, k, v, alpha, beta)))

    def test_sink_mass_decreases_and_gradients_are_finite(self):
        q, k, v = (torch.randn(2, 2, 5, 4, requires_grad=True) for _ in range(3))
        sink = torch.zeros(2, requires_grad=True)
        out, mass = sink_attention(q, k, v, sink, 3)
        _, higher_mass = sink_attention(q, k, v, sink + 2, 3)
        self.assertTrue((higher_mass < mass).all())
        self.assertTrue((mass > 0).all() and (mass < 1).all())
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(sink.grad).all())

    def test_key_as_value_step_parity_for_every_prefix(self):
        q, kv = torch.randn(2, 2, 12, 4), torch.randn(2, 2, 12, 4)
        all_outputs = key_as_value_attention(q, kv)
        for length in range(1, 13):
            prefix = key_as_value_attention(q[:, :, :length], kv[:, :, :length])
            scores = q[:, :, length - 1:length] @ kv[:, :, :length].transpose(-2, -1) / 2
            decoded = scores.softmax(-1) @ kv[:, :, :length]
            torch.testing.assert_close(prefix[:, :, -1:], decoded, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(prefix, all_outputs[:, :, :length], atol=1e-6, rtol=1e-5)
        random_values = torch.randn_like(kv)
        alternate = (q @ kv.transpose(-2, -1) / 2).masked_fill(~causal_window_mask(12, None), -torch.inf).softmax(-1) @ random_values
        self.assertFalse(torch.allclose(alternate, all_outputs))

    def test_selective_ssm_zero_decay_forgets_prefix(self):
        s = torch.randn(2, 2, 3, 4)
        x, b, c = torch.randn(2, 2, 4), torch.randn(2, 2, 3), torch.randn(2, 2, 3)
        y, next_state = selective_ssm_step(s, x, torch.zeros(2, 2), b, c)
        y2, next_state2 = selective_ssm_step(s + 10, x, torch.zeros(2, 2), b, c)
        torch.testing.assert_close(y, y2, atol=0, rtol=0)
        torch.testing.assert_close(next_state, next_state2, atol=0, rtol=0)

    def test_hybrid_state_restores_and_reorders_all_state_kinds(self):
        s = HybridState(torch.randn(2, 3, 4), torch.randn(2, 2, 5, 4), torch.randn(2, 2, 5, 4), 5)
        restored = HybridState.restore(s.snapshot())
        for name in ("recurrent", "keys", "values"):
            torch.testing.assert_close(getattr(restored, name), getattr(s, name))
            self.assertNotEqual(getattr(restored, name).data_ptr(), getattr(s, name).data_ptr())
        reordered = s.reorder(torch.tensor([1, 0, 1]))
        self.assertEqual(reordered.recurrent.shape[0], 3)
        torch.testing.assert_close(reordered.values[2], s.values[1])
        bad = s.snapshot()
        bad["tokens"] = 6
        with self.assertRaises(ValueError):
            HybridState.restore(bad)

    def test_all_case_studies_have_executable_constituent_checks(self):
        self.assertEqual(set(CASE_OPERATORS), set(range(51, 74)))
        for lesson in range(51, 74):
            with self.subTest(lesson=lesson):
                result = run_case(lesson)
                self.assertEqual(result["scope"], "constituent_operator_checks_not_full_family_port")
                self.assertTrue(result["checks"])


if __name__ == "__main__":
    unittest.main()
