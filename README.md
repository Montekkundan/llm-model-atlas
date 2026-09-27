# Model atlas: a tiny architecture-switching lab

This is a standalone **student project starter** for the model-family case studies in lectures 51–73. It shows how one `TinyLanguageModel` and one synthetic training loop can select two distinct, validated text-path mechanisms by preset. The presets are scaled teaching designs, **not** published-model configurations, pretrained weights, or checkpoint-compatible reproductions.

## What runs today

| Preset | Mechanisms actually executed | Source | Deliberate simplifications |
| --- | --- | --- | --- |
| `olmo2` | Dense MHA, QK RMSNorm, adjacent-pair RoPE, output-side/reordered RMSNorm, SwiGLU, full causal attention | [OLMo 2 report, §§2.1 and 3.3.2](https://arxiv.org/html/2501.00656) | Tiny 32-wide/two-layer text model; toy 64-token vocabulary; no OLMo tokenizer, official weights, z-loss, real data, or training recipe. This preset represents an MHA-sized OLMo 2 variant, not the later 32B GQA scale. |
| `gemma3` | GQA, QK RMSNorm, adjacent-pair RoPE, pre-and-post RMSNorm, a five-local/one-global attention schedule, distinct local/global RoPE bases | [Gemma 3 report, §2](https://arxiv.org/html/2503.19786) | Tiny 32-wide/six-layer text model; four-token local window instead of the report's 1,024; toy vocabulary; no vision encoder, real tokenizer, distillation, official weights, or full training recipe. |

The OLMo 2 paper describes output-side/reordered normalization and QK norm; Gemma 3 describes GQA, pre/post norms, a 5:1 local/global pattern, 1,024-token local windows, and different local/global RoPE bases. The preset files name those sources and carry `schema_version=1`. Their validation rejects unsupported families and illegal head/schedule combinations. The only currently implemented family slugs are `olmo2` and `gemma3`; the [lesson map](LESSON_MAP.md) marks the other 21 family case studies unimplemented.

Both tiny paths intentionally tie embedding and output weights, omit biases in the projections, use a tiny synthetic token vocabulary, and perform full-sequence attention. These choices make a compact comparison, but they cannot be loaded from official checkpoints or used to claim published-model parity. The local window is enforced in a dense mask: this checks visibility, **not** the memory or speed of an optimized sliding-window kernel.

## Run on CPU

With Python 3.10+ and an installed PyTorch 2.x:

```bash
python -m unittest discover -s tests -v
python -m atlas.demo olmo2 --steps 2
python -m atlas.demo gemma3 --steps 2
```

The demo does two updates on one fixed batch of random token IDs. Its printed loss shows that forward, backward, and optimizer steps run; it is **not** language-model pretraining or a quality metric. `cache_bytes` reports an **analytical raw KV capacity** for a hypothetical rolling cache. The executable model does not create or decode from a KV cache, so that printed number is not observed allocation.

`presets/*.json` are the versioned student-editable entry points. To compare a mechanism, copy a preset under a new name, keep `schema_version`, record the source revision and deliberate simplifications, and update the registry/validation only after the corresponding operator and tests exist. Changing a JSON `family` string alone cannot implement MLA, MoE, recurrent state, sparse attention, or a new tokenizer. Switching families begins a new random-weight run; weight transfer requires a separate documented conversion.

## What the tests establish

The tests check validated schema, distinct block styles and attention schedules, causal/sliding masks, a real local-versus-global forward difference, RoPE norm preservation, finite forward/backward passes for both tiny models, nonzero Q/K/V gradients, prefix causality under a future-token change, and raw KV accounting. They do **not** establish model quality, exact published architecture equivalence, cache decode parity, long-context performance, GPU throughput, or distributed correctness. A useful next lab adds an actual cache interface and full-versus-incremental parity per family before benchmarking memory or latency.

### Verification record

On 27 September 2026, the seven unit tests passed with Python 3.12 and cached PyTorch 2.13.0 on CPU. Each demo completed two synthetic optimizer steps. `olmo2` printed losses 14.5915 and 11.1278, `trainable_parameters=22720`, and `analytical_raw_kv_bytes_at_32_tokens=16384`; `gemma3` printed losses 12.0554 and 6.7070, `trainable_parameters=58240`, and `analytical_raw_kv_bytes_at_32_tokens=6656`. Those counts and losses describe only these tiny configurations and fixed random batch. No GPU profiling, dataset evaluation, or official checkpoint verification was performed.

## Primary references

- Team OLMo. [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656), especially §§2.1 and 3.3.2 for architecture choices.
- Gemma Team. [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786), especially §2 for GQA and local/global schedule.
- Ainslie et al. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), for KV-head sharing semantics.
- Su et al. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), for rotary query/key positions.
