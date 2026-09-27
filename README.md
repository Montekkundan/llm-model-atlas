# Model atlas: a tiny architecture-switching lab

This is a standalone **student project starter** for the model-family case studies in lectures 51–73. It shows how one `TinyLanguageModel` and one synthetic training loop select five validated text-path designs by preset. The presets are scaled teaching designs, **not** published-model configurations, pretrained weights, or checkpoint-compatible reproductions.

## Find a case study

The [lesson map](LESSON_MAP.md) is the short reading guide. The [machine-readable catalog](catalog/model_families.json) has one card for each of the 23 proposed lessons, in lesson order, with the title and slug, mechanisms to investigate, source URLs, and `implementationStatus`. A card marked `documented_pending` is a research/implementation target, not an executable model. Lesson 72 is a survey of releases, not a single model family. Lessons 51–54 and 56 are marked `runnable_tiny_text_path`; Qwen3's runnable path covers its dense model only, and the DeepSeek path covers V3-style structure only, not R1 training.

To look up one card without installing another package:

```bash
python -c 'import json; c=json.load(open("catalog/model_families.json")); print(next(x for x in c["cards"] if x["lessonNumber"] == 62))'
```

## What runs today

| Preset | Mechanisms actually executed | Source | Deliberate simplifications |
| --- | --- | --- | --- |
| `deepseek_v3_style` | Low-rank query and KV projections, decoupled RoPE key, full-sequence MLA reconstruction, three dense FFN layers then sigmoid top-2 routed experts plus a shared expert | [DeepSeek-V3 report, §§2.1 and 4.2](https://arxiv.org/html/2412.19437), [official inference code](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py) | Tiny 32-wide/four-layer model. No cached latent decode, weight absorption, load-balancing bias, node-limited routing, MTP, YaRN extension, FP8, distributed training, official weights, or R1 post-training. |
| `olmo2` | Dense MHA, full-projection QK RMSNorm, adjacent-pair RoPE, output-side/reordered RMSNorm, SwiGLU, full causal attention | [OLMo 2 report, §§2.1 and 3.3.2](https://arxiv.org/html/2501.00656) | Tiny 32-wide/two-layer text model; toy 64-token vocabulary; no OLMo tokenizer, official weights, z-loss, real data, or training recipe. This preset represents an MHA-sized OLMo 2 variant, not the later 32B GQA scale. |
| `gemma3` | GQA, headwise QK RMSNorm, adjacent-pair RoPE, pre-and-post RMSNorm, GeGLU, a five-local/one-global attention schedule, distinct local/global RoPE bases | [Gemma 3 report, §2](https://arxiv.org/html/2503.19786) | Tiny 32-wide/six-layer text model; four-token local window instead of the report's 1,024; toy vocabulary; no vision encoder, real tokenizer, distillation, official weights, or full training recipe. |
| `mistral_small31` | Global GQA without QK norm, adjacent-pair RoPE, pre-norm SwiGLU, untied output | [Mistral Small 3.1 config](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/config.json), [Mistral inference code](https://github.com/mistralai/mistral-inference/tree/main/src/mistral_inference) | Tiny text-only path; no vision, Tekken tokenizer, official dimensions or weights, or optimized cache. |
| `qwen3_dense` | Global GQA, headwise QK RMSNorm, split-half RoPE, pre-norm SwiGLU, tied output | [Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json), [Qwen3 model code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modular_qwen3.py) | Tiny **dense-only** path; no Qwen3 MoE routing, tokenizer, official head dimensions or weights, or cache. |

The DeepSeek-V3 paper specifies low-rank KV and query paths, a decoupled rotary key, sigmoid top-k expert affinity, and three initial dense FFN layers. The OLMo 2 paper describes output-side/reordered normalization and QK norm; Gemma 3 describes GQA, pre/post norms, a 5:1 local/global pattern, 1,024-token local windows, and different local/global RoPE bases. Mistral Small 3.1's published text configuration has no sliding window, while Qwen3-0.6B has headwise QK norm. Mistral and Qwen specify head dimensions independently of hidden width, so their tiny presets also make the attention projection narrower and wider than the residual stream, respectively. The preset files carry `schema_version=4` with explicit head dimension, QK norm axis, RoPE layout, normalization epsilon, output-weight tying, gate activation, and optional MLA/MoE dimensions. Earlier schemas are rejected. Validation rejects unsupported families and incompatible mechanisms. The [lesson map](LESSON_MAP.md) marks the other 18 case studies or surveys pending.

All five paths use a tiny synthetic token vocabulary and full-sequence attention. Embedding/output weights are untied in the Mistral and DeepSeek-style presets. These choices make a compact comparison, but they cannot be loaded from official checkpoints or used to claim published-model parity. The local window is enforced in a dense mask: this checks visibility, **not** the memory or speed of an optimized sliding-window kernel.

## Run on CPU

With Python 3.10+ and an installed PyTorch 2.x:

```bash
python -m unittest discover -s tests -v
python -m atlas.demo deepseek_v3_style --steps 2
python -m atlas.demo olmo2 --steps 2
python -m atlas.demo gemma3 --steps 2
python -m atlas.demo mistral_small31 --steps 2
python -m atlas.demo qwen3_dense --steps 2
```

The demo does two updates on one fixed batch of random token IDs. Its printed loss shows that forward, backward, and optimizer steps run; it is **not** language-model pretraining or a quality metric. `cache_bytes` reports an **analytical raw KV capacity** for a hypothetical rolling cache. The executable model does not create or decode from a KV cache, so that printed number is not observed allocation.

`presets/*.json` are the versioned student-editable entry points. To compare a mechanism, copy a preset under a new name, keep `schema_version`, record the source revision and deliberate simplifications, and update the registry/validation only after the corresponding operator and tests exist. Changing a JSON `family` string alone cannot implement recurrent state, sparse attention, or a new tokenizer. Switching families begins a new random-weight run; weight transfer requires a separate documented conversion.

## What the tests establish

The tests check validated schema, distinct block styles and attention schedules, causal/sliding masks, a real local-versus-global forward difference, both RoPE layouts, finite forward/backward passes for all five tiny models, MLA projection and router gradients, normalized top-k weights, prefix causality under a future-token change, and analytical KV accounting. They do **not** establish model quality, exact published architecture equivalence, cache decode parity, long-context performance, GPU throughput, or distributed correctness. A useful next lab adds an actual cache interface and full-versus-incremental parity per family before benchmarking memory or latency.

### Verification record

On 27 September 2026, the unit tests and all five demos were run with PyTorch 2.11.0 on CPU. Each demo completed two synthetic optimizer steps. These outputs are smoke tests, **not comparable model-quality scores**. No GPU profiling, dataset evaluation, or official checkpoint verification was performed.

## Primary references

- DeepSeek-AI. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), especially §§2.1 and 4.2 for MLA and expert routing, and [official inference code](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py) for the naive attention path.
- Team OLMo. [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656), especially §§2.1 and 3.3.2 for architecture choices.
- AllenAI. [OLMo model code](https://github.com/allenai/OLMo/blob/main/olmo/model.py), for full-projection Q/K normalization before head reshaping.
- Gemma Team. [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786), especially §2 for GQA and local/global schedule.
- Google. [Gemma PyTorch model code](https://github.com/google/gemma_pytorch/blob/main/gemma/model.py), for headwise Q/K normalization and the tanh-approximate GELU gate.
- Ainslie et al. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), for KV-head sharing semantics.
- Su et al. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), for rotary query/key positions.
- Mistral AI. [Mistral Small 3.1 configuration](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/config.json) and [inference implementation](https://github.com/mistralai/mistral-inference/tree/main/src/mistral_inference), for the text decoder path.
- Qwen. [Qwen3-0.6B configuration](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json) and [Qwen3 PyTorch implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modular_qwen3.py), for dense GQA and headwise QK normalization.
