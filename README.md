# Model atlas: a tiny architecture-switching lab

This is a standalone **student project starter** for the model-family case studies in lectures 51–73. It shows how one `TinyLanguageModel` and one synthetic training loop select five validated text-path designs by preset. The presets are scaled teaching designs, **not** published-model configurations, pretrained weights, or checkpoint-compatible reproductions.

## What is runnable, and what is not

| Lessons | State | What you can run |
| --- | --- | --- |
| 51, 52, 53, 54, 56 | **5 tiny text-path presets** (`deepseek_v3_style`, `olmo2`, `gemma3`, `mistral_small31`, `qwen3_dense`) | `python -m atlas.demo PRESET` trains a toy model for two synthetic steps; `python -m atlas.case_study N` runs the lesson's operator checks |
| 55, 57 to 73 (18 lessons) | **Reference operators only** | `python -m atlas.case_study N` runs constituent operators (routing, masks, recurrent updates) on small random tensors; there is no model graph, tokenizer or weights for these families |

The repository holds **no published model dimensions**: no layer counts, hidden sizes,
head counts, expert counts or parameter totals are stored or checked, apart from the
structural constants that three operator checks apply to toy tensors (Llama 4 Scout: 16
routed experts, top-1, 48 layers with a NoPE layer every fourth; Grok-2: 8 experts,
top-2; SmolLM3: 36 layers, NoPE every fourth). The presets are 32-wide toys that copy a
few published choices (RoPE bases, normalisation epsilon, Gemma 3's five-local/one-global
pattern), and the catalog stores source URLs and mechanism names, not numbers. A lesson
that asks you to audit source-reported dimensions or to cost a family's specification
therefore has to read those numbers from the cited configuration and compute them
yourself; nothing here checks them.

Several lessons print **identical operator results** because they run the same
operators: lessons 51, 58 and 67 (`mla`, `bias_route`), 53 and 63 (`window`), and
66 and 71 (`sparse`, `mla`) differ only in the `lesson` field of the JSON. A matching
output is not evidence that the families are alike; `tests/test_catalog.py` pins this
list so it stays accurate.

## Find a case study

The [lesson map](LESSON_MAP.md) is the short reading guide. The [machine-readable catalog](catalog/model_families.json) has one card for each of the 23 proposed lessons, in lesson order, with the title and slug, mechanisms to investigate, source URLs, and `implementationStatus`. A card marked `runnable_reference_mechanisms` runs constituent operators and numerical contracts, not a complete family model. Lesson 72 is a survey of releases, not a single model family. Lessons 51–54 and 56 are marked `runnable_tiny_text_path`; Qwen3's runnable path covers its dense model only, and the DeepSeek path covers V3-style structure only, not R1 training.

To look up one card without installing another package:

```bash
python -c 'import json; c=json.load(open("catalog/model_families.json")); print(next(x for x in c["cards"] if x["lessonNumber"] == 62))'
```

## What runs today

| Preset | Mechanisms actually executed | Source | Deliberate simplifications |
| --- | --- | --- | --- |
| `deepseek_v3_style` | Low-rank query and KV projections, decoupled RoPE key, full-sequence MLA reconstruction, three dense FFN layers then sigmoid top-2 routed experts plus a shared expert | [DeepSeek-V3 report, §§2.1 and 4.2](https://arxiv.org/html/2412.19437), [official inference code](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py) | Tiny 32-wide/four-layer model. No cached latent decode, weight absorption, load-balancing bias, node-limited routing, MTP, YaRN extension, FP8, distributed training, official weights, or R1 post-training. |
| `olmo2` | Dense MHA, full-projection QK RMSNorm, split-half RoPE, output-side/reordered RMSNorm, SwiGLU, untied output, full causal attention | [OLMo 2 report, §§2.1 and 3.3.2](https://arxiv.org/html/2501.00656) | Tiny 32-wide/two-layer text model; toy 64-token vocabulary; no OLMo tokenizer, official weights, z-loss, real data, or training recipe. This preset represents an MHA-sized OLMo 2 variant, not the later 32B GQA scale. |
| `gemma3` | GQA, headwise QK RMSNorm, split-half RoPE, pre-and-post RMSNorm, GeGLU, a five-local/one-global attention schedule, distinct local/global RoPE bases | [Gemma 3 report, §2](https://arxiv.org/html/2503.19786) | Tiny 32-wide/six-layer text model; four-token local window instead of the report's 1,024; toy vocabulary; no vision encoder, real tokenizer, distillation, official weights, or full training recipe. |
| `mistral_small31` | Global GQA without QK norm, adjacent-pair RoPE, pre-norm SwiGLU, untied output | [Mistral Small 3.1 config](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/68faf511d618ef198fef186659617cfd2eb8e33a/config.json), [Mistral inference code](https://github.com/mistralai/mistral-inference/tree/9eaeb91c17450e09021b6065a1d5cc69876507c8/src/mistral_inference) | Tiny text-only path; no vision, Tekken tokenizer, official dimensions or weights, or optimized cache. |
| `qwen3_dense` | Global GQA, headwise QK RMSNorm, split-half RoPE, pre-norm SwiGLU, tied output | [Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json), [Qwen3 model code](https://github.com/huggingface/transformers/blob/35dff0957a99d50eaf85d7852a96fd29e59052d6/src/transformers/models/qwen3/modular_qwen3.py) | Tiny **dense-only** path; no Qwen3 MoE routing, tokenizer, official head dimensions or weights, or cache. |

The DeepSeek-V3 paper specifies low-rank KV and query paths, a decoupled rotary key, sigmoid top-k expert affinity, and three initial dense FFN layers. The OLMo 2 paper describes output-side/reordered normalization and QK norm; Gemma 3 describes GQA, pre/post norms, a 5:1 local/global pattern, 1,024-token local windows, and different local/global RoPE bases. Mistral Small 3.1's published text configuration has no sliding window, while Qwen3-0.6B has headwise QK norm. Mistral and Qwen specify head dimensions independently of hidden width, so their tiny presets also make the attention projection narrower and wider than the residual stream, respectively. The preset files carry `schema_version=4` with explicit head dimension, QK norm axis, RoPE layout, normalization epsilon, output-weight tying, gate activation, and optional MLA/MoE dimensions. Earlier schemas are rejected. Validation rejects unsupported families and incompatible mechanisms. The other 18 case studies or surveys now have executable constituent-operator checks; their complete family graphs remain porting work.

All five paths use a tiny synthetic token vocabulary and full-sequence attention. Embedding/output weights are untied in the OLMo 2, Mistral and DeepSeek-style presets (the OLMo 2 1B, 7B and 32B configurations all set `tie_word_embeddings=false`). These choices make a compact comparison, but they cannot be loaded from official checkpoints or used to claim published-model parity. The local window is enforced in a dense mask: this checks visibility, **not** the memory or speed of an optimized sliding-window kernel.

**RoPE pairing.** The reference code of OLMo 2 (`rotate_half`), Gemma 3 (`torch.chunk` into halves) and Qwen3 pairs channel `i` with `i + D/2` (split-half), so the `olmo2`, `gemma3` and `qwen3_dense` presets use `rope_layout="half"`. Mistral's native inference code and DeepSeek-V3's official inference code rotate adjacent channels `(2i, 2i+1)`, so `mistral_small31` and `deepseek_v3_style` use `"adjacent"`. The two layouts are the same rotation applied to permuted channels, so a model trained from scratch behaves identically under either; a ported checkpoint does not, because its Q/K weights were trained for one layout. `tests/test_atlas.py` shows both: with the Q/K projection rows permuted per head the two layouts give identical attention, and without the permutation they do not.

## Case-study operator checks

`python -m atlas.case_study 62` runs the gated Delta recurrence and hybrid-state contracts. Every number from 51 to 73 is accepted; the JSON result names the operations and measured assertions. Read `atlas/mechanisms.py` alongside `atlas/case_study.py`. The latter maps a family lesson to operations; it does not create a family-compatible checkpoint.

The references cover NoPE/RoPE schedules, sigmoid/softmax top-k routing with selection-only bias (normalised gates, as in DeepSeek-V3) and the two unnormalised gates that appear in the Llama 4 text router (sigmoid of the selected logit, top-1 of 16 with a shared expert) and the Grok-1 code (full softmax, top-2 of 8, `normalize=False`), a Llama 4 style chunked causal mask, causal sparse gather, probability-absorbing attention sinks, key-as-value sharing, scalar/vector gated Delta updates, a discretized per-head selective SSM update, and hybrid-state fork/reorder/snapshot. Sparse `k=T` equals dense attention; recurrent split/resume and key-as-value prefix/step outputs agree. The SSM reference supplies the algebraic state update, not Mamba-2 convolution, parameter generation, discretization or fused scan kernels. Recurrent and attention tensors retain distinct layouts even when their lifecycle interface is shared.

Dense masking and Python loops are intentional CPU references. They establish shape, causality, gradients and state contracts, not sparse-kernel speed, real cache allocation, pretrained quality, MTP, multimodal support or the complete source training recipe.

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

Every matrix and embedding is drawn from N(0, 0.02^2), the `initializer_range` of the OLMo 2, Gemma 3, Mistral Small 3.1, Qwen3 and DeepSeek-V3 configurations, so the first loss is close to ln 64 = 4.16. The demo does two updates on one fixed batch of random token IDs. Its printed loss shows that forward, backward, and optimizer steps run; it is **not** language-model pretraining or a quality metric. `cache_bytes` reports an **analytical raw KV capacity** for a hypothetical rolling cache. The executable model does not create or decode from a KV cache, so that printed number is not observed allocation.

`presets/*.json` are the versioned student-editable entry points. To compare a mechanism, copy a preset under a new name, keep `schema_version`, record the source revision and deliberate simplifications, and update the registry/validation only after the corresponding operator and tests exist. Changing a JSON `family` string alone cannot implement recurrent state, sparse attention, or a new tokenizer. Switching families begins a new random-weight run; weight transfer requires a separate documented conversion.

## What the tests establish

The tests check validated schema, distinct block styles and attention schedules, causal/sliding masks, a real local-versus-global forward difference, both RoPE layouts, finite forward/backward passes for all five tiny models, MLA projection and router gradients, normalized top-k weights, prefix causality under a future-token change, and analytical KV accounting. The operator tests additionally establish sparse all-keys/dense parity, biased-selection versus unbiased-gating semantics, sink mass, KDA update order, SSM forgetting, hybrid state ownership, and key-as-value prefix/step parity. The mechanism tests also cover unnormalised top-1 (Llama 4) and top-2 (Grok-1 code) gates, the chunked causal mask, and the published Llama 4 Scout and SmolLM3 NoPE schedules. The model tests compare grouped and latent attention with per-head loops and torch SDPA (scale and KV-head grouping), show that the adjacent and split-half RoPE layouts agree up to a channel permutation (and disagree without it), and check the N(0, 0.02^2) initialisation. They do **not** establish model quality, exact published architecture equivalence, cached decoding of the five complete tiny models, long-context performance, GPU throughput, or distributed correctness.

### Verification record

On 29 September 2026, the unit tests, all 23 operator cases and all five demos were run with PyTorch 2.9.1 on CPU. On 2 October 2026, after the OLMo 2 untying, the split-half RoPE layout for OLMo 2 and Gemma 3, the N(0, 0.02^2) initialisation and the Llama 4, SmolLM3 and Grok operator changes, the 31 unit tests, all 23 operator cases and all five demos were re-run with PyTorch 2.14.1 on CPU; the demo losses, the `olmo2` parameter count (22,816 to 24,864) and the case outputs for lessons 55, 57, 60 and 72 differ from the 29 September run for that reason. Each demo completed two synthetic optimizer steps. These outputs are smoke tests, **not comparable model-quality scores**. No GPU profiling, dataset evaluation, or official checkpoint verification was performed.

## Primary references

- DeepSeek-AI. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), especially §§2.1 and 4.2 for MLA and expert routing, and [official inference code](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py) for the naive attention path.
- Team OLMo. [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656), especially §§2.1 and 3.3.2 for architecture choices.
- AllenAI. [OLMo model code](https://github.com/allenai/OLMo/blob/090253dac6688f2532509daa7aa2eb5fae50e956/olmo/model.py), for full-projection Q/K normalization before head reshaping.
- Gemma Team. [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786), especially §2 for GQA and local/global schedule.
- Google. [Gemma PyTorch model code](https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py), for headwise Q/K normalization and the tanh-approximate GELU gate.
- Ainslie et al. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), for KV-head sharing semantics.
- Su et al. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), for rotary query/key positions.
- Mistral AI. [Mistral Small 3.1 configuration](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/68faf511d618ef198fef186659617cfd2eb8e33a/config.json) and [inference implementation](https://github.com/mistralai/mistral-inference/tree/9eaeb91c17450e09021b6065a1d5cc69876507c8/src/mistral_inference), for the text decoder path.
- Qwen. [Qwen3-0.6B configuration](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json) and [Qwen3 PyTorch implementation](https://github.com/huggingface/transformers/blob/35dff0957a99d50eaf85d7852a96fd29e59052d6/src/transformers/models/qwen3/modular_qwen3.py), for dense GQA and headwise QK normalization.
