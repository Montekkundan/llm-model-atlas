# Lectures 51–73: architecture case-study project map

This map follows the proposed `CURRICULUM_V2_PLAN.md` order. **Tiny text path** means only the listed teaching mechanisms are executable here; it does not mean the published model is reproduced. **Not implemented** means there is no preset or operator for that case study in this starter. It is not a claim that a cited model is impossible or undocumented. Primary sources beyond the two runnable families require case-by-case curation before adding presets.

| # | Case study | Project state | Next required work |
| --- | --- | --- | --- |
| 51 | DeepSeek V3 and R1 | Not implemented | MLA, fine-grained MoE, MTP, and separate R1 post-training account. [V3 report](https://arxiv.org/abs/2412.19437). |
| 52 | OLMo 2 | **Tiny text path** | Official tokenizer/config audit, z-loss/training recipe, cache parity and evaluation. [Report](https://arxiv.org/abs/2501.00656). |
| 53 | Gemma 3 | **Tiny text path** | Vision path, real tokenizer/config, optimized rolling cache, official local span and evaluation. [Report](https://arxiv.org/abs/2503.19786). |
| 54 | Mistral Small 3.1 | Not implemented | Source-backed text spec and tokenizer audit. |
| 55 | Llama 4 | Not implemented | Published MoE/attention operator audit and text spec. |
| 56 | Qwen3 | Not implemented | Dense and MoE presets, routing and source-backed dimensions. |
| 57 | SmolLM3 | Not implemented | Published NoPE/position schedule and exact tests. |
| 58 | Kimi K2 and K2 Thinking | Not implemented | Architecture-versus-post-training comparison; source audit. |
| 59 | GPT-OSS | Not implemented | Published MoE routing and attention spec. |
| 60 | Grok 2.5 | Not implemented | Public-detail audit before claiming a faithful preset. |
| 61 | GLM-4.5 | Not implemented | Published text path and training/structure separation. |
| 62 | Qwen3-Next | Not implemented | DeltaNet/recurrent state, hybrid schedule and MTP tests. |
| 63 | MiniMax-M2 | Not implemented | Source-backed full-attention architecture and cost comparison. |
| 64 | Kimi Linear | Not implemented | Kimi Delta Attention state update and hybrid cache. |
| 65 | Olmo 3 Thinking | Not implemented | Base architecture and post-training distinction. |
| 66 | DeepSeek V3.2 | Not implemented | Documented sparse-attention selection and comparison to V3. |
| 67 | Mistral 3 | Not implemented | Source-backed operator reuse and family differences. |
| 68 | Nemotron 3 Nano and Super | Not implemented | Mamba-2/recurrent path and latent experts. |
| 69 | Xiaomi MiMo-V2-Flash | Not implemented | Published hybrid schedule and distinct operator audit. |
| 70 | Arcee AI Trinity Large | Not implemented | Source-backed expert and attention preset. |
| 71 | GLM-5 | Not implemented | Published mechanism audit and spec. |
| 72 | February 2026 roundup | Not implemented | A dated, sourced comparison table; no fictitious single model preset. |
| 73 | Gemma 4 | Not implemented | Published differences from Gemma 3 and new-mechanism tests. |

For each future family, accept a preset only when its primary paper or official configuration supports the named choices, the code includes missing operators, and tests cover shapes, forward/backward, causal behavior, cache parity, and one short train step. “Not implemented” remains the honest state until those checks exist.
