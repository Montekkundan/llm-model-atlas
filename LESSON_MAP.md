# Lectures 51–73: architecture case-study project map

This map follows the 96-lesson course order. **Tiny text path** means only the listed teaching mechanisms are executable here; it does not mean the published model is reproduced. **Reference operators** means the constituent mechanisms run through `python -m atlas.case_study NUMBER`; a complete family graph and checkpoint loader remain separate work.

| # | Case study | Project state | Next required work |
| --- | --- | --- | --- |
| 51 | DeepSeek V3 and R1 | **Tiny V3-style text path** | Low-rank MLA with decoupled RoPE and sigmoid top-k/shared-expert MoE run; no MTP, cache decode, expert balancing, distributed training, or R1 post-training. [V3 report](https://arxiv.org/abs/2412.19437). |
| 52 | OLMo 2 | **Tiny text path** | Official tokenizer/config audit, z-loss/training recipe, cache parity and evaluation. [Report](https://arxiv.org/abs/2501.00656). |
| 53 | Gemma 3 | **Tiny text path** | Vision path, real tokenizer/config, optimized rolling cache, official local span and evaluation. [Report](https://arxiv.org/abs/2503.19786). |
| 54 | Mistral Small 3.1 | **Tiny text path** | Vision/Tekken tokenizer, official config dimensions, cache parity and evaluation. [Config](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/68faf511d618ef198fef186659617cfd2eb8e33a/config.json). |
| 55 | Llama 4 | Reference operators | Top-1 of 16 sigmoid-gated routing (unnormalised), the 48-layer NoPE-every-fourth schedule and a chunked mask run; shared-expert MoE layer, attention temperature tuning, dimensions and a text spec remain. |
| 56 | Qwen3 | **Tiny dense text path** | MoE routing, official dimensions/tokenizer, cache parity and evaluation. [Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json). |
| 57 | SmolLM3 | Reference operators | The 36-layer, every-fourth-layer NoPE schedule runs (zero-based 3, 7, ..., 35); a full graph with exact tests remains. |
| 58 | Kimi K2 and K2 Thinking | Reference operators | Architecture-versus-post-training comparison; source audit. |
| 59 | GPT-OSS | Reference operators | Published MoE routing and attention spec. |
| 60 | Grok 2.5 | Reference operators | Softmax-over-all-experts, top-2-of-8 unnormalised gate (Grok-1 code) runs; router and attention soft-capping, residual MoE and a public-detail audit remain. |
| 61 | GLM-4.5 | Reference operators | Published text path and training/structure separation. |
| 62 | Qwen3-Next | Reference operators | DeltaNet/recurrent state, hybrid schedule and MTP tests. |
| 63 | MiniMax-M2 | Reference operators | Source-backed full-attention architecture and cost comparison. |
| 64 | Kimi Linear | Reference operators | Kimi Delta Attention state update and hybrid cache. |
| 65 | Olmo 3 Thinking | Reference operators | Base architecture and post-training distinction. |
| 66 | DeepSeek V3.2 | Reference operators | Documented sparse-attention selection and comparison to V3. |
| 67 | Mistral 3 | Reference operators | Source-backed operator reuse and family differences. |
| 68 | Nemotron 3 Nano and Super | Reference operators | Mamba-2/recurrent path and latent experts. |
| 69 | Xiaomi MiMo-V2-Flash | Reference operators | Published hybrid schedule and distinct operator audit. |
| 70 | Arcee AI Trinity Large | Reference operators | Source-backed expert and attention preset. |
| 71 | GLM-5 | Reference operators | Published mechanism audit and spec. |
| 72 | February 2026 roundup | Reference operators | A dated, sourced comparison table; no fictitious single model preset. |
| 73 | Gemma 4 | Reference operators | Published differences from Gemma 3 and new-mechanism tests. |

For each future family, accept a preset only when its primary paper or official configuration supports the named choices, the code includes the distinguishing operators, and tests cover shapes, forward/backward, causal behavior, and one short train step. Cache parity becomes a separate requirement when an actual decode cache is added. Constituent checks establish only their stated mathematical contracts; they do not establish a full-family implementation or released-checkpoint parity.
