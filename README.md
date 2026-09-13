# NCP-64M

**A ~64M-parameter replication of NCP-ArchPreview — latent-space language modeling with Next Concept Prediction, in the style of [MiniMind](https://github.com/jingyaogong/minimind).**

[![Paper](https://img.shields.io/badge/arXiv-2609.10715-b31b1b.svg)](https://arxiv.org/abs/2609.10715)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)]()

[NCP-ArchPreview](https://arxiv.org/abs/2609.10715) (Intern-NCP Team, 2026) pushes
autoregressive pretraining beyond next-token prediction: alongside NTP, the model
learns **Next Concept Prediction (NCP)** — predicting discrete concepts that span
multiple tokens — inside a product-quantized latent space built from its own
hidden states. At 8.9B parameters / 5.73T tokens it reached OLMo-3-7B's final
pretraining loss with only 51.3% of the tokens and beat its downstream
macro-average by +2.45 (incl. +5.99 GSM8K).

This repo re-implements the architecture at laptop scale (~70M params, one
consumer GPU / Apple Silicon) with MiniMind's minimalist, from-scratch-PyTorch
conventions: one model file, one trainer, a packed token dataset, and tests that
actually verify the concept pathway is causal.

## Architecture

```
tokens ──> Token Encoder (3L) ──mean-pool k=4──> concepts c_m ──> Concept Module (2L, causal)
                                                                │  softmax over S codebooks,
                                                                │  weighted codeword mix -> ĉ_m
Token Decoder (3L) <────────── h_t + ĉ (repeated k×, shifted) <─┘
      │
      └─> logits
```

| Component | This repo (64M class) | Paper (8.9B) |
|---|---|---|
| Token Encoder / Concept Module / Token Decoder | 3 / 2 / 3 layers | 16 / 8 / 16 layers |
| Hidden size, heads | 768, 12 (MHA) | 4096, 32 |
| Concept compression `k` | 4 tokens | 4 tokens |
| Product quantization | S=6 codebooks × N=128 × 128d | S=32 × N=128 × 128d |
| Residual routing | IRC + CRC (all 3 paths) | IRC + CRC |
| Optimizer | AdamW or Muon (`--optimizer`) | Moonlight Muon |
| Total params | ~70M | 8.94B |

Mechanisms implemented faithfully to the paper's equations:

- **VQ concept vocabulary** — each pooled concept is split into `S` segments,
  each assigned its nearest codeword (Eq. 3-6). `L_VQ = mean‖sg(c) − d‖²` moves
  codebook entries toward concepts with a stop-gradient on `c`, so it never
  distorts the token encoder (Eq. 21).
- **NCP objective** — the Concept Module outputs a softmax distribution over
  each codebook; `ĉ` is the expectation over codewords (fully differentiable,
  Eq. 8-10). `L_NCP` is MSE against `sg(c)` of the *next* concept (Eq. 22).
- **Causal concept feedback** — `ĉ_m` is repeated `k`× and shifted `k`, so the
  position predicting a token in chunk `m` sees only concepts predicted from
  chunks `< m` (Eq. 11-12). A test proves no future leakage.
- **Hierarchical residuals** — IRC: token-conditioned unnormalized mix over all
  previous depths, initialized to `[0,…,0,1]` = plain residual (Eq. 14-18).
  CRC: Enc→CM (chunk-pooled), Enc→Dec, CM→Dec (same causal shift); softmax
  routing over source depths, RMSNorm, learned diagonal gate (Eq. 19-20).
- **Joint loss** — `L = L_NTP + α·L_NCP + β·L_VQ` (Eq. 24), each term logged
  separately during training.
- OLMo-3 backbone recipe: RoPE (θ=5e5), SwiGLU, RMSNorm(ε=1e-6), per-head
  QK-norm (the stabilization variant identified in §4.6), every-4th-layer full
  attention with optional sliding window.

## Quickstart

```bash
uv sync          # or: pip install -r requirements.txt

# 1) Data -> packed uint16 token stream (any jsonl {"text": ...} or a HF dataset)
uv run python scripts/prepare_data.py \
    --hf_dataset roneneldan/TinyStories --hf_split train \
    --text_field text --max_docs 30000 --out dataset/pretrain.bin

# 2) Train the NCP model
uv run python trainer/train_pretrain.py --data_path dataset/pretrain.bin --device mps

# 3) Generate
uv run python eval_llm.py --weight pretrain

# Tests (causality, gradient flow, param count)
uv run pytest tests/
```

minimind's own corpus also works: download `pretrain_hq.jsonl` from
[huggingface.co/datasets/jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset)
and pass `--data_path <file>` to `prepare_data.py`.

### Ablations (Sec. 4.3.2 of the paper)

```bash
# Vanilla — parameter-matched plain transformer baseline
uv run python trainer/train_pretrain.py --arch vanilla --use_irc 0 --save_weight vanilla ...

# Vanilla + Concept Module (no residual routing, no NCP loss)
... --use_irc 0 --use_crc 0 --ncp_loss_weight 0 --vq_loss_weight 0

# Vanilla + CM + Residual (routing but no NCP supervision)
... --ncp_loss_weight 0 --vq_loss_weight 0

# Full NCP-ArchPreview — defaults
```

`--optimizer muon` applies the paper's recipe: Muon for matrix parameters,
AdamW for embeddings/heads/codebook.

## Monitoring (wandb)

```bash
uv run python trainer/train_pretrain.py --use_wandb 1 --wandb_project ncp \
    --data_path dataset/pretrain.bin --device mps
```

`--wandb_mode offline` logs locally without an account; `--from_resume 1`
continues the same run (the run id is stored in the resume checkpoint). Three
cadences, all in optimizer steps: `--log_interval` scalars,
`--diag_interval` (250) diagnostics, `--eval_interval` (500) val split,
`--showcase_interval` (1000) tables/images. `--eval_batches 16` carves a fixed
holdout off the tail of the token stream (0 disables it).

What gets logged (see `trainer/metrics.py`):

| Group | Metrics |
|---|---|
| `loss/`, `val/` | total / NTP / NCP / VQ, train and held-out split |
| `opt/`, `grad/` | lr, global grad-norm (pre-clip), clip-event rate, weight norm, per-module grad norms |
| `perf/` | tokens/sec, tokens seen (use as x-axis for ablations), est. TFLOPs, MFU with `--peak_flops` |
| `vq/` | per-segment codebook perplexity `exp(H)`, active-code fraction, dead-code count |
| `ncp/` | **concept top-1 accuracy** (`argmax π` vs next chunk's true code, per segment), prediction confidence + normalized entropy |
| `routing/` | IRC off-residual mass & last-state weight per module; CRC alpha entropy/last-depth per consumer←source; `crc_scale` injection strength; `concept_gain` |
| `update/` | `‖ΔW‖/‖W‖` per param group, update-RMS/lr (Muon & AdamW ≈0.2 target), spectral entropy of ΔW (early collapse signal) |
| `attn/`, `act/` | per-layer attention entropy + max logit (probe forward, flash off), per-block residual RMS |
| `gns/` | gradient noise scale ≈ critical batch size (two-half-batch estimator) |
| `show/` | generation samples table, codebook table (open with wandb's Embedding Projector), IRC/CRC routing heatmap tables, per-chunk-position accuracy, attention-map images |
| watch | `wandb.watch` per-layer gradient/parameter histograms (`--wandb_watch`, `--watch_freq`) |

Grad-norm spikes (>3× running median) and NaN loss/grad fire `wandb.alert`.

**VQ-only domain adaptation** (Sec. 5.1 — freeze the backbone, train only the
VQ codebooks + prediction heads):

```bash
uv run python trainer/train_pretrain.py --from_weight pretrain --train_vq_only 1 \
    --data_path dataset/<new_domain>.bin --save_weight vq_adapt
```

**Inference concept feedback** (Sec. 2.3): `eval_llm.py` defaults to
`--concept_feedback predicted`, feeding the Concept Module's own predictions
back autoregressively as the paper specifies. `--concept_feedback pooled`
teacher-forces encoder-pooled concepts instead.

## Verification evidence

| Check | Result |
|---|---|
| Unit tests | 25/25 pass — forward shapes, joint-loss identity, **strict no-future-leakage** through the concept path, VQ→codebook-only grads, NCP→CM+encoder grads, concept-history override wiring, predicted-feedback generation, VQ-only freeze coverage, label-alignment pin, W&B metric helpers |
| Concept-channel isolation | with `sliding_window=4`, a perturbation in chunk 0 changes distant logits only via the concept pathway |
| Short pretraining run | TinyStories 30k docs / 9.5M tokens on Apple MPS, AdamW: `ntp 6.73 → 3.89` over 150 steps, `ncp`/`vq` bounded and decreasing |
| Generation | `eval_llm.py` produces continuations end-to-end (early-checkpoint gibberish, as expected) |

## Deviations from the paper

- **Concepts are RMS-normalized** (parameter-free) before quantization,
  prediction, and feedback. The paper doesn't disclose its scale handling;
  without it the MSE objectives are unbounded and diverge at small scale.
  A learnable diagonal `concept_gain` lets the decoder set injection strength.
- **Aux losses use elementwise-mean MSE** — the paper's segment-L2/S convention
  differs by a constant `seg_dim` factor, absorbed into α/β.
- **α, β are undisclosed** — both default to 1.0 and are CLI flags.
- **Autoregressive concept feedback is sequential**: `generate` rebuilds the
  predicted-concept history one chunk at a time (paper-accurate), so crossing a
  chunk boundary costs one extra forward. No KV cache — simple and correct at
  this scale.
- Per-head QK-norm (used by MiniMind) replaces OLMo-3's layer-wise QK-norm —
  this is the variant the paper itself found *more* stable under Muon (§4.6).

## Roadmap

- [ ] Controlled NCP-vs-vanilla convergence comparison at 64M (the paper's
      headline 1.95× speedup claim)
- [x] VQ-only domain adaptation — freeze the backbone, train only codebooks +
      prediction heads (§5.1) — `--train_vq_only 1`
- [ ] Concept injection into a block-parallel speculative drafter (§5.3)
- [ ] KV-cache generation (currently full-forward per step — simple and correct
      at this scale)

## Repository layout

```
model/model_ncp.py          config + model: encoder/CM/decoder, PQ-VQ, IRC/CRC, generate
model/tokenizer*.json       bundled BPE tokenizer (vocab 6400, from MiniMind)
dataset/lm_dataset.py       fixed-length windows over a packed token stream
scripts/prepare_data.py     jsonl / HF dataset -> packed .bin
trainer/train_pretrain.py   training loop (AdamW|Muon, per-loss logging, resume, wandb)
trainer/train_tokenizer.py  train a fresh BPE tokenizer on your own corpus
trainer/trainer_utils.py    Muon optimizer, lr schedule, checkpointing
trainer/metrics.py          wandb catalog: VQ/concept stats, routing probes, dW spectra, GNS
eval_llm.py                 sampling / generation
tests/test_ncp.py           causality, gradient-flow, param-count checks
```

## Citation

```bibtex
@article{ncp-archpreview-2026,
  title   = {NCP-ArchPreview Technical Report: Moving towards Latent Space
             Language Models through Next Concept Prediction},
  author  = {{The Intern-NCP Team}},
  journal = {arXiv:2609.10715},
  year    = {2026}
}
```

## Acknowledgments

- [NCP-ArchPreview](https://arxiv.org/abs/2609.10715) — the paper this replicates
- [MiniMind](https://github.com/jingyaogong/minimind) — repo layout, trainer
  conventions, and the bundled tokenizer (`model/tokenizer*.json`, Apache-2.0)

## License

Apache-2.0. See [LICENSE](LICENSE).
