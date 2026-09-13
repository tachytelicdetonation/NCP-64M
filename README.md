# NCP-64M

A ~64M-class replication of **NCP-ArchPreview** (arXiv:2609.10715) — a latent-space
language model that adds **Next Concept Prediction (NCP)** on top of standard
next-token prediction (NTP). Repo layout and training pipeline follow
[minimind](https://github.com/jingyaogong/minimind).

## Architecture

Three modules around a product-quantized concept vocabulary:

```
tokens ──> Token Encoder (3L) ──mean-pool k=4──> concepts c_m ──> Concept Module (2L, causal)
                                                                │  softmax over S codebooks,
                                                                │  weighted codeword mix -> ĉ_m
Token Decoder (3L) <────────── h_t + ĉ (repeated k×, shifted) <─┘
      │
      └─> logits
```

- **VQ concept vocabulary** — `S=6` codebooks × `N=128` codewords × 128 dims
  (product quantization; `N^S` latent capacity). Codebook trained by
  `L_VQ = mean ||sg(c) - d||²` — moves codebook entries toward concepts only
  (stop-gradient on `c`), matching Eq. 21.
- **NCP objective** — `L_NCP = mean ||ĉ_m - sg(c_m)||²` over concept positions;
  `ĉ` is the softmax-expectation over codewords (differentiable, Eq. 8-10).
- **Concept feedback** — each `ĉ_m` is repeated `k=4`× and shifted by `k`, so
  position `t` predicting a token in chunk `m` sees the concept predicted from
  chunks `< m` only (Eq. 11-12). No future leakage — enforced by a test.
- **Hierarchical residuals** — IRC: per-layer token-conditioned weighted mix of
  all previous depths, initialized as the plain residual `[0,…,0,1]` (Eq. 14-18).
  CRC: Enc→CM (chunk-pooled), Enc→Dec, CM→Dec (same causal shift) — softmax
  routing over source depths + RMSNorm + learnable diagonal scale (Eq. 19-20).
- **Joint loss** — `L = L_NTP + α·L_NCP + β·L_VQ` (Eq. 24).
- Backbone follows the paper's OLMo-3 recipe at small scale: RoPE (θ=5e5),
  SwiGLU, RMSNorm(ε=1e-6), per-head QK-norm (the stable variant from §4.6),
  every-4th-layer full attention with optional sliding window elsewhere, MHA.

Default config: `d=768, 3/2/3 encoder/concept/decoder layers, k=4, S=6, N=128,
vocab=6400, tied embeddings → ~70M params` (minimind's own 768×8 counts ~68.6M
under the same "≈64M" label).

### Deviations from the paper

- **Concepts are RMS-normalized** (`F.rms_norm`, no params) before quantization,
  prediction, and feedback. Without this the MSE objectives are unbounded and
  diverge at small scale; the paper doesn't disclose its handling. A learnable
  diagonal `concept_gain` lets the decoder set injection strength.
- **Aux losses are elementwise-mean MSE** (paper writes segment-L2 / S; the
  `1/seg_dim` difference is absorbed into α, β).
- **Inference feedback uses pooled concepts** of generated chunks rather than
  feeding predicted `ĉ` back — strictly causal and equivalent at train time;
  the paper's predicted-feedback variant is a one-line change in `generate`.
- α, β are not disclosed in the report; both default to 1.0 and are flags.

## Layout

```
model/model_ncp.py        config + model (encoder/CM/decoder, VQ, IRC/CRC, generate)
model/tokenizer*.json     bundled tokenizer (vocab 6400, from minimind)
dataset/lm_dataset.py     fixed-length windows over a packed uint16 token stream
scripts/prepare_data.py   jsonl/HF dataset -> packed .bin
trainer/train_pretrain.py training loop (AdamW or Muon, joint loss breakdown)
trainer/train_tokenizer.py  train a fresh BPE tokenizer on your corpus
trainer/trainer_utils.py  Muon optimizer, lr schedule, checkpoints
eval_llm.py               sampling/generation
tests/test_ncp.py         shapes, causality, gradient-flow, param-count checks
```

## Usage

```bash
uv sync                                    # or: pip install -r requirements.txt

# 1. data: local jsonl with {"text": ...} OR stream a HF dataset
uv run python scripts/prepare_data.py \
    --hf_dataset roneneldan/TinyStories --hf_split train \
    --text_field text --max_docs 30000 --out dataset/pretrain.bin
# minimind's own corpus: download pretrain_hq.jsonl from
# huggingface.co/datasets/jingyaogong/minimind_dataset, then --data_path <file>

# 2. train (NCP model)
uv run python trainer/train_pretrain.py --data_path dataset/pretrain.bin --device mps

# parameter-matched vanilla baseline (paper's "Vanilla"):
uv run python trainer/train_pretrain.py --arch vanilla --use_irc 0 --save_weight vanilla ...

# paper's optimizer recipe (Muon for matrices, AdamW for the rest):
uv run python trainer/train_pretrain.py --optimizer muon ...

# ablations from Sec. 4.3.2:
#   Vanilla+CM            -> --use_irc 0 --use_crc 0 --ncp_loss_weight 0 --vq_loss_weight 0
#   Vanilla+CM+Residual   -> --ncp_loss_weight 0 --vq_loss_weight 0
#   full NCP              -> defaults

# 3. generate
uv run python eval_llm.py --weight pretrain

# tests
uv run pytest tests/
```

## Status / evidence

- 9/9 tests pass, including a strict no-future-leakage check on the concept path
  and gradient-flow checks on all three objectives.
- Short run on TinyStories (30k docs, 9.5M tokens, MPS, AdamW): NTP loss
  6.33 → 4.31 over 150 steps; NCP/VQ losses bounded and decreasing.
- Not yet done: a controlled NCP-vs-vanilla convergence comparison at this
  scale, VQ-only domain adaptation (§5.1), drafter injection (§5.3).
