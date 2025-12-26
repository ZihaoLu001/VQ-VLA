# UniTok × VQ-VLA Integration (WIP)

This branch integrates **UniTok visual tokens** into the **VQ-VLA / OpenVLA** training pipeline, aiming for a *fully-tokenized* VLA model:
- **Vision**: image → **UniTok discrete tokens** (offline precompute) → embedding/projection → OpenVLA backbone  
- **Action**: continuous actions → **VQ-VLA residual VQ-VAE action tokens** (existing pipeline)

> Key design choice: **Route A (recommended)** — *offline UniTok tokenization* (precompute once, train from disk).  
> This keeps training fast, reproducible, and avoids running UniTok inside the training loop.

---

## 0. Repository layout (current)

Relevant new/modified parts in this repo:

```
VQ-VLA/
  scripts_unitok/
    sanity_check_tokens.py          # single-image encode/decode + save tokens (idx/seq)
    precompute_unitok_tokens.py     # batch precompute tokens for an image folder
  vla_unitok/
    vision_tokenizer/
      unitok_wrapper.py             # wrapper around externals/UniTok tokenizer ckpt
      token_packing.py              # pack/unpack indices and offset packing metadata
    models/
      openvla_unitok_adapter.py     # (WIP) adapter to feed vision tokens into OpenVLA
  externals/
    UniTok/                         # upstream UniTok code (submodule or vendored copy)
```

---

## 1. Milestones & progress

### Milestone 1 — Reproduce VQ-VLA baseline
- **Status**: *not executed in this branch yet* (baseline training/eval not rerun here).
- **Next**: run a minimal LIBERO evaluation to ensure baseline still works before adding UniTok tokens.

### Milestone 2 — Load dataset (e.g., GATE-VLAP) and extract (image, instruction, actions)
- **Status**: *pending* (dataset adapter not implemented yet).
- **Next**: add a dataset wrapper that can optionally read precomputed UniTok tokens by image path.

### Milestone 3 — Precompute UniTok visual tokens offline
- **Status**: ✅ **done** (tokenization + packing validated on sample images)

What is confirmed:
- UniTok ckpt loads correctly and can **reconstruct** images (`inference.py` & `sanity_check_tokens.py --decode`).
- Encoder output indices shape is **(B, K, N) = (1, 8, 256)** on 256×256 images.
- Using **offset packing**, we flatten to **(B, 2048)** tokens where:
  - `K = 8` codebooks
  - `N = 256` spatial tokens
  - `vocab_size = 32768`
  - packed id range example: min/max ≈ `43 … 233468`

Example outputs (generated on cluster):
- Reconstruction image:
  - `/datasets/v2p/current/lucas/tmp/unitok/vqvla_pipeline_rec.png`
- Packed tokens (example single image):
  - `/datasets/v2p/current/lucas/tmp/unitok/vqvla_pipeline_seq.pt`
  - `/datasets/v2p/current/lucas/tmp/unitok/vqvla_pipeline_idx.pt`
- Batch precompute results (cosmos assets):
  - `/datasets/v2p/current/lucas/workspace/unitok_tokens/cosmos_assets_offsetpack/*.pt`

### Milestone 4 — Train VLA with UniTok vision tokens + VQ-VLA action tokens
- **Status**: *WIP / not started*.
- **Next**: implement the model adapter and dataloader changes to consume `vision_tokens`.

---

## 2. Environment setup

We recommend **two conda envs** (stable and reproducible):

### (A) `vqvla` env (training / VQ-VLA baseline)
Use the original VQ-VLA instructions (torch==2.2.0 + flash-attn, etc.).

### (B) `unitok` env (offline token precompute)
This env is used only to run UniTok inference and precompute tokens.

Example (already used on the cluster):
```bash
conda activate unitok
python --version
# Python 3.10.x

# UniTok deps are installed under externals/UniTok/requirements.txt
# plus repo-local requirements if needed.
```

> Notes:
> - You may see warnings like `No pretrained configuration specified for vitamin_large model...` from timm registry.
>   This is non-fatal for inference/tokenization.
> - `torch.utils.checkpoint` warnings about `use_reentrant` are warnings only (no gradients in inference).

---

## 3. UniTok tokenizer checkpoint

We use the released UniTok tokenizer checkpoint:
- `unitok_tokenizer.pth` (downloaded separately)

Path used in experiments:
```bash
CKPT=/datasets/v2p/current/lucas/workspace/ckpts/unitok/unitok_tokenizer.pth
```

---

## 4. Token format & packing (important)

### 4.1 Raw indices from UniTok
UniTok returns indices shaped like:
- `idx`: **(B, K, N)**, where
  - `K` = number of codebooks (e.g. 8)
  - `N` = number of spatial tokens (e.g. 256)
  - each element is in `[0, vocab_size-1]`

### 4.2 Packed sequence (offset packing)
To feed a single discrete stream to an LLM/VLA, we pack `(B, K, N)` into a flat `(B, N*K)` sequence with offsets:
- for codebook `k`, token ids are shifted by `k * vocab_size`

This avoids collisions where the same id from different codebooks would mean different things.

Packed layout used here (BKN → BNK → flatten):
```python
# idx: (B,K,N)
offset = torch.arange(K).view(1,K,1) * vocab_size
seq = (idx + offset).permute(0,2,1).reshape(B, -1)   # (B, N*K)
```

**Saved metadata** includes:
- original shape
- packing mode (`offset`)
- vocab size
- layout (`BKN`)

---

## 5. Usage

### 5.1 Sanity check on a single image (encode/decode/save tokens)

```bash
conda activate unitok
cd /datasets/v2p/current/lucas/workspace/VQ-VLA

mkdir -p /datasets/v2p/current/lucas/tmp/unitok

python scripts_unitok/sanity_check_tokens.py \
  --ckpt_path /datasets/v2p/current/lucas/workspace/ckpts/unitok/unitok_tokenizer.pth \
  --src_img assets/vqvla_pipeline.png \
  --out_dir /datasets/v2p/current/lucas/tmp/unitok \
  --prefix vqvla_pipeline \
  --decode --save_idx --save_seq
```

Outputs:
- `${out_dir}/${prefix}_rec.png`
- `${out_dir}/${prefix}_idx.pt`  (raw indices)
- `${out_dir}/${prefix}_seq.pt`  (packed sequence)

### 5.2 Precompute tokens for an image directory

```bash
conda activate unitok
cd /datasets/v2p/current/lucas/workspace/VQ-VLA

IMG_DIR=/datasets/v2p/current/lucas/cosmos-predict2/assets
OUT_DIR=/datasets/v2p/current/lucas/workspace/unitok_tokens/cosmos_assets_offsetpack
mkdir -p $OUT_DIR

python scripts_unitok/precompute_unitok_tokens.py \
  --ckpt_path /datasets/v2p/current/lucas/workspace/ckpts/unitok/unitok_tokenizer.pth \
  --img_dir  $IMG_DIR \
  --out_dir  $OUT_DIR \
  --pack --pack_mode auto
```

Each output file is a `.pt` with:
```python
{
  "indices": Tensor,         # packed or raw depending on args
  "meta": {...},             # packing + orig shape
  "src": "/abs/path/to/image"
}
```

---

## 6. Integration plan (what’s next)

### 6.1 Dataset side
Goal: for each training sample, provide:
- `instruction` (string)
- `image` (optional, for baseline comparisons)
- **`vision_tokens`**: packed UniTok token sequence (LongTensor)
- `actions` / `action_tokens` (existing VQ-VLA pipeline)

Implementation idea:
- Store tokens as `.pt` files keyed by image stem or relative path.
- In dataset `__getitem__`, load token `.pt` by image path and return `vision_tokens`.
- Add a config flag: `use_unitok_tokens: true/false`

### 6.2 Model side (OpenVLA adapter)
Goal: replace/augment the visual embedding input with UniTok token embeddings.
Minimal viable approach:
- `nn.Embedding(unitok_vocab_total, d_model)` where `unitok_vocab_total = K * vocab_size`
- optionally a small projector (MLP or 2-layer Transformer) to match OpenVLA expected conditioning format
- concatenate with text tokens in the same causal stream

Open questions to decide soon:
- Where to insert vision tokens in OpenVLA prompt format (e.g., `<image>` placeholder vs direct concatenation).
- Whether to keep the original image encoder path for ablations (`pixel` vs `unitok`).

### 6.3 Training plan (recommended order)
1. Re-run **baseline** VQ-VLA small eval (sanity).
2. Overfit a tiny dataset split (e.g., 64 samples) with UniTok tokens to validate plumbing.
3. Scale to full dataset, compare:
   - baseline OpenVLA vision encoder
   - UniTok vision tokens

---

## 7. Known issues & fixes applied

### 7.1 `pos_embed` incompatibility (timm VisionTransformer)
We hit:
```
TypeError: VisionTransformer.__init__() got an unexpected keyword argument 'pos_embed'
```
This comes from a mismatch between UniTok’s ViTamin model factory and the installed `timm` API.

**Resolution in this branch** (Route A):
- we avoid passing `pos_embed='none'` into the timm VisionTransformer constructor in the relevant UniTok model creation path.
- UniTok inference now works and tokenization succeeds.

> If syncing UniTok upstream or changing timm versions, re-check this issue.

---

## 8. References
- VQ-VLA (ICCV 2025): action tokenizer scaling + OpenVLA integration.
- UniTok (NeurIPS 2025 spotlight): unified visual tokenizer for generation + understanding.

---

## 9. TODO checklist

- [ ] Baseline: run VQ-VLA LIBERO eval in this branch and record results.
- [ ] Dataset adapter: add token loading and caching strategy (pt / webdataset / lmdb).
- [ ] Model adapter: implement `openvla_unitok_adapter.py` to consume `vision_tokens`.
- [ ] Training sanity: tiny overfit run + logs.
- [ ] Full training: run 1–2 configs and report success rates / speed.
- [ ] Documentation: add config examples and expected directory structure for token dumps.
