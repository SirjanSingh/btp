# Domain Pitfalls: Aerial Rooftop & Solar Panel Segmentation

**Domain:** Aerial imagery segmentation pipeline (rooftop + PV panels)
**Researched:** 2026-03-30
**Confidence:** MEDIUM-HIGH (training knowledge through Aug 2025; WebSearch unavailable — flags where live verification is recommended)

---

## Critical Pitfalls

Mistakes that cause silent failure, silent metric inflation, or major rewrites.

---

### Pitfall 1: Tile Overlap Causes Train/Val Leakage

**What goes wrong:** When tiling 10,000x10,000px AIRS images with 10% overlap (stride ~460px for 512px tiles), adjacent tiles share ~50px of pixels on each edge. If you tile the full image first and then split tiles into train/val, adjacent tiles from the same source image end up in both splits. The model memorizes boundary context rather than generalizing, IoU inflates silently, and you only discover the problem post-submission.

**Why it happens:** Tiling is done at the pixel level without tracking which source image each tile belongs to. A random 80/20 split of all tiles then distributes siblings across splits.

**Consequences:** Reported val IoU can be 3-8 points higher than true generalization. The inflated number may appear to beat the PSPNet baseline (0.899) when it hasn't.

**Prevention:**
- Split at the source image level first (the 857/94/96 split is already given — use it as-is).
- Tile each split independently. Never tile the whole dataset and then split the tiles.
- AIRS already provides an official train/val/test file list; respect it exactly.
- If you add custom augmentation that creates new tiles, ensure they derive from train-split source images only.

**Detection:** Compare val IoU with and without overlap. A suspiciously high score (>0.92 early in training) is a red flag. Also visualize which source image each tile comes from.

---

### Pitfall 2: Mask-Image Spatial Misalignment

**What goes wrong:** AIRS masks are GeoTIFF rasters aligned to the corresponding image via georeference metadata. If you load images with PIL/OpenCV (which ignores geotransform) and masks with rasterio (which honors it), or if you use different resampling methods, the mask can be offset by several pixels relative to the image — enough to corrupt boundary-heavy losses like Dice.

**Why it happens:** PIL, OpenCV, and rasterio all handle coordinate origins differently. TIFF files sometimes embed geotransforms; naively reading both channels with mismatched libraries produces a silent offset.

**Consequences:** Boundary pixels become mislabeled. Loss converges but F1 plateaus well below expectations. Hard to diagnose visually unless you overlay at high zoom.

**Prevention:**
- Use rasterio (or GDAL) for both image and mask. Read both with `rasterio.open()`, call `.read()` on each, and verify `src_img.transform == src_mask.transform` before tiling.
- Alternatively use `cv2.imread` for both but strip geotransform — just ensure crop windows are applied identically to both arrays.
- After tiling, write a sanity check: overlay mask contours on image tiles and visually inspect 20 random samples.

**Detection:** Overlay mask boundary on image. If building edges are offset by 1-5 pixels, you have a misalignment issue.

---

### Pitfall 3: GSD Hardcoded Wrong — Area Estimates Are Off by a Factor

**What goes wrong:** AIRS is 7.5cm/pixel (GSD = 0.075m). Area = `pixel_count × (0.075)²` = `pixel_count × 0.005625 m²`. If you accidentally use 0.75 (10x error), area estimates are 100x inflated. If inference is run on a different-resolution image without updating GSD, every area number is wrong.

**Why it happens:** GSD is often hardcoded as a constant in one place. Stage 2 solar panel datasets are at 0.8/0.3/0.1m — 10-80x coarser. If Stage 1 and Stage 2 share an area estimation function that reads the same hardcoded GSD, solar panel areas are wildly wrong.

**Consequences:** Final capacity estimates (kW) are meaningless. The demo fails silently — numbers look plausible but are incorrect.

**Prevention:**
- Never hardcode GSD as a bare float. Use a named config constant: `GSD_AIRS_M = 0.075`.
- Pass GSD explicitly to every area estimation call.
- For Stage 2, define separate GSD constants per dataset/resolution tier.
- At inference time, require the caller to supply GSD (or read it from the image metadata via rasterio).
- Add a unit test: `assert area_from_mask(ones_100x100, gsd=0.075) == pytest.approx(56.25, rel=1e-4)`.

**Detection:** Cross-check one manually measured building against the estimate.

---

### Pitfall 4: Non-Persistent `/scratch/` Wipes Checkpoints

**What goes wrong:** On the LNMIIT DGX, `/scratch/` is a fast local disk that is not persistent across container restarts or scheduled wipes. An overnight training run saves checkpoints only to `/scratch/`. The container is restarted (or the disk is wiped), and the entire run is lost.

**Why it happens:** `/scratch/` is fast (NVMe-class), so it is the natural target for intermediate files. Its non-persistence is a DGX cluster policy, not a bug, and is easy to overlook.

**Consequences:** Full training run lost. On a BTP deadline, this can cost days.

**Prevention:**
- Use `/scratch/` only for in-epoch temp files (e.g., prefetched tiles).
- Save every checkpoint (or at minimum best + latest) to Google Drive using `gdown`/`rclone` at the end of each epoch callback.
- Implement a `CheckpointCallback` that does: save to `/scratch/checkpoints/`, then `rclone copy /scratch/checkpoints/ gdrive:rooftop_solar/checkpoints/`.
- Test the rclone copy path before starting a long run.
- Use `screen -S training` (already planned) but also log the screen session name so you can reattach after disconnect.

**Detection:** Check `/scratch/` contents after reconnecting to a screen session — if empty, the wipe already occurred.

---

### Pitfall 5: Docker glibc Mismatch Breaks Modern Binaries

**What goes wrong:** The DGX host has glibc 2.17 (CentOS 7 vintage). Binaries compiled against glibc >=2.28 (most pip wheels built 2022+) segfault or print `GLIBC_2.28 not found`. This includes recent versions of torch, gdal, and rasterio.

**Why it happens:** The Ubuntu 22.04 Docker container has glibc 2.35, which is fine for packages installed inside the container. The problem appears when host paths are bind-mounted into the container, or when a host-installed Python is used instead of the container's Python — both of which are easy to do accidentally.

**Consequences:** Silent segfaults during DataLoader worker initialization (which uses fork). Workers die, DataLoader hangs or returns empty batches, training appears to run but produces garbage.

**Prevention:**
- Always work inside the Docker container. Never call the host Python.
- `pip install` everything inside the container. Do not bind-mount a host venv.
- Pin your container image tag (e.g., `nvcr.io/nvidia/pytorch:23.10-py3`) and record it in a `Dockerfile` or `docker_run.sh` script so the environment is reproducible.
- Set `num_workers=0` initially to rule out DataLoader fork issues; increase only after confirming stability.

**Detection:** Run `python -c "import torch; print(torch.__version__)"` inside the container. If it fails, you are outside the container or using the wrong Python.

---

### Pitfall 6: rclone / gdown Authentication Expires Mid-Run

**What goes wrong:** Google Drive OAuth tokens expire (typically 1 hour for `gdown`, or if rclone token file is not refreshed). A training run that saves checkpoints to Drive at epoch end silently fails to upload after hour 1. The run appears to complete; Drive has only epoch-1 checkpoint.

**Why it happens:** OAuth2 access tokens are short-lived. rclone uses a refresh token automatically if configured, but only if the `rclone.conf` file is accessible and the refresh token has not been revoked (e.g., by password change or security audit).

**Consequences:** Lost checkpoints. If the run completes and `/scratch/` is not persistent, no checkpoint survives.

**Prevention:**
- Use rclone (not gdown) for write operations — rclone handles token refresh automatically.
- Store `rclone.conf` inside the container or in a persistent bind-mounted path (e.g., `/home/23ucs715/.config/rclone/`).
- After setting up rclone, test upload/download of a 100MB dummy file during a short session to confirm refresh works.
- As a belt-and-suspenders measure, also save a checkpoint to `/home/` if it is on a persistent volume on your DGX setup.
- Do not use `gdown` for uploads — it is download-only from public links.

**Detection:** After a training run, check Drive file timestamps. If last upload is from hour 1 of a 6-hour run, token expired.

---

## Moderate Pitfalls

---

### Pitfall 7: Class Imbalance — Background Overwhelms Building Pixels

**What goes wrong:** In AIRS tiles, building pixels are a minority (urban density varies; many tiles are mostly background). BCE loss on a pixel-classifier is dominated by the majority background class. The model learns to predict "background everywhere" and achieves ~95% pixel accuracy while having near-zero IoU for buildings.

**Prevention:**
- Use BCE + Dice combined loss (already planned). Dice is insensitive to class imbalance for the positive class.
- Add `pos_weight` to `BCEWithLogitsLoss` to upweight building pixels: estimate from training set statistics (typical building/background ratio in AIRS is roughly 1:4 to 1:8).
- Consider dropping tiles with zero building pixels (fully-background tiles) or capping their fraction in each batch.
- Log per-class pixel counts during the first epoch to verify actual imbalance ratio.

---

### Pitfall 8: Overlapping Tile Aggregation at Inference Produces Seam Artifacts

**What goes wrong:** At inference, the 10,000x10,000px image is tiled with overlap. Predictions for overlapping regions from adjacent tiles are averaged (or max-pooled). If averaging is not done properly, a hard tile boundary shows as a visible seam in the output mask, causing false negatives at boundaries of large buildings.

**Prevention:**
- Use a Gaussian blending window (not flat averaging) over the overlap region. Libraries like `patchify` or custom implementations weight center pixels higher than edges.
- Alternatively, use a stride that keeps overlap small (10% is already planned) and only use the center crop of each tile prediction (ignore the 10% border).
- Implement a `merge_tiles` function and test it on a synthetic image where the expected output is known.

---

### Pitfall 9: Albumentations Geometric Transforms Break Mask Integrity

**What goes wrong:** Albumentations applies identical transforms to image and mask only when both are passed together via the `image` and `mask` keys in the same call. If you accidentally apply transforms to image and mask in separate calls (common when copy-pasting augmentation code), geometric transforms (flip, rotate, elastic) desynchronize image and mask.

**Prevention:**
- Always use `A.Compose([...])` with the transform applied as `transformed = aug(image=img, mask=mask)`. Never call `aug(image=img)` and `aug(mask=mask)` separately.
- For multi-mask scenarios (Stage 2 with rooftop + solar masks), use `additional_targets={'mask2': 'mask'}`.
- Write a test: apply a known transform (90-degree rotate) and assert that a hand-placed pixel in image and mask end up at the same coordinates.

---

### Pitfall 10: Pretrained Encoder Learning Rate — Differential LR Required

**What goes wrong:** ResNet-34 encoder weights from ImageNet are well-trained. Using a single global learning rate (e.g., 1e-3) on the entire U-Net applies the same LR to both the pretrained encoder and the randomly initialized decoder. The encoder catastrophically forgets ImageNet features in early epochs, and the model performs worse than a frozen encoder baseline.

**Prevention:**
- Use differential learning rates: encoder LR = 1e-4 (or lower), decoder LR = 1e-3.
- In PyTorch, create two parameter groups: `[{'params': model.encoder.parameters(), 'lr': 1e-4}, {'params': model.decoder.parameters(), 'lr': 1e-3}]`.
- SMP models expose `model.encoder` and `model.decoder` parameter groups directly.
- Optionally freeze the encoder for the first 2-3 epochs, then unfreeze with a lower LR.

---

### Pitfall 11: Batch Size Too Small → Unstable BatchNorm Statistics

**What goes wrong:** ResNet-34 uses BatchNorm layers. With small batch sizes (e.g., 2-4 on a single GPU with 512x512 tiles due to memory), BatchNorm running statistics are noisy. This manifests as high loss variance across batches and slower convergence. The problem is worse when tiles have heterogeneous content (background-heavy vs. building-dense).

**Prevention:**
- Use effective batch size >= 8. On DGX with 80GB A100, 512x512 tiles allow batch size 16-32 easily.
- If memory is tight, use gradient accumulation: accumulate 4 steps with batch_size=4 to simulate batch_size=16.
- Alternatively, switch to GroupNorm or use SyncBatchNorm if using multi-GPU training.
- Log loss curve variance per epoch. High variance with small batch is the signature.

---

### Pitfall 12: Large Factory Roofs Exceed Crop Size — Hard to Detect

**What goes wrong:** Some factory roofs in AIRS span areas larger than 512x512 pixels (>38.4m x 38.4m). A 512x512 crop captures only a partial interior of such a roof, with no boundary visible. The model sees a large homogeneous region with no edges and struggles to classify it. This is a known AIRS failure mode (cited in project context).

**Prevention:**
- For large roofs, use multi-scale inference: run the model at 256x256 effective resolution (downsample input to 50%) and then upsample the prediction. This gives more context.
- Alternatively, use a second pass at 1024x1024 crop size if GPU memory permits.
- Flag large-roof tiles in evaluation separately to understand their contribution to overall IoU.
- Accept this as an architectural limitation in the write-up — it is a documented AIRS challenge.

---

### Pitfall 13: IoU Metric Computed Over Batches vs. Globally

**What goes wrong:** A common implementation error: compute IoU per batch and average across batches (mean of per-batch IoU). This gives a different (usually higher) number than the correct approach: accumulate TP/FP/FN counts across all batches and compute IoU globally. The difference is significant when batch sizes vary or when building pixel distribution is uneven across batches.

**Prevention:**
- Use `torchmetrics.JaccardIndex` with `average='micro'` (global accumulation) rather than computing per-batch and averaging.
- Or manually accumulate `tp_sum, fp_sum, fn_sum` across all batches and compute `iou = tp_sum / (tp_sum + fp_sum + fn_sum)` at epoch end.
- The AIRS baselines report global IoU. Your implementation must match this to make the comparison valid.

**Detection:** If your val IoU is suspiciously high compared to loss, check whether you are averaging per-batch IoU.

---

### Pitfall 14: Test-Time Augmentation Applied Inconsistently

**What goes wrong:** TTA (flipping, rotating inputs and averaging predictions) can add 0.5-2 IoU points. If TTA is used during hyperparameter search but not reported correctly, or used in some evaluations but not others, published numbers become inconsistent. Alternatively, TTA applied naively to asymmetric augmentations (e.g., non-orthogonal rotations) introduces artifacts.

**Prevention:**
- Decide upfront whether the reported number is with or without TTA, and be consistent.
- If using TTA, limit to horizontal flip + vertical flip + 90/180/270 rotations (8-fold for binary masks — all are invertible for square tiles).
- Apply the inverse transform to each prediction before averaging.

---

## Minor Pitfalls

---

### Pitfall 15: gdown Fails on Large Files / Quota Exceeded

**What goes wrong:** AIRS is a large dataset. Google Drive has a download quota that can be exceeded when multiple team members or multiple DGX sessions download simultaneously. `gdown` raises a quota error that looks like a network error, not a quota error.

**Prevention:**
- Download AIRS once to `/scratch/` (if persistent enough for the session) and copy to your home directory or a persistent mount.
- Use `rclone copy` from your own Drive mount rather than gdown for large files — it is resumable and handles rate limits.
- Alternatively pre-tile the dataset on Colab and upload the tiles (~220k files) to Drive; then only download tiles, not raw 10,000x10,000px images.

---

### Pitfall 16: Albumentations Version API Breaks

**What goes wrong:** Albumentations made breaking API changes between v1.x and v2.x (released late 2023). Some transform names and parameter names changed. Code written for v1.x fails silently or raises cryptic errors under v2.x.

**Prevention:**
- Pin `albumentations==1.3.x` if using code examples from 2022-2023 tutorials.
- Or migrate to v2.x and use the new API (check the official migration guide).
- Record the exact pinned version in `requirements.txt` or `environment.yml`.
- Confidence: MEDIUM — verify against current albumentations changelog.

---

### Pitfall 17: SMP Model Output is Logits, Not Probabilities

**What goes wrong:** `segmentation_models_pytorch` models return raw logits (unbounded float). If you apply a threshold of 0.5 directly to logits (instead of applying sigmoid first), predictions will be wrong — logits near 0 map to ~0.5 probability, so many true positives are thresholded away.

**Prevention:**
- Apply `torch.sigmoid(output)` before thresholding for binary segmentation.
- Or use `activation='sigmoid'` in the SMP model constructor to get probabilities directly.
- Confirm: if loss is `BCEWithLogitsLoss` (preferred, numerically stable), do NOT apply sigmoid before the loss — only apply sigmoid at inference time.

---

### Pitfall 18: Screen Session Disconnects Without Output Redirect

**What goes wrong:** `screen -S training python train.py` keeps the session alive, but if training output is not also written to a log file, you cannot review output after reconnecting. Print statements from multiple epochs may be lost if the screen scrollback buffer overflows.

**Prevention:**
- Redirect output: `python train.py 2>&1 | tee /scratch/logs/train_$(date +%Y%m%d_%H%M).log`
- Copy logs to Drive periodically (or at the end of each epoch alongside checkpoints).

---

### Pitfall 19: Normalizing Aerial Images with Wrong Mean/Std

**What goes wrong:** Pretrained ResNet-34 encoders expect ImageNet normalization (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`). AIRS aerial imagery has a different distribution (more gray/brown, less blue-sky). Using ImageNet stats is a pragmatic default, but using zero normalization (or incorrect custom stats computed on only a few images) degrades encoder feature quality.

**Prevention:**
- Default: use ImageNet normalization (standard practice when using ImageNet pretrained encoders, even on non-natural images — it is what the encoder weights expect).
- Optional: compute AIRS dataset mean/std from training set tiles and fine-tune with those values. But benchmark against ImageNet norm first.
- SMP applies `preprocessing_fn` per encoder automatically when you use `smp.create_model(..., encoder_weights='imagenet')`.

---

## Stage 2 Specific: Solar Panel Detection

---

### Pitfall 20: Resolution Mismatch Between Stage 1 and Stage 2 Training Data

**What goes wrong:** AIRS (Stage 1) is at 0.075m/pixel. BDAPPV and multi-resolution PV datasets are at 0.8/0.3/0.1m/pixel — 4x to 107x coarser. A model trained on AIRS-resolution crops cannot directly process BDAPPV images, and vice versa. Running Stage 2 model on AIRS-resolution rooftop crops (which is what the end-to-end pipeline does) means the model sees data at a resolution it was never trained on.

**Why it happens:** The natural instinct is to train Stage 2 on the PV dataset and then apply it to Stage 1 outputs. But Stage 1 outputs are cropped from AIRS (0.075m), while PV datasets are at coarser resolutions.

**Consequences:** Stage 2 model sees solar panels that appear much smaller than during training (because GSD is finer). Recall collapses.

**Prevention:**
- Before training Stage 2, decide the canonical inference resolution. Options:
  - (A) Downsample AIRS rooftop crops to match the PV dataset resolution (e.g., 0.1m) before Stage 2 input — straightforward but loses detail.
  - (B) Upsample PV dataset training images to 0.075m equivalent resolution — adds no real information.
  - (C) Train Stage 2 only on PV data at its native resolution, and at inference resample AIRS crops to that same resolution.
- Option A or C are most principled. Document the chosen GSD in config and enforce it.
- Add an assertion at Stage 2 inference entry point: `assert abs(input_gsd - STAGE2_EXPECTED_GSD) < 1e-4`.

---

### Pitfall 21: Solar Panel Pixels Are a Tiny Minority — Extreme Class Imbalance

**What goes wrong:** Within a rooftop crop, solar panel pixels are sparse (typical coverage 5-30% of roof area, often less). Pure BCE loss leads to predicting "no panels" universally with high accuracy.

**Prevention:**
- Use Focal Loss or BCE + Dice (same approach as Stage 1).
- Oversample rooftop crops that contain solar panels during training.
- In BDAPPV, positive/negative split is known — ensure training batches contain at least 50% positive tiles.
- Consider a two-stage approach: first classify whether a rooftop crop contains any panel (binary classifier), then segment only within positive crops.

---

### Pitfall 22: Rooftop Crop Boundary Bleeds Into Non-Roof Background

**What goes wrong:** Stage 1 produces a binary rooftop mask. When cropping the region for Stage 2, a tight bounding box around the mask will include some background pixels outside the roof (because real roofs are non-rectangular). If Stage 2 model trains on clean rectangular crops but receives irregular-shaped input at inference, non-roof context pixels confuse the model.

**Prevention:**
- Apply the Stage 1 mask as a binary multiplicative mask to the crop before Stage 2 inference (zero out non-roof pixels).
- Or pad the bounding box crop and supply the Stage 1 mask as an additional input channel to Stage 2.
- At a minimum, black-fill non-roof pixels so the model does not process sky/ground/neighboring-building texture.

---

### Pitfall 23: Multi-Resolution PV Dataset — Train/Test Contamination Across Resolution Tiers

**What goes wrong:** The multi-resolution PV dataset (Zenodo 5171712) contains images at three resolutions (0.8/0.3/0.1m). Some images at different resolutions may be derived from the same geographic area (multi-scale captures of the same site). If you split randomly across all resolution tiers, same-site images at different scales can end up in both train and test.

**Prevention:**
- Check the dataset documentation for site-level identifiers.
- If site-level splits are not provided, split by geographic tile/coordinate rather than by file index.
- Alternatively, train and evaluate on one resolution tier at a time to avoid confusion.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| AIRS tiling pipeline | Tile overlap leakage (Pitfall 1) | Split source images before tiling |
| AIRS tiling pipeline | Mask-image misalignment (Pitfall 2) | Use rasterio for both channels; verify transforms match |
| DGX environment setup | glibc mismatch (Pitfall 5) | Always work inside Docker; pin container image |
| DGX environment setup | Checkpoint loss to /scratch/ wipe (Pitfall 4) | Per-epoch rclone to Drive |
| DGX environment setup | rclone auth expiry (Pitfall 6) | Test upload before long run; use rclone not gdown |
| Training loop | Class imbalance (Pitfall 7) | BCE+Dice + pos_weight |
| Training loop | Encoder LR too high (Pitfall 10) | Differential LR: encoder 1e-4, decoder 1e-3 |
| Training loop | BatchNorm instability (Pitfall 11) | Batch size >=8; gradient accumulation |
| Evaluation | Per-batch IoU averaging (Pitfall 13) | Global TP/FP/FN accumulation |
| Evaluation | SMP logit thresholding (Pitfall 17) | Sigmoid before threshold at inference |
| Inference / demo | GSD hardcoded wrong (Pitfall 3) | Named config constant; explicit GSD parameter |
| Inference / demo | Tile seam artifacts (Pitfall 8) | Gaussian blending window |
| Stage 2 training | Resolution mismatch (Pitfall 20) | Explicit canonical GSD for Stage 2 |
| Stage 2 training | Solar panel class imbalance (Pitfall 21) | Focal/Dice loss; oversample positive crops |
| Stage 2 inference | Non-roof background bleed (Pitfall 22) | Mask-fill non-roof pixels before Stage 2 |

---

## Confidence Notes

| Area | Confidence | Notes |
|------|------------|-------|
| Tile overlap leakage | HIGH | Well-documented in remote sensing literature; AIRS split is pre-defined |
| Mask-image misalignment | HIGH | Known GDAL/PIL interaction issue; rasterio fix is standard |
| GSD error | HIGH | Arithmetic is deterministic; verified against project constants |
| /scratch/ non-persistence | HIGH | Stated explicitly in project context |
| Docker glibc | HIGH | CentOS 7 glibc 2.17 is a well-known DGX constraint |
| rclone auth | MEDIUM | Token behavior is well-documented; exact DGX config may vary |
| Albumentations API break | MEDIUM | v1→v2 changes are documented; pin version to mitigate |
| Stage 2 resolution mismatch | HIGH | Dataset GSD values are stated in project context; arithmetic is exact |
| Multi-res PV site contamination | MEDIUM | Depends on Zenodo 5171712 dataset structure — verify against dataset README |
| BatchNorm batch size | HIGH | Standard deep learning training knowledge |
| IoU metric accumulation | HIGH | torchmetrics JaccardIndex behavior is documented |

---

## Sources

- AIRS dataset paper: Chen et al., 2019, ISPRS (cited in PROJECT.md)
- segmentation_models_pytorch documentation (smp.readthedocs.io)
- albumentations documentation and v2 migration guide (albumentations.ai)
- torchmetrics JaccardIndex documentation (torchmetrics.readthedocs.io)
- Zenodo dataset 5171712: Multi-resolution dataset for photovoltaic panel segmentation
- BDAPPV dataset: Nature Scientific Data (aerial PV panel masks)
- PyTorch BCEWithLogitsLoss documentation (numerical stability note)
- NVidia DGX documentation: /scratch/ volume non-persistence behavior
- rclone documentation: OAuth2 token refresh behavior (rclone.org)

*Note: WebSearch was unavailable during this research session. All findings are based on training knowledge (cutoff Aug 2025). Claims marked MEDIUM confidence should be verified against current documentation before implementation.*
