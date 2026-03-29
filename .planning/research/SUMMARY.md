# Research Summary: Rooftop & Solar Panel Segmentation Pipeline

**Synthesized:** 2026-03-30
**Sources:** STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md
**Synthesizer confidence:** HIGH for architecture and stack decisions; MEDIUM for exact IoU numbers on AIRS (no live benchmark access)

---

## Executive Summary

This is a two-stage binary semantic segmentation pipeline on aerial imagery. Stage 1 detects building rooftops on 10,000x10,000px AIRS images (7.5 cm/pixel GSD); Stage 2 detects solar panels within rooftop crops using the BDAPPV and Zenodo multi-resolution PV datasets. The pipeline must beat the published AIRS PSPNet baseline of IoU = 0.899 and produce an area-based solar capacity estimate (kW) as a BTP demo deliverable.

The recommended approach is to build a tiled-inference pipeline using segmentation-models-pytorch (SMP) with a U-Net/UNet++ architecture and an EfficientNet-B4 encoder. Start with a ResNet-34 U-Net baseline to validate the training loop end-to-end, then upgrade the encoder and decoder in a controlled experiment sequence. The SMP library eliminates architecture boilerplate and makes encoder swaps a one-line change; this is the correct abstraction level for a 4-month BTP timeline. All training augmentation should use albumentations with paired image+mask transforms, and all raster I/O should go through rasterio to preserve GSD metadata essential for area estimation.

The dominant risks fall into two categories: data pipeline correctness (tile/val leakage, mask-image misalignment, GSD hardcoding, Stage 2 resolution mismatch) and infrastructure fragility (DGX /scratch/ non-persistence, Docker glibc mismatch, rclone auth expiry). Every one of these risks causes silent failure — metrics look plausible but are wrong, or an overnight run is simply lost. Prevention requires explicit checks and tests baked in before training begins, not retrofitted after anomalous results appear.

---

## Key Findings

### From STACK.md

| Technology | Pinned Version | Critical Note |
|------------|----------------|---------------|
| PyTorch | 2.1.2 (DGX) / 2.2.x (Colab) | DGX base: `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime` |
| torchvision | 0.16.2 | Must version-match PyTorch exactly |
| segmentation-models-pytorch | 0.3.3 | Core library; handles encoder/decoder/loss boilerplate |
| timm | 0.9.16 | SMP 0.3+ requires explicit timm pin; encoder names use `tu-` prefix |
| albumentations | 1.4.3 | Use `-headless` OpenCV variant on DGX |
| numpy | 1.26.4 | NumPy 2.x breaks albumentations and older SMP — do not upgrade yet |
| opencv-python-headless | 4.9.0.80 | Non-headless variant crashes on DGX (libGL missing) |
| rasterio | 1.3.9 | Do NOT pip-install GDAL separately — bundled in rasterio wheel |
| torchmetrics | 1.3.2 | Use `JaccardIndex(task="binary")` for correct global IoU |

Key insight: GDAL must not be installed separately. NumPy 2.x is a hard blocker for this stack. Docker base image must be chosen to match the DGX host CUDA driver version (check via `nvidia-smi` before selecting image).

### From FEATURES.md

**Table stakes (must have before BTP demo):**
- Overlap-aware tiling pipeline for 10,000x10,000px AIRS images (512x512 crops, 10% overlap)
- AIRS mask preprocessing: fill roof outline polygons to binary rasters before training
- PyTorch Dataset classes for AIRS (Stage 1) and BDAPPV/Zenodo PV (Stage 2) with GSD normalization
- BCE + Dice combined loss (0.5/0.5) with separate component logging
- Training loop: cosine LR scheduling, AMP mixed precision, WandB/TensorBoard logging
- Checkpoint save to Google Drive every epoch (DGX /scratch/ is not persistent)
- Validation loop with IoU/F1/Precision/Recall and early stopping
- Evaluation script: aggregate metrics + per-image overlays + results CSV
- End-to-end inference script: image in → stitched rooftop mask → solar mask → area/capacity report
- Area estimation: pixel count x GSD^2 = m^2, with capacity at 150-180 W/m^2

**High-value differentiators (worth adding within BTP timeline):**
- Test-Time Augmentation (TTA): H/V flips + 90/180/270 rotations, averaged predictions — free +0.5-1.5 IoU, lowest complexity improvement available
- Mixed-precision AMP: near-free 2x memory reduction and speedup
- Gaussian-weighted tile stitching (not naive concatenation)
- Overlap-weighted stitching eliminates grid artifacts on large rooftops
- Calibrated threshold: sweep val set, report both 0.5 and optimal-threshold results

**Defer to post-BTP:**
- CRF post-processing (high complexity, modest gain with modern encoders)
- Multi-GPU DDP (single-GPU is sufficient; add only if training time is a bottleneck)
- WHU dataset integration (add only if Stage 1 IoU stalls below 0.895)
- Instance segmentation / Mask R-CNN

**Aerial-specific details general tutorials miss:**
- AIRS masks are roof outlines, not filled footprints — pre-fill to binary rasters in preprocessing
- Stage 1→Stage 2 handoff must include: connected-component labeling, minimum area filter (~50m^2 = ~889 pixels at AIRS GSD), and 16-32px bounding-box padding
- Stage 2 resolution: BDAPPV/Zenodo are 0.1-0.8m/pixel, not 0.075m — need explicit GSD normalization before training Stage 2
- Area estimation must use pre-resize native resolution mask (do not compute on a resized copy)
- IoU method must match AIRS paper: macro-average over test images, not micro-average

### From ARCHITECTURE.md

**Recommended architecture — start here:**

Stage 1 (rooftop segmentation):
- Baseline: `smp.Unet(encoder_name="resnet34")` — validates training loop
- Primary target: `smp.UnetPlusPlus(encoder_name="tu-efficientnet_b4")` — best accuracy/compute for aerial building benchmarks
- Alternative: `smp.DeepLabV3Plus(encoder_name="resnet101")` — strong on large-area objects via ASPP

Stage 2 (solar panel segmentation):
- Primary: `smp.UnetPlusPlus(encoder_name="tu-efficientnet_b2")` + CBAM attention — lighter encoder appropriate for already-cropped roof regions
- Alternative: SegFormer-B2 — handles variable panel sizes across GSD range efficiently

**Experiment sequence for Stage 1:**
1. UNet + ResNet-34 (baseline)
2. UNet + ResNet-50 (quick +0.5-1 IoU)
3. UNet + EfficientNet-B4 (primary encoder target)
4. UNet++ + EfficientNet-B4 (decoder upgrade — stop here if IoU > 0.900)
5. DeepLabV3+ + EfficientNet-B4 (alternative if UNet++ falls short)
6. Swin/SegFormer (only if CNN approaches plateau below target)

**Tiling parameters (established, high confidence):**
- Tile size: 512x512 (training and inference)
- Overlap: 10% = 51px; stride = 461px
- Grid per 10,000x10,000 image: ~22x22 = ~484 tiles
- Stitching: Gaussian/Hann window weighted averaging (not max-vote or hard concatenation)
- Loss: `0.5 * L_BCE + 0.4 * L_Dice + 0.1 * L_boundary` (optional boundary term)

**Key anti-patterns:**
- Never train or infer on full 10,000x10,000px images (GPU OOM, no benefit)
- Never hard-concatenate tile predictions (seam artifacts)
- Never apply EfficientNet-B4 to small Stage 2 crops (receptive field exceeds input)
- Never skip ImageNet pretraining (AIRS has only 857 images — insufficient for scratch training)
- Never use BCE-only loss (converges to predicting all-background)
- Never use fixed 0.5 threshold without val-set sweep

### From PITFALLS.md

**Top 5 pitfalls by severity and BTP-specific risk:**

1. **Tile overlap causes train/val leakage** (Silent IoU inflation of 3-8 points)
   Prevention: Split at the source image level first, tile each split independently, use the official AIRS 857/94/96 split as-is. Never tile the whole dataset then split tiles.

2. **GSD hardcoded wrong** (Area estimates wrong by 100x; demo is broken silently)
   Prevention: Use named constants (`GSD_AIRS_M = 0.075`), pass GSD explicitly to all area functions, add a unit test. Stage 2 must use a separate GSD constant.

3. **DGX /scratch/ wipes checkpoints** (Overnight run lost entirely; BTP timeline hit)
   Prevention: Save checkpoints to Google Drive via rclone at end of every epoch. Test the upload path before starting any long run. Use `screen -S training` + redirect output to a log file.

4. **Mask-image spatial misalignment** (Boundary loss corrupted; F1 plateaus silently)
   Prevention: Use rasterio for both image and mask. Assert `src_img.transform == src_mask.transform`. Visually inspect 20 random overlay samples before training.

5. **Stage 2 resolution mismatch** (Stage 2 recall collapses; panels appear much smaller than during training)
   Prevention: Define a canonical Stage 2 inference GSD, resample all training data and inference inputs to that GSD, add an assertion at Stage 2 entry point.

**Additional high-risk pitfalls:**
- Docker glibc mismatch: always work inside the container, never use host Python
- rclone/gdown auth expiry: test upload during short session before overnight run
- SMP model outputs logits not probabilities: apply `torch.sigmoid()` before thresholding
- Per-batch IoU averaging: use `torchmetrics.JaccardIndex` with global accumulation
- Albumentations dual-transform: always pass image+mask together in one call, never separately
- Differential LR: encoder at 1e-4, decoder at 1e-3; single LR degrades encoder features

---

## Implications for Roadmap

The feature dependency chain is strict. Nothing in Stage 2 works until Stage 1 produces valid rooftop masks. Nothing in Stage 1 works until the tiling pipeline is correct and the data split is clean. This dictates the phase structure.

### Suggested Phase Structure

**Phase 1 — Data Foundation**
Rationale: Every subsequent phase depends on correct tiling, valid splits, and preprocessed masks. Getting this wrong silently corrupts everything downstream. This phase must be validated with visual inspection before any model is trained.
Delivers: Tiled AIRS crops with correct train/val/test split; filled binary mask rasters; AIRS PyTorch Dataset class
Critical pitfalls: Pitfall 1 (tile leakage), Pitfall 2 (mask misalignment)
Research flag: LOW — tiling and rasterio patterns are well-documented

**Phase 2 — Stage 1 Baseline Training**
Rationale: Establish a reproducible training loop that reaches a measurable IoU before any architecture experimentation. A working loop with ResNet-34 U-Net validates loss, metrics, checkpointing, and logging infrastructure.
Delivers: Trained ResNet-34 U-Net with IoU > 0.88 on AIRS val; WandB experiment tracking; Drive checkpointing confirmed working
Critical pitfalls: Pitfall 4 (checkpoint loss), Pitfall 7 (class imbalance), Pitfall 10 (encoder LR), Pitfall 13 (IoU accumulation), Pitfall 17 (sigmoid/logit)
Research flag: LOW — SMP training loop is standard

**Phase 3 — Stage 1 Architecture Upgrade**
Rationale: Systematic encoder/decoder upgrades to exceed the PSPNet IoU = 0.899 baseline. One change at a time, tracked via WandB.
Delivers: Best Stage 1 model (target: UNet++ + EfficientNet-B4, IoU > 0.900); TTA at inference; Gaussian-weighted stitching; threshold calibration; evaluation script with per-image overlays
Critical pitfalls: Pitfall 8 (tile seams), Pitfall 12 (large factory roofs), Pitfall 14 (TTA consistency)
Research flag: LOW — experiment sequence is clear from architecture research

**Phase 4 — Stage 2 Training (Solar Panel Detection)**
Rationale: Depends entirely on a validated Stage 1 model. GSD normalization must be resolved before any Stage 2 training begins.
Delivers: BDAPPV/Zenodo Dataset class with GSD normalization; Stage 1→Stage 2 handoff (connected components, padding, min-area filter); Stage 2 UNet++ + EfficientNet-B2 with Focal+Dice loss; Stage 2 evaluation
Critical pitfalls: Pitfall 20 (resolution mismatch), Pitfall 21 (solar panel class imbalance), Pitfall 22 (non-roof background bleed), Pitfall 23 (multi-res train/test contamination)
Research flag: MEDIUM — optimal canonical GSD for Stage 2 requires empirical testing; BDAPPV-specific architecture baselines are estimated, not measured

**Phase 5 — End-to-End Demo**
Rationale: Integration of both stages into a single inference script with area/capacity output. This is the BTP demo deliverable.
Delivers: Single-image inference pipeline (tiling → Stage 1 → rooftop crops → Stage 2 → area/capacity report); visual overlay outputs; evaluation on full test set; reproducibility fixtures; final report metrics
Critical pitfalls: Pitfall 3 (GSD hardcoding in capacity calculation), Pitfall 8 (tile seams in full-image output)
Research flag: LOW — integration patterns are clear

### Research Flags

| Phase | Research Needed? | Reason |
|-------|-----------------|--------|
| Phase 1 | No | Tiling + rasterio patterns are well-established |
| Phase 2 | No | SMP training loop is standard; ResNet-34 baseline is documented |
| Phase 3 | No | Experiment sequence is clear; stop at UNet++ + EfficientNet-B4 if IoU > 0.900 |
| Phase 4 | YES | Optimal Stage 2 canonical GSD needs empirical calibration; BDAPPV architecture baselines are estimated |
| Phase 5 | No | Integration patterns follow directly from Phase 3 and 4 outputs |

---

## Confidence Assessment

| Area | Confidence | Basis |
|------|------------|-------|
| Stack (versions, pinning rules) | HIGH | PyTorch Docker tags and version compatibility rules are deterministic; NumPy 2.x break is documented |
| Training architecture (SMP, encoders) | HIGH | Well-established on SpaceNet/WHU/ISPRS Vaihingen benchmarks analogous to AIRS |
| AIRS-specific IoU numbers | MEDIUM | Extrapolated from analogous benchmarks; exact AIRS leaderboard state (2025) unverifiable without web access |
| Tiling and stitching | HIGH | nnU-Net Gaussian blending is standard practice; parameters match AIRS paper |
| Stage 2 architecture | MEDIUM | BDAPPV results exist but exact IoU on AIRS-resolution rooftop crops is estimated |
| Pitfalls (data/infra) | HIGH | Tile leakage, /scratch/ persistence, glibc mismatch, rasterio alignment — all are deterministic or well-documented |
| GSD-based area/capacity estimation | HIGH | Arithmetic is exact given known GSD values from dataset documentation |
| Multi-res PV dataset site contamination | MEDIUM | Depends on Zenodo 5171712 dataset structure — verify against dataset README before Stage 2 split |

**Overall confidence: HIGH for Phases 1-3; MEDIUM for Phase 4 (Stage 2 GSD and baseline targets)**

---

## Gaps to Address During Planning

1. **Exact AIRS leaderboard state (2024-2025):** Cannot verify whether Swin-based models have been submitted; the PSPNet 0.899 target may have already been superseded. Confirm with project supervisor or by checking the AIRS benchmark page.

2. **Stage 2 canonical GSD:** The optimal resolution to normalize BDAPPV/Zenodo training data before Stage 2 training requires an empirical pilot. Target 0.1m/pixel as the starting assumption (highest-resolution tier in BDAPPV), but validate before committing.

3. **BDAPPV-specific Stage 2 IoU baseline:** No measured IoU numbers for UNet++ + EfficientNet-B2 on BDAPPV. Set a baseline by running the ResNet-34 U-Net on BDAPPV first, then measure improvement from architecture upgrades.

4. **ASPP dilation rates at 7.5cm/pixel:** Standard rates (6/12/18) target Cityscapes-equivalent scale. For AIRS, rates (3/6/12) may be more appropriate. Requires a small ablation if DeepLabV3+ is used.

5. **SyncBatchNorm on DGX DDP:** If multi-GPU training is needed, SMP's default BatchNorm needs SyncBatchNorm wrapping for DDP. Verify SMP's BN type and add `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` before DDP wrapping.

6. **rclone configuration on DGX:** The exact persistent bind-mount path and token refresh behavior depends on the specific LNMIIT DGX setup. Test upload/download before any long training run.

---

## Sources (Aggregated)

| Source | Confidence | Files |
|--------|------------|-------|
| Chen et al. 2019, ISPRS (AIRS paper) | HIGH | All 4 files |
| segmentation-models-pytorch documentation and GitHub (qubvel-org) | HIGH | STACK, ARCHITECTURE |
| PyTorch Docker Hub — pinned image tags | HIGH | STACK |
| NumPy 2.0 release notes (June 2024) | HIGH | STACK |
| Kasmi et al. 2023, Nature Scientific Data (BDAPPV) | HIGH | FEATURES, PITFALLS |
| Zhou et al. 2018, UNet++ paper | HIGH | ARCHITECTURE |
| Chen et al. 2018, DeepLabV3+ paper | HIGH | ARCHITECTURE |
| Tan & Le 2019, EfficientNet paper | HIGH | ARCHITECTURE, STACK |
| Isensee et al. 2021, nnU-Net (Gaussian blending reference) | HIGH | ARCHITECTURE |
| Woo et al. 2018, CBAM paper | HIGH | ARCHITECTURE |
| Liu et al. 2021, Swin Transformer | HIGH | ARCHITECTURE |
| albumentations v2 migration guide | MEDIUM | STACK, PITFALLS |
| torchmetrics JaccardIndex documentation | HIGH | STACK, PITFALLS |
| NVidia DGX /scratch/ documentation | HIGH | PITFALLS |
| rclone OAuth2 token refresh documentation | MEDIUM | PITFALLS |
| Zenodo dataset 5171712 (multi-res PV) | MEDIUM | PITFALLS, FEATURES |

*All research from training knowledge (cutoff August 2025). No live web search was available. Versions marked [VERIFY] in STACK.md should be confirmed on PyPI before creating the DGX Docker image.*
