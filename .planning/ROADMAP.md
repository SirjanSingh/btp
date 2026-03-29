# Roadmap: Rooftop & Solar Panel Segmentation

## Overview

Five phases that build the two-stage aerial segmentation pipeline from scratch to BTP demo. Phase 1
is a pure data engineering phase — no model training until it passes visual inspection. Phases 2 and
3 are the Stage 1 gauntlet: establish a reproducible training loop, then drive IoU past the PSPNet
0.899 baseline through systematic architecture upgrades. Phase 4 builds Stage 2 solar panel
detection on top of validated Stage 1 outputs. Phase 5 stitches both stages into the end-to-end demo
that is the BTP deliverable.

---

## Phases

- [ ] **Phase 1: Data Foundation** — Tile AIRS images correctly, validate splits, produce clean training crops
- [ ] **Phase 2: Stage 1 Baseline Training** — End-to-end U-Net (ResNet-34) training loop on AIRS, IoU > 0.88
- [ ] **Phase 3: Stage 1 Architecture Upgrade** — Systematic encoder/decoder upgrades to beat PSPNet (IoU > 0.900)
- [ ] **Phase 4: Stage 2 Solar Panel Segmentation** — Train PV detection model on BDAPPV/Zenodo using Stage 1 rooftop crops
- [ ] **Phase 5: End-to-End Demo & Evaluation** — Full-image inference pipeline, area/capacity output, BTP demo

---

## Phase Details

### Phase 1: Data Foundation

**Goal**: Produce correctly-tiled, split-safe AIRS crops with filled binary masks, ready for model
training without any data leakage or spatial misalignment.

**Depends on**: Nothing — start here today

**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, INFRA-01, INFRA-02, INFRA-03

**Deliverables:**
- Google Drive folder structure matching project spec (`rooftop_solar/datasets/airs/{train,val,test}/{images,masks}/`)
- `gdown` / Drive-mount download script for AIRS (Colab-compatible)
- Tile cropping script: 10,000×10,000px → 512×512 crops, 10% overlap (stride 461px), per official
  857/94/96 split — source images split BEFORE tiling, never after
- AIRS mask pre-fill script: roof outline GeoTIFF → filled binary raster (critical: do before training)
- PyTorch `AIRSDataset` class with albumentations augmentation pipeline (H/V flip, RandomRotate90,
  ShiftScaleRotate, GaussNoise, ImageNet normalization)
- `DataLoader` config for Colab (batch 8, num_workers 2) and DGX (batch 32, num_workers 8)
- DGX Docker environment: Dockerfile based on `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime` with
  pinned requirements.txt (numpy==1.26.4, opencv-python-headless, no standalone GDAL)
- `screen` session guide + `rclone` checkpoint-sync script tested against Drive

**Success Criteria:**
- [ ] Visual inspection of 20 random image-mask pairs shows mask contours align precisely with
      building edges (no pixel-level offset)
- [ ] No tile from the val or test crop directories shares pixel content with any train crop (verified
      by checking source image filenames in crop metadata)
- [ ] Tiled crop counts: train ~430k+, val ~47k+, test ~48k+ (or proportional subset if capped for
      speed) with near-empty crops discarded
- [ ] `AIRSDataset` loads a batch and `augmented_mask.sum() > 0` passes for every sample in a
      20-batch smoke test
- [ ] rclone uploads a 100 MB test file to Drive from inside the DGX Docker container without error

**Notes on pitfall prevention (Phase 1):**
- Pitfall 1 (tile leakage): split source images first, tile each split independently
- Pitfall 2 (mask misalignment): use rasterio for both image and mask; assert `transform` equality
- Pitfall 5 (Docker glibc): always work inside container; never call host Python

**Plans**: TBD

---

### Phase 2: Stage 1 Baseline Training

**Goal**: Establish a reproducible Stage 1 training loop that reaches IoU > 0.88 on the AIRS
validation set, confirming that loss, metrics, checkpointing, and logging infrastructure are all
correct before any architecture experimentation.

**Depends on**: Phase 1

**Requirements**: ROOF-01, ROOF-02, ROOF-03, ROOF-04, ROOF-05

**Deliverables:**
- U-Net with ResNet-34 encoder (`smp.Unet(encoder_name="resnet34", encoder_weights="imagenet")`)
- Combined BCE + Dice loss (0.5/0.5), differential LR (encoder 1e-4, decoder 1e-3)
- Training loop: cosine LR scheduler, AMP mixed precision, per-epoch IoU/F1/Precision/Recall logging
  (global TP/FP/FN accumulation via `torchmetrics.JaccardIndex`, NOT per-batch averaging)
- Checkpoint save to Google Drive every epoch via rclone callback (`best` + `latest` files)
- WandB (or TensorBoard) experiment tracking with loss curves, metric curves, sample predictions
- `screen -S training ... | tee` log redirect for persistent DGX sessions

**Success Criteria:**
- [ ] Validation IoU reaches > 0.880 within 30 epochs (matches FPN baseline tier from Chen et al. 2019)
- [ ] Loss curve is monotonically decreasing over the first 10 epochs with no NaN or inf values
- [ ] Best-model checkpoint verified loadable: reload weights and re-run one val batch, IoU matches
      saved value within 0.001
- [ ] Drive checkpoint directory shows a new file after each epoch (rclone working correctly)
- [ ] Training run survives a simulated SSH disconnect and resumes from the `screen` session without
      data loss

**Notes on pitfall prevention (Phase 2):**
- Pitfall 4 (/scratch/ wipe): per-epoch rclone to Drive; test before overnight run
- Pitfall 7 (class imbalance): BCE+Dice + `pos_weight` tuned from training set pixel stats
- Pitfall 10 (encoder LR): differential LR parameter groups from SMP's `model.encoder` / `model.decoder`
- Pitfall 13 (IoU accumulation): global JaccardIndex, not mean-of-batch-means
- Pitfall 17 (logit vs sigmoid): `torch.sigmoid()` applied at inference only; loss uses logits

**Plans**: TBD

---

### Phase 3: Stage 1 Architecture Upgrade

**Goal**: Systematically upgrade the Stage 1 model until test-set IoU exceeds 0.900, beating the
PSPNet baseline of 0.899 (Chen et al. 2019, ISPRS).

**Depends on**: Phase 2

**Requirements**: ROOF-06, ROOF-07, ROOF-08, AREA-01, AREA-02, AREA-03, AREA-04, DEMO-01, DEMO-02, DEMO-03

**Deliverables:**
- Encoder upgrade experiments tracked in WandB: ResNet-34 → ResNet-50 → EfficientNet-B4 (one change
  at a time; stop when IoU > 0.900)
- UNet++ decoder swap (`smp.UnetPlusPlus`) as the next lever after encoder upgrade
- Gaussian-weighted tile stitching for 10,000×10,000px full-image inference (replaces naive paste)
- Sliding-window inference pipeline: tiling → per-tile inference → Gaussian blend → binary mask
- Test-Time Augmentation (TTA): H/V flip + 90/180/270 rotations, predictions averaged before
  thresholding
- Validation-set threshold sweep: report both threshold=0.5 and optimal-threshold IoU
- Evaluation script: aggregate IoU/F1/Precision/Recall on full test split, per-image overlay PNG
  saved to Drive, results CSV
- Area estimation functions: `pixel_count × GSD²` using named constant `GSD_AIRS_M = 0.075`
- Rooftop area (m²), estimated solar capacity (kW, default 150 W/m²) output per image
- Visual output: original aerial + rooftop overlay + boundary-only view in one figure

**Success Criteria:**
- [ ] Best Stage 1 model achieves IoU > 0.900 on the official AIRS test split (96 images), measured
      using global TP/FP/FN accumulation matching the Chen et al. 2019 evaluation method
- [ ] Full-image (10,000×10,000px) inference on a test image produces a mask with no visible
      grid-pattern seam artifacts when viewed at full resolution
- [ ] Area estimation unit test passes: `area_from_mask(ones_100x100, gsd=0.075) == 56.25 m²`
      (within 1e-4 relative tolerance)
- [ ] Evaluation script produces a results CSV covering all 96 test images with per-image IoU, and at
      least 3 overlay PNGs saved to Drive for qualitative inspection

**Notes on pitfall prevention (Phase 3):**
- Pitfall 3 (GSD hardcoding): use `GSD_AIRS_M = 0.075` named constant; pass explicitly to area functions
- Pitfall 8 (tile seams): Gaussian blending window, not hard concatenation
- Pitfall 12 (large factory roofs): document as known limitation; optionally flag in per-image CSV
- Pitfall 14 (TTA consistency): decide and document whether reported numbers include TTA or not

**Plans**: TBD
**UI hint**: yes

---

### Phase 4: Stage 2 Solar Panel Segmentation

**Goal**: Train a solar panel segmentation model on BDAPPV and Zenodo multi-resolution PV data, with
GSD normalization that makes it compatible with Stage 1 rooftop crops at AIRS resolution.

**Depends on**: Phase 3

**Requirements**: DATA-07, PV-01, PV-02, PV-03, PV-04

**Deliverables:**
- PV dataset download and preparation script for BDAPPV + Zenodo 5171712
- `PVDataset` PyTorch class with GSD-normalized crops: all training images resampled to canonical
  Stage 2 GSD (target starting assumption: 0.1 m/pixel — validate empirically before committing)
- Site-level train/test split for Zenodo 5171712 (by geographic coordinate, not file index) to
  prevent cross-resolution contamination
- Stage 1 → Stage 2 handoff pipeline:
  - Connected-component labeling on Stage 1 mask (`scipy.ndimage.label`)
  - Minimum area filter: 50 m² (~889 pixels at AIRS GSD) to drop noise blobs
  - Bounding-box crop with 16–32 px padding
  - Non-roof pixel masking (zero-fill) before Stage 2 input
  - Resample crop to canonical Stage 2 GSD before inference
- Stage 2 model: `smp.UnetPlusPlus(encoder_name="tu-efficientnet_b2")` with Focal + Dice loss
  (gamma=2.0; panels are sparse — pure BCE collapses to all-background)
- GSD assertion at Stage 2 entry point: `assert abs(input_gsd - STAGE2_EXPECTED_GSD) < 1e-4`
- Stage 2 evaluation: IoU and F1 on held-out PV test split; per-image overlays saved to Drive

**Success Criteria:**
- [ ] Stage 2 model trains to IoU > 0.70 on PV test set (baseline target; establish ResNet-34 U-Net
      baseline first, then measure improvement from UNet++ + EfficientNet-B2)
- [ ] Handoff pipeline correctly isolates at least 3 individual rooftop instances from a test AIRS
      image: each crop contains only roof pixels (non-roof regions zeroed), at canonical Stage 2 GSD
- [ ] GSD assertion fires (raises AssertionError) when a raw AIRS crop (0.075 m/pixel) is passed
      directly to Stage 2 inference without resampling
- [ ] Stage 2 inference on a positive-label PV test crop produces a mask with recall > 0.50 (panels
      are detected, not suppressed to background)

**Notes on pitfall prevention (Phase 4):**
- Pitfall 20 (resolution mismatch): define `STAGE2_EXPECTED_GSD` constant; enforce at inference entry
- Pitfall 21 (solar panel class imbalance): Focal + Dice loss; oversample positive crops in batches
- Pitfall 22 (non-roof background bleed): zero-fill non-roof pixels from Stage 1 mask before Stage 2
- Pitfall 23 (multi-res site contamination): split Zenodo 5171712 by site coordinate, not file index

**Plans**: TBD

---

### Phase 5: End-to-End Demo & Evaluation

**Goal**: Integrate Stage 1 and Stage 2 into a single Colab notebook that accepts an aerial image
and returns rooftop mask, solar panel mask, rooftop area, solar area, and estimated solar capacity —
this is the BTP demo deliverable.

**Depends on**: Phase 4

**Requirements**: DEMO-01, DEMO-02, DEMO-03

**Deliverables:**
- End-to-end inference function: aerial image → tiled Stage 1 inference → Gaussian-blended rooftop
  mask → per-rooftop crop extraction → Stage 2 inference → per-rooftop panel mask → area/capacity report
- Colab demo notebook (`notebooks/demo_rooftop_segmentation.ipynb` extended or replaced) with:
  - File upload cell for any aerial image
  - Automatic tiling and full-image reconstruction
  - Side-by-side figure: original | rooftop overlay | solar panel overlay | capacity table
  - Works end-to-end in < 5 minutes on Colab T4 for a single 10,000×10,000px image
- Evaluation run on full AIRS test set (96 images): Stage 1 IoU/F1, Stage 2 IoU/F1, per-image
  overlays saved to Drive, summary table for BTP report
- Results on 3 diverse test images (residential, commercial/industrial, mixed) demonstrating the
  pipeline across scene types
- Fixed random seeds and pinned requirements.txt committed for reproducibility

**Success Criteria:**
- [ ] Single-image demo runs end-to-end (upload → capacity report) on a previously-unseen AIRS test
      image without manual intervention
- [ ] Combined output figure shows original image, rooftop overlay (orange), and solar panel overlay
      (blue/green) in one saved PNG
- [ ] Area and capacity numbers are plausible: a 200m² rooftop with 30% panel coverage reports
      ~30 m² solar area and ~4.5 kW estimated capacity (within 5% of manual calculation using
      GSD_AIRS_M = 0.075 and 150 W/m²)
- [ ] Pipeline tested on at least 3 test images from distinct scene types (residential, commercial,
      industrial) with IoU reported for each
- [ ] Stage 1 test-set IoU and F1 exceed PSPNet baselines: IoU > 0.899, F1 > 0.947

**Notes on pitfall prevention (Phase 5):**
- Pitfall 3 (GSD in demo): all area calls use named constants; demo cell displays GSD assumption explicitly
- Pitfall 8 (tile seams in full image): Gaussian blending confirmed in Phase 3 carries through here

**Plans**: TBD
**UI hint**: yes

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Foundation | 0/TBD | Not started | - |
| 2. Stage 1 Baseline Training | 0/TBD | Not started | - |
| 3. Stage 1 Architecture Upgrade | 0/TBD | Not started | - |
| 4. Stage 2 Solar Panel Segmentation | 0/TBD | Not started | - |
| 5. End-to-End Demo & Evaluation | 0/TBD | Not started | - |

---

## Requirement Traceability

All 28 v1 requirements mapped. No orphans.

| Requirement | Description | Phase |
|-------------|-------------|-------|
| DATA-01 | Drive folder structure matches spec | Phase 1 |
| DATA-02 | AIRS download script (gdown / Drive mount) | Phase 1 |
| DATA-03 | Tile cropping: 10k×10k → 512×512, 10% overlap, discard near-empty | Phase 1 |
| DATA-04 | PyTorch Dataset for AIRS crops (load + augment) | Phase 1 |
| DATA-05 | DataLoader with configurable batch/workers for Colab and DGX | Phase 1 |
| DATA-06 | Augmentation pipeline (flips, rotations, scale, noise) | Phase 1 |
| DATA-07 | PV dataset preparation (BDAPPV + Zenodo) | Phase 4 |
| ROOF-01 | U-Net ResNet-34 via SMP, end-to-end training on AIRS | Phase 2 |
| ROOF-02 | BCE + Dice loss implemented and verified | Phase 2 |
| ROOF-03 | Training loop: per-epoch loss, IoU, F1, Precision, Recall logging | Phase 2 |
| ROOF-04 | Checkpoint save to Drive every epoch | Phase 2 |
| ROOF-05 | Best-model checkpoint tracked by validation IoU | Phase 2 |
| ROOF-06 | Full-image inference via tiled sliding window with overlap stitching | Phase 3 |
| ROOF-07 | Test-set evaluation: IoU and F1 comparable to AIRS baselines | Phase 3 |
| ROOF-08 | Visualization: predicted mask overlaid on aerial image | Phase 3 |
| PV-01 | Fine-tune Stage 1 backbone on BDAPPV + multi-res PV dataset | Phase 4 |
| PV-02 | Input is cropped rooftop region from Stage 1 prediction | Phase 4 |
| PV-03 | PV model produces binary mask for solar panels within rooftop crop | Phase 4 |
| PV-04 | Test-set evaluation with IoU and F1 on PV dataset | Phase 4 |
| AREA-01 | Rooftop area from pixel count × GSD² | Phase 3 |
| AREA-02 | Solar panel area from Stage 2 mask using same formula | Phase 3 |
| AREA-03 | Solar capacity estimate: solar_area × efficiency_factor | Phase 3 |
| AREA-04 | Output summary: rooftop area (m²), solar area (m²), capacity (kW) | Phase 3 |
| DEMO-01 | End-to-end Colab notebook: image → rooftop mask → PV mask → area/capacity | Phase 5 |
| DEMO-02 | Visual output: original + rooftop overlay + solar overlay in one figure | Phase 5 |
| DEMO-03 | Works on 3 diverse sample images (residential, commercial, industrial) | Phase 5 |
| INFRA-01 | DGX Docker environment setup script | Phase 1 |
| INFRA-02 | `screen` session management guide | Phase 1 |
| INFRA-03 | rclone/gdown checkpoint sync script | Phase 1 |

**Coverage:** 28/28 v1 requirements mapped. 0 orphans.

---

## Note on `notebooks/demo_rooftop_segmentation.ipynb`

This notebook was created as an emergency BTP demo script before full pipeline development. It covers
the Stage 1 training path end-to-end on a single Colab session. Specifically:

**What it covers (Phase 1 + Phase 2 scope):**
- GPU check and dependency install (SMP, albumentations, rasterio, gdown)
- Google Drive mount and path configuration
- Configuration block: crop size, overlap, batch size, epochs, LR, GSD constant
- Tile cropping pipeline: `tile_image_mask()` and `prepare_crops()` — produces 512×512 crops from
  10k×10k source images, discards near-empty mask crops (min_mask_frac=0.01)
- `AIRSDataset` PyTorch class with albumentations augmentation (flips, rotations, noise, ImageNet normalization)
- `DataLoader` setup for both train and val splits
- U-Net ResNet-34 via SMP with combined BCE + Dice loss (0.5/0.5)
- Differential LR: encoder at `LR * 0.1`, decoder at `LR`
- Training loop with AMP, cosine LR scheduler, per-epoch IoU/F1/Precision/Recall logging
- Checkpoint saving to Drive (best model by val IoU)
- Training curve plots with PSPNet baseline reference lines
- Single-crop inference and area/capacity estimation from predicted mask
- Interactive upload cell for user-provided aerial images
- Full test-set evaluation loop

**What it does NOT cover (Phases 3–5 scope):**
- Architecture upgrade experiments (ResNet-50, EfficientNet-B4, UNet++)
- Gaussian-weighted tile stitching for full 10k×10k inference (uses crop-level inference only)
- Test-Time Augmentation (TTA)
- Validation-set threshold sweep
- Stage 2 solar panel model (PV dataset, handoff pipeline, PV training loop)
- Connected-component labeling and per-rooftop crop extraction
- GSD normalization for Stage 2 (BDAPPV / Zenodo resolution mismatch handling)
- Combined two-stage inference producing both rooftop and panel masks in one pass

**Known limitations vs. full pipeline:**
- IoU metric uses per-batch averaging, not global TP/FP/FN accumulation — will produce slightly
  different numbers from the AIRS paper evaluation method (addressed in Phase 2)
- Area estimation computes on resized inference crops, not native-GSD masks — may have minor error
  for large images where the full tiling pipeline is needed (addressed in Phase 3)
- No protection against tile leakage if the official AIRS 857/94/96 split is not respected when
  setting up the Drive folder (addressed in Phase 1 as the first task)

**Action for Phase 1:** Verify that the Drive folder structure loaded by this notebook matches the
official split before using it for any training run. The notebook's tiling code is otherwise valid
as a starting point for the Phase 1 tile cropping deliverable.
