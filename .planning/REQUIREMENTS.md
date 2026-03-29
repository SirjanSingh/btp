# Requirements: Rooftop & Solar Panel Segmentation

**Defined:** 2026-03-30
**Core Value:** Given an aerial image, produce accurate rooftop masks and solar panel masks — and translate pixel counts into actionable solar capacity estimates.

## v1 Requirements

### Data Pipeline

- [ ] **DATA-01**: Dataset folder structure on Google Drive matches project spec (`rooftop_solar/datasets/airs/{train,val,test}/{images,masks}/`)
- [ ] **DATA-02**: Download script for AIRS dataset via `gdown` or Drive mount (Colab-compatible)
- [ ] **DATA-03**: Tile cropping pipeline converts 10,000×10,000px AIRS images into 512×512 crops with 10% overlap, discarding near-empty mask crops
- [ ] **DATA-04**: PyTorch `Dataset` class for AIRS crops (loads image + binary mask, applies augmentation)
- [ ] **DATA-05**: `DataLoader` with configurable batch size, num_workers, pin_memory for both Colab and DGX
- [ ] **DATA-06**: Augmentation pipeline: horizontal/vertical flips, 0/90/180/270° rotation, random scale, Gaussian noise (matches AIRS paper)
- [ ] **DATA-07**: PV dataset preparation (BDAPPV + Zenodo 5171712) with same crop/augment pipeline

### Stage 1 — Rooftop Segmentation

- [ ] **ROOF-01**: U-Net with ResNet-34 encoder (via `segmentation_models_pytorch`) trains end-to-end on AIRS
- [ ] **ROOF-02**: Combined BCE + Dice loss function implemented and verified
- [ ] **ROOF-03**: Training loop with per-epoch logging of loss, IoU, F1, Precision, Recall (train and val)
- [ ] **ROOF-04**: Checkpoint saving to Google Drive after each epoch (DGX non-persistent `/scratch/` constraint)
- [ ] **ROOF-05**: Best-model checkpoint tracked by validation IoU
- [ ] **ROOF-06**: Full-image inference via tiled sliding window with overlap stitching (handles 10k×10k inputs)
- [ ] **ROOF-07**: Test-set evaluation produces IoU and F1 scores comparable to AIRS paper baselines
- [ ] **ROOF-08**: Visualization: predicted mask overlaid on aerial image (color overlay, side-by-side, boundary only)

### Stage 2 — Solar Panel Segmentation

- [ ] **PV-01**: Fine-tune Stage 1 U-Net backbone on BDAPPV + multi-res PV dataset
- [ ] **PV-02**: Input is cropped rooftop region extracted from Stage 1 prediction
- [ ] **PV-03**: PV model produces binary mask for solar panels within rooftop crop
- [ ] **PV-04**: Test-set evaluation with IoU and F1 on PV dataset

### Area Estimation & Output

- [ ] **AREA-01**: Rooftop area calculation: `pixel_count × GSD²` (AIRS: 0.075m/pixel → 56.25cm²/pixel)
- [ ] **AREA-02**: Solar panel area calculation from Stage 2 mask using same formula
- [ ] **AREA-03**: Solar capacity estimate: `solar_area_m2 × efficiency_factor` (configurable, default 150W/m²)
- [ ] **AREA-04**: Output summary: rooftop area (m²), solar area (m²), estimated capacity (kW)

### Demo & Evaluation

- [ ] **DEMO-01**: End-to-end Colab notebook: upload/select aerial image → Stage 1 rooftop mask → Stage 2 PV mask → area/capacity output
- [ ] **DEMO-02**: Visual output: original image + rooftop overlay + solar panel overlay in single figure
- [ ] **DEMO-03**: Works on at least 3 diverse sample images from AIRS test set (residential, commercial, industrial)

### Infrastructure

- [ ] **INFRA-01**: DGX Docker environment setup script (Ubuntu 22.04, install all dependencies)
- [ ] **INFRA-02**: `screen` session management guide for persistent DGX training
- [ ] **INFRA-03**: `rclone` or `gdown` script to sync checkpoints Drive ↔ DGX `/scratch/`

## v2 Requirements

### Architecture Improvements

- **ARCH-01**: FPN encoder comparison against U-Net baseline
- **ARCH-02**: PSPNet comparison to verify beating paper baseline
- **ARCH-03**: Multi-Scale Feature Fusion (MSFF) module as per FPN+MSFF from AIRS paper
- **ARCH-04**: Test-time augmentation (TTA) with averaging for improved inference IoU
- **ARCH-05**: Larger crop size (640×640 or 768×768) experiment to address large factory roof failure

### Data & Training

- **DATA-08**: WHU Building Dataset integration as secondary training data source
- **DATA-09**: Hard negative mining for CBD failure cases
- **TRAIN-01**: Learning rate scheduler experiments (cosine annealing, OneCycleLR)
- **TRAIN-02**: Mixed precision training (torch.cuda.amp) for DGX throughput

### Analysis

- **EVAL-01**: Per-category error analysis (residential vs commercial vs industrial)
- **EVAL-02**: Failure case visualization gallery (tree occlusion, construction, factory roofs)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-spectral / LiDAR input | AIRS is RGB only; adds sensor dependency |
| Real-time video inference | Single-image pipeline sufficient for BTP demo |
| Web deployment / hosted app | Colab notebook demo is sufficient for BTP |
| Financial ROI calculations | Beyond engineering scope; solar capacity (kW) is sufficient |
| Instance segmentation (individual buildings) | Semantic segmentation sufficient for area estimation |
| WHU dataset in v1 training | AIRS alone is sufficient; WHU deferred to v2 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| DATA-05 | Phase 1 | Pending |
| DATA-06 | Phase 1 | Pending |
| INFRA-01 | Phase 1 | Pending |
| INFRA-02 | Phase 1 | Pending |
| INFRA-03 | Phase 1 | Pending |
| ROOF-01 | Phase 2 | Pending |
| ROOF-02 | Phase 2 | Pending |
| ROOF-03 | Phase 2 | Pending |
| ROOF-04 | Phase 2 | Pending |
| ROOF-05 | Phase 2 | Pending |
| ROOF-06 | Phase 3 | Pending |
| ROOF-07 | Phase 3 | Pending |
| ROOF-08 | Phase 3 | Pending |
| AREA-01 | Phase 3 | Pending |
| AREA-02 | Phase 3 | Pending |
| AREA-03 | Phase 3 | Pending |
| AREA-04 | Phase 3 | Pending |
| DATA-07 | Phase 4 | Pending |
| PV-01 | Phase 4 | Pending |
| PV-02 | Phase 4 | Pending |
| PV-03 | Phase 4 | Pending |
| PV-04 | Phase 4 | Pending |
| DEMO-01 | Phase 5 | Pending |
| DEMO-02 | Phase 5 | Pending |
| DEMO-03 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 28 total
- Mapped to phases: 28
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-30*
*Last updated: 2026-03-30 — traceability corrected after roadmap creation (INFRA-01/02/03 → Phase 1; DEMO-01/02/03 → Phase 5)*
