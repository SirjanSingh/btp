# Rooftop & Solar Panel Segmentation

## What This Is

A two-stage deep learning pipeline for aerial imagery: Stage 1 segments rooftops from RGB aerial photos, Stage 2 segments solar panels within detected rooftops. Final output is rooftop area (m²), usable solar area (m²), and estimated solar capacity (kW). Built as a BTP (B.Tech Project) deliverable with an end-to-end interactive demo.

## Core Value

Given an aerial image, produce accurate rooftop masks and solar panel masks — and translate those pixel counts into actionable solar capacity estimates.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Stage 1: U-Net (ResNet-34) rooftop segmentation on AIRS dataset exceeding PSPNet baseline (IoU > 0.899)
- [ ] Stage 2: Solar panel segmentation within cropped rooftop regions using BDAPPV / multi-res PV dataset
- [ ] End-to-end inference demo: upload aerial image → rooftop mask → solar mask → area/capacity output
- [ ] Tile cropping pipeline: 10,000×10,000px AIRS images → 512×512 crops with 10% overlap
- [ ] PyTorch Dataset + DataLoader for AIRS and PV datasets
- [ ] Training loop with BCE+Dice loss, IoU/F1/Precision/Recall logging
- [ ] Evaluation script with visual mask overlays on test set
- [ ] Area estimation from predicted mask using GSD (0.075m/pixel → 56.25 cm²/pixel)
- [ ] DGX-compatible training setup (Docker Ubuntu 22.04, screen sessions, Drive checkpointing)

### Out of Scope

- Multi-spectral or LiDAR input — RGB only (AIRS constraint)
- Real-time video inference — single image at a time
- WHU dataset training in v1 — AIRS primary, WHU secondary/optional
- Web deployment — demo via Colab notebook, not a hosted app
- Financial ROI calculations beyond kW capacity estimate

## Context

**BTP project** at LNMIIT (username: 23ucs715 on lnmdgx1 DGX server). Target: beat AIRS paper baselines and deliver a working end-to-end demo.

**Compute setup:**
- Google Colab (prototyping, dataset prep, quick experiments) + Google Drive storage
- LNMIIT DGX server (`lnmdgx1`) for full training — Docker container (Ubuntu 22.04) required due to glibc 2.17 host constraint. Use `screen -S training` for persistent sessions.
- Local RTX 4050 laptop (light testing only)

**Datasets:**
- AIRS: 220k+ buildings, 7.5cm/pixel RGB, Christchurch NZ. 857/94/96 train/val/test images, each 10,000×10,000px. Roof outlines (not footprints). Stored on Google Drive, accessed via `gdown` or Drive mount.
- WHU Building Dataset (Aerial): 860km², 0.45m resolution — secondary/optional
- BDAPPV: aerial PV masks + metadata (Nature Scientific Data)
- Multi-resolution PV dataset (Zenodo: 5171712): 0.8/0.3/0.1m, 3,716 samples

**Baselines to beat (AIRS paper — Chen et al., 2019, ISPRS):**
- FPN: IoU=0.882, F1=0.937
- FPN+MSFF: IoU=0.888, F1=0.941
- PSPNet: IoU=0.899, F1=0.947 ← primary target

**Known failure modes:** tree occlusion, buildings under construction, large factory roofs (crop size limitation), central business district (irregular roofs, low recall).

**Key libraries:** `segmentation-models-pytorch`, `albumentations`, `rasterio`, `geopandas`, `shapely`, `torch`, `torchvision`, `opencv-python`, `gdown`

**Drive folder structure:**
```
MyDrive/rooftop_solar/
├── datasets/airs/{train,val,test}/{images,masks}/
├── datasets/whu/
├── datasets/pv_panels/
├── checkpoints/
├── logs/
└── notebooks/
```

## Constraints

- **Compute**: DGX requires Docker container (Ubuntu 22.04) — glibc 2.17 on host blocks modern binaries
- **Storage**: DGX `/scratch/` is not persistent — checkpoint frequently to Google Drive
- **Input resolution**: 512×512 or 640×640 crops — large factory roofs may be clipped
- **GSD**: AIRS is 7.5cm/pixel (56.25 cm²/pixel) — area estimates specific to this resolution
- **Deadline**: BTP demo required — end-to-end inference must work on real aerial image

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| U-Net with ResNet-34 encoder | AIRS paper baselines use FPN/PSPNet; U-Net is well-supported by SMP, good starting point | — Pending |
| BCE + Dice combined loss | BCE handles class imbalance signal, Dice directly optimizes overlap metric | — Pending |
| Stage 2 uses same backbone | Reduces complexity; rooftop crops normalize input domain for PV detection | — Pending |
| Colab for prototyping, DGX for full training | Colab has 15GB Drive + fast iteration; DGX has multi-GPU for full 857-image training | — Pending |
| 512×512 crop size | Matches AIRS paper setup; fits in GPU memory on both Colab and DGX | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-30 after initialization*
