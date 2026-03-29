# Feature Landscape: Aerial Rooftop & Solar Panel Segmentation Pipeline

**Domain:** Two-stage aerial image segmentation (rooftop detection + solar panel detection)
**Researched:** 2026-03-30
**Confidence:** MEDIUM-HIGH (training knowledge through Aug 2025; no live web search available)

---

## Table Stakes

Features that a working pipeline must have. Missing any of these and the pipeline is incomplete or unreliable.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Tile/crop pipeline for large images | AIRS images are 10,000×10,000px — must be sliced to 512×512 crops; naive resizing destroys fine structure at 7.5cm/pixel | Medium | Overlap (~10%) prevents border artifacts; reconstruct full mask by stitching crops back with max-voting on overlapping regions |
| Dataset class (PyTorch) for AIRS | Loads image+mask pairs, applies crop indexing, handles train/val/test splits | Low | Must handle `.tif` rasters (rasterio) and paired binary mask files |
| Dataset class for BDAPPV / Zenodo PV | Different folder layout and mask format from AIRS; Stage 2 expects rooftop-cropped input patches | Low | Multi-resolution dataset (0.1/0.3/0.8m) requires GSD-aware normalization |
| BCE + Dice combined loss | Pure BCE does not directly optimize IoU; Dice handles class imbalance from sparse panel pixels | Low | Typical weight: 0.5 BCE + 0.5 Dice; log both components separately for debugging |
| IoU / F1 / Precision / Recall per epoch | Standard segmentation metrics; IoU is the primary AIRS benchmark metric | Low | Compute at threshold=0.5; also compute mIoU across foreground+background classes |
| Training loop with LR scheduler | Cosine annealing or ReduceLROnPlateau with warmup; without it, runs diverge or plateau early | Low | Log LR per step for debugging |
| Validation loop with early stopping | Prevents overfitting; AIRS train set is large (857 images × many crops) | Low | Patience=10 epochs on val IoU; save best checkpoint |
| Checkpoint save/load | DGX /scratch/ is not persistent — must push to Google Drive every N epochs | Low | Save: model weights, optimizer state, epoch, best val IoU; Drive sync via rclone or gdown |
| Standard augmentation suite | AIRS paper uses: H-flip, V-flip, 0/90/180/270° rotation, random scale, Gaussian noise | Low | Implemented in albumentations; use `ReplayBuffer` so same transform applies to image+mask |
| Visual mask overlay outputs | Required for BTP demo evaluation — qualitative inspection of predictions | Low | Overlay predicted mask (semi-transparent) on original RGB crop; color-code by class |
| Area estimation from mask | Pixel count × GSD² → m²; GSD=0.075m → 56.25cm²/pixel for AIRS | Low | Stage 1: rooftop area; Stage 2: solar panel area per rooftop; aggregate to image-level totals |
| Capacity estimation | Solar area (m²) × efficiency factor → kW; standard assumption: ~150–180 W/m² for commercial panels | Low | Must be clearly labelled as an estimate; document assumed efficiency |
| End-to-end inference script | Single image in → rooftop mask + solar mask + area/capacity report out | Medium | Handles tiling, stitching, Stage 1→Stage 2 handoff, and output rendering automatically |
| Evaluation script on test set | Runs inference on full test split, computes aggregate IoU/F1, saves per-image overlay images | Medium | Produces a results CSV + summary table for report |
| Confusion matrix / per-class breakdown | Needed to diagnose known failure modes (trees, construction, CBD) | Low | Log TP/FP/FN/TN counts; derive precision/recall per image then aggregate |
| Reproducibility fixtures | Fixed random seeds (torch, numpy, random); pinned library versions in requirements.txt | Low | Critical for BTP academic reporting |

---

## Differentiators

Features that meaningfully improve performance or usability beyond the baseline. Not required for a first working version, but often what separates IoU=0.88 from IoU=0.91+.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Test-Time Augmentation (TTA) | Inference-time H/V flips + rotations → average predictions; typically +0.5–1.5 IoU points at zero training cost | Low | ttach library or manual; apply to both stages; most impactful single inference improvement |
| Multi-scale inference | Run crops at multiple resolutions, upsample and average logits; helps catch both large factory roofs and fine panel edges | Medium | Requires careful GSD tracking; most useful for Stage 2 where panel size varies |
| Overlap-weighted tile stitching | Instead of simple paste, weight tile contributions by distance to tile edge (Gaussian or linear ramp); eliminates seam artifacts visible at tile boundaries | Medium | Essential when tiles overlap — naive max-vote leaves grid-pattern artifacts on large roofs |
| CRF post-processing (DenseCRF) | Sharpens mask boundaries using color/position compatibility; historically +0.3–0.8 IoU on building datasets | High | pydensecrf library; adds ~2s/image inference overhead; less impactful with modern encoders but still useful for jagged roof edge cleanup |
| Mixed-precision training (AMP) | 2× speedup + 2× memory reduction on DGX A100 GPUs; enables larger batch sizes | Low | `torch.cuda.amp.autocast()` + GradScaler; now standard in PyTorch; almost free to add |
| Multi-GPU DataParallel / DDP | Utilizes all DGX GPUs; linear batch size scaling; reduces full AIRS training from ~8h to ~2h | Medium | Use DistributedDataParallel (DDP) over DataParallel; requires launcher script (torchrun) |
| Cosine annealing with warm restarts | Better final accuracy than step LR; avoids plateau trapping; pairs well with Dice loss | Low | `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` |
| Boundary-aware loss (boundary Dice / edge weighting) | Penalizes errors near roof boundaries more heavily; improves edge sharpness metrics | Medium | Compute binary edge map from mask, upweight loss at boundary pixels; important for accurate area estimation |
| Hard negative / hard positive mining | AIRS CBD failure mode = many false negatives on irregular roofs; focal loss or OHEM addresses this | Medium | Focal loss is the simpler drop-in; OHEM requires custom sampler |
| Focal loss for Stage 2 (panel detection) | Solar panels are rare pixels (~2–10% of rooftop area); BCE/Dice both struggle; focal loss suppresses easy negatives | Low | gamma=2.0 is standard starting point; combine with Dice: FocalDice loss |
| Pretrained encoder fine-tuning schedule | Freeze encoder for first N epochs, then unfreeze with lower LR; prevents destroying ImageNet features on small datasets | Low | Relevant for Stage 2 where PV training set is smaller than AIRS |
| WandB / TensorBoard experiment tracking | Logs loss curves, metric curves, LR, sample predictions per epoch; essential for comparing runs | Low | WandB is easier for remote DGX; TensorBoard works offline; pick one and be consistent |
| Per-image IoU histogram | Shows distribution of easy vs hard images; flags which geographic areas (CBD, industrial) are failing | Low | Produce at eval time; one extra plot in the BTP report |
| Rooftop instance separation | Current binary mask merges adjacent rooftops; watershed or connected-component labeling separates them for per-building capacity estimates | Medium | `scipy.ndimage.label` or `skimage.segmentation.watershed`; needed for realistic per-building output |
| GSD-adaptive crop size | Stage 2 PV dataset spans 0.1–0.8m/pixel; crop size should scale with GSD to keep physical footprint constant | Medium | E.g., at 0.3m/pixel use 256×256 crops to match ~19m physical footprint of 512×512 at 0.075m/pixel |
| Calibrated confidence threshold selection | Default 0.5 threshold is rarely optimal; sweep threshold on val set, pick threshold maximizing F1 | Low | Plot precision-recall curve; report threshold used in final evaluation |

---

## Anti-Features

Features to explicitly NOT build for this BTP project.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| End-to-end single-stage detection (Mask R-CNN, etc.) | Adds instance detection complexity without clear accuracy benefit over two-stage approach for this dataset; harder to explain in BTP report | Keep two-stage: rooftop binary seg → crop → panel binary seg |
| Web deployment / REST API | Out of scope per PROJECT.md; significant devops overhead with no academic benefit | Colab notebook demo is sufficient for BTP |
| Financial ROI / energy production forecasting | Requires weather data, orientation data, shadowing calculations — out of scope | Cap output at kW capacity estimate |
| Multi-spectral / LiDAR fusion | AIRS is RGB-only; adding modalities requires dataset surgery and re-architecture | Document as future work |
| Video / streaming inference | Real-time is out of scope; single image is the deliverable | N/A |
| Custom architecture design (novel blocks) | BTP timeline is short; architectural novelty is not the goal — beating baselines with a well-tuned U-Net is | Use SMP's well-tested implementations |
| WHU dataset as primary training data | Adds dataset management complexity; AIRS is primary; WHU may be added for Stage 1 augmentation only if AIRS results are below target | Add WHU only if IoU < 0.895 on AIRS val |
| Panoptic segmentation | Requires instance + semantic joint training; far more complex than binary segmentation | Out of scope |

---

## Aerial-Specific Features General Tutorials Miss

These are features that standard segmentation tutorials omit but are critical for aerial rooftop/panel work.

### Tile Pipeline Details

**Overlap-aware stitching is not trivial.** A naive tile-and-paste pipeline produces grid artifacts at tile boundaries that are visible as discontinuous masks across the 10,000×10,000px image. Required: overlap percentage tracking, blend weights (Gaussian or linear taper at tile edges), and mask aggregation with float accumulation before final thresholding.

**Mask label format for AIRS.** AIRS masks are roof outlines (not filled footprints). The mask file encodes roof polygon boundaries. Filling these to binary raster masks is a preprocessing step that must happen before training — not during the DataLoader.

**Coordinate-correct crop indexing.** When tiling, the (row, col) index of each crop must be stored alongside the crop for stitching reconstruction. This metadata is not optional — without it, reconstructed masks are spatially scrambled.

### GSD Calibration

**Area estimation requires GSD fidelity.** Any resizing (interpolation) of the mask before area calculation corrupts the pixel count. The mask must be resized back to original resolution at native pixel scale before multiplying by GSD². Report area estimates with an explicit ±N% uncertainty band covering GSD calibration error.

**Stage 2 GSD mismatch.** AIRS is 7.5cm/pixel; BDAPPV and Zenodo 5171712 contain images at 0.8/0.3/0.1m/pixel. These cannot be mixed naively. Either: (a) normalize all Stage 2 crops to a single target GSD by rescaling, or (b) condition the model on GSD as an auxiliary input. Option (a) is simpler.

### Class Imbalance in Stage 2

Solar panels occupy a small fraction of rooftop area (typically 5–30% of the rooftop mask). Standard BCE loss will converge to predicting all-background. This must be addressed with Dice loss, Focal loss, or positive-weight rescaling before any meaningful training signal appears.

### Augmentation Constraints

**Rotation augmentation must be label-preserving.** A 90° rotation applied to an image must apply the identical 90° rotation to its paired mask. albumentations handles this correctly with the dual-transform API. OpenCV-based augmentation written naively often drops the mask transform.

**Scale augmentation in aerial context.** Random scale augmentation changes the effective GSD. This is valid as a data augmentation strategy (the model sees apparent object size variation) but the augmented crops must NOT be used for area estimation calibration — only inference on unaugmented images at known GSD.

### Evaluation Against Published Baselines

**IoU calculation method must match the AIRS paper.** Chen et al. (2019) compute IoU at the image level and average across test images (macro-average). Some implementations compute IoU over the entire concatenated test set (micro-average), which gives different numbers. Clarify and use the same method as the baseline paper.

**Threshold matters for F1/Precision/Recall.** IoU is threshold-sensitive. The AIRS baselines use 0.5. Report metrics at 0.5 for comparability but also report the optimal-threshold result separately.

### Stage 1 → Stage 2 Handoff

**Error propagation.** If Stage 1 misses a rooftop, Stage 2 never sees it — the false negative propagates silently. At evaluation time, report Stage 1 FN rate and its estimated impact on total solar capacity detection.

**Rooftop crop padding.** When extracting Stage 2 crops from Stage 1 predictions, add a small padding (e.g., 16–32px) around the predicted rooftop bounding box. Tight crops cut off panel edges near the roof boundary, reducing Stage 2 recall.

**Minimum area threshold.** Stage 1 produces noisy small blobs in addition to real rooftops (shadow patches, parked cars, etc.). Apply a minimum connected-component area threshold (e.g., 50m² = ~889 pixels at AIRS GSD) before feeding Stage 2 to avoid wasted inference on false positives.

---

## Feature Dependencies

```
Tile pipeline (overlap-aware)
  └─→ AIRS Dataset class
        └─→ Stage 1 training loop
              └─→ Stage 1 checkpointing
                    └─→ Stage 1 evaluation (IoU/F1)
                          └─→ Stage 1→Stage 2 handoff (rooftop crops)
                                └─→ PV Dataset class (BDAPPV/Zenodo)
                                      └─→ Stage 2 training loop
                                            └─→ Stage 2 evaluation (IoU/F1)
                                                  └─→ Area estimation (GSD calibration)
                                                        └─→ Capacity estimation
                                                              └─→ End-to-end demo

Visual overlays ──────────────────────────────────────→ Required by: evaluation script, demo
WandB/TensorBoard ────────────────────────────────────→ Required by: training loop (both stages)
Mixed-precision AMP ──────────────────────────────────→ Required by: training loop (both stages)
TTA ──────────────────────────────────────────────────→ Required by: evaluation / demo inference
Overlap-weighted stitching ───────────────────────────→ Required by: end-to-end inference
Connected-component labeling ─────────────────────────→ Required by: Stage 1→Stage 2 handoff
```

---

## MVP Recommendation

Build in this order to reach a working BTP demo fastest:

**Phase 1 — Data Foundation (build first, everything else depends on it)**
1. AIRS tile crop pipeline with overlap — correctly paired image+mask crops
2. AIRS Dataset class (PyTorch) — loads crops, applies albumentations augmentations
3. AIRS mask preprocessing — fill roof outlines to binary rasters

**Phase 2 — Stage 1 Training**
4. U-Net ResNet-34 (SMP) with BCE+Dice loss
5. Training loop with cosine LR, AMP, WandB logging
6. Checkpoint save to Drive every epoch
7. Validation loop with IoU/F1/Precision/Recall

**Phase 3 — Stage 1 Evaluation**
8. Evaluation script on AIRS test set — aggregate metrics, per-image IoU histogram
9. Visual overlay output for qualitative inspection
10. TTA at inference (highest ROI improvement for least code)

**Phase 4 — Stage 2 Training**
11. PV Dataset class (BDAPPV + Zenodo) with GSD normalization
12. Stage 1→Stage 2 handoff: rooftop crop extraction with padding + min-area filter
13. Stage 2 training loop with Focal+Dice loss (class imbalance)

**Phase 5 — End-to-End Demo**
14. Inference script: image in → stitched rooftop mask → solar mask per rooftop → area/capacity report
15. Overlap-weighted stitching for Stage 1 reconstruction
16. Connected-component labeling for per-rooftop instance output

**Defer to post-BTP:**
- CRF post-processing (high complexity, modest gain)
- Multi-GPU DDP (single-GPU training is sufficient if DGX schedule allows)
- GSD-adaptive crop sizing (simplify: normalize all Stage 2 data to one GSD)
- WHU dataset integration (add only if Stage 1 IoU stuck below 0.895)

---

## Sources

**Confidence notes:**

| Claim | Confidence | Basis |
|-------|------------|-------|
| AIRS tile/overlap pipeline requirements | HIGH | PROJECT.md + AIRS paper conventions (Chen et al. 2019 ISPRS) |
| BCE+Dice loss for building segmentation | HIGH | Standard in SMP documentation and building segmentation literature |
| TTA giving +0.5–1.5 IoU points | MEDIUM | Consistent across multiple aerial segmentation papers; not verified against current live sources |
| CRF +0.3–0.8 IoU | MEDIUM | Historical DenseCRF results; less impactful with modern encoders — may be lower in practice |
| Solar panel class imbalance (5–30%) | MEDIUM | Derived from BDAPPV dataset characteristics in published paper; exact figure varies by scene |
| Focal gamma=2.0 standard | MEDIUM | RetinaNet paper default; widely adopted |
| Minimum rooftop area threshold (50m²) | LOW | Engineering heuristic; requires calibration on AIRS val set |
| Capacity estimate 150–180 W/m² | MEDIUM | Standard commercial panel efficiency range as of 2024; verify with project supervisor for academic citation |

- Chen, J. et al. (2019). "Aerial Imagery for Roof Segmentation: A Large-Scale Dataset towards Real-World Applications." ISPRS Journal of Photogrammetry and Remote Sensing. (AIRS paper — baseline source)
- Kasmi, G. et al. (2023). "BDAPPV: A dataset for rooftop photovoltaic panels detection." Nature Scientific Data.
- Ioffe & Szegedy conventions for batch norm / encoder freezing
- PyTorch AMP documentation (torch.cuda.amp) — HIGH confidence, stable API
- albumentations dual-transform API for paired image+mask augmentation — HIGH confidence
- segmentation-models-pytorch (SMP) U-Net + ResNet-34 architecture — HIGH confidence
