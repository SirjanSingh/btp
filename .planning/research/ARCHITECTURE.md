# Architecture Patterns

**Project:** Rooftop & Solar Panel Segmentation (AIRS + BDAPPV)
**Dimension:** Architecture
**Researched:** 2026-03-30
**Overall confidence:** HIGH for encoder/decoder choices (well-established literature), MEDIUM for exact AIRS leaderboard positions (no live benchmark access)

---

## Context and Target

- Must beat PSPNet baseline: IoU > 0.899, F1 > 0.947
- Input: 512×512 RGB crops from 7.5 cm/pixel aerial imagery
- Stage 1: building/rooftop segmentation (binary, large objects)
- Stage 2: solar panel segmentation within rooftop crops (binary, small objects, sparse)
- Framework: segmentation_models_pytorch (SMP) — primary library
- Compute: Colab T4/A100 + DGX A100 multi-GPU

---

## Recommended Architecture

### Stage 1: Rooftop Segmentation

**Primary recommendation: UNet++ with EfficientNet-B4 encoder**

Rationale: UNet++ adds dense skip connections that capture multi-scale roof geometry better than vanilla U-Net. EfficientNet-B4 gives a strong accuracy/compute tradeoff with compound scaling. On the SpaceNet building footprint benchmark and WHU aerial dataset (architecturally similar to AIRS), UNet++ with EfficientNet-B4 consistently reaches IoU 0.91–0.93, above the PSPNet 0.899 target.

**Secondary recommendation: DeepLabV3+ with ResNet-101 or EfficientNet-B5 encoder**

Rationale: ASPP (Atrous Spatial Pyramid Pooling) in the decoder handles large-area rooftops well by aggregating multi-scale context without losing resolution. ResNet-101 with output stride 16 + ASPP is what PSPNet competes against in most aerial benchmarks. DeepLabV3+ adds a lightweight decoder that sharpens roof boundaries.

**Tertiary for experimentation: Swin-T / Swin-S backbone with UPerNet decoder**

Rationale: Swin Transformer captures long-range dependencies across large flat industrial rooftops. On ISPRS Vaihingen and Potsdam (closest public benchmarks to AIRS), Swin-based segmenters consistently rank above CNN baselines by 1–2 IoU points. Caveat: requires pre-trained weights and higher GPU memory (~12 GB for Swin-S at 512×512 batch 8).

### Stage 2: Solar Panel Segmentation (within rooftop crops)

**Primary recommendation: UNet++ with EfficientNet-B2 encoder + CBAM attention**

Rationale: Solar panels are small, spectrally distinct objects. A lighter encoder (B2) is appropriate because crops are already normalized to roof regions. CBAM (Convolutional Block Attention Module) in skip connections helps suppress non-panel roof texture. On the BDAPPV and DeepSolar/multi-res PV datasets, attention-UNet variants outperform plain U-Net by 2–4 IoU points for panel detection.

**Alternative: SegFormer-B2**

Rationale: SegFormer's hierarchical transformer encoder with all-MLP decoder handles variable panel sizes well across the 0.1–0.8 m/pixel range. At 0.075 m/pixel (AIRS resolution), panels may be 20–80 pixels wide — SegFormer B2's mix transformer captures this range efficiently. Inference is faster than Swin due to no windowed attention overhead at small input sizes.

---

## Encoder Comparison for Stage 1

| Encoder | Params | ImageNet Top-1 | Aerial Building IoU (est.) | Memory (batch=8, 512px) | Notes |
|---------|--------|---------------|---------------------------|------------------------|-------|
| ResNet-34 | 21M | 73.3% | 0.89–0.91 | ~4 GB | Current starting point; good baseline |
| ResNet-50 | 25M | 76.1% | 0.90–0.92 | ~6 GB | Worthwhile upgrade from R34 |
| ResNet-101 | 45M | 77.4% | 0.91–0.92 | ~9 GB | Diminishing returns over R50 for binary seg |
| EfficientNet-B4 | 19M | 82.9% | 0.91–0.93 | ~7 GB | Recommended: best accuracy/param ratio |
| EfficientNet-B5 | 30M | 83.6% | 0.92–0.94 | ~10 GB | Strong alternative if memory allows |
| MobileNetV3-Large | 5.4M | 75.2% | 0.87–0.89 | ~3 GB | Only for fast prototyping, likely below target |
| Swin-T | 28M | 81.3% | 0.92–0.94 | ~11 GB | Transformer, stronger long-range but slower |
| Swin-S | 50M | 83.0% | 0.93–0.95 | ~14 GB | Best expected IoU, needs A100 |

Confidence: MEDIUM. Ranges synthesized from SpaceNet, WHU, ISPRS Vaihingen benchmarks — not AIRS-specific measurements.

---

## Decoder Comparison for Stage 1

| Decoder | Key Mechanism | Strength | Weakness | SMP Support |
|---------|--------------|----------|----------|-------------|
| UNet | Skip connections + bilinear upsampling | Simple, strong boundary preservation | Limited global context | Yes |
| UNet++ | Dense nested skips | Better multi-scale feature reuse | More memory, slower | Yes |
| DeepLabV3+ | ASPP + lightweight decoder | Strong context aggregation | Less precise boundaries | Yes |
| FPN | Multi-scale feature pyramid | Fast, good for objects at varied scales | Weaker than UNet++ for binary | Yes |
| PSPNet | Pooling pyramid module | Global context (AIRS baseline) | Fixed pooling scales | Yes |
| PAN | Feature aggregation from all levels | Good balance | Less common, less community support | Yes |
| MAnet | Multi-scale attention | Strong for aerial imagery | Compute-heavy | Yes |

**Recommended decoder progression:**
1. Start: UNet (baseline, fast)
2. Experiment 1: UNet++ (primary upgrade path)
3. Experiment 2: DeepLabV3+ (alternative strategy)
4. If resources allow: MAnet or UPerNet+Swin

---

## Attention Mechanisms Worth Adding

### CBAM (Convolutional Block Attention Module)
- Can be inserted into any CNN encoder's residual blocks
- Channel attention suppresses irrelevant spectral features (vegetation, shadows)
- Spatial attention sharpens boundary localization
- Cost: +3–5% params, +1–2 IoU points on boundary-critical tasks
- Confidence: HIGH (well-documented on aerial datasets)

### Self-Attention / Non-Local Blocks
- Useful for industrial flat rooftops where spatial context matters
- Place after encoder stage 4 (deepest features)
- Memory cost is quadratic in spatial size — only viable at stride-16 feature maps
- Confidence: MEDIUM

### Squeeze-Excitation (SE)
- Already built into EfficientNet variants — free channel recalibration
- If using ResNet, upgrade to SE-ResNet-50 for minimal cost
- Confidence: HIGH

---

## Multi-Scale Fusion Strategy

The AIRS paper's best model (FPN+MSFF, IoU=0.888) uses multi-scale feature fusion. To surpass PSPNet (0.899), multi-scale handling is mandatory.

**Recommended approach: ASPP-style pooling at decoder bottleneck**

```
Encoder bottleneck features (C x H/32 x W/32)
    → ASPP block (rates: 6, 12, 18 for 512px input, effectively 48, 96, 144 RF)
    → Concatenate with decoder path
    → Standard decoder upsampling
```

At 512×512 input, ASPP rates of 6/12/18 correspond to receptive fields of ~48/96/144 pixels — appropriate for medium (5m) to large (10m) roof spans at 7.5 cm/pixel.

Alternative: Use SPP (Spatial Pyramid Pooling) with pool sizes [1, 2, 4, 8] — simpler than ASPP, comparable for very large objects.

---

## Component Boundaries

```
Stage 1 Pipeline
├── Input: 512×512 RGB crop (from 10k×10k tile)
├── Encoder: EfficientNet-B4 pretrained on ImageNet
│   ├── Stage 1: 112×112 features (stride 4)
│   ├── Stage 2: 56×56 features (stride 8)
│   ├── Stage 3: 28×28 features (stride 16)
│   └── Stage 4: 16×16 features (stride 32)
├── Bottleneck: ASPP or SPP module
├── Decoder: UNet++ nested dense blocks
│   ├── Skip from stride-32 → 32×32
│   ├── Skip from stride-16 → 64×64
│   ├── Skip from stride-8 → 128×128
│   └── Skip from stride-4 → 256×256
├── Head: 1×1 conv → sigmoid → binary mask 512×512
└── Loss: BCE + Dice (0.5/0.5 weight) + optional Boundary loss

Stage 2 Pipeline
├── Input: variable-size rooftop crop (resized to 256×256 or 384×384)
├── Encoder: EfficientNet-B2 pretrained on ImageNet
├── Attention: CBAM in skip connections
├── Decoder: UNet++ or standard UNet
├── Head: 1×1 conv → sigmoid → binary panel mask
└── Loss: BCE + Dice (panels are sparse, high FP risk)
```

---

## Tiling and Stitching Strategy for 10k×10k Images

This is a critical implementation concern. The AIRS images are 10,000×10,000 pixels — inference must be done in tiles then stitched.

### Tiling Parameters

```
Tile size:   512×512  (matches training resolution)
Overlap:     10% → 51px  (matches AIRS paper)
Stride:      512 - 51 = 461px
Grid:        ceil((10000 - 51) / 461) ≈ 22 tiles per axis = ~484 tiles per image
```

For training, random crops are acceptable. For inference, use systematic sliding window with overlap.

### Stitching Strategy: Weighted Average Blending

Problem: Raw tile boundaries produce seam artifacts — the network has seen only centered context during training, so edges are lower quality.

Solution: Gaussian or Hann window weighting before averaging overlapping regions.

```python
# Concept (not runnable pseudocode)
weight_map = gaussian_2d(tile_size=512, sigma=0.4)  # peaks at center, near-zero at edges

output_sum = zeros(10000, 10000)
weight_sum = zeros(10000, 10000)

for each tile at (row, col):
    pred = model(tile)  # [0,1] probability map
    output_sum[row:row+512, col:col+512] += pred * weight_map
    weight_sum[row:row+512, col:col+512] += weight_map

final_mask = (output_sum / weight_sum) > threshold
```

This is the standard approach in medical imaging (nnU-Net uses it) and transfers directly to aerial imagery. Confidence: HIGH.

### Alternative: Test-Time Augmentation (TTA) per tile

For each tile, run inference on: original + horizontal flip + vertical flip + 90/180/270 rotations, then average predictions. Adds 8x compute but typically gains 0.3–0.8 IoU points on test set. Use for final evaluation runs, not development.

### Tile Size Trade-off

| Tile Size | Coverage | Memory | Context | Notes |
|-----------|----------|--------|---------|-------|
| 256×256 | Narrow | ~2 GB | Limited global context | Under-performs on large roofs |
| 512×512 | Standard | ~5 GB | Good | Matches AIRS paper, recommended |
| 640×640 | Wider | ~8 GB | Better for factory roofs | YOLOv5/8 default, worth trying |
| 1024×1024 | Full factory | ~18 GB | Captures large objects whole | A100 only, may need batch=1 |

Recommendation: Train at 512×512. For inference on DGX A100 (80 GB), try 1024×1024 with Gaussian blending to handle large factory roofs — this specifically addresses the known failure mode of large factory rooftops being clipped.

### Boundary Loss for Tile Seams

Add a boundary-weighted loss term during training to force the model to learn precise edge predictions. This reduces stitching artifacts at tile boundaries by improving per-edge prediction quality:

```
L_total = 0.5 * L_BCE + 0.4 * L_Dice + 0.1 * L_boundary
```

Boundary loss weights pixels near the ground-truth mask edge more heavily. SMP does not include this natively — implement with morphological dilation of the mask as the weight map.

---

## Data Flow

```
10k×10k AIRS image
    ↓ Sliding window tiling (512×512, stride 461, overlap 51)
    ↓ Normalization (ImageNet mean/std)
    ↓ [Stage 1 Model] → per-tile probability maps
    ↓ Gaussian-weighted stitching → full-image probability map
    ↓ Threshold (0.5, or optimize on val set)
    → Full-image binary rooftop mask

Full-image rooftop mask
    ↓ Connected component labeling (scipy.ndimage or cv2)
    ↓ Bounding box extraction per component (+ 10px padding)
    ↓ Crop and resize to 256×256
    ↓ [Stage 2 Model] → panel probability map
    ↓ Threshold
    → Per-rooftop panel mask

Panel mask
    ↓ Count foreground pixels × (0.075m)² = area in m²
    ↓ Panel efficiency factor (default 0.15–0.20 kW/m²)
    → Estimated solar capacity in kW
```

---

## Stage 2 Specific Considerations

### Input Resolution Issue

At 7.5 cm/pixel (AIRS resolution), a 1.6m × 1.0m residential solar panel occupies approximately 21 × 13 pixels. After cropping a 512×512 rooftop region and resizing it for Stage 2, panels may shrink further.

Recommended approach:
- Do NOT resize the full rooftop crop to a fixed small size
- Instead, slide a 256×256 window over the rooftop crop at the original resolution
- This preserves panel pixel density for detection

For BDAPPV dataset (which has varying GSD 0.1–0.8 m/pixel), apply GSD normalization: resample all images to a common GSD before training Stage 2. Target GSD: 0.1 m/pixel (highest resolution in the dataset, preserves panel detail).

### Class Imbalance in Stage 2

Even within a rooftop crop, panels typically occupy 5–30% of the area. Use:
1. Focal loss component (gamma=2) alongside Dice loss to suppress easy background negatives
2. Positive pixel oversampling: during training, ensure crops contain at least one panel (reject crops with zero panel coverage)

### Panel-Specific Augmentations

Panels have fixed aspect ratios and appear at regular grid angles. Useful augmentations:
- GridDistortion (simulates panel warping/lens distortion)
- RandomRotate90 (panels appear at all orientations)
- ColorJitter (panels have spectrally similar appearance to sky-reflected surfaces)
- Do NOT use heavy elastic transforms — destroys the geometric regularity that helps the model

### Architecture Note: Instance Segmentation is Overkill for Stage 2

Tempting to use Mask R-CNN or YOLOv8-seg for individual panel instances, but:
- Semantic segmentation (binary mask) is sufficient for area estimation
- BDAPPV provides semantic masks, not instance annotations
- Instance segmentation adds detection overhead without improving capacity estimates
- Defer to instance segmentation only if panel counting per roof is a hard requirement

---

## Scalability Considerations

| Concern | Colab T4 (16 GB) | DGX A100 (80 GB×8) |
|---------|-----------------|---------------------|
| Batch size (512px) | 8–12 | 32–64 per GPU (256+ total) |
| Swin-S training | Marginal (batch=4) | Comfortable (batch=16/GPU) |
| Full 857-image training | ~6 hrs/epoch | ~40 min/epoch |
| 1024px tile inference | OOM at batch>1 | Comfortable batch=4 |
| Multi-GPU | Not available | DDP via PyTorch DistributedDataParallel |

DGX setup note: Use `torchrun --nproc_per_node=N` for DDP training. SMP models are standard PyTorch nn.Module — DDP wraps directly without modification.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Training on Full 10k×10k Images
**What:** Passing full-resolution tiles through the model directly.
**Why bad:** GPU OOM, meaningless context beyond 512px anyway, incompatible with SMP decoders.
**Instead:** Always tile. Use the sliding window pipeline described above.

### Anti-Pattern 2: Naive Tile Boundary Concatenation
**What:** Stitching predicted tiles by hard-concatenation at boundary edges.
**Why bad:** Produces visible seam artifacts; inconsistent predictions near tile edges lower full-image IoU.
**Instead:** Gaussian-weighted blending in overlapping regions.

### Anti-Pattern 3: Using a Deep Encoder for Stage 2 on Small Crops
**What:** Applying the same EfficientNet-B4 Stage 1 encoder to 64×64 panel crops.
**Why bad:** Receptive field exceeds input size; deep features are spatially meaningless; wasteful.
**Instead:** EfficientNet-B2 or B0 for small crops; retain spatial resolution with output_stride=8.

### Anti-Pattern 4: Skipping ImageNet Pretraining
**What:** Training from random initialization on AIRS.
**Why bad:** AIRS has 857 training images — insufficient for learning good low-level features from scratch. ImageNet pretraining provides essential texture/edge detectors.
**Instead:** Always use imagenet pretrained weights. SMP sets this by default.

### Anti-Pattern 5: Binary Cross-Entropy Only
**What:** Using only BCE as the training loss.
**Why bad:** BCE is sensitive to class imbalance (large background vs small rooftop boundaries). The model will predict mostly background and still achieve low loss.
**Instead:** BCE + Dice (0.5/0.5) as minimum. Add boundary-weighted loss for boundary IoU improvement.

### Anti-Pattern 6: Fixed Threshold at 0.5
**What:** Applying threshold=0.5 at inference without validation-set optimization.
**Why bad:** For imbalanced classes, optimal threshold is often 0.35–0.45. Can swing IoU by 0.005–0.02.
**Instead:** Sweep threshold on validation set and pick argmax(IoU).

---

## Recommended Experiment Order

Phase 1 (baseline establishment):
1. UNet + ResNet-34 (current starting point) — establish reproducible training loop
2. UNet + ResNet-50 — quick encoder upgrade, should +0.5–1 IoU
3. UNet + EfficientNet-B4 — primary encoder upgrade target

Phase 2 (decoder upgrade):
4. UNet++ + EfficientNet-B4 — primary architecture recommendation
5. DeepLabV3+ + EfficientNet-B4 — alternative decoder strategy
6. MAnet + EfficientNet-B4 — if UNet++ and DeepLabV3+ both fall short

Phase 3 (transformer, if CNN approaches plateau below target):
7. SegFormer-B2 (via mmsegmentation or HuggingFace transformers)
8. Swin-S + UPerNet (via mmsegmentation)

Stop at Phase 2 if UNet++ + EfficientNet-B4 exceeds 0.900 IoU on val set.

---

## Key Papers and References

All citations from training knowledge — confidence HIGH for papers published before August 2025.

| Paper | Relevance | Finding |
|-------|-----------|---------|
| Chen et al. 2019, ISPRS (AIRS paper) | Direct baseline | PSPNet = 0.899 IoU, FPN+MSFF = 0.888 |
| Zhou et al. 2018, UNet++ | Primary decoder recommendation | Dense nested skips improve boundary IoU on medical/aerial |
| Chen et al. 2018, DeepLabV3+ | Decoder alternative | ASPP + lightweight decoder, strong on large objects |
| Tan & Le 2019, EfficientNet | Encoder recommendation | Compound scaling; B4 is best accuracy/compute point |
| Woo et al. 2018, CBAM | Attention mechanism | Channel + spatial attention, 1–2 IoU improvement |
| Liu et al. 2021, Swin Transformer | Transformer backbone | SOTA on Vaihingen/Potsdam aerial segmentation |
| Xie et al. 2021, SegFormer | Stage 2 alternative | Hierarchical mix-transformer, efficient at small scales |
| Isensee et al. 2021, nnU-Net | Tiling/stitching reference | Gaussian blending for overlap regions — medical imaging |
| Ji et al. 2018, BDAPPV | Stage 2 dataset | Aerial PV masks, semantic segmentation ground truth |

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Encoder ranking (CNN) | HIGH | Well-established on aerial building benchmarks through 2024 |
| Decoder comparison | HIGH | SMP docs + multiple benchmark papers |
| AIRS-specific IoU numbers | MEDIUM | Extrapolated from analogous benchmarks, not AIRS leaderboard |
| Swin/SegFormer aerial perf | MEDIUM | ISPRS benchmarks are close but not identical to AIRS setup |
| Stage 2 panel detection | MEDIUM | BDAPPV results exist but exact IoU on AIRS-resolution crops is estimated |
| Tiling/stitching | HIGH | Standard practice in nnU-Net and aerial inference pipelines |
| DGX DDP setup | HIGH | Standard PyTorch DDP, SMP is vanilla nn.Module |

---

## Gaps to Address in Phase-Specific Research

1. **Exact AIRS leaderboard state (2024–2025):** Cannot verify current top entries without web access. Swin-based models likely appear; exact IoU ceilings unknown.
2. **BDAPPV-specific architecture ablations:** Stage 2 IoU targets are estimated. Need to run baseline on BDAPPV to set concrete targets before architecture experiments.
3. **Optimal ASPP dilation rates for 7.5 cm/pixel:** Standard rates (6/12/18) designed for Cityscapes (~3 cm/pixel equivalent). May need adjustment to (3/6/12) for AIRS scale.
4. **GSD mismatch between AIRS (7.5 cm) and BDAPPV (10–80 cm):** Stage 2 will need a GSD normalization preprocessing step whose optimal target GSD requires empirical testing.
5. **Multi-GPU training on DGX with SMP:** SyncBatchNorm wrapping required for DDP; batch norm statistics are per-GPU by default. Verify SMP's default BN type and wrap if needed.
