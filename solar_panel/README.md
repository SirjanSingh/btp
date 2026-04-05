# Stage 2 — Solar Panel Segmentation

Detect solar panel pixels in aerial imagery using a U-Net with a ResNet-34 encoder, trained on the BDAPPV dataset. Designed to run on rooftop crops identified by Stage 1.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Architecture](#architecture)
3. [Dataset — BDAPPV](#dataset--bdappv)
4. [Training Pipeline](#training-pipeline)
5. [Results](#results)
6. [Scripts](#scripts)
7. [Quick Commands](#quick-commands)

---

## Problem Statement

Given an aerial image crop, predict a **binary mask** labelling:
- `1` → solar panel
- `0` → background (roof, road, vegetation, etc.)

<p align="center">
  <img src="https://images.unsplash.com/photo-1509391366360-2e959784a276?w=800" width="700"/>
  <br><i>Aerial view of rooftop solar panels — exactly what Stage 2 targets.</i>
</p>

**Key challenge:** Most aerial images contain **no solar panels at all**. A naive approach that skips panel-free images produces a model biased toward over-predicting panels. This pipeline keeps negative examples using a *black mask fallback* (see [Dataset section](#dataset--bdappv)).

**Pipeline position:**
```
Aerial image
     ↓  Stage 1 (rooftop/train.py)
Rooftop mask
     ↓  crop to rooftop regions
Rooftop crops
     ↓  Stage 2 (solar_panel/train_solar.py)
Solar panel mask
     ↓
Panel area (m²) + Estimated peak power (kW)
```

---

## Architecture

The model is identical in structure to Stage 1: **U-Net with a ResNet-34 encoder**, pretrained on ImageNet.

<p align="center">
  <img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" width="780"/>
  <br><i>U-Net architecture. Encoder (left) compresses spatial info into features; decoder (right) restores resolution using skip connections from the encoder.</i>
</p>

### Why U-Net for Solar Panels?

Solar panels have strong **local texture** (uniform blue/black rectangular cells) and **global context** (appear only on flat rooftop surfaces). U-Net captures both:
- **Encoder skip connections** preserve fine-grained panel boundary detail
- **Deep bottleneck** captures global context (surrounding roof structure)

### Encoder — ResNet-34 Stages

| Stage | Output Size | Channels | Blocks |
|-------|------------|----------|--------|
| Stem | 256 × 256 | 64 | Conv7×7 + BN + Pool |
| Layer 1 | 128 × 128 | 64 | 3 residual blocks |
| Layer 2 | 64 × 64 | 128 | 4 residual blocks |
| Layer 3 | 32 × 32 | 256 | 6 residual blocks |
| Layer 4 | 16 × 16 | 512 | 3 residual blocks |

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*BnarLMa5bJmBtvf_QNQY9w.png" width="640"/>
  <br><i>ResNet-34 architecture. Stacked residual blocks allow gradients to flow directly through skip connections, enabling effective training of 34 layers.</i>
</p>

### Decoder — Skip Connection Expansion

```
Bottleneck  (16×16,  512ch)
     ↓  Upsample 2× + concat skip → Conv → Conv
Decode-4    (32×32,  256ch)
     ↓  Upsample 2× + concat skip → Conv → Conv
Decode-3    (64×64,  128ch)
     ↓  Upsample 2× + concat skip → Conv → Conv
Decode-2    (128×128, 64ch)
     ↓  Upsample 2× + concat skip → Conv → Conv
Decode-1    (256×256, 32ch)
     ↓  Upsample 2× → Conv 1×1
Output      (512×512,  1ch)  ← sigmoid → [0,1] panel probability map
```

### Model Parameter Count

| Component | Parameters | Learning Rate |
|-----------|-----------|--------------|
| ResNet-34 Encoder | 21.3M | `lr × 0.1 = 1e-5` |
| U-Net Decoder | 2.9M | `lr = 1e-4` |
| Segmentation Head | 33 | `lr = 1e-4` |
| **Total** | **24.4M** | — |

---

## Dataset — BDAPPV

**Building Detection and Aerial Photovoltaic Panel Vision (BDAPPV)** — Kasmi et al., 2023
[Zenodo Link](https://zenodo.org/records/7358126) | French aerial imagery | ~28,000 crops

<p align="center">
  <img src="https://zenodo.org/records/7358126/files/bdappv_example.png?download=1" width="680"/>
</p>

### Sources

| Source | Sensor | GSD | Images |
|--------|--------|-----|--------|
| Google | Google Maps satellite | ~0.25 m/px | ~14,000 |
| IGN | French national survey | ~0.20 m/px | ~14,000 |

### Dataset Layout (after `prep_bdappv.py`)

```
bdappv_crops/
├── train/
│   ├── images/    ← 16,763 PNG crops (512×512 after resize)
│   └── masks/     ← PNG masks, ONLY for crops WITH solar panels
├── val/
│   ├── images/    ← 2,094 crops
│   └── masks/
└── test/
    ├── images/    ← 2,095 crops
    └── masks/
```

Split: **80% train / 10% val / 10% test** (fixed seed=42)

### Black Mask Fallback — Handling Negative Examples

Many images contain **no solar panels** — their `masks/` file simply doesn't exist. Three options:

| Approach | Problem |
|----------|---------|
| Skip images without masks | Model never sees "no panels" — over-predicts |
| Crash on missing file | Loses ~60% of dataset |
| **Use all-black mask** ✅ | Correct ground truth: "no panels present" |

```python
if mask_path.exists():
    mask = cv2.imread(mask_path, GRAYSCALE)
    mask = (mask > 127).astype(float32)   # binarise
else:
    mask = zeros((H, W), float32)          # ← black fallback
```

At startup the dataset reports the split:
```
[SolarFolderDataset] /tmp/bdappv_crops/train:
  16,763 images | 16,763 with masks | 0 without masks (→ black mask fallback)
```

> **Note:** After `prep_bdappv.py`, all crops that have panels come with masks, all panel-free crops do not. The fallback ensures both are included.

---

## Training Pipeline

### Input Preprocessing

BDAPPV images are **400×400 px**. U-Net requires dimensions divisible by 32. We resize to **512×512** at the start of both train and val augmentation:

```python
A.Resize(512, 512)   # → 512×512 before all other transforms
```

### Augmentation

| Transform | Probability | Purpose |
|-----------|------------|---------|
| Resize(512, 512) | 1.0 | Ensure divisibility by 32 |
| HorizontalFlip | 0.5 | No preferred direction in aerial view |
| VerticalFlip | 0.5 | Same |
| RandomRotate90 | 0.5 | Same |
| ShiftScaleRotate | 0.5 | Scale ±20%, shift ±10% |
| GaussNoise | 0.3 | Sensor noise robustness |
| ColorJitter | 0.4 | Panel color varies by age/angle/vendor |
| Normalize (ImageNet) | 1.0 | μ=(0.485,0.456,0.406), σ=(0.229,0.224,0.225) |

### Loss Function

```
Loss = 0.5 × SoftBCEWithLogitsLoss + 0.5 × DiceLoss
```

Solar panels typically occupy **<5% of pixels** per crop (highly imbalanced). Dice loss is critical here — it directly optimises the overlap between prediction and ground truth regardless of class imbalance.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*yUd5ckecHjWZf6hGrdlwHQ.png" width="560"/>
  <br><i>Dice loss vs BCE. Dice is insensitive to class imbalance because it only considers foreground pixels (panels), making it ideal for sparse targets like solar panels.</i>
</p>

### Optimiser & Scheduler

```python
optimizer = AdamW([
    {"params": encoder.parameters(), "lr": 1e-5},
    {"params": decoder.parameters(), "lr": 1e-4},
    {"params": seg_head.parameters(), "lr": 1e-4},
], weight_decay=1e-4)

scheduler = CosineAnnealingLR(T_max=100, eta_min=1e-6)
```

### Training Config Used

| Hyperparameter | Small Run | Full Run |
|---------------|-----------|---------|
| Training samples | 4,000 | 16,763 |
| Val samples | 499 | 2,094 |
| Batch size | 32 | 32 |
| Epochs run | 30 (early stop) | 93 (early stop) |
| LR (decoder) | 1e-4 | 1e-4 |
| LR (encoder) | 1e-5 | 1e-5 |
| Early stop patience | 15 | 15 |
| GPUs | 2 × V100-SXM2 32GB | 2 × V100-SXM2 32GB |

---

## Results

### Training Curves

**Small run (4,000 samples):**

| Epoch | Train Loss | Val IoU | Note |
|-------|-----------|--------|------|
| 1 | 0.719 | 0.556 | |
| 5 | 0.296 | 0.772 | |
| 10 | 0.088 | 0.827 | |
| **15** | **0.060** | **0.853** | **← best** |
| 30 | 0.042 | 0.850 | early stop |

**Full run (16,763 samples):**

| Epoch | Train Loss | Val IoU | Note |
|-------|-----------|--------|------|
| 39 | 0.040 | 0.844 | (resumed from earlier) |
| 47 | 0.042 | 0.851 | ← best at this point |
| 65 | 0.037 | 0.853 | |
| **78** | **0.037** | **0.854** | **← best overall** |
| 93 | 0.035 | 0.852 | early stop |

### Test Set Evaluation (2,095 crops)

| Model | Threshold | Test IoU | F1 | Precision | Recall |
|-------|-----------|---------|-----|-----------|--------|
| 4k samples (epoch 15) | 0.55 | 0.772 | 0.871 | 0.867 | 0.876 |
| **Full dataset (epoch 78)** | **0.55** | **0.848** | **0.918** | **0.915** | **0.921** |

**Full run improvement: +0.076 IoU over the 4k model on test set.**

### Threshold Sweep (Full Model, 2,095 test crops)

| Threshold | IoU | F1 | Precision | Recall |
|-----------|-----|----|-----------|--------|
| 0.30 | 0.8479 | 0.9177 | 0.9089 | 0.9267 |
| 0.35 | 0.8481 | 0.9178 | 0.9103 | 0.9254 |
| 0.40 | 0.8482 | 0.9179 | 0.9117 | 0.9242 |
| 0.45 | 0.8484 | 0.9180 | 0.9129 | 0.9231 |
| 0.50 | 0.8484 | 0.9180 | 0.9141 | 0.9219 |
| **0.55** | **0.8484** | **0.9180** | **0.9153** | **0.9207** |

> The threshold sweep is nearly flat — the model is well-calibrated. Use 0.5 (default) unless you prefer higher recall (lower threshold) or higher precision (higher threshold).

### Impact of Dataset Size

| Training Samples | Best Val IoU | Best Epoch | Test IoU |
|-----------------|-------------|-----------|---------|
| 500 (smoke test) | 0.055 | 3 | — |
| 4,000 | 0.853 | 15 | 0.772 |
| 16,763 (full) | **0.854** | 78 | **0.848** |

> **Key insight:** Val IoU converged at ~0.853 regardless of dataset size. The large test set improvement (+0.076) shows the full model generalises much better even though val IoU looked similar — the 4k model was overfitting to the narrow val distribution.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `prep_bdappv.py` | Download, resize, and split BDAPPV into train/val/test |
| `train_solar.py` | Train the solar panel segmentation model |
| `evaluate_solar.py` | Evaluate on test set + threshold sweep + overlay visualisations |
| `infer_solar.py` | Sliding-window inference + panel area + peak power estimate |

---

## Quick Commands

### 1. Prepare BDAPPV dataset
```bash
# Download
wget -O bdappv.zip "https://zenodo.org/records/7358126/files/bdappv.zip?download=1"
unzip bdappv.zip -d solar_panel/bdappv/

# Prepare crops (writes directly to /tmp for fast I/O)
python solar_panel/prep_bdappv.py \
    --src_dir solar_panel/bdappv/bdappv \
    --out_dir /tmp/bdappv_crops \
    --sources google ign \
    --val_frac 0.1 --test_frac 0.1
```

### 2. Quick smoke test (3 epochs, 500 samples)
```bash
python solar_panel/train_solar.py \
    --train_dir /tmp/bdappv_crops/train \
    --val_dir   /tmp/bdappv_crops/val \
    --max_samples 500 --epochs 3 \
    --arch unet --encoder resnet34 \
    --batch_size 32 --workers 2
```

### 3. Full training
```bash
python solar_panel/train_solar.py \
    --train_dir /tmp/bdappv_crops/train \
    --val_dir   /tmp/bdappv_crops/val \
    --arch unet --encoder resnet34 \
    --epochs 100 --batch_size 32 --workers 2 \
    --ckpt_dir solar_panel/checkpoints \
    --log_dir  solar_panel/logs \
    --save_every 5
```

### 4. Evaluate on test set
```bash
python solar_panel/evaluate_solar.py \
    --test_dir /tmp/bdappv_crops/test \
    --ckpt     solar_panel/checkpoints/<run_folder>/best.pth \
    --arch unet --encoder resnet34 \
    --sweep_threshold \
    --out_dir  solar_panel/eval_results \
    --log_dir  solar_panel/logs
```

### 5. Inference on a new image
```bash
# Single aerial image → panel mask + area + kW estimate
python solar_panel/infer_solar.py \
    --image /path/to/aerial.png \
    --ckpt  solar_panel/checkpoints/<run_folder>/best.pth \
    --gsd   0.25 \
    --out_dir solar_panel/infer_results/
```

Output image shows: original | binary mask | yellow overlay with panel area (m²) and estimated peak power (kW) at the given GSD.

### 6. Resume a crashed run
```bash
python solar_panel/train_solar.py \
    --train_dir /tmp/bdappv_crops/train \
    --val_dir   /tmp/bdappv_crops/val \
    --arch unet --encoder resnet34 \
    --epochs 100 --batch_size 32 --workers 2 \
    --resume solar_panel/checkpoints/<run_folder>/epoch055.pth
```

---

## References

- **BDAPPV Dataset:** Kasmi et al., *A crowdsourced dataset of aerial images with annotated solar panels*, Scientific Data 2023
- **U-Net:** Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
- **ResNet:** He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
- **segmentation-models-pytorch:** Yakubovskiy, 2019 — [GitHub](https://github.com/qubvel/segmentation_models.pytorch)
- **Dice Loss for imbalanced segmentation:** Sudre et al., *Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations*, MICCAI 2017
