# BTP — Rooftop & Solar Panel Segmentation

Semantic segmentation pipeline for detecting **rooftop areas** and **solar panels**
from aerial/satellite imagery. Built for DGX multi-GPU training using
[segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch).

---

## Repository Structure

```
btp/
├── rooftop/                        # Rooftop segmentation (Stage 1)
│   ├── train.py                    # Training script (DGX-ready, multi-GPU, AMP)
│   ├── evaluate.py                 # Test-set evaluation + visual overlays
│   ├── infer.py                    # Single-image / batch inference + area estimation
│   ├── tile_airs.py                # Preprocess: tile 10k×10k images → 512×512 crops
│   ├── checkpoints/                # Saved model checkpoints (best + periodic)
│   ├── logs/                       # TensorBoard logs
│   ├── notebooks/
│   │   ├── rooftop_segmentation.ipynb      # Exploratory analysis
│   │   ├── rooftop_area.ipynb              # Area estimation experiments
│   │   └── demo_rooftop_segmentation.ipynb # End-to-end demo
│   └── airs_text.txt               # AIRS dataset reference
│
├── solar_panel/                    # Solar panel segmentation (Stage 2)
│   ├── train_solar.py              # Training script with black-mask fallback
│   ├── checkpoints/                # Saved model checkpoints
│   └── logs/                       # TensorBoard logs
│
├── Dockerfile                      # Docker image (PyTorch 2.1.2, CUDA 11.8)
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Rooftop Segmentation

### Data Preparation

Tile large aerial images (10,000 × 10,000 px) into 512 × 512 crops:

```bash
python rooftop/tile_airs.py \
    --src_dir /data/airs/train \
    --out_dir /data/airs_crops/train \
    --crop_size 512 --overlap 0.1
```

Expected input layout for `src_dir`:
```
src_dir/
├── images/    ← .tif / .png source images
└── masks/     ← .png binary masks (0 / 255)
```

### Training

```bash
python rooftop/train.py \
    --train_dir /data/airs_crops/train \
    --val_dir   /data/airs_crops/val   \
    --ckpt_dir  rooftop/checkpoints    \
    --arch unet --encoder resnet34     \
    --epochs 50 --batch_size 8
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--arch` | `unet` | Architecture: `unet`, `unetplusplus`, `fpn`, `pspnet`, `deeplabv3plus` |
| `--encoder` | `resnet34` | Encoder backbone (any SMP-supported, e.g. `efficientnet-b4`) |
| `--epochs` | `50` | Training epochs |
| `--batch_size` | `8` | Batch size |
| `--lr` | `1e-4` | Decoder learning rate (encoder gets `lr × 0.1`) |
| `--patience` | `15` | Early-stopping patience |
| `--resume` | — | Path to checkpoint to resume from |

### Evaluation

```bash
python rooftop/evaluate.py \
    --test_dir /data/airs_crops/test \
    --ckpt     rooftop/checkpoints/unet_resnet34_best.pth \
    --arch unet --encoder resnet34
```

### Inference

```bash
# Single image
python rooftop/infer.py \
    --input /path/to/image.png \
    --ckpt  rooftop/checkpoints/unet_resnet34_best.pth \
    --arch unet --encoder resnet34

# Directory of images
python rooftop/infer.py \
    --input /path/to/images/ \
    --ckpt  rooftop/checkpoints/unet_resnet34_best.pth \
    --out_dir rooftop/predictions/
```

### Target Metrics (AIRS dataset)

| Model | IoU | F1 |
|-------|-----|----|
| PSPNet (paper baseline) | 0.899 | 0.947 |
| FPN | 0.882 | 0.937 |

---

## Solar Panel Segmentation

### Key Difference from Rooftop

Many aerial images contain **no solar panels**, so no mask file exists for them.
`train_solar.py` handles this automatically:

- If a mask file **exists** → load it as ground truth.
- If a mask file **does not exist** → use an **all-black mask** (correct ground truth: "no panels here").

This keeps negative examples in training rather than skipping them, which prevents
the model from over-predicting panels.

At startup the dataset reports the split:
```
[SolarFolderDataset] /data/solar/train: 1200 images, 400 with masks, 800 without masks (→ black mask fallback)
```

### Data Layout

```
data_dir/
├── images/    ← all 512×512 image crops (PNG)
└── masks/     ← only crops that contain solar panels (PNG, 0/255)
                  missing masks → black mask fallback
```

### Training

```bash
python solar_panel/train_solar.py \
    --train_dir /data/solar/train \
    --val_dir   /data/solar/val   \
    --arch unet --encoder resnet34 \
    --epochs 50 --batch_size 8
```

Checkpoints save to `solar_panel/checkpoints/` and TensorBoard logs to
`solar_panel/logs/` by default (configurable via `--ckpt_dir` / `--log_dir`).

---

## Model Architectures

All architectures are from `segmentation-models-pytorch` with ImageNet-pretrained
encoders. The decoder and segmentation head are trained at full learning rate;
the encoder at `lr × 0.1` (differential learning rates).

| Architecture | `--arch` value |
|--------------|----------------|
| UNet | `unet` |
| UNet++ | `unetplusplus` |
| FPN | `fpn` |
| PSPNet | `pspnet` |
| DeepLabV3+ | `deeplabv3plus` |

**Loss**: `0.5 × SoftBCEWithLogitsLoss + 0.5 × DiceLoss`

**Metrics**: Global IoU, F1, Precision, Recall (accumulated across all batches per epoch).

---

## DGX Setup & Docker

### 0. Check your driver / CUDA version

```bash
nvidia-smi
# Look for "CUDA Version: 11.x" → use the default Dockerfile tag (cu118)
# If "CUDA Version: 12.x" → edit Dockerfile line 1:
#   pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
```

### 1. Build the image

```bash
# Run from the repo root on the DGX node
docker build -t btp_seg .
```

### 2. Run an interactive container

```bash
docker run --gpus all -it --rm \
    -v /scratch:/scratch \
    -v $(pwd):/workspace \
    -w /workspace \
    btp_seg bash
```

- `--gpus all` — expose all DGX GPUs inside the container
- `-v /scratch:/scratch` — mount shared scratch storage (datasets, checkpoints)
- `-v $(pwd):/workspace` — mount the repo so edits are reflected inside
- `-w /workspace` — set working directory to the repo root

### 3. Commands inside the container

**Tile raw AIRS images → 512×512 crops**
```bash
python rooftop/tile_airs.py \
    --src_dir /scratch/airs/train \
    --out_dir /scratch/airs_crops/train \
    --crop_size 512 --overlap 0.1

python rooftop/tile_airs.py \
    --src_dir /scratch/airs/val \
    --out_dir /scratch/airs_crops/val \
    --crop_size 512 --overlap 0.1
```

**Train rooftop segmentation**
```bash
python rooftop/train.py \
    --train_dir /scratch/airs_crops/train \
    --val_dir   /scratch/airs_crops/val \
    --ckpt_dir  /scratch/checkpoints/rooftop \
    --arch unet --encoder resnet34 \
    --epochs 50 --batch_size 8 --workers 8
```

**Evaluate rooftop model on test set**
```bash
python rooftop/evaluate.py \
    --test_dir /scratch/airs_crops/test \
    --ckpt     /scratch/checkpoints/rooftop/unet_resnet34_best.pth \
    --arch unet --encoder resnet34
```

**Run rooftop inference**
```bash
# Single image
python rooftop/infer.py \
    --input /scratch/test_images/sample.png \
    --ckpt  /scratch/checkpoints/rooftop/unet_resnet34_best.pth \
    --arch unet --encoder resnet34

# Whole folder
python rooftop/infer.py \
    --input   /scratch/test_images/ \
    --ckpt    /scratch/checkpoints/rooftop/unet_resnet34_best.pth \
    --out_dir /scratch/rooftop_predictions/
```

**Train solar panel segmentation**
```bash
python solar_panel/train_solar.py \
    --train_dir /scratch/solar/train \
    --val_dir   /scratch/solar/val \
    --ckpt_dir  /scratch/checkpoints/solar_panel \
    --log_dir   /scratch/logs/solar_panel \
    --arch unet --encoder resnet34 \
    --epochs 50 --batch_size 8 --workers 8
```

### 4. Resume a training run

```bash
python rooftop/train.py \
    --train_dir /scratch/airs_crops/train \
    --val_dir   /scratch/airs_crops/val \
    --ckpt_dir  /scratch/checkpoints/rooftop \
    --arch unet --encoder resnet34 \
    --resume /scratch/checkpoints/rooftop/unet_resnet34_best.pth
```

### 5. Run training detached (long jobs on DGX)

```bash
# Use screen so the job survives SSH disconnection
screen -S train_rooftop

docker run --gpus all --rm \
    -v /scratch:/scratch \
    -v $(pwd):/workspace \
    -w /workspace \
    btp_seg \
    python rooftop/train.py \
        --train_dir /scratch/airs_crops/train \
        --val_dir   /scratch/airs_crops/val \
        --ckpt_dir  /scratch/checkpoints/rooftop \
        --arch unet --encoder resnet34 \
        --epochs 50 --batch_size 8

# Detach from screen : Ctrl+A then D
# Reattach later     : screen -r train_rooftop
```

### 6. Monitor with TensorBoard

```bash
# Rooftop logs
tensorboard --logdir /scratch/checkpoints/rooftop/tb_logs --port 6006 --bind_all

# Both tasks side by side
tensorboard \
    --logdir rooftop:/scratch/checkpoints/rooftop/tb_logs,solar:/scratch/logs/solar_panel \
    --port 6006 --bind_all
```

Open `http://<dgx-node-ip>:6006` in your browser.

---

## Local Setup (without Docker)

```bash
pip install -r requirements.txt
```
