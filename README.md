# BTP — Rooftop & Solar Panel Segmentation

Semantic segmentation pipeline for detecting **rooftop areas** (Stage 1) and **solar panels** (Stage 2)
from aerial/satellite imagery. Built for DGX multi-GPU training using
[segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch).

---

## Repository Structure

```
btp/
├── rooftop/                        # Stage 1 — Rooftop segmentation
│   ├── train.py                    # Training script (DGX-ready, multi-GPU, AMP, tqdm)
│   ├── evaluate.py                 # Test-set evaluation + threshold sweep + overlays
│   ├── infer.py                    # Sliding-window inference for any image size
│   ├── tile_airs.py                # Preprocess: tile 10k×10k AIRS images → 512×512 crops
│   ├── checkpoints/                # Saved model weights (.pth) — gitignored
│   ├── logs/                       # Per-run JSON + TXT logs (epoch metrics) + TensorBoard
│   ├── dataset/                    # Raw AIRS dataset — gitignored, placeholder only
│   ├── dataset_crops/              # Tiled 512×512 crops — gitignored, placeholder only
│   ├── infer_results/              # Inference outputs — gitignored, placeholder only
│   └── notebooks/
│       ├── rooftop_segmentation.ipynb
│       ├── rooftop_area.ipynb
│       └── demo_rooftop_segmentation.ipynb
│
├── solar_panel/                    # Stage 2 — Solar panel segmentation
│   ├── train_solar.py              # Training script (RunLogger, tqdm, black-mask fallback)
│   ├── evaluate_solar.py           # Test-set evaluation + threshold sweep + overlays
│   ├── infer_solar.py              # Sliding-window inference for any image size
│   ├── prep_bdappv.py              # Prepare BDAPPV dataset (resize, split train/val/test)
│   ├── checkpoints/                # Saved model weights (.pth) — gitignored
│   ├── logs/                       # Per-run JSON + TXT logs + TensorBoard
│   ├── bdappv/                     # Raw BDAPPV dataset — gitignored, placeholder only
│   └── bdappv_crops/               # Prepared crops — gitignored, placeholder only
│
├── prep_bdappv.py                  # Prepare BDAPPV solar panel dataset (resize, split)
├── Dockerfile                      # Docker image (PyTorch 2.1.2, CUDA 11.8)
├── requirements.txt                # Python dependencies
└── README.md
```

> **Gitignored:** All dataset folders, model weights (`.pth`), and generated outputs are excluded from git.
> Each folder has a `.gitkeep` file so the directory structure is visible in the repo.

---

## Quickstart (DGX)

### 0. Build the Docker image

```bash
# From repo root on the DGX node
docker build -t btp_seg .
```

### 1. Start a screen session on the HOST (not inside Docker)

```bash
screen -S train
# This survives SSH disconnects. Detach: Ctrl+A D  |  Reattach: screen -r train
```

### 2. Launch the container inside screen

```bash
docker run --gpus '"device=0,1"' -it --rm \
    --shm-size=16g \
    -v /scratch:/scratch \
    -v $(pwd):/workspace \
    -w /workspace \
    btp_seg bash
```

- `--gpus '"device=0,1"'` — pick specific free GPUs (check `nvidia-smi` first)
- `--shm-size=16g` — prevents bus errors with multiple DataLoader workers
- `-v /scratch:/scratch` — mount shared scratch storage

### 3. Copy datasets to /tmp for fast I/O

NFS (`/scratch`) is slow (~4s/it). Copy crops to local NVMe `/tmp` before training:

```bash
cp -r /scratch/airs_crops /tmp/airs_crops
cp -r /scratch/bdappv_crops /tmp/bdappv_crops
```

`/tmp` is local NVMe (~10× faster), does not count toward your quota, and is cleared on reboot.

---

## Stage 1 — Rooftop Segmentation

### Data Preparation

Tile large AIRS images (10,000 × 10,000 px) into 512 × 512 crops:

```bash
python rooftop/tile_airs.py \
    --src_dir /scratch/airs/train \
    --out_dir /scratch/airs_crops/train \
    --crop_size 512 --overlap 0.1

python rooftop/tile_airs.py \
    --src_dir /scratch/airs/val \
    --out_dir /scratch/airs_crops/val

python rooftop/tile_airs.py \
    --src_dir /scratch/airs/test \
    --out_dir /scratch/airs_crops/test
```

Expected AIRS input layout:
```
src_dir/
├── image/    ← .tif source images (use --img_subdir image)
└── label/    ← .png binary masks  (use --mask_subdir label)
```

AIRS masks use values 0/1 (not 0/255) — `tile_airs.py` handles this automatically.

### Training

```bash
python rooftop/train.py \
    --train_dir /tmp/airs_crops/train \
    --val_dir   /tmp/airs_crops/val   \
    --ckpt_dir  rooftop/checkpoints   \
    --log_dir   rooftop/logs          \
    --arch unet --encoder resnet34    \
    --epochs 100 --batch_size 32 --workers 2
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--arch` | `unet` | Architecture: `unet`, `unetplusplus`, `fpn`, `pspnet`, `deeplabv3plus` |
| `--encoder` | `resnet34` | Encoder backbone (any SMP-supported, e.g. `efficientnet-b4`) |
| `--epochs` | `50` | Training epochs |
| `--batch_size` | `8` | Batch size (32 works well on V100 32GB) |
| `--lr` | `1e-4` | Decoder LR (encoder gets `lr × 0.1`) |
| `--workers` | `4` | DataLoader workers (keep at 2 to avoid OOM) |
| `--max_samples` | — | Cap training samples (val capped proportionally) — useful for quick tests |
| `--simulate_low_res` | off | Downscale augmentation to simulate 30cm/px from 7.5cm/px AIRS data |
| `--patience` | `15` | Early-stopping patience |
| `--resume` | — | Path to checkpoint to resume from |
| `--log_dir` | `rooftop/logs` | Directory for TensorBoard + JSON/TXT run logs |

After each epoch a `.json` and `.txt` log are written to `--log_dir`. Watch live:
```bash
tail -f rooftop/logs/$(ls -t rooftop/logs/*.txt | head -1)
```

### Evaluation

```bash
python rooftop/evaluate.py \
    --test_dir /tmp/airs_crops/test \
    --ckpt     rooftop/checkpoints/unet_resnet34_best.pth \
    --arch unet --encoder resnet34  \
    --log_dir  rooftop/logs
```

Runs a threshold sweep (0.30–0.55), auto-picks best, saves per-sample IoU CSV and overlay PNGs.

### Inference (any image size)

`infer.py` uses sliding-window tiling with Hann-window blending — works on Google Maps screenshots or any arbitrary resolution.

```bash
# Single image
python rooftop/infer.py \
    --input /path/to/image.png \
    --ckpt  rooftop/checkpoints/unet_resnet34_best.pth \
    --arch unet --encoder resnet34 \
    --threshold 0.35

# Whole folder
python rooftop/infer.py \
    --input   /path/to/images/ \
    --ckpt    rooftop/checkpoints/unet_resnet34_best.pth \
    --out_dir rooftop/infer_results/
```

Use `--threshold 0.35` for best results (model is under-confident on limited training data).

### Results (AIRS dataset)

| Model | Samples | IoU |
|-------|---------|-----|
| PSPNet (Chen et al., 2019 — paper baseline) | full | 0.899 |
| UNet ResNet-34 (ours, 2000 samples 100ep) | 210 test | **0.9016** ✅ |
| UNet ResNet-34 (ours, 2000 samples 100ep) | full test | 0.8664 |

> Training on more samples and with `--simulate_low_res` is expected to close the full-test gap.

---

## Stage 2 — Solar Panel Segmentation

### Dataset — BDAPPV

Download from [Zenodo](https://zenodo.org/records/7358126):

```bash
wget -O bdappv.zip "https://zenodo.org/records/7358126/files/bdappv.zip?download=1"
unzip bdappv.zip -d solar_panel/bdappv/
```

BDAPPV structure after unzip:
```
solar_panel/bdappv/bdappv/
├── google/
│   ├── img/    ← ~400×400px aerial images (Google Maps)
│   └── mask/   ← binary masks (only for images WITH solar panels)
└── ign/
    ├── img/    ← higher-res IGN images
    └── mask/
```

### Data Preparation

```bash
python prep_bdappv.py \
    --src_dir solar_panel/bdappv/bdappv \
    --out_dir /tmp/bdappv_crops \
    --sources google ign \
    --size 400 \
    --val_frac 0.1 --test_frac 0.1
```

Output: `/tmp/bdappv_crops/train|val|test/images/` and `.../masks/`

`prep_bdappv.py` options:

| Flag | Default | Description |
|------|---------|-------------|
| `--src_dir` | `./bdappv/bdappv` | Path to BDAPPV root |
| `--out_dir` | `./bdappv_crops` | Output directory |
| `--sources` | `google ign` | Which sources to use |
| `--size` | `400` | Resize to N×N px (400 = keep original) |
| `--min_panel_frac` | `0.001` | Skip crops with < this fraction of panel pixels |

### Key Difference from Rooftop

Many images have **no solar panels**, so no mask file exists for them.
`train_solar.py` handles this with a black-mask fallback:

- Mask file **exists** → load it (solar panels present)
- Mask file **missing** → use all-zeros mask (correct ground truth: "no panels here")

This keeps negative examples in training and prevents the model from over-predicting panels.

At startup the dataset reports the breakdown:
```
[SolarFolderDataset] /tmp/bdappv_crops/train: 18432 images, 6200 with masks, 12232 without masks (→ black mask fallback)
```

### Training

```bash
python solar_panel/train_solar.py \
    --train_dir /tmp/bdappv_crops/train \
    --val_dir   /tmp/bdappv_crops/val   \
    --arch unet --encoder resnet34      \
    --epochs 100 --batch_size 32 --workers 2 \
    --ckpt_dir solar_panel/checkpoints  \
    --log_dir  solar_panel/logs
```

Quick smoke-test (3 epochs, 500 samples):
```bash
python solar_panel/train_solar.py \
    --train_dir /tmp/bdappv_crops/train \
    --val_dir   /tmp/bdappv_crops/val   \
    --max_samples 500 --epochs 3        \
    --arch unet --encoder resnet34      \
    --batch_size 32 --workers 2         \
    --ckpt_dir solar_panel/checkpoints  \
    --log_dir  solar_panel/logs
```

### Evaluation

```bash
python solar_panel/evaluate_solar.py \
    --test_dir /tmp/bdappv_crops/test \
    --ckpt     solar_panel/checkpoints/unet_resnet34_best.pth \
    --arch unet --encoder resnet34 \
    --sweep_threshold \
    --out_dir  solar_panel/eval_results \
    --log_dir  solar_panel/logs
```

Runs a threshold sweep (0.30–0.55), auto-picks best, saves per-sample IoU CSV and 4-panel overlay PNGs (image | GT | prediction | yellow blend).

### Inference (any image size)

```bash
# Single image
python solar_panel/infer_solar.py \
    --image /path/to/aerial.png \
    --ckpt  solar_panel/checkpoints/unet_resnet34_best.pth \
    --gsd   0.25

# Whole folder
python solar_panel/infer_solar.py \
    --image   /path/to/images/ \
    --ckpt    solar_panel/checkpoints/unet_resnet34_best.pth \
    --out_dir solar_panel/infer_results/
```

Outputs per-image PNG: original | binary mask | yellow overlay + panel area (m²) + estimated peak power (kW).

---

## Model Architectures

All from `segmentation-models-pytorch` with ImageNet-pretrained encoders.
Decoder + segmentation head train at full LR; encoder at `lr × 0.1`.

| Architecture | `--arch` value |
|--------------|----------------|
| UNet | `unet` |
| UNet++ | `unetplusplus` |
| FPN | `fpn` |
| PSPNet | `pspnet` |
| DeepLabV3+ | `deeplabv3plus` |

**Loss**: `0.5 × SoftBCEWithLogitsLoss + 0.5 × DiceLoss`

**Metrics**: Global IoU, F1, Precision, Recall — accumulated across all batches per epoch (not averaged per-batch).

---

## Monitoring

### Live log tail
```bash
tail -f solar_panel/logs/$(ls -t solar_panel/logs/*.txt | head -1)
```

### TensorBoard
```bash
# Rooftop
tensorboard --logdir rooftop/logs --port 6006 --bind_all

# Both stages side by side
tensorboard \
    --logdir rooftop:rooftop/logs,solar:solar_panel/logs \
    --port 6006 --bind_all
```

Open `http://<dgx-node-ip>:6006` in your browser.

### Watch GPU usage
```bash
watch -n 2 nvidia-smi
```

---

## Resume a crashed run

```bash
python rooftop/train.py \
    --train_dir /tmp/airs_crops/train \
    --val_dir   /tmp/airs_crops/val   \
    --ckpt_dir  rooftop/checkpoints   \
    --arch unet --encoder resnet34    \
    --resume rooftop/checkpoints/unet_resnet34_best.pth
```

Periodic checkpoints (`unet_resnet34_epoch010.pth`, etc.) are saved every `--save_every` epochs for crash recovery.

---

## Local Setup (without Docker)

```bash
pip install -r requirements.txt
```
