# Technology Stack

**Project:** Rooftop & Solar Panel Segmentation Pipeline
**Researched:** 2026-03-30
**Confidence note:** Web tools unavailable. All findings from training data (cutoff August 2025). Versions marked [VERIFY] should be pinned after checking PyPI on target environment.

---

## Recommended Stack

### Deep Learning Core

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| PyTorch | 2.2.x (Colab) / 2.1.x (DGX Docker) | Tensor ops, autograd, training loop | De-facto standard for research segmentation. 2.2 adds torch.compile() gains; 2.1 is the last fully stable release with CUDA 11.8 Docker images on Ubuntu 22.04. Do NOT use 2.3+ on DGX without verifying CUDA 12.x driver availability. |
| torchvision | 0.17.x (matches torch 2.2) | Image tensor utilities, pretrained weight loading | Must version-match PyTorch exactly. Mismatches cause silent weight loading failures. |
| segmentation-models-pytorch | 0.3.3 [VERIFY latest] | U-Net, FPN, DeepLabV3+ architectures with swappable encoders | Purpose-built for this exact use case. Handles encoder pretraining, decoder construction, loss functions. Saves ~500 lines of boilerplate per model. The SMP API is stable and the community maintains it actively. |
| timm | 0.9.x | ImageNet pretrained encoder weights (resnet34, efficientnet, etc.) | SMP 0.3+ uses timm as its encoder registry. Required when using timm-prefixed encoder names. Pin timm separately or SMP will pull a potentially incompatible version. |

### Data Augmentation

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| albumentations | 1.4.x [VERIFY] | Spatial and pixel augmentations for image+mask pairs | Aerial imagery requires simultaneous image+mask transforms. albumentations handles this natively. Far faster than torchvision transforms for this pattern. The `A.Compose([...], additional_targets={"mask": "mask"})` API is the correct pattern for segmentation. |

### Geospatial / Raster I/O

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| rasterio | 1.3.x | Reading GeoTIFF tiles, accessing CRS and GSD metadata | AIRS tiles are GeoTIFF. rasterio reads them while preserving spatial metadata. Prefer over PIL/cv2 for anything that touches projection or GSD. |
| GDAL | bundled via rasterio wheels | Raster format support | Do NOT pip-install gdal separately. Use the GDAL version bundled into the rasterio wheel (`pip install rasterio` handles this). Installing GDAL independently on Ubuntu 22.04 Docker is a dependency hell source. |
| shapely | 2.0.x | Polygon operations for rooftop mask post-processing | If rooftop polygons need to be vectorized from raster masks (e.g. for crop region definition in Stage 2), shapely provides clean geometry operations. |
| geopandas | 0.14.x | Tabular spatial data (if AIRS labels are in shapefile/GeoJSON) | Only needed if working with AIRS vector labels. If masks are already rasterized, this is optional. |

### Dataset Acquisition

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| gdown | 5.x | Downloading AIRS from Google Drive | AIRS dataset is hosted on Drive. gdown 5.x handles large file quotas better than 4.x. Always use the file ID form, not the sharing URL form, for reliability. |

### Training Utilities

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| numpy | 1.26.x | Array ops | Pin to 1.26.x. NumPy 2.0 introduced breaking API changes (np.bool_ removal, etc.) that broke many downstream packages including albumentations < 1.4 and older SMP versions. If using NumPy 2.0+, verify all three are updated together. |
| opencv-python-headless | 4.9.x | Image decoding, resizing, contour operations | Use the `-headless` variant on DGX and Colab — the full `opencv-python` package requires a display server (libGL). Headless avoids ImportError on serverless environments. |
| Pillow | 10.x | PNG/JPEG I/O, visualization | Required by torchvision. Let torchvision pull the version rather than pinning separately. |
| tqdm | 4.x | Training loop progress bars | Standard; no version constraints. |
| matplotlib | 3.8.x | Mask overlay visualization, metric plotting | For the evaluation script and demo overlay generation. |

### Metrics and Logging

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| torchmetrics | 1.3.x | IoU, F1, Precision, Recall computation | Handles batched metric accumulation correctly. Avoids the common bug of computing mean-of-batch-means instead of true dataset-level metrics. Use `JaccardIndex(task="binary")` for IoU. |
| wandb OR tensorboard | latest | Experiment tracking | wandb preferred for Colab (browser dashboard, no port forwarding). tensorboard is simpler on DGX when port forwarding is inconvenient. Pick one and stay consistent. wandb free tier is sufficient for BTP. |

### Environment / Reproducibility

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Docker (Ubuntu 22.04) | image: pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime | DGX container base | Use an official PyTorch Docker image as the base — it guarantees CUDA/cuDNN/NCCL alignment. Do NOT start from ubuntu:22.04 and pip-install PyTorch — CUDA linking errors are common. The 2.1.2-cuda11.8 tag is the safest match for DGX infrastructure that may be running CUDA 11.x drivers. |
| pip + requirements.txt | — | Dependency locking | No conda on DGX Docker — pip only. Generate requirements.txt with `pip freeze` after a working Colab prototype to lock exact versions before DGX training run. |

---

## Alternatives Considered and Rejected

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Segmentation framework | segmentation-models-pytorch | mmsegmentation (OpenMMLab) | mmsegmentation is powerful but requires mmcv, which has notoriously fragile version coupling with PyTorch/CUDA. Setup time on a Docker-locked DGX is high. SMP is pip-installable with no system dependencies. |
| Segmentation framework | segmentation-models-pytorch | detectron2 | detectron2 targets instance/panoptic segmentation, not binary semantic segmentation. Overhead is unjustified for this task. |
| Encoder | ResNet-34 (starting point) | EfficientNet-B4 or MobileNetV3 | ResNet-34 is the correct baseline to match the AIRS paper's encoder tier. Once baseline is established, EfficientNet-B4 is the natural next upgrade for better feature extraction at similar parameter count. MobileNetV3 is too lightweight for 7.5cm/pixel detail. |
| Encoder | ResNet-34 | Vision Transformer (SegFormer) | SegFormer (mit_b2/b3) is a legitimate upgrade path and has beaten U-Net on aerial datasets in 2023-2024. However, ViT encoders are significantly more memory-hungry and slower to train. Beat the ResNet-34 baseline first; then experiment with mit_b2 via SMP's timm encoder registry. |
| Augmentation | albumentations | torchvision.transforms v2 | torchvision transforms v2 added segmentation support in 0.16 but the API is less mature and the library of available transforms is narrower. albumentations has `GridDistortion`, `ElasticTransform`, `RandomSunFlare`, `OpticalDistortion` — all relevant for aerial imagery. |
| Raster I/O | rasterio | PIL / imageio | PIL doesn't read GeoTIFF metadata (GSD, CRS, projection). rasterio is mandatory for anything that uses spatial metadata. imageio is acceptable only for pure RGB array reading without spatial context. |
| Loss function | BCE + Dice (combined) | Focal Loss | Focal Loss addresses class imbalance but BCE+Dice already handles this well for binary segmentation. Combined loss is standard in the segmentation literature since 2020. Tversky Loss is an option if false negatives are specifically problematic (partial rooftop recall). |
| Metrics | torchmetrics | sklearn metrics | sklearn computes metrics on CPU from flattened arrays — correct but slow. torchmetrics stays on GPU and accumulates across batches correctly. |

---

## Critical Version Pinning Gotchas for DGX Docker Ubuntu 22.04

### 1. CUDA Driver vs Runtime Mismatch
The DGX host may expose a CUDA driver version (visible via `nvidia-smi`) that differs from what's inside the Docker image. The rule: Docker image's CUDA runtime version must be <= host driver version. PyTorch 2.1.x with CUDA 11.8 requires driver >= 450.80. PyTorch 2.2.x with CUDA 12.1 requires driver >= 525.85. Verify with `nvidia-smi` on the DGX host before choosing the Docker base image.

### 2. NumPy 1.x vs 2.x Break
NumPy 2.0 (released June 2024) is a major breaking change. Many packages pinned to NumPy 1.x behavior will fail silently or crash on import. Safe rule: use numpy==1.26.4 until all dependencies explicitly declare NumPy 2.x support. Check with `pip check` after install.

### 3. albumentations OpenCV Dependency
albumentations requires opencv — specifically it imports `cv2`. Install `opencv-python-headless` not `opencv-python`. On Colab, `opencv-python` works because Colab has a virtual display. On DGX Docker without a display server, `import cv2` in `opencv-python` (non-headless) throws `libGL.so.1: cannot open shared object file`. This is the single most common DGX setup error for this stack.

### 4. rasterio GDAL Bundling
`pip install rasterio` on Linux downloads a wheel that bundles GDAL internally. Do NOT run `apt-get install gdal-bin` or `pip install GDAL` in the same Docker layer — they will conflict. If you need the GDAL CLI tools (gdal_translate, etc.), install `gdal-bin` via apt before rasterio, then install rasterio from source or accept the potential conflict. For this project, the bundled rasterio wheel is sufficient — no CLI tools needed.

### 5. SMP + timm Encoder Name Changes
In SMP 0.3+, encoders are accessed via timm under the prefix `tu-`. For example, `encoder_name="tu-efficientnet_b4"`. The old `timm-efficientnet-b4` naming (with hyphens) was deprecated. If using an old tutorial as reference, this name change will cause a silent fallback to resnet34. Always verify with `smp.encoders.get_encoder_names()`.

### 6. gdown Quota Errors
Google Drive imposes download quotas on files with many downloads. For AIRS (large dataset, widely used), quota errors are common. Workaround: use `gdown --fuzzy` with the direct file ID, or use `gdown` with a service account token. Alternatively, copy the Drive folder to your own Drive and download from your copy — this bypasses the original file's quota.

### 7. screen Session Persistence on DGX
DGX `/scratch/` is cleared between sessions (not guaranteed persistent). Checkpoint saving must go to a mounted Google Drive path or an NFS path. Use `screen -S training` to keep training running after SSH disconnect. Verify the screen session survives with `screen -ls` from a second SSH connection before closing the first.

---

## Exact Dependency Block (requirements.txt template)

```
# Deep Learning
torch==2.1.2
torchvision==0.16.2
# Install PyTorch via: pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Segmentation
segmentation-models-pytorch==0.3.3
timm==0.9.16

# Augmentation
albumentations==1.4.3

# Geospatial
rasterio==1.3.9
shapely==2.0.3
geopandas==0.14.3   # optional — only if working with vector labels

# Image I/O
opencv-python-headless==4.9.0.80
Pillow==10.2.0

# Numerics
numpy==1.26.4

# Metrics / Logging
torchmetrics==1.3.2
wandb==0.16.x       # or: tensorboard==2.16.x

# Data acquisition
gdown==5.1.0

# Utilities
tqdm==4.66.x
matplotlib==3.8.x
```

All versions marked x.x (minor) should be latest patch at time of environment creation. Run `pip freeze > requirements_lock.txt` after the first successful training run to capture exact patch versions.

---

## Installation — Colab

```python
# Run once per Colab session (or bake into a setup cell)
!pip install -q \
    torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 \
    segmentation-models-pytorch==0.3.3 \
    timm==0.9.16 \
    albumentations==1.4.3 \
    rasterio==1.3.9 \
    shapely==2.0.3 \
    opencv-python-headless==4.9.0.80 \
    numpy==1.26.4 \
    torchmetrics==1.3.2 \
    gdown==5.1.0
```

Note: Colab pre-installs some of these (torch, numpy, opencv). If Colab's preinstalled torch version is newer (2.3+), do not downgrade — instead use Colab's version and verify compatibility with SMP and albumentations. The conflict is most likely to appear with NumPy 2.x on newer Colab runtimes.

## Installation — DGX Docker (Ubuntu 22.04)

```dockerfile
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# System deps (apt)
RUN apt-get update && apt-get install -y \
    git \
    screen \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Note: do NOT apt-get install gdal-bin if rasterio bundled wheel is used
# Only include libgdal-dev if building rasterio from source
```

---

## Upgrade Path (Post-Baseline)

Once ResNet-34 U-Net baseline is established and exceeds IoU=0.899:

1. **Encoder upgrade**: swap `encoder_name="resnet34"` for `encoder_name="tu-efficientnet_b4"` in SMP — one line change, same training loop.
2. **Architecture upgrade**: Try `smp.Unet` → `smp.UnetPlusPlus` — generally +1-2 IoU points on building datasets.
3. **ViT exploration**: `encoder_name="tu-mit_b2"` (SegFormer encoder) via timm. Expect 2-3x training time increase but potentially +3-5 IoU on fine-grained structures like rooftop edges.
4. **Test-Time Augmentation (TTA)**: flip horizontal/vertical and average predictions — free +0.5-1 IoU at inference.
5. **Loss upgrade**: Add Lovász loss for directly optimizing IoU if BCE+Dice plateaus.

---

## Sources

- Training knowledge (August 2025 cutoff) — HIGH confidence for library relationships, API patterns, version compatibility rules
- SMP GitHub (qubvel-org/segmentation_models.pytorch) — version 0.3.3 was current as of mid-2025 [VERIFY latest on PyPI]
- albumentations changelog — 1.4.x series was current as of mid-2025 [VERIFY]
- PyTorch Docker Hub — pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime is a real, pinned tag [HIGH confidence]
- NumPy 2.0 migration guide — breaking changes documented in NumPy 2.0 release notes (June 2024) [HIGH confidence]
- AIRS paper (Chen et al., 2019, ISPRS) — PSPNet IoU=0.899 baseline [HIGH confidence, from project context]
- gdown quota workarounds — known issue, widely documented in Drive API community [MEDIUM confidence]
