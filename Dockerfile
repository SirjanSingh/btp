# Dockerfile — Rooftop & Solar Panel Segmentation
#
# Base: official PyTorch image with CUDA 11.8 + cuDNN 8 (Ubuntu 22.04)
# Check your DGX host driver first:
#   nvidia-smi   →  "CUDA Version: 11.x" → use cu118 tag
#                   "CUDA Version: 12.x" → change tag to pytorch:2.2.0-cuda12.1-cudnn8-runtime
#
# Build:
#   docker build -t rooftop_seg .
#
# Run (interactive, mounts /scratch and current dir):
#   docker run --gpus all -it --rm \
#       -v /scratch:/scratch \
#       -v $(pwd):/workspace \
#       -w /workspace \
#       rooftop_seg bash

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        libgdal-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        screen \
        rclone \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /workspace

# ── Default command ───────────────────────────────────────────────────────────
CMD ["bash"]
