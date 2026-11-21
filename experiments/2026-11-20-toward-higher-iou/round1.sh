#!/bin/bash
# Round 1 of the parallel IoU experiments (2026-11-20). Each run changes ONE thing vs the
# r34_c0_shift baseline (seed 42, 40 ep, bs 16, val = OB conf>=0.75 unshifted val tiles).
#   edge       Sirko 2021 edge-weighted BCE (x200, sigma 5.6 px) on the c0_shift labels
#   dino       U-Net with a DINOv3 ConvNeXt-Tiny encoder (timm 1.0.20 / smp 0.5.0) on c0_shift
#   lab_align / lab_sam / lab_align_sam   r34 on the refined label sets from scripts/refine_labels.py
# Checkpoints go to data/iou/ckpt (= host /tmp/btp_data/iou, outside the home quota).
# Usage (DGX, ~/btp): bash experiments/2026-11-20-toward-higher-iou/round1.sh TAG GPU
set -uo pipefail
cd ~/btp
E=experiments/2026-11-20-toward-higher-iou
mkdir -p $E/logs
TAG=$1; GPU=$2
COMMON="--val_dir data/jaipur_weak/val --arch unet --epochs 40 --batch_size 16 --workers 3 --cache_ram \
  --save_every 999 --seed 42 --ckpt_dir data/iou/ckpt/$TAG --log_dir $E/logs/$TAG"
case $TAG in
  edge) CMD="python -u rooftop/train.py --train_dir data/jaipur_weak_c0_shift/train --encoder resnet34 --edge_weight 200 --edge_sigma 5.6 $COMMON" ;;
  dino) CMD="env HF_HOME=data/iou/hf HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/data/iou/pylib_dino python -u rooftop/train.py --train_dir data/jaipur_weak_c0_shift/train --encoder tu-convnext_tiny.dinov3_lvd1689m $COMMON" ;;
  lab_*) CMD="python -u rooftop/train.py --train_dir data/iou/$TAG/train --encoder resnet34 $COMMON" ;;
  *) CMD="$3" ;;
esac
./run_docker.sh $GPU "$CMD" > $E/logs/train_$TAG.txt 2>&1
echo DONE > $E/logs/DONE_$TAG
