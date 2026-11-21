#!/bin/bash
# round2.sh TAG GPU "train.py args"  -- same fixed settings as round 1, any extra args.
set -uo pipefail
cd ~/btp
E=experiments/2026-11-20-toward-higher-iou
mkdir -p $E/logs
TAG=$1; GPU=$2; ARGS=$3
COMMON="--val_dir data/jaipur_weak/val --arch unet --epochs 40 --batch_size 16 --workers 3 --cache_ram \
  --save_every 999 --seed ${SEED:-42} --ckpt_dir data/iou/ckpt/$TAG --log_dir $E/logs/$TAG"
PRE=""
[[ "$ARGS" == *dinov3* ]] && PRE="env HF_HOME=data/iou/hf HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/data/iou/pylib_dino"
./run_docker.sh $GPU "$PRE python -u rooftop/train.py $ARGS $COMMON" > $E/logs/train_$TAG.txt 2>&1
echo DONE > $E/logs/DONE_$TAG
