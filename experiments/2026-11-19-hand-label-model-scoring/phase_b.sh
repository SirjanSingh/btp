#!/usr/bin/env bash
# Phase B, launched 2026-11-20 after the shifted labels won (r34_c0_shift test IoU 0.697 @0.4).
# All in parallel on different GPUs:
#   - energy rerun with r34_c0_shift @0.4 (and @0.5 for comparison)
#   - seeds 43 and 44 of r34_c0_shift (is the +0.038 robust?)
#   - MiT-B2 on the c0_shift labels (does the bigger encoder help on better labels?)
# Usage: nohup bash .../phase_b.sh GPU_ENERGY GPU_S43 GPU_S44 GPU_MIT > .../logs/phase_b.txt 2>&1 &
set -uo pipefail
GE=${1:-4}; G43=${2:-7}; G44=${3:-6}; GM=${4:-3}
E=experiments/2026-11-19-hand-label-model-scoring
CK=$E/ckpt/r34_c0_shift/unet_resnet34_40ep_20261119_172916/best.pth

energy() {
  for t in 0.4 0.5; do
    ./run_docker.sh $GE "python -u scripts/d16_capacity_estimate.py --ckpt $CK --encoder resnet34 --threshold $t \
      --out $E/energy_r34_c0_shift_t${t}.json" > $E/logs/energy_r34_c0_shift_t${t}.txt 2>&1
  done
}
train() {  # gpu tag encoder batch seed
  ./run_docker.sh "$1" "python -u rooftop/train.py --train_dir data/jaipur_weak_c0_shift/train --val_dir data/jaipur_weak/val \
    --arch unet --encoder $3 --epochs 40 --batch_size $4 --workers 3 --save_every 999 --seed $5 \
    --ckpt_dir $E/ckpt/$2 --log_dir $E/logs/$2" > $E/logs/train_$2.txt 2>&1
}
energy &
train $G43 r34_c0_shift_s43 resnet34 16 43 &
train $G44 r34_c0_shift_s44 resnet34 16 44 &
train $GM mitb2_c0_shift mit_b2 12 42 &
wait
touch $E/logs/PHASE_B_DONE
