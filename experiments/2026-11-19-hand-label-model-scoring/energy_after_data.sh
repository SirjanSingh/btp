#!/usr/bin/env bash
# Waits for the rebuilt data/jaipur_weak, then reruns the capacity/energy chain (d16) three ways.
set -uo pipefail
E=experiments/2026-11-19-hand-label-model-scoring
while [ ! -f $E/logs/DATA_DONE ]; do sleep 60; done
X=experiments
d16() {  # tag ckpt encoder threshold
  ./run_docker.sh 4 "python -u scripts/d16_capacity_estimate.py --ckpt $2 --encoder $3 --threshold $4 --out $E/energy_$1.json" > $E/logs/energy_$1.txt 2>&1
}
d16 default_t05   $X/2026-09-10-eroded-labels-rerun/checkpoints/unet_mit_b2_40ep_20260910_053557/best.pth mit_b2 0.5
d16 r34_c0_t03    $X/2026-09-10-label-quantity-vs-quality/checkpoints/unet_resnet34_40ep_20260909_210501/best.pth resnet34 0.3
d16 r34_c075_t03  $X/2026-09-09-weak-supervision-jaipur/checkpoints/unet_resnet34_40ep_20260909_152425/best.pth resnet34 0.3
touch $E/logs/ENERGY_DONE
