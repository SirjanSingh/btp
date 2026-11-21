#!/usr/bin/env bash
# Phase A of the Nov 2026 plan, run on the DGX from ~/btp.
#   1. rebuild data/jaipur_weak (images + conf>=0.75 masks); the images were wiped by the Oct 8 reboot
#   2. three extra mask-only label sets that share those images:
#        jaipur_weak_c0        all Open Buildings polygons (re-creates the Sept c0 set)
#        jaipur_weak_c0_shift  all polygons, moved 7 px south and 3 px west onto the roofs
#        jaipur_weak_shift     conf>=0.75, moved the same way
#   3. train ResNet34 (identical config to 2026-09-10-label-quantity-vs-quality) on c0_shift and
#      shift, two GPUs in parallel. Checkpoints are still selected on the OB conf>=0.75 val tiles,
#      so no hand label touches training or model selection.
# Usage: nohup bash experiments/2026-11-19-hand-label-model-scoring/prep_and_train.sh GPU_A GPU_B > .../logs/prep_and_train.txt 2>&1 &
set -uo pipefail
GA=${1:-1}; GB=${2:-5}
E=experiments/2026-11-19-hand-label-model-scoring
mkdir -p $E/logs $E/ckpt

run() { ./run_docker.sh "$1" "$2"; }

if [ ! -f data/jaipur_weak/weak_labels_summary.json ]; then
  [ -L data/jaipur_weak ] && rm data/jaipur_weak          # dangling symlink left by the reboot
  run $GA "python -u scripts/make_weak_labels.py --out_dir data/jaipur_weak --stride 512 --min_conf 0.75" \
    > $E/logs/build_jaipur_weak.txt 2>&1
fi
for spec in "jaipur_weak_c0:0.0:0 0" "jaipur_weak_c0_shift:0.0:7 -3" "jaipur_weak_shift:0.75:7 -3"; do
  IFS=: read -r name conf shift <<< "$spec"
  if [ ! -f data/$name/weak_labels_summary.json ]; then
    for s in train val; do mkdir -p data/$name/$s; ln -sfn ../../jaipur_weak/$s/images data/$name/$s/images; done
    run $GA "python -u scripts/make_weak_labels.py --out_dir data/$name --stride 512 --min_conf $conf --shift_px $shift --masks_only" \
      > $E/logs/build_$name.txt 2>&1
  fi
done
touch $E/logs/DATA_DONE

train() {  # gpu tag train_dir
  run "$1" "python -u rooftop/train.py --train_dir data/$3/train --val_dir data/jaipur_weak/val \
    --epochs 40 --batch_size 16 --workers 3 --save_every 999 --seed 42 \
    --ckpt_dir $E/ckpt/$2 --log_dir $E/logs/$2" > $E/logs/train_$2.txt 2>&1
}
train $GA r34_c0_shift jaipur_weak_c0_shift &
train $GB r34_shift jaipur_weak_shift &
wait
touch $E/logs/TRAIN_DONE
