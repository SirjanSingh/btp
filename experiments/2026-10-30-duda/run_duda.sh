#!/usr/bin/env bash
# Arm E: DUDA (large mit_b5 teacher/student distilled into the resnet34 seed).
# Same data, seed checkpoint and AdaBN start as arms B-D, so rows are comparable.
# Launch from the repo root:  nohup ./run_docker.sh 2 "bash experiments/2026-10-30-duda/run_duda.sh" &
set -uo pipefail
E=experiments/2026-10-30-duda
D=data/st_run
mkdir -p $E/logs
# Freeze the helpers this run imports, so a later edit to arms A-D's script
# cannot change what arm E did. The hash goes in the log.
cp experiments/2026-10-30-annotation-free-selftrain/self_train.py $E/st_common.py
sha1sum $E/st_common.py > $E/logs/st_common.sha1
env PYTHONPATH=rooftop:$E python -u $E/duda.py \
  --ss_ckpt rooftop/checkpoints/unet_resnet34_best.pth \
  --src_dir $D/airs_crops/train --target_dir $D/target_tiles --val_target_dir $D/eval/val \
  --simulate_low_res --adabn_batches 40 --large_encoder mit_b5 \
  --pre_iters 3000 --large_src_warmup 1000 --ft_iters 1500 --eval_every 500 \
  --batch_size 4 --workers 3 --seed 42 \
  --log_dir $E/logs --ckpt_dir $E/ckpt 2>&1 | tr "\r" "\n" | grep --line-buffered -v "it/s\|s/it\|arn" > $E/logs/duda.txt
touch $E/logs/DUDA_DONE
