#!/bin/bash
# Waits for the refined label sets, trains r34 on each, then scores all round-1 models on the
# 152 hand crops in one table (threshold and choice on calib, IoU on test, bootstrap vs baseline).
# Usage (DGX, ~/btp): nohup bash experiments/2026-11-20-toward-higher-iou/round1_after.sh > .../logs/round1_after.txt 2>&1 &
set -uo pipefail
cd ~/btp
E=experiments/2026-11-20-toward-higher-iou
while [ $(ls data/iou/build_summary_*of3.json 2>/dev/null | wc -l) -lt 3 ]; do sleep 60; done
for v in align sam align_sam; do echo "$v masks: $(ls data/iou/lab_$v/train/masks | wc -l)"; done
date; echo "labels ready, training lab_*"
bash $E/round1.sh lab_align 5 &
bash $E/round1.sh lab_sam 5 &
bash $E/round1.sh lab_align_sam 1 &
while [ $(ls $E/logs/DONE_{edge,dino,lab_align,lab_sam,lab_align_sam} 2>/dev/null | wc -l) -lt 5 ]; do sleep 120; done
date; echo "all 5 trained, scoring"
BASE=experiments/2026-11-19-hand-label-model-scoring/ckpt/r34_c0_shift/unet_resnet34_40ep_20261119_172916/best.pth
{ echo "tag,group,ckpt"; echo "r34_c0_shift,weak,$BASE"
  for t in edge dino lab_align lab_sam lab_align_sam; do
    c=$(ls data/iou/ckpt/$t/*/best.pth 2>/dev/null | head -1); [ -n "$c" ] && echo "$t,round1,$c"; done; } > $E/models_r1.csv
cat $E/models_r1.csv
./run_docker.sh 5 "env HF_HOME=data/iou/hf HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/data/iou/pylib_dino python -u scripts/hand_ensemble_budget.py --index data/hand_r12b2/index.csv --models $E/models_r1.csv --baseline r34_c0_shift --out $E/round1_scores.json" > $E/logs/score_r1.txt 2>&1
date; touch $E/logs/R1_SCORED
