cd ~/btp
E=experiments/2026-11-20-toward-higher-iou
true
BASE=experiments/2026-11-19-hand-label-model-scoring/ckpt/r34_c0_shift/unet_resnet34_40ep_20261119_172916/best.pth
{ echo "tag,group,ckpt"; echo "r34_c0_shift,weak,$BASE"
  for t in lab_sam lab_align_sam lab_sam_s43 lab_align_sam_s43 st_soft_pred; do
    c=$(ls /tmp/btp_data/iou/ckpt/$t/*/best.pth 2>/dev/null | head -1); [ -n "$c" ] && echo "$t,overnight,${c/\/tmp\/btp_data\//data/}"; done; } > $E/models_seeds.csv
./run_docker.sh 5 "python -u scripts/hand_ensemble_budget.py --index data/hand_r12b2/index.csv --models $E/models_seeds.csv --baseline r34_c0_shift --out $E/seeds_scores.json" > $E/logs/score_seeds.txt 2>&1
touch $E/logs/SEEDS_SCORED
