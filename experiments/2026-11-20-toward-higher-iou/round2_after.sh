set -uo pipefail
cd ~/btp
E=experiments/2026-11-20-toward-higher-iou
while [ ! -f $E/logs/PSEUDO_DONE ]; do sleep 60; done
date; echo "pseudo labels ready"; ls /tmp/btp_data/iou/ | grep soft
bash $E/round2.sh st_soft_pred 6 "--train_dir data/iou/lab_soft_pred/train --encoder resnet34 --soft_masks" &
sleep 120
bash $E/round2.sh st_soft_fuse08 7 "--train_dir data/iou/lab_soft_fuse0.8/train --encoder resnet34 --soft_masks" &
while [ $(ls $E/logs/DONE_{edge,dino,lab_align,lab_sam,lab_align_sam,st_soft_pred,st_soft_fuse08} 2>/dev/null | wc -l) -lt 7 ]; do sleep 120; done
date; echo "all trained, scoring round 1 + 2"
BASE=experiments/2026-11-19-hand-label-model-scoring/ckpt/r34_c0_shift/unet_resnet34_40ep_20261119_172916/best.pth
{ echo "tag,group,ckpt"; echo "r34_c0_shift,weak,$BASE"
  for t in edge dino lab_align lab_sam lab_align_sam st_soft_pred st_soft_fuse08; do
    c=$(ls /tmp/btp_data/iou/ckpt/$t/*/best.pth 2>/dev/null | head -1); [ -n "$c" ] && echo "$t,overnight,${c/\/tmp\/btp_data\//data/}"; done; } > $E/models_r12.csv
cat $E/models_r12.csv
./run_docker.sh 6 "env HF_HOME=data/iou/hf HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/data/iou/pylib_dino python -u scripts/hand_ensemble_budget.py --index data/hand_r12b2/index.csv --models $E/models_r12.csv --baseline r34_c0_shift --out $E/round12_scores.json" > $E/logs/score_r12.txt 2>&1
date; touch $E/logs/R12_SCORED
