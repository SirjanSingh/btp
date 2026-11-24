#!/bin/bash
# Step 6: fine-tune the zero-label r34_c0_shift model on hand-labelled crops, 2-fold.
#   fold A: train on the 76 calib crops, report on the 76 test crops
#   fold B: train on the 76 test crops,  report on the 76 calib crops
# No model selection touches the held-out half: the training half doubles as the "val"
# set (its number means nothing) and the epoch-30 checkpoint is used, not best.pth.
# The scoring threshold is picked on the training half and reported on the held-out half.
# Usage (DGX, from ~/btp): bash experiments/2026-11-20-toward-higher-iou/step6_finetune.sh GPU_A GPU_B
set -e
cd ~/btp
E=experiments/2026-11-20-toward-higher-iou
D=data/hand_ft
INIT=experiments/2026-11-19-hand-label-model-scoring/ckpt/r34_c0_shift/unet_resnet34_40ep_20261119_172916/best.pth
mkdir -p $E/logs $E/ckpt

python3 - <<'PY'
import csv, os, shutil
rows = list(csv.DictReader(open("data/hand_r12b2/index.csv")))
for fold, half in (("A", "calib"), ("B", "test")):
    for sub in ("images", "masks"):
        os.makedirs(f"data/hand_ft/{fold}/{sub}", exist_ok=True)
    n = 0
    for r in rows:
        if r["half"] != half:
            continue
        shutil.copy(r["img"], f"data/hand_ft/{fold}/images/{r['name']}")
        shutil.copy(r["mask"], f"data/hand_ft/{fold}/masks/{r['name']}")
        n += 1
    print(fold, half, n, "crops")
# swapped index for scoring fold B: its training half (test) becomes "calib"
with open("data/hand_r12b2/index_swapped.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    for r in rows:
        w.writerow({**r, "half": {"calib": "test", "test": "calib"}[r["half"]]})
PY

ft() {  # gpu fold
  ./run_docker.sh "$1" "python -u rooftop/train.py --train_dir $D/$2 --val_dir $D/$2 \
    --init_weights $INIT --epochs 30 --batch_size 8 --lr 1e-4 --workers 3 \
    --save_every 30 --patience 999 --seed 42 \
    --ckpt_dir $E/ckpt/ft_$2 --log_dir $E/logs/ft_$2" > $E/logs/train_ft_$2.txt 2>&1
}
ft "$1" A &
ft "$2" B &
wait
echo TRAIN_DONE

A=$(ls $E/ckpt/ft_A/*/epoch030.pth); B=$(ls $E/ckpt/ft_B/*/epoch030.pth)
printf "tag,group,ckpt\nr34_c0_shift,weak,%s\nft_A,hand-ft,%s\n" "$INIT" "$A" > $E/models_ftA.csv
printf "tag,group,ckpt\nr34_c0_shift,weak,%s\nft_B,hand-ft,%s\n" "$INIT" "$B" > $E/models_ftB.csv
./run_docker.sh "$1" "python scripts/hand_ensemble_budget.py --index data/hand_r12b2/index.csv \
  --models $E/models_ftA.csv --baseline r34_c0_shift --out $E/step6_foldA.json" > $E/logs/score_ftA.txt 2>&1 &
./run_docker.sh "$2" "python scripts/hand_ensemble_budget.py --index data/hand_r12b2/index_swapped.csv \
  --models $E/models_ftB.csv --baseline r34_c0_shift --out $E/step6_foldB.json" > $E/logs/score_ftB.txt 2>&1 &
wait
echo ALL_DONE
