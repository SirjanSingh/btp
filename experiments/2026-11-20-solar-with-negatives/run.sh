#!/usr/bin/env bash
# Solar retrain with no-panel images. Run on the DGX from ~/btp:
#   nohup bash experiments/2026-11-20-solar-with-negatives/run.sh 0 > experiments/2026-11-20-solar-with-negatives/logs/run.txt 2>&1 &
# 1. wait for the BDAPPV zip (8,165,112,873 bytes) to finish downloading to host /tmp
# 2. build /tmp/btp_data/solar_neg (host python; run_docker.sh mounts it at data/solar_neg)
# 3. train ResNet34 U-Net, same recipe as 2026-09-10-solar-google-to-ign, 30 epochs
# 4. run the new model on the 152 hand-labelled Jaipur crops and write a gallery
set -uo pipefail
G=${1:-0}
E=experiments/2026-11-20-solar-with-negatives
Z=/tmp/bdappv_23ucs715/bdappv.zip
mkdir -p $E/logs $E/ckpt

until [ "$(stat -c %s $Z 2>/dev/null)" = "8165112873" ]; do sleep 60; done
echo "zip complete $(date)"
python3 -c "import zipfile,sys; z=zipfile.ZipFile('$Z'); print('zip entries', len(z.namelist()))" || exit 1

if [ ! -f /tmp/btp_data/solar_neg/counts.json ]; then
  python3 -u $E/build_solar_neg.py $Z > $E/logs/build.txt 2>&1 || { echo build failed; exit 1; }
fi
cat /tmp/btp_data/solar_neg/counts.json
touch $E/logs/DATA_DONE

./run_docker.sh $G "python -u solar_panel/train_solar.py \
    --train_dir data/solar_neg/train --val_dir data/solar_neg/val \
    --epochs 30 --batch_size 16 --workers 3 --save_every 999 --cache_ram --seed 42 \
    --ckpt_dir $E/ckpt --log_dir $E/logs/tb" > $E/logs/train.txt 2>&1
touch $E/logs/TRAIN_DONE

CK=$(ls -t $E/ckpt/*/best.pth | head -1)
# The trainer resizes 400 px BDAPPV crops to 512, so the model sees ~0.078 m/px.
# Jaipur is 0.266 m/px: scale 3.4 matches that, 2.6 matches raw BDAPPV.
./run_docker.sh $G "python -u scripts/solar_on_hand_crops.py --index data/hand_r12b2/index.csv \
    --ckpt $CK --scales 1 2.6 3.4 --n 20 --out diagnostics/solar_on_hand_neg" > $E/logs/jaipur_eval.txt 2>&1
touch $E/logs/ALL_DONE
