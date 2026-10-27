# Phase 2 — Stage 1 Baseline Training (Runbook)

**Goal (from `.planning/ROADMAP.md`):** reproducible U-Net (ResNet-34) training loop on AIRS
reaching **val IoU > 0.88** within 30 epochs, with correct loss/metrics/checkpoint/logging infra.

**Status of the code:** `rooftop/train.py` already satisfies every Phase-2 deliverable —
- global TP/FP/FN IoU/F1/P/R accumulation (`train.py:338`) — matches AIRS paper method
- differential LR: encoder `lr×0.1`, decoder/head `lr` (`train.py:437`)
- AdamW + CosineAnnealingLR, AMP GradScaler, DataParallel multi-GPU
- BCE(0.5)+Dice(0.5) loss, per-epoch JSON/TXT logs + TensorBoard, `--save_every`, `--resume`

So Phase 2 is **infra + data**, not code. Blockers below, in order.

---

## 0. Fix file ownership (one time)  ⚠️ blocker

Earlier training ran in Docker as root, so many files under `~/btp` are `root:root` and
can't be moved/cleaned. Fix from the host:

```bash
docker run --rm -v /home/23ucs715/btp:/w ubuntu:22.04 chown -R 1205:1206 /w
docker run --rm -v /home/23ucs715:/h  ubuntu:22.04 chown -R 1205:1206 /h/checkpoints
```

Then clean the stray leftovers:

```bash
cd ~/btp
git rm -r --cached logs                       # root-level logs/ = dup of rooftop/logs/
rm -rf logs rooftop/checkpoints/solar_panel rooftop/logs/solar_panel solar_panel/solar_panel
rmdir solar_panel/checkpoints/unet_resnet34_100ep_20260403_180709 \
      solar_panel/checkpoints/unet_resnet34_100ep_20260404_050947 \
      solar_panel/checkpoints/unet_resnet34_100ep_20260404_051053
git add -A && git commit -m "chore: consolidate logs, drop stray Stage-2 artifacts from rooftop/"
```

## 1. Build the Docker image  ⚠️ blocker (no image built yet)

```bash
cd ~/btp
docker build -t btp_seg .          # pytorch 2.1.2 + cu118, ~10 min, needs network
```

Host driver is CUDA 11.x-compatible (V100). If `docker build` fails on the base image,
check `nvidia-smi` CUDA version and adjust the `FROM` tag in `Dockerfile`.

## 2. Get the AIRS dataset  ⚠️ blocker (not on this server)

`rooftop/dataset/{train,val}.txt` hold the official split (857 / 94 `christchurch_*.tif`).
The imagery itself is on Google Drive (see `.planning/PROJECT.md` → Drive folder structure).
**Put it under `~/btp/data/` — NOT `/scratch` or `/tmp`** (`/` is full, `/scratch` unusable here).

```bash
# inside the container (gdown is in requirements.txt), or configure rclone to your Drive
mkdir -p ~/btp/data/airs
# download AIRS train/val/test → data/airs/{train,val,test}/{image,label}/
```

Expected layout: `data/airs/<split>/image/*.tif` + `data/airs/<split>/label/*.png` (0/1 masks).

## 3. Tile to 512×512 crops

```bash
docker run --gpus '"device=7"' -it --rm --shm-size=16g \
  -v /home/23ucs715/btp:/workspace -w /workspace btp_seg bash

python rooftop/tile_airs.py --src_dir data/airs/train --out_dir data/airs_crops/train \
    --crop_size 512 --overlap 0.1
python rooftop/tile_airs.py --src_dir data/airs/val   --out_dir data/airs_crops/val
python rooftop/tile_airs.py --src_dir data/airs/test  --out_dir data/airs_crops/test
```

## 4. Launch Phase 2 training

Pick a free GPU first (`nvidia-smi`); the node is usually 7/8 busy. Run inside `screen`.

```bash
screen -S phase2
docker run --gpus '"device=<FREE_GPU>"' -it --rm --shm-size=16g \
  -v /home/23ucs715/btp:/workspace -w /workspace btp_seg bash

python rooftop/train.py \
    --train_dir data/airs_crops/train \
    --val_dir   data/airs_crops/val \
    --ckpt_dir  rooftop/checkpoints \
    --log_dir   rooftop/logs \
    --arch unet --encoder resnet34 \
    --epochs 30 --batch_size 32 --lr 1e-3 --workers 2 \
    --save_every 5
```

`--lr 1e-3` → encoder 1e-4 / decoder 1e-3, exactly the roadmap spec.
Start with `--max_samples 2000` for a fast sanity run (matches the earlier 0.90-IoU result),
then drop it for the full run.

Watch: `tail -f rooftop/logs/$(ls -t rooftop/logs/*.txt | head -1)`

## 5. Phase 2 exit criteria (verify before calling it done)

- [ ] val IoU > 0.880 within 30 epochs
- [ ] loss strictly decreasing first 10 epochs, no NaN/inf
- [ ] reload `best.pth`, re-run one val batch → IoU matches saved value ±0.001
- [ ] checkpoints appear every `--save_every` epochs; `--resume` works after a kill
- [ ] (optional per roadmap) rclone push of checkpoints to Drive after each epoch — **not
      configured yet**, `rclone config` needed if you want offsite checkpoints

---

## Open items not covered by current code

| Item | Roadmap phase | Notes |
|------|---------------|-------|
| rclone → Drive checkpoint sync | 2 | `rclone` not configured on host or in a persistent config |
| WandB tracking | 2 (optional) | only TensorBoard is wired; add `wandb` if you want it |
| `--simulate_low_res` full run | 3 | closes the full-test-set gap (0.866 → ~0.90) |
