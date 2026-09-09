# 2026-09-09-boundary-relaxed-loss — stop penalising the roof/footprint offset

| | |
|---|---|
| **Status** | running |
| **Date** | 2026-09-09 |
| **Commit** | `HEAD` on `feat/init-project-setup` |

## Question

Ignore a ±4 px band around every weak-label edge when computing the loss, then repeat the
weak-supervision run.

**Why it matters:** the first weak-supervision run showed **recall ~0.11 above precision
for all 40 epochs**. The labels are Open Buildings *ground footprints*; the model predicts
*roof* outlines; off-nadir at 26.6 cm those disagree by roughly 8 px (Gap 4). So the loss
was actively teaching the model to shrink roofs down to footprint size — punishing it for
roof area that is genuinely there. Relaxing the boundary removes that pressure without
pretending the labels are exact.

**Prediction (before running):** **+0.02 to +0.05 IoU** (so ~0.67–0.70), with **precision
rising** and the precision/recall gap narrowing. Risk: a 4 px band on a 512 px crop removes
a large share of the informative pixels at this building density, so it may instead blur
edges and cost IoU.

## Setup

Identical to [`2026-09-09-weak-supervision-jaipur`](../2026-09-09-weak-supervision-jaipur/)
plus `--boundary_relax 4`. Loss change in `rooftop/train.py`: a morphological gradient
(max-pool dilate minus erode) gives the band; BCE is masked per-pixel, and Dice is masked
**after** the sigmoid — scaling logits by zero maps to 0.5, not background, which would
quietly poison the term.

```bash
./run_docker.sh 2 "python -u rooftop/train.py \
    --train_dir data/jaipur_weak/train --val_dir data/jaipur_weak/val \
    --init_weights rooftop/checkpoints/unet_resnet34_best.pth \
    --boundary_relax 4 --epochs 40 --batch_size 16 --workers 4 --save_every 999 \
    --ckpt_dir experiments/2026-09-09-boundary-relaxed-loss/checkpoints \
    --log_dir  experiments/2026-09-09-boundary-relaxed-loss/outputs"
```

## Results

*pending*

## Threats to validity

- **Validation uses the unrelaxed metric**, so the score stays comparable with every other
  run. Only the training loss changes. If val IoU were also relaxed the comparison would be
  meaningless.
- Same footprint-label caveat: agreement with footprints, not roof accuracy.
- 4 px is a guess anchored on the ~8 px offset estimate. A sweep over 2/4/8 px would be the
  honest follow-up if this helps.
