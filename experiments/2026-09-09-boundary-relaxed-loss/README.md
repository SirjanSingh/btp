# 2026-09-09-boundary-relaxed-loss — stop penalising the roof/footprint offset

| | |
|---|---|
| **Status** | ✅ done — **negative result** |
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

| | baseline | **relax = 4 px** | delta |
|---|---|---|---|
| best val IoU | **0.6475** @ep33 | 0.6366 @ep22 | **−0.0110** |
| precision | 0.7361 | 0.7402 | +0.0042 |
| recall | 0.8434 | 0.8197 | **−0.0237** |
| F1 | 0.7861 | 0.7779 | −0.0082 |
| recall − precision gap | +0.1073 | **+0.0795** | −0.0278 |

**Predicted +0.02 to +0.05 IoU with precision up. Measured −0.011.** Wrong.

## Interpretation

**The mechanism worked; the trade did not.** The precision/recall gap did narrow, from
+0.107 to +0.080 — so ignoring the band genuinely did relieve the pressure to shrink roofs
down to footprint size, exactly as intended. But it bought **+0.004 precision at a cost of
−0.024 recall**, roughly a 6:1 bad exchange, and IoU fell.

The reading: **at this building density a 4 px band removes far more true signal than label
noise.** D3 measured the median Jaipur building at 913 px, about 30×30. A ±4 px band around
such a shape covers a large fraction of it — the boundary *is* most of the object. The idea
is sound for large buildings and self-defeating for small ones, and Jaipur is small ones.

It also converged early (best at epoch 22, then 18 epochs of drift) which fits a model given
less to learn from.

## Decision

- [x] **Do not use boundary relaxation at 4 px.** Keep the standard loss.
- [ ] Not fully dead: a **1–2 px** band might sit on the right side of the trade, and the
      result would differ on larger buildings. But it is no longer a priority — R4's early
      numbers suggest architecture is the better lever.
- [x] The roof-vs-footprint offset is real (the gap moved) but **cannot be fixed by throwing
      pixels away**. Shift correction or SAM2 refinement — moving the labels rather than
      deleting them — remains the right approach (`plan/03` Tier 4).

## Threats to validity

- Single band width, single seed. A sweep over 1/2/4/8 px is the honest follow-up.
- Validation was deliberately **unrelaxed**, so this is a fair comparison to the baseline —
  but it also means the relaxed model is scored on the very pixels it was told to ignore.
  That is the correct choice (the task is roofs, not footprints-minus-boundaries) and it does
  disadvantage the method by construction.
- Footprint labels, not roof ground truth.

## Threats to validity

- **Validation uses the unrelaxed metric**, so the score stays comparable with every other
  run. Only the training loss changes. If val IoU were also relaxed the comparison would be
  meaningless.
- Same footprint-label caveat: agreement with footprints, not roof accuracy.
- 4 px is a guess anchored on the ~8 px offset estimate. A sweep over 2/4/8 px would be the
  honest follow-up if this helps.
