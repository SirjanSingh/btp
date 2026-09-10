# 2026-09-10-erosion-sweep — how far can erosion go before it starts splitting buildings?

| | |
|---|---|
| **Status** | running |
| **Date** | 2026-09-10 |

## Question

Erode footprints by **0.8 m** (~3 px) instead of 0.4 m, and retrain MiT-B2.

**Why:** [eroded-labels](../2026-09-10-eroded-labels/) cut merging 29 % and took `pred/label`
from 0.760 to 0.9745 — but **split rate stayed at exactly 0.0**, in all four cells of the
2 × 2. The failure mode erosion is supposed to risk has not appeared at all, which means the
useful range has not been explored to its end. If 0.8 m keeps split at 0 while cutting merges
further, 0.4 m was simply too timid.

It also re-tests dilation. At 0.4 m, dilating predictions back **re-merged** buildings
(merge 0.3256 → 0.4163) because neighbour gaps were only ~3 px and a 2 px dilation from each
side closes them. At 0.8 m the gap should be ~6 px, wide enough that a 3 px dilation might
recover misses without re-merging. That is the specific geometric claim being tested.

**Predictions (before running):**

- **merge rate below 0.25**, down from 0.3256.
- **split rate finally rises above 0** — somewhere around 0.02–0.08. If it stays at exactly
  0.0 again, that is a real surprise and means the model simply never over-segments at any
  erosion this side of destroying the labels.
- **missed rate rises further**, ~0.36–0.42; more erosion means more conservatism.
- **IoU 0.61–0.63**, below 0.6405.
- **`pred/label` near or slightly above 1.0.** Above 1.0 would mean splitting has begun.
- **Dilation at 3 px now helps** rather than hurts — this is the claim I got wrong at 0.4 m,
  restated with the geometry that should make it true.

## Setup

Masks: 318,207 polygons eroded 0.8 m, mean foreground **17.42 %** (vs 20.17 % at 0.4 m,
23.06 % un-eroded). Training only; val stays the original un-eroded set, as in every other
arm.

```bash
./run_docker.sh 7 "python -u rooftop/train.py \
    --train_dir data/jaipur_weak_erode8/train --val_dir data/jaipur_weak/val \
    --arch unet --encoder mit_b2 --epochs 40 --batch_size 12 --workers 3 --save_every 999 ..."
```

## Results

*pending — IoU, then merge/split at dilate 0 and 3 px*

## Threats to validity

- Two points (0.4, 0.8 m) plus the un-eroded baseline is a coarse sweep; the optimum could
  sit anywhere between.
- Erosion removes small buildings entirely — at 0.4 m only 6 of 318,207 vanished, but 0.8 m
  will drop more, which changes the label set as well as its geometry. **Measured: 123 of 318,207
  vanish at 0.8 m** (vs 6 at 0.4 m) — a 20× rise, but still only 0.04 % of the set, so
  shrinkage rather than deletion remains the dominant effect.
- Same footprint-not-roof caveat throughout.
