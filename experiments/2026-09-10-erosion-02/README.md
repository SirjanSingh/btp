# 2026-09-10-erosion-02 — fill the gap between un-eroded and 0.4 m

| | |
|---|---|
| **Status** | running |
| **Date** | 2026-09-10 |

## Question

Erode by **0.2 m** (~0.75 px). The sweep so far is 0 / 0.4 / 0.8 m and 0.4 m is the crossing
point; this fills the interval below it to check the optimum is not actually finer.

**Why it matters:** 0.4 m already achieves `pred/label` **0.9914** — within 0.9 % of one
prediction per building — but costs 0.018 IoU and 8.4 % split rate. If 0.2 m gets most of the
count benefit for half the cost, it is the better default. If it barely moves, 0.4 m is
confirmed as a genuine threshold rather than an arbitrary point on a slope.

**Predictions (before running):** everything should land **between** the un-eroded and 0.4 m
values, since the sweep has been monotonic in every metric so far.

| | un-eroded | **0.2 m predicted** | 0.4 m |
|---|---|---|---|
| IoU | 0.6569 | **0.648 – 0.654** | 0.6393 |
| merge | 0.4615 | **0.38 – 0.42** | 0.3155 |
| split | 0.0341 | **0.05 – 0.06** | 0.0840 |
| pred/label | 0.7600 | **0.85 – 0.90** | 0.9914 |

If any metric lands *outside* that bracket, the relationship is not monotonic and the sweep
needs more points, not fewer.

## Setup

Masks: 318,207 polygons eroded 0.2 m, mean foreground **21.61 %** (vs 23.06 / 20.17 / 17.42 %
for 0 / 0.4 / 0.8 m). Training only; val stays the original un-eroded set.

**First run to use `--cache_ram`** — measured 3.68× faster (74.4 → 20.2 s/epoch) by decoding
the crop set into RAM once instead of re-decoding every epoch.

```bash
./run_docker.sh 2 "python -u rooftop/train.py --train_dir data/jaipur_weak_erode2/train \
    --val_dir data/jaipur_weak/val --arch unet --encoder mit_b2 --cache_ram \
    --epochs 40 --batch_size 12 --workers 3 --save_every 999 ..."
```

## Results

*pending*

## Threats to validity

- 0.2 m is ~0.75 px at 26.6 cm — **less than one pixel**, so the rasterised effect may be
  closer to "sometimes shrinks by 1 px" than a clean 0.2 m shrink. That quantisation could
  make this arm behave more like un-eroded than the interpolation suggests.
- Same footprint-not-roof caveat as everything else (R12).
- Single seed.
