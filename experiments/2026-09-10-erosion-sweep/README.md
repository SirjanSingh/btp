# 2026-09-10-erosion-sweep — how far can erosion go before it starts splitting buildings?

| | |
|---|---|
| **Status** | ✅ done — **0.8 m over-erodes** |
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

Best val IoU **0.5911** @ep10 (early-stopped ep25). Merge/split on the full val set:

| erosion | IoU | **merge** | split | missed | **pred/label** |
|---|---|---|---|---|---|
| none | 0.6569 | 0.4615 | 0.0 | 0.2423 | 0.760 |
| **0.4 m** | 0.6405 | 0.3256 | 0.0 | 0.3223 | **0.9745** |
| 0.8 m | 0.5911 | **0.1339** | 0.0 | **0.4689** | **1.4743** |
| 0.8 m + dilate 3 px | — | 0.3562 | 0.0 | 0.2744 | 0.9288 |

**Predictions vs measured:** merge below 0.25 → **0.1339** ✅ · split above 0 → **still exactly
0.0**, wrong for the third time ❌ · missed 0.36–0.42 → **0.4689**, worse than predicted ❌ ·
IoU 0.61–0.63 → 0.5911, worse ❌ · pred/label near 1.0 → **1.4743**, badly over ❌ · dilation
helps → **partially**, see below.

## Interpretation

**0.8 m over-erodes, and `pred/label` is what reveals it.** Merge rate looks spectacular —
0.4615 → 0.1339, a 71 % reduction — and taken alone it would read as a triumph. But the model
now emits **47 % more components than there are buildings** and misses nearly half of them.
It is not separating buildings; it is **fragmenting** them.

**★ My split-rate definition cannot see that, and that is a flaw in the metric.** A label is
"split" only when ≥ 2 predictions each cover ≥ 50 % of it. Fragments smaller than half a
building never qualify, so a model shattering buildings into thirds scores **split = 0.0**
while `pred/label` climbs past 1.4. The three consecutive "split stayed at 0" results I kept
reporting as surprising were partly an artifact of my own threshold. **`pred/label` is the
honest over/under-segmentation signal; split rate as defined here is not.**

**0.4 m remains the best operating point.** `pred/label` of 0.9745 is nearest 1.0 from either
side, with the least damage to IoU and missed rate. The sweep found the optimum by bracketing
it: 0.760 under, 0.9745 near-perfect, 1.4743 over.

**Dilation at 0.8 m works better than at 0.4 m — but is still a trade, not a fix.** It pulls
`pred/label` from 1.4743 to 0.9288 and cuts missed from 0.4689 to 0.2744, exactly as the
geometry predicted with a ~6 px gap. But merge climbs back 0.1339 → 0.3562. So dilation
reliably converts merges into misses and back; it never produces a configuration better than
plain 0.4 m erosion on any axis.

## Decision

- [x] **Adopt 0.4 m erosion.** The sweep brackets it as the optimum.
- [x] **Drop inference-time dilation.** Tested at both erosion levels; always a trade, never
      a win.
- [ ] ★ **Fix the split-rate metric** — count a label as split if ≥ 2 predictions overlap it
      at all, or report fragment count per label. As defined it has been silently blind.
- [ ] Re-check the 0.4 m and un-eroded numbers once the metric is fixed; their split rates of
      0.0 may also be artifacts.

## Threats to validity

- The metric flaw above affects **every split rate in this repo**.
- Three erosion points plus dilation is still a coarse sweep; the optimum is bracketed, not
  located precisely.
- 123 of 318,207 polygons vanish entirely at 0.8 m (vs 6 at 0.4 m) — a 20× rise, though still
  0.04 %, so shrinkage rather than deletion remains dominant.
- Single seed throughout.

## Threats to validity

- Two points (0.4, 0.8 m) plus the un-eroded baseline is a coarse sweep; the optimum could
  sit anywhere between.
- Erosion removes small buildings entirely — at 0.4 m only 6 of 318,207 vanished, but 0.8 m
  will drop more, which changes the label set as well as its geometry. **Measured: 123 of 318,207
  vanish at 0.8 m** (vs 6 at 0.4 m) — a 20× rise, but still only 0.04 % of the set, so
  shrinkage rather than deletion remains the dominant effect.
- Same footprint-not-roof caveat throughout.
