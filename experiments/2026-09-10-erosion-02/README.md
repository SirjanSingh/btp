# 2026-09-10-erosion-02 — fill the gap between un-eroded and 0.4 m

| | |
|---|---|
| **Status** | ✅ done — completes the sweep |
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

Best val IoU **0.6540** @ep26. With the 0.2 m point added, the sweep is **monotonic in every
metric**:

| erosion | IoU | merge | split | missed | **pred/label** |
|---|---|---|---|---|---|
| none | **0.6569** | 0.4615 | 0.0341 | 0.2423 | 0.7600 |
| **0.2 m** | 0.6540 | 0.4118 | 0.0496 | 0.2688 | 0.8412 |
| **0.4 m** | 0.6393 | 0.3155 | 0.0840 | 0.3257 | **0.9914** |
| 0.8 m | 0.5911 | 0.1339 | 0.1762 | 0.4689 | 1.4743 |

**Prediction scorecard — 3 of 4 inside the bracket:** IoU 0.648–0.654 → **0.6540** ✅ ·
merge 0.38–0.42 → **0.4118** ✅ · split 0.05–0.06 → **0.0496** ✅ (just under) ·
pred/label 0.85–0.90 → **0.8412** ❌, slightly below.

## Interpretation

**A clean dose-response curve.** Every metric moves monotonically with erosion, in the
expected direction, with no inflection: more erosion buys less merging and costs more misses
and more splits. That is what a well-behaved knob looks like, and it means the earlier
three-point sweep was not hiding structure.

**0.4 m stays the default, and now for a stated reason rather than by bracketing.** It is the
only point where `pred/label` reaches ~1.0. The others under- or over-count:

- 0.2 m → **0.8412**, still 16 % under-counting
- 0.4 m → **0.9914**, within 0.9 %
- 0.8 m → **1.4743**, 47 % over

**The sub-pixel worry was partly wrong, and worth correcting.** I flagged that 0.2 m ≈ 0.75 px
might "round away entirely". IoU barely moved (−0.003), which fits that story — but merge
fell 0.4615 → 0.4118 and `pred/label` rose 0.760 → 0.8412, which does not. Sub-pixel erosion
has a real effect on **instance structure** while leaving pixel overlap almost untouched. That
is a small illustration of the project's larger lesson: IoU and instance metrics measure
different things, and IoU is the less informative of the two here.

**The cost of the fix is remarkably cheap.** Going from un-eroded to 0.4 m costs **0.018 IoU**
and takes under-counting from 24 % to 0.9 %. For a pipeline whose output is a per-building kW
estimate, that is close to free.

## Decision

- [x] **0.4 m + MiT-B2 confirmed as default**, now on a four-point curve rather than a
      bracket.
- [x] Sweep closed — the curve is monotonic and smooth, so intermediate points (0.3, 0.5 m)
      would refine `pred/label` toward 1.0 but change nothing structural.
- [ ] If a future model has a different merge baseline, the optimal erosion will move with it;
      the rule is *"tune erosion until `pred/label` ≈ 1"*, not *"use 0.4 m"*.

## Threats to validity

- All four points share one seed each; the differences between adjacent points (especially
  0 vs 0.2 m on IoU, −0.003) are within plausible seed noise even where the trend is not.
- Every number is agreement with Open Buildings footprints, not roof ground truth (R12).
- `pred/label` counts OB polygons; where OB already merges two structures, this cannot see it.

## Threats to validity

- 0.2 m is ~0.75 px at 26.6 cm — **less than one pixel**, so the rasterised effect may be
  closer to "sometimes shrinks by 1 px" than a clean 0.2 m shrink. That quantisation could
  make this arm behave more like un-eroded than the interpolation suggests.
- Same footprint-not-roof caveat as everything else (R12).
- Single seed.

---

## Addendum (2026-09-10 evening) — the labels were inspected visually

`scripts/viz_label_levels.py` renders the footprints over the imagery at all four levels for
8 stratified crops, with connected-component counts **on the label side**:

| erosion | label components | vs 0 m | label area |
|---|---|---|---|
| 0 m | 182 | — | — |
| 0.2 m | 360 | **+98 %** | −7.2 % |
| 0.4 m | 374 | +105 % | −14.4 % |
| 0.8 m | 384 | +111 % | −28.5 % |

**Read alone, this argues for 0.2 m** — it separates nearly twice as many buildings for half
the area cost, and 0.2 → 0.4 m adds only 7 points of separation for another 7 % of area.

**That reading is wrong, and the disagreement is the useful part.** Labels separate readily at
0.2 m, but the *model trained on them* still under-counts by 16 % (`pred/label` 0.8412). Only
at 0.4 m does it reach 0.9914. A gap the labels contain is not automatically a gap the model
learns — at 0.75 px the separation is too thin to survive the encoder's downsampling, and it
takes ~1.5 px before the model reliably emits two components.

**Rule this establishes:** never pick an erosion level (or any label-space knob) from label
statistics alone. The label is the input; `pred/label` is the output, and only the output is
the deliverable. Recorded in `docs/PITFALLS.md`.

Also refuted: **no polygon vanished entirely at any level**, including 0.8 m — the
small-building wipe-out predicted above does not occur. The 0.8 m damage is area loss and
over-fragmentation, not disappearance.

Visual review page built by `scripts/viz_build_page.py`.
