# Session — 2026-09-10 — autonomous experiment runner

**Branch:** `feat/init-project-setup`
**Scope:** A 29-minute loop running experiments continuously. 10 experiments completed,
several infrastructure faults found and fixed, and two long-held assumptions overturned.

---

## The results that change the project

### 1. Self-training works — but the recommended recipe destroys it

| method | pseudo-label threshold | target (IGN) IoU | precision | recall |
|---|---|---|---|---|
| S1 source-only | — | 0.5611 | 0.741 | 0.698 |
| S2 **CBST ratio-matched** | 0.0004 | **0.3752** | 0.391 | 0.902 |
| S4 **confidence threshold** | 0.4563 | **0.6165** | 0.789 | 0.738 |

`MASTER_CONTEXT` prescribes CBST class-ratio thresholding to prevent foreground collapse. On
this domain gap it *causes* the collapse: matching the source's 1.83 % foreground forces the
threshold to **0.0004**, so pseudo-labels are noise. A plain confidence threshold instead
**beat the source-only baseline**, closing 17.8 % of the 31-point gap with *both* precision
and recall improving.

**CBST has an unstated precondition — it is safe only while the model retains calibrated
confidence on the target.** A large gap is exactly what destroys that. `plan/03` Tier 3 and
`MASTER_CONTEXT` both need correcting.

**On Jaipur this would have been undetectable.** S2's *source* score barely moved
(0.8723 → 0.8678) and would have looked healthy while the target degraded by a third.

### 2. Half of all buildings were being merged, and no metric could see it

D4 measured **78 % of Jaipur buildings touching a neighbour**. Implementing merge/split rate
showed a model at 0.6483 IoU was fusing **50 %** of buildings and emitting **21 % fewer
components than exist** — fatal for a per-building kW estimate, invisible to IoU.

Eroding footprints by 0.4 m before rasterising fixes it. Complete sweep:

| erosion | IoU | merge | split | missed | **pred/label** |
|---|---|---|---|---|---|
| none | 0.6569 | 0.4615 | 0.0341 | 0.2423 | 0.7600 |
| 0.2 m | 0.6540 | 0.4118 | 0.0496 | 0.2688 | 0.8412 |
| **0.4 m** | 0.6393 | 0.3155 | 0.0840 | 0.3257 | **0.9914** |
| 0.8 m | 0.5911 | 0.1339 | 0.1762 | 0.4689 | 1.4743 |

**0.018 IoU buys under-counting falling from 24 % to 0.9 %.** The rule is *"tune erosion
until pred/label ≈ 1"*, not *"use 0.4 m"*.

### 3. Current default configuration

**MiT-B2 encoder + 0.4 m eroded Open Buildings labels + ImageNet init.** MiT beat ResNet-34
by +0.0086 and converged 2× faster; the AIRS seed was worth nothing (identical trajectories
to ImageNet init).

---

## What was fixed in the infrastructure

- **`--cache_ram`** on both trainers — training was dataloader-bound with GPU utilisation a
  0→97 % sawtooth; PNG decode costs 11.6 ms/image. Measured **3.68×**.
- **The cache then had a copy-on-write bug**: a list of 33k arrays is copied per DataLoader
  worker because refcounting writes object headers. Wall clock was **8.4× worse than the
  per-epoch timer reported**. Fixed with a verified contiguous array.
- **Quota parking**: `data/jaipur_weak` (4.2 GB, regenerable) moved to host `/tmp`, mounted
  back by `run_docker.sh`. Quota 2.0 → 6.1 GB free.
- **`.gitignore` now covers `*.pth`** — a 280 MB checkpoint had reached GitHub and the
  rejection looked like a network error for four ticks.
- **Split-rate metric fixed** — the strict ≥50 % definition was blind to fragmentation and
  had reported 0.0 three times while a model shattered buildings.

Fifteen distinct traps are recorded in [`docs/PITFALLS.md`](../PITFALLS.md), grouped into
three root patterns in its §0.

---

## Still open, in priority order

1. **R12 — hand-label ~30 Jaipur tiles.** There is *no ground truth*: every Jaipur number is
   agreement with machine-generated Open Buildings footprints, which are also what the models
   train on. Relative comparisons survive; absolute ones do not. Three results dead-end here.
2. **Confidence self-training on Jaipur** — first UDA method with measured evidence behind it.
3. **S3 — the BDAPPV zero-negatives bug** is unfixable from disk (raw data absent); every
   Stage-2 precision number measures the wrong task until BDAPPV is re-downloaded.
4. Multi-round self-training (S7); reconcile the three planning docs with three phase schemes.

**Running at session end:** two self-training threshold-cliff arms (pseudo-label thresholds
0.0100 and 0.0015) mapping where the method flips from helping to harmful. The confidence
distribution on IGN is **bimodal** — a 1.36× change in target ratio moves the threshold 45× —
so ratio is a dangerous control variable and threshold is a safe one.
