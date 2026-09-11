# 2026-09-11-mit-b5 — does more encoder capacity help small buildings?

| | |
|---|---|
| **Status** | ❌ negative — +0.0063 IoU (1.5× noise) for 3.6× compute; B2 stays |
| **Date** | 2026-09-11 |

## Question

R4 swapped ResNet-34 → MiT-B2 and gained **+0.0086 IoU**, with the gain entirely in precision,
and erosion turned out to be *synergistic* with MiT (−0.064 vs −0.136). `MASTER_CONTEXT` names
MiT-B5 as the prerequisite for the DAFormer/HRDA/MIC family. This tests the next size up on the
current default (0.4 m eroded labels).

**Why it matters:** the median Jaipur building is ~30×30 px (D2/D3: 913 px against AIRS's
21,084). Small-object segmentation is usually capacity- and receptive-field-limited, and MiT-B5
has a wider global receptive field — the property that made B2 exploit label gaps ResNet could
not.

## Predictions (before running)

**IoU +0.000 to +0.010 over the 0.6393 teacher.** Bigger encoders usually help, but 7,371 crops
is a small set for an 82 M-parameter backbone, and the labels are noisy machine-generated
footprints — capacity spent fitting label noise is wasted.

**`pred/label` will move less than IoU.** The counting behaviour is set mainly by label erosion,
which is unchanged here.

**Genuine uncertainty about direction.** Unlike most arms tonight I would not be surprised
either way; if B5 overfits the noisy labels it could land *below* B2. That makes the seed
replicate running alongside this directly relevant — any gain under ~0.008 will need the noise
floor to interpret.

## Setup

Identical to the default except `--encoder mit_b5` and `--batch_size 8` (B5 will not fit at 12).
Same 0.4 m eroded labels, 40 epochs, same val tiles, scored on IoU plus the full instance set.

**Confound noted up front:** batch size changes with the encoder, so this is "B5 at batch 8"
against "B2 at batch 12", not a clean encoder-only swap. A batch-8 B2 control would separate
them and is not being run yet.

## Results

**Compared against the three-seed MiT-B2 distribution**, not a single run — the first arm in
this project able to do that.

| | IoU | merge | split | frag/label | missed | pred/label |
|---|---|---|---|---|---|---|
| MiT-B2, seed 42 | 0.6393 | 0.3155 | 0.0840 | 0.9203 | 0.3257 | 0.9914 |
| MiT-B2, seed 43 | 0.6423 | 0.3650 | 0.0634 | 0.8962 | 0.3074 | 0.9317 |
| MiT-B2, seed 44 | 0.6434 | 0.3329 | 0.0793 | 0.9310 | 0.3045 | 1.0027 |
| **MiT-B2 mean** | **0.6417** | 0.3378 | 0.0756 | 0.9158 | 0.3125 | **0.9753** |
| **MiT-B5** | **0.6480** | 0.3070 | 0.0922 | 0.9438 | 0.3063 | **1.0509** |
| | **+0.0063** | −0.0308 | +0.0166 | +0.0280 | −0.0062 | +0.0756 |
| *as × 2 sd* | **1.5×** | 0.6× | 0.8× | 0.8× | 0.3× | **1.0×** |

**Prediction scorecard.** IoU +0.000 to +0.010 → **+0.0063** against the B2 mean (+0.0087
against seed 42 alone) ✅. "`pred/label` will move less than IoU" — ambiguous as I wrote it:
in absolute terms it moved *more* (0.0756 vs 0.0063), but relative to each metric's own noise
it moved **less** (1.0× vs 1.5× its 2 sd). The intended meaning holds; the wording did not.

## Interpretation

**A real but small gain that does not justify the cost.** +0.0063 IoU is **1.5× the B2 2 sd** —
above noise, but not comfortably, and it rests on a **single B5 run whose own variance is
unmeasured**. If B5 varies like B2, a second seed could plausibly halve or double it.

**Every other metric is inside noise.** merge 0.6×, split 0.8×, frag 0.8×, missed 0.3×. The
counting behaviour is unchanged, as predicted — it is set by label erosion, not encoder capacity.

**The cost side is decisive.** B5 runs at **530 s/epoch against B2's 147 s** (3.6×) and writes a
**971 MB checkpoint against 315 MB** (3.1×) — the checkpoint that nearly breached the quota
floor this morning. Paying 3.6× compute and 3× storage for a 1.5×-noise IoU gain, on a metric
this project has explicitly demoted below `pred/label`, is a bad trade.

**And `pred/label` moves the wrong way.** B5's 1.0509 is further from 1.0 than the B2 mean's
0.9753 (0.051 vs 0.025 absolute error) — within noise, so not a finding, but certainly not an
argument for B5 either.

**The batch confound is unresolved.** B5 ran at batch 8 and B2 at batch 12, for VRAM. A
batch-8 B2 control would separate encoder from batch size, and is not worth running given the
cost verdict above.

## Decision

- [x] **MiT-B2 stays the default.** B5's gain is 1.5× noise, single-seed, confounded with batch
      size, and costs 3.6× compute for it.
- [x] **`MASTER_CONTEXT`'s "swap to MiT-B5" prerequisite is answered for *this* stage**: the
      capacity is not what limits Jaipur rooftops. Small buildings are not a capacity problem
      here — erosion and label quality dominate.
- [ ] If the DAFormer/HRDA/MIC family is attempted later it may need B5 for its own reasons;
      that is a separate question from whether B5 helps the current model.

## Threats to validity

- **Single B5 seed against a 3-seed B2 distribution.** The comparison uses the baseline's
  variance, which is the right thing to do, but says nothing about B5's own.
- Batch-size confound (8 vs 12).
- Agreement with Open Buildings, not accuracy (R12).
- 40 epochs may suit B2 better than B5; no schedule tuning was attempted.

## Threats to validity

- Batch-size confound above.
- Single seed, and the concurrent seed-variance run exists precisely to bound how much that
  matters.
- Agreement with Open Buildings, not accuracy (R12).
