# 2026-09-11-mit-b5 — does more encoder capacity help small buildings?

| | |
|---|---|
| **Status** | running |
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

*pending*

## Threats to validity

- Batch-size confound above.
- Single seed, and the concurrent seed-variance run exists precisely to bound how much that
  matters.
- Agreement with Open Buildings, not accuracy (R12).
