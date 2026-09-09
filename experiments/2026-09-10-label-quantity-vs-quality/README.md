# 2026-09-10-label-quantity-vs-quality — are the low-confidence buildings worth having?

| | |
|---|---|
| **Status** | running |
| **Date** | 2026-09-10 |

## Question

Train on weak labels built from **all 523,283** Open Buildings polygons (confidence ≥ 0.0)
instead of the **318,207** at confidence ≥ 0.75, and compare.

**Why it matters:** the baseline discards **205,076 buildings** — 39% of the dataset — on a
confidence threshold nobody has justified with a measurement. Open Buildings is least
confident about buildings that are small, irregular, or densely packed, which in Jaipur are
plausibly the **hard and important** ones. If discarding them costs accuracy, the threshold is
throwing away exactly the signal the project needs; if it does not, the current setting is
vindicated and the choice is defensible in the report rather than arbitrary.

Label prior confirms the sets differ as intended: conf ≥ 0.0 gives **28.17 %** mean
foreground (matching D1's 28.19 % for all polygons), versus **23.06 %** at ≥ 0.75.

**Prediction (before running):** within **±0.02** of the 0.6475 baseline — more labels but
noisier, roughly cancelling. Slight lean to a small *gain* in recall and a small *loss* in
precision, since the extra buildings are real but their outlines are less reliable.

## Setup

Identical to the baseline except the training masks.

**Design fix, important:** the model trains on conf-0.0 masks but validates against the
**original conf-0.75 val masks** — the same targets the baseline was scored on. Validating on
conf-0.0 masks would change the eval target as well as the training data, and the resulting
number would not be comparable to 0.6475 at all. Only one thing may vary.

| | |
|---|---|
| Train | `data/jaipur_weak_c0/train` — 7,371 crops, masks from 523,283 polygons |
| Val | `data/jaipur_weak/val` — **unchanged**, 1,701 crops at conf ≥ 0.75 |
| Init | ImageNet (R2 showed the AIRS seed is worthless) |

Images are shared between the two label sets via relative symlinks — `--masks_only` writes
masks alone, since images are 3.5 GB of the 4.2 GB set and a confidence sweep would otherwise
exhaust the quota.

```bash
./run_docker.sh 2 "python -u rooftop/train.py \
    --train_dir data/jaipur_weak_c0/train --val_dir data/jaipur_weak/val \
    --epochs 40 --batch_size 16 --workers 3 --save_every 999 \
    --ckpt_dir experiments/2026-09-10-label-quantity-vs-quality/checkpoints \
    --log_dir  experiments/2026-09-10-label-quantity-vs-quality/outputs"
```

## Results

*pending*

## Threats to validity

- **Train and val now come from different label distributions** (conf 0.0 vs 0.75). That is
  deliberate — it keeps the comparison to baseline honest — but it does mean the model is
  optimised against slightly different targets than it is scored on, which mildly
  disadvantages this arm.
- Both label sets are footprints, not roofs. Unchanged caveat.
- Single seed. A confidence *sweep* (0.0 / 0.5 / 0.75 / 0.9) would be the fuller answer; this
  is the two-point version.
