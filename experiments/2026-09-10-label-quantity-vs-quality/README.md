# 2026-09-10-label-quantity-vs-quality — are the low-confidence buildings worth having?

| | |
|---|---|
| **Status** | ✅ done — **confounded; see cross-eval** |
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

### The comparison as designed (both scored on the conf-0.75 val set)

| | conf ≥ 0.75 (318k) | conf ≥ 0.0 (523k) | delta |
|---|---|---|---|
| best val IoU | **0.6483** @ep33 | 0.6281 @ep36 | **−0.0202** |
| precision | 0.7363 | 0.6771 | **−0.0592** |
| recall | 0.8443 | 0.8967 | **+0.0524** |

Predicted within ±0.02; measured −0.0202, right at the edge. The directional lean
(recall up, precision down) held — but both moves were far larger than "slight".

### Cross-evaluation — both models, both label sets

The precision drop above has an obvious alternative explanation: **a building present only in
the conf-0.0 label set is absent from the conf-0.75 val masks, so predicting it correctly
scores as a false positive.** Scoring both checkpoints against both label sets separates
"the extra labels are noise" from "the extra labels are real buildings the val set refuses to
credit". Raw: [`diagnostics/r3_cross_eval.json`](../../diagnostics/r3_cross_eval.json).

| trained ↓ · evaluated → | val @ 0.75 | val @ 0.00 |
|---|---|---|
| **conf ≥ 0.75** | **0.6483** (P 0.736 / R 0.844) | 0.6769 (P 0.835 / R 0.781) |
| **conf ≥ 0.00** | 0.6281 (P 0.677 / R 0.897) | **0.7205** (P 0.806 / R 0.871) |

## Interpretation

**Each model wins on its own label distribution.** That is the signature of a metric
measuring *label agreement* rather than accuracy, and it means the headline −0.0202 does not
support the conclusion it appears to.

**The low-confidence buildings are largely real, not noise.** The conf-0.0 model's precision
jumps from **0.677 → 0.806** simply by switching to a val set that includes the buildings it
was trained to find. Roughly 0.13 of what looked like false positives were real structures
missing from the conf-0.75 labels. If the extra 205k polygons were mostly junk, that number
would not move like that.

**But 0.7205 is not "the best model in the project".** IoU rises with foreground fraction,
and val@0.00 is a denser, easier target — visible in the fact that even the *conf-0.75* model
scores higher on it (0.6769) than on its own val set (0.6483). **Comparing IoU across
different label sets is meaningless.** Anyone quoting 0.7205 alongside 0.6483 would be
comparing two different tasks.

**The honest position: this question cannot be settled with these labels.** Neither val set is
ground truth; both are footprint approximations at different recall levels. Choosing a
confidence threshold by scoring against a val set built at *some* confidence threshold is
circular by construction.

## Decision

- [x] **Keep conf ≥ 0.75 as the default** — not because it is proven better, but because the
      comparison is unresolved and it is the incumbent.
- [x] **Do not quote 0.7205.** It is not comparable to any other number in this repo.
- [ ] ★ **This is a concrete argument for the 400-tile hand-labelling task.** It is the only
      way to break the circularity, and it now blocks a real decision rather than being
      generic good practice. `MASTER_CONTEXT` already lists it on the critical path; this
      experiment gives it a specific, dated justification.
- [ ] When ground truth exists, re-run this as a proper confidence sweep (0.0 / 0.5 / 0.75 /
      0.9) scored against it.

## Threats to validity

- **The original design was confounded** — one variable was meant to change and two did
  (training labels *and*, implicitly, what counts as correct). I wrote "mildly disadvantages
  this arm" when predicting; the cross-eval shows it was the dominant effect, not a mild one.
- Cross-eval used a single fixed threshold (0.5), no sweep.
- Neither val set is ground truth. Every number here is agreement-with-footprints.
- Single seed per arm.

## Threats to validity

- **Train and val now come from different label distributions** (conf 0.0 vs 0.75). That is
  deliberate — it keeps the comparison to baseline honest — but it does mean the model is
  optimised against slightly different targets than it is scored on, which mildly
  disadvantages this arm.
- Both label sets are footprints, not roofs. Unchanged caveat.
- Single seed. A confidence *sweep* (0.0 / 0.5 / 0.75 / 0.9) would be the fuller answer; this
  is the two-point version.
