# 2026-09-10-split-metric-audit — is the split rate lying, and what is the 0.8 m damage really?

| | |
|---|---|
| **Status** | ✅ done — metric vindicated, interpretation corrected |
| **Date** | 2026-09-10 |

## Question

R11 flagged every split rate in the repo as suspect. The strict definition counts a label as
split only when **two predictions each cover ≥ 50 %** of it, so a building shattered into
thirds scores zero. At 0.8 m erosion the split rate read **0.0** while `pred/label` hit
**1.4743** — a contradiction that should be impossible if the metric worked.

**Why it matters:** merge and split rates are the metrics this project reports *instead of*
IoU, on the grounds that IoU cannot see instance errors. If they are themselves blind, the
argument collapses.

**Prediction (before re-running):** the published split rates are wrong and will move once
recomputed with the loose (≥ 10 %) definition.

## Setup

Re-ran `scripts/merge_split_rate.py` on all three surviving MiT-B2 rooftop checkpoints
(0.2 / 0.4 / 0.8 m erosion) against the same 1,701-crop val set, reporting both definitions
plus `fragments_per_label`.

## Results

| erosion | merge | **split (≥10 %)** | split_strict (≥50 %) | frag/label | missed | pred/label |
|---|---|---|---|---|---|---|
| 0.2 m | 0.4118 | 0.0496 | **0.0** | 0.8992 | 0.2688 | 0.8412 |
| 0.4 m | 0.3155 | 0.0840 | **0.0** | 0.9203 | 0.3257 | 0.9914 |
| 0.8 m | 0.1339 | 0.1762 | **0.0** | 1.0354 | 0.4689 | 1.4743 |

**Prediction wrong, in the good direction.** The published numbers are unchanged — every split
rate already in the repo was computed with the loose definition, because the fix landed before
those runs. Nothing needs correcting.

**But `split_strict` is 0.0 at every single level**, including 0.8 m where the model emits
47 % more components than there are labels. That is the metric's blindness demonstrated rather
than argued: a definition that returns the same value across the entire useful range of a
parameter carries **zero information**.

## Interpretation

**A correction to what I wrote earlier today.** I described the 0.8 m failure as
"over-fragmentation" — buildings shattering into pieces. The numbers say otherwise:

- `fragments_per_label` at 0.8 m is only **1.0354**, i.e. labels that are hit are hit by
  ~1 component. Buildings are *not* shattering.
- Yet `pred/label` is **1.4743** — 41,160 predicted components against 27,918 labels.
- Components that touch a label ≈ 28,900. So roughly **12,250 predicted components — 30 % of
  all predictions — touch no label at all.**

**Over-erosion does not fragment buildings; it hallucinates new ones.** Shrinking labels
teaches the model that gaps belong between buildings, and past a point it starts inserting
buildings into gaps. That is a different failure with a different fix, and the merge/split
framing hid it: merge *improves* monotonically (0.4118 → 0.1339) all the way to 0.8 m, so on
merge alone 0.8 m looks best.

**`pred/label` is doing the real work, and it is doing it for two different reasons.** Below
0.4 m it moves because merging falls; above it, because false positives rise. A single ratio
conflating both is fine as a *tuning target* (1.0 is right either way) but misleading as a
diagnosis. `frag_sum` vs `pred_count` separates them, and should be reported alongside.

## Decision

- [x] **Keep the loose (≥ 10 %) split definition.** It is already what every published number
      uses.
- [x] **Drop `split_strict` from reporting** — retain it in the JSON as a documented null so
      the blindness stays visible, but never quote it.
- [x] **Report `fragments_per_label` beside `pred/label`** from now on: their divergence is
      what distinguishes fragmentation from hallucination.
- [x] R11 closed — the metric is sound; the interpretation of 0.8 m was not.

## Threats to validity

- Labels are Open Buildings footprints, so "touches no label" includes real buildings that
  Open Buildings missed. The 30 % figure is an upper bound on true false positives, and R12's
  hand labels are the only way to split that.
- Component counting is connectivity-8 on a 0.5-thresholded mask; a different threshold moves
  all counts, though the 0.2/0.4/0.8 ordering is stable.
- The un-eroded (0 m) checkpoint no longer exists on disk, so the sweep here starts at 0.2 m.
