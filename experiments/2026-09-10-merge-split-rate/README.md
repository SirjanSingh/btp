# 2026-09-10-merge-split-rate — half of all buildings are merged, and IoU never said so

| | |
|---|---|
| **Status** | ✅ done |
| **Date** | 2026-09-10 |

## Question

Implement merge rate and split rate, and measure them on the two best rooftop checkpoints.

**Why:** [D4](../2026-09-10-d4-adjacency/) measured **78 % of Jaipur buildings touching a
neighbour**. `MASTER_CONTEXT` §6.2 states that pixel IoU cannot detect instance merging and
boundary IoU largely cannot either. So every IoU in this repo has been silent about a failure
mode affecting most buildings — and the pipeline ends in a **per-building kW estimate**, where
merging two houses corrupts the count, the per-roof area distribution and `k_usable` without
moving IoU at all.

It also decides whether [R4](../2026-09-10-segformer-backbone/)'s precision gain (+0.017) can
be attributed to better separation of adjacent buildings, which was the interesting
explanation and was unverifiable until now.

**Prediction:** merge rate 20–35 % for ResNet-34, with MiT-B2 a few points lower.

## Setup

Definitions, stated because the literature varies. A predicted component **P** and a label
component **L** are *associated* when their intersection covers ≥ **50 %** of L's area.

- **Merge** — one P associated with ≥ 2 label components (under-segmentation)
- **Split** — one L associated with ≥ 2 predicted components (over-segmentation)
- **Missed** — an L with no associated P

Rates are reported as a fraction of **label components**, not of events, so they read as
"what fraction of real buildings are mis-instanced" — the quantity the kW estimate depends on.
Label blobs under 50 px (~3.5 m² at 26.6 cm) are ignored as specks. Full val set, 8-connected
components, threshold 0.5.

```bash
./run_docker.sh 7 "python -u scripts/merge_split_rate.py --ckpt <best.pth> --encoder <enc>"
```

## Results

Raw: [`diagnostics/merge_split/`](../../diagnostics/merge_split/) · **27,918** label components.

| | val IoU | **merge rate** | split rate | missed | pred/label |
|---|---|---|---|---|---|
| ResNet-34 | 0.6483 | **0.5040** | 0.0000 | 0.2364 | 0.789 |
| MiT-B2 | 0.6569 | **0.4615** | 0.0000 | 0.2423 | 0.760 |

**Predicted 20–35 %. Measured 46–50 %** — far worse than expected.

## Interpretation

**★ Half of all buildings are fused into a neighbour, by models whose IoU looks respectable.**
A model reporting 0.6483 IoU is merging **50.4 %** of the buildings it is supposed to count.
This is the single most important number produced in this project so far, and no metric used
before today could see it.

**The count is wrong, not just the shape.** `pred/label = 0.789` — the model outputs **21 %
fewer components than there are buildings**. Any per-building kW estimate built on this
under-counts by roughly a fifth before any other error is considered. That is a direct hit on
the project's stated goal, which is energy, not segmentation.

**Split rate is exactly 0.0 for both models.** The failure is purely under-segmentation, never
over-segmentation. That is consistent with D2's finding that the seed was trained where
buildings are rare and sparse: the model's prior is "few, large blobs", and Jaipur is "many,
small, touching". It also means any fix should push toward *separating* predictions, with no
need to guard against the opposite.

**R4's precision gain is now partly attributable.** MiT-B2 merges **4.3 points less**
(0.4615 vs 0.5040, an 8.4 % relative reduction) while missing marginally more. So the
transformer's global receptive field does separate adjacent buildings better — the
explanation offered in R4 holds, and could not have been confirmed with IoU alone. The effect
is real but modest; it does not come close to solving the problem.

## Decision

- [x] **Report merge/split rate alongside IoU from now on.** An IoU-only result table is
      misleading for this project.
- [x] R4's precision gain is attributed to separation, with a measured basis.
- [ ] ★ **The three-class relabel (building / boundary / background) is now clearly the
      priority**, not the architecture search. A 50 % merge rate is a bigger problem than
      0.009 of IoU.
- [ ] Re-examine the headline weak-supervision result: 0.6483 IoU reads far better than
      "half the buildings merged, 21 % under-counted" does.

## Threats to validity

- **Labels are footprints, and Open Buildings sometimes already merges two structures into
  one polygon.** Where it does, a genuine merge is invisible here — so 50 % is a **lower
  bound**.
- The 50 % overlap association rule is a choice. A stricter rule would raise the missed rate
  and lower the merge rate; a sweep would show sensitivity.
- Threshold fixed at 0.5. A lower threshold grows predictions and would likely *increase*
  merging.
- Components under 50 px excluded. At 26.6 cm the median Jaipur building is ~913 px (D3), so
  this removes specks rather than real buildings.
- Single seed per architecture.
