# 2026-09-10-eroded-labels — can shrinking the labels stop the model fusing buildings?

| | |
|---|---|
| **Status** | running |
| **Date** | 2026-09-10 |

## Question

Shrink every Open Buildings footprint by **0.4 m** before rasterising, so that touching
buildings get a visible gap, and train on that. Does the **merge rate** fall?

**Why:** [R8](../2026-09-10-merge-split-rate/) measured a **50.4 % merge rate with a 0.0 %
split rate** — the model fuses neighbours and never over-segments. [D4](../2026-09-10-d4-adjacency/)
explains it: 78 % of Jaipur buildings have a neighbour within half a metre, under two pixels,
so frequently **there is no gap in the label for the model to learn**. The cheapest possible
intervention is to put one there.

This matters more than IoU. The pipeline ends in a **per-building kW estimate**, and R8 found
the model emits 21 % fewer components than there are buildings. Fixing counts is worth more
than fixing pixels.

**Predictions (before running):**

- **Merge rate 0.30–0.40**, down from 0.5040. This is the number the experiment lives or
  dies by.
- **IoU 0.60–0.64**, i.e. slightly *worse* than 0.6483 — predictions will be systematically
  smaller than the un-eroded val targets. An IoU drop is an acceptable price and is expected.
- **Split rate rises above 0**, possibly to a few percent. If erosion overshoots it will
  start cutting single buildings in two.
- Erosion should help **ResNet-34 more than MiT-B2**, since MiT already merges 4.3 points
  less on its own.

## Setup

A clean **2 × 2**: {un-eroded, eroded 0.4 m} × {ResNet-34, MiT-B2}. The two un-eroded arms are
already measured, so this run adds the other two.

| | |
|---|---|
| Train masks | `data/jaipur_weak_erode/train` — 318,207 polygons eroded 0.4 m; only **6** vanished; mean fg **20.17 %** (vs 23.06 %) |
| Val | `data/jaipur_weak/val` — **un-eroded, unchanged** |
| Init | ImageNet |

**Evaluation design, learned from [R3](../2026-09-10-label-quantity-vs-quality/):** the eroded
masks are used for **training only**. Validation stays on the original masks so IoU remains
comparable with every other run in the repo. Changing both training and evaluation targets is
exactly the confound that made R3 unresolvable.

```bash
./run_docker.sh 7 "python -u rooftop/train.py --train_dir data/jaipur_weak_erode/train \
    --val_dir data/jaipur_weak/val --epochs 40 --batch_size 16 --workers 3 --save_every 999 ..."
./run_docker.sh 2 "... --encoder mit_b2 --batch_size 12 ..."
```

## Results

### MiT-B2 arm (complete; early-stopped epoch 34, best @19)

| | un-eroded | **eroded 0.4 m** | delta |
|---|---|---|---|
| val IoU | 0.6569 | 0.6405 | −0.0164 |
| **merge rate** | 0.4615 | **0.3256** | **−0.1359 (−29 % relative)** |
| split rate | 0.0000 | 0.0000 | 0 |
| missed rate | 0.2423 | 0.3223 | +0.0800 |
| **pred / label** | 0.760 | **0.9745** | **+0.215** |

ResNet-34 arm still running.

**Predictions vs measured:** merge rate 0.30–0.40 → **0.3256** ✅ · IoU 0.60–0.64 → 0.6405
(slightly better than predicted) · split rate rises above 0 → **it did not, still exactly
0.0** ❌.

## Interpretation

**★ The count is very nearly fixed.** `pred/label` went from **0.760 to 0.9745** — the model
now emits about one component per real building instead of three for every four.
Under-counting fell from **24 % to 2.6 %**. Since the pipeline ends in a per-building kW
estimate, this is the single most consequential improvement the project has produced, and it
came from a one-line change to label generation rather than any modelling work.

**Merging fell 29 % relative** (0.4615 → 0.3256) for **1.6 points of IoU**. Judged on IoU
alone this experiment looks like a small regression; judged on what the project actually
needs, it is the largest win so far. That is precisely why R8 had to exist before this could
be run.

**The cost is misses, not false splits.** Missed rate rose 8 points (0.242 → 0.322): eroded
labels teach the model to be conservative, so small or faint buildings drop below its
threshold entirely. Split rate stayed at **exactly 0.0**, contradicting my prediction — 0.4 m
does not overshoot into cutting single buildings apart. That headroom suggests **more erosion
is worth trying**, since the failure mode I feared has not appeared at all.

**The natural next step is inference-time dilation.** Predictions are systematically shrunk by
construction; dilating them back by ~0.4 m should recover much of the missed rate and IoU
while keeping the instances separate. It is not implemented, so every number here understates
the approach.

## Decision

- [x] **Erosion works for its intended purpose.** Adopt for count-sensitive outputs.
- [ ] **Sweep erosion 0.2 / 0.4 / 0.8 m** — split rate is still 0, so the upper bound has not
      been found.
- [ ] **Implement inference-time dilation** before treating the IoU/missed cost as real.
- [ ] Report `pred/label` in every future result table; it tracks the project's actual goal
      more directly than IoU does.

## Threats to validity

- 0.4 m (~1.5 px) is one guess at the erosion amount. Too little does nothing; too much cuts
  buildings apart. A sweep (0.2 / 0.4 / 0.8 m) is the honest follow-up if this helps.
- Erosion shrinks predictions, so IoU against un-eroded targets is **biased against this
  method by construction**. Judge it on merge rate; treat the IoU as a cost, not a verdict.
- A real deployment would dilate predictions back afterwards; that step is not implemented,
  so these numbers understate what the approach could deliver.
- Merge rate is itself a lower bound — Open Buildings sometimes already merges two structures
  into one polygon (R8 threats).
