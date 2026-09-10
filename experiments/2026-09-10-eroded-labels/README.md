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

*pending — IoU, then merge/split rate via `scripts/merge_split_rate.py`*

## Threats to validity

- 0.4 m (~1.5 px) is one guess at the erosion amount. Too little does nothing; too much cuts
  buildings apart. A sweep (0.2 / 0.4 / 0.8 m) is the honest follow-up if this helps.
- Erosion shrinks predictions, so IoU against un-eroded targets is **biased against this
  method by construction**. Judge it on merge rate; treat the IoU as a cost, not a verdict.
- A real deployment would dilate predictions back afterwards; that step is not implemented,
  so these numbers understate what the approach could deliver.
- Merge rate is itself a lower bound — Open Buildings sometimes already merges two structures
  into one polygon (R8 threats).
