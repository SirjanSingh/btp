# 2026-09-10-eroded-labels-rerun — restore the adopted configuration's weights

| | |
|---|---|
| **Status** | running |
| **Date** | 2026-09-10 |

## Question

Retrain **MiT-B2 on 0.4 m eroded labels** — the configuration adopted as the project default
— because `git filter-branch` deleted its checkpoint from disk (`docs/PITFALLS.md` §3.14).

**Why it needs redoing rather than shrugging off:** the original run's *metrics* survived in
`RUN_LEDGER` (best IoU **0.6405** @ep19), so nothing scientific was lost. But two things need
the weights themselves:

1. **The corrected split-rate metric** (R11) cannot be applied to a checkpoint that no longer
   exists, so the 0.4 m row of the erosion table is stuck on the old, blind definition.
2. It is the **default configuration** — the one any later experiment would build on. A
   project cannot adopt a config it cannot load.

**Prediction:** should reproduce **0.6395–0.6415** — the same recipe, same data, differing
only by seed noise. If it lands outside that, something is non-deterministic that I have not
accounted for, which would itself be worth knowing. Expect merge ≈ 0.33, `pred/label` ≈ 0.97
as before, plus a split rate that is now *visible* (the old run reported 0.0 under the strict
definition; the true value is probably a few percent, as the un-eroded arms turned out to be).

## Setup

Identical to the [original](../2026-09-10-eroded-labels/) MiT-B2 arm.

```bash
./run_docker.sh 3 "python -u rooftop/train.py \
    --train_dir data/jaipur_weak_erode/train --val_dir data/jaipur_weak/val \
    --arch unet --encoder mit_b2 --epochs 40 --batch_size 12 --workers 3 --save_every 999 ..."
```

## Results

*pending*

## Threats to validity

- Different seed from the original run, so an exact match is not expected and a small
  difference is not evidence of a problem.
- Same footprint-not-roof caveat as everything else in this project — see R12.
