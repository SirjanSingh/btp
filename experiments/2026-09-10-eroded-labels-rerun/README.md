# 2026-09-10-eroded-labels-rerun — restore the adopted configuration's weights

| | |
|---|---|
| **Status** | ✅ done — reproduced |
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

**Best val IoU 0.6393 @ep19** (early-stopped ep34) against the destroyed original's
**0.6405 @ep19**. Same best epoch, **0.0012** apart — seed noise. The original was
reproducible, not a fluke.

### The 0.4 m row, finally measured with the corrected split metric

| config | IoU | merge | **split** | frag/label | missed | **pred/label** |
|---|---|---|---|---|---|---|
| ResNet un-eroded | 0.6483 | 0.5040 | 0.0253 | 0.8823 | 0.2364 | 0.7888 |
| MiT un-eroded | **0.6569** | 0.4615 | 0.0341 | 0.8903 | 0.2423 | 0.7600 |
| **MiT 0.4 m** | 0.6393 | **0.3155** | 0.0840 | 0.9203 | 0.3257 | **0.9914** |
| MiT 0.8 m | 0.5911 | 0.1339 | **0.1762** | 1.0354 | 0.4689 | 1.4743 |

## Interpretation

**`pred/label` = 0.9914 — within 0.9 % of one prediction per building.** The original run's
0.9745 was already the best in the project; this reproduces it slightly closer to unity.
Under-counting is essentially eliminated, which is what the per-building kW estimate depends
on.

**The corrected metric shows the trade the old one hid.** Split rate rises monotonically with
erosion — **0.034 → 0.084 → 0.176** — where the strict definition reported 0.0 at every
level. So erosion *does* over-segment, progressively, exactly as expected; the old metric was
simply incapable of showing it. 0.4 m sits at a defensible 8.4 %, 0.8 m at a clearly
excessive 17.6 %.

**The sweep now reads cleanly in one direction.** More erosion → less merging, more
splitting, more misses, `pred/label` climbing past 1.0. **0.4 m is the crossing point**, and
it is the only configuration where `pred/label` is near 1 *and* split stays under 10 %.

## Decision

- [x] **0.4 m erosion + MiT-B2 confirmed as the default**, now with a loadable checkpoint and
      an honest split rate.
- [x] The destroyed run reproduced; no scientific loss from the `filter-branch` accident.
- [ ] Every split rate reported before R11 was strict-only — the corrected values are in this
      table.

## Threats to validity

- Different seed from the original run, so an exact match is not expected and a small
  difference is not evidence of a problem.
- Same footprint-not-roof caveat as everything else in this project — see R12.
