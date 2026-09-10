# 2026-09-10-selftrain-threshold-cliff — where does self-training stop working?

| | |
|---|---|
| **Status** | running |
| **Date** | 2026-09-10 |

## Question

S4 (pseudo-label threshold **0.4563**) reached **0.6165**, beating the source-only baseline.
S2 (**0.0004**) collapsed to **0.3752**. Somewhere between them the method flips from helping
to destroying. Two intermediate points: **0.0100** and **0.0015**.

**Why:** a Jaipur transfer has no labels, so the threshold must be chosen blind. Knowing
*where* the cliff is — and how sharp — determines how much margin to leave.

## ★ The cliff is in the confidence distribution, before any training

Choosing the target ratio and reading off the threshold it implies:

| target foreground ratio | implied threshold |
|---|---|
| 0.0059 (S4) | **0.4563** |
| 0.008 | **0.0100** |
| 0.012 | 0.0015 |
| 0.0183 (S2, = source prior) | 0.0004 |

**A 1.36× change in ratio (0.0059 → 0.008) drops the threshold 45×.** That is not a smooth
knob — the model's confidence distribution on IGN is **bimodal**: a small confident mass, then
almost nothing between ~0.01 and ~0.45. Ratio-matching lands in the empty middle and falls
straight through to the noise floor.

This explains CBST's failure mechanically rather than just empirically. On a well-calibrated
target the ratio→threshold map is gentle; here it is a cliff, so **ratio is a dangerous
control variable and threshold is a safe one.**

**Predictions:** thr **0.0100** → **0.42–0.52** (already deep in the noise floor, so most of
the damage should already be done); thr **0.0015** → **0.37–0.45** (essentially S2). If
0.0100 lands near 0.6 instead, the cliff is sharper and further down than the distribution
suggests, and threshold choice is safer than I think.

## Setup

Identical to S4 except `--class_ratio`. Train google-real + ign-pseudo, select on source val,
report on `ign_val`. Both arms use the new `--cache_ram` on `train_solar.py` (S5), which
removes the ~340 s/epoch dataloader bound.

## Results

*pending*

## Threats to validity

- Same C1 caveat: every BDAPPV crop contains a panel.
- Two intermediate points only; the cliff is bracketed, not located.
- One round, one seed per arm.
