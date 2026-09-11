# 2026-09-11-seed-variance — how big is the noise floor everything is being compared against?

| | |
|---|---|
| **Status** | running |
| **Date** | 2026-09-11 |

## Question

Across ~20 experiments this project has compared single runs and repeatedly said things like
*"+0.0039 is inside seed noise"* or *"the S6b drop is far too large to be noise"*. **Not one of
those statements was measured.** No configuration in this repo has ever been run twice.

Re-run the exact default config (MiT-B2, 0.4 m eroded labels, 40 epochs) with `--seed 43`
instead of 42. The difference between the two runs *is* the noise floor.

**Why it matters:** it retroactively determines which of tonight's conclusions are real. Three
in particular hinge on it:

- R5 arm B beat the teacher by **+0.0039 IoU** — I dismissed it as noise.
- R5b E02 vs E04 differ by **0.023 `pred/label`** — I treated that as a real ordering.
- R4 chose MiT-B2 over ResNet-34 on **+0.0086 IoU** — a live default rests on it.

If seed spread turns out to be ±0.01 IoU, the first two readings are unsupportable and R4's
margin is inside the noise. If it is ±0.002, all three stand.

## Predictions (before running)

**IoU spread |Δ| = 0.003–0.008.** Segmentation runs on ~7k crops with a fixed schedule are
usually reproducible to a few thousandths, but the best-epoch selection over 40 noisy epochs
adds variance of its own.

**Instance metrics will vary more than IoU.** `pred/label` and merge depend on connected
components, and a small probability shift near the decision boundary can split or join a blob
outright. I expect `pred/label` spread of **0.01–0.03** — i.e. *larger* than several
differences I have already interpreted as meaningful.

**The uncomfortable prediction:** I expect this to invalidate at least one claim I made
tonight. E02-vs-E04's 0.023 `pred/label` gap is the most exposed.

## Setup

Identical to the default: `--train_dir data/jaipur_weak_erode4/train`-equivalent (the same
0.4 m eroded set the teacher used), MiT-B2, 40 epochs, batch 12, `--seed 43`. Scored the same
way — val IoU plus the full instance metric set at threshold 0.5.

## Results

*pending*

## Threats to validity

- **Two runs give a range, not a standard deviation.** This bounds the noise floor loosely;
  it does not estimate it properly. Three or more seeds would, and are worth queuing if the
  spread turns out to matter.
- Seed controls init, shuffling and augmentation, but cuDNN autotuning is not deterministic
  either, so this is total run-to-run variance rather than seed variance specifically — which
  is the quantity actually wanted here.
