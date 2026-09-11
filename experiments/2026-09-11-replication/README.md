# 2026-09-11-replication — settle the two claims sitting closest to the noise floor

| | |
|---|---|
| **Status** | claim 2 done · claim 1 running |
| **Date** | 2026-09-11 |

## Question

The seed-variance run put a 2 sd floor of **0.0763 `pred/label`** and **0.0502 merge** on this
project's instance metrics. Two live conclusions sit close to that line and are worth more than
a caveat:

1. **Erosion is necessary** (the central methodological claim). Its matched-operating-point
   evidence is 1.2–1.9× 2 sd at individual points, carried mainly by *three points moving the
   same direction*. It compares a 3-seed eroded distribution against a **single** un-eroded run.
2. **Self-training regresses counting** (R5 arm B, `pred/label` 0.9134 vs 0.9914). At 1.0× 2 sd
   this is exactly marginal. Its direction is corroborated by R5b's eroded arms moving it back,
   but the primary comparison is one run against one run.

Both are fixed the same way: give each a second seed, so a distribution is compared against a
distribution.

**Why these two and not something new:** every remaining backlog item is blocked (R12 on hand
labels, S3 on absent data) or low-value (R7, since the AIRS seed was shown worthless). Turning
the project's central claim from "probably" into "measured" is worth more than another arm.

## Predictions (before running)

**Un-eroded seed 43: IoU 0.652–0.662, `pred/label` 0.72–0.79.** The seed-42 run gave 0.6580 and
0.7507; B2's IoU 2 sd is 0.0042 and `pred/label` 2 sd is 0.0763, so these are just
seed-42 ± 2 sd.

**The erosion conclusion will survive and strengthen.** The eroded/un-eroded `pred/label` gap at
matched merge is 0.093–0.147. Even if the un-eroded model's own spread matches B2's, the gap at
merge 0.3155 (0.1468) should stay ≥ 1.5× a pooled 2 sd.

**R5 arm B seed 43: `pred/label` 0.88–0.99.** Seed 42 gave 0.9134. **This is the one I expect to
go badly** — the interval straddles the teacher's 0.9753 mean, so a second seed could easily
land where the regression disappears. If the two arm-B seeds average above ~0.95, **the R5
headline is not supported** and the paper claim has to become "self-training did not help"
rather than "self-training regresses counting".

**Honest position:** I expect claim 1 to firm up and claim 2 to weaken. R5's *decision*
(reject self-training) does not depend on claim 2 — no arm ever beat the teacher — but the
stated mechanism does.

## Setup

Two arms, both `--seed 43`, otherwise byte-identical to the originals:

- **un-eroded**: `data/jaipur_weak/train`, MiT-B2, 40 ep, batch 12
- **selftrain-t050**: `data/jaipur_pseudo_t050/train`, MiT-B2, 40 ep, batch 12

Scored on the same val tiles with IoU plus the full instance set at threshold 0.5.

## Results

### Claim 2 — self-training regresses counting: **direction survives, precision does not**

| run | IoU | merge | missed | **pred/label** |
|---|---|---|---|---|
| teacher seed 42 / 43 / 44 | 0.6393 / 0.6423 / 0.6434 | — | — | 0.9914 / 0.9317 / 1.0027 |
| **teacher mean (n=3)** | 0.6417 | 0.3378 | 0.3125 | **0.9753** (sd 0.0382) |
| self-train t050 seed 42 | 0.6432 | 0.3371 | 0.3156 | 0.9134 |
| **self-train t050 seed 43** | 0.6442 | 0.4501 | 0.2680 | **0.7702** |
| **self-train mean (n=2)** | 0.6437 | 0.3936 | 0.2918 | **0.8418** (sd 0.1013) |

**Prediction missed, and in the direction opposite to my worry.** I predicted seed 43 would land
0.88–0.99 and feared the regression would evaporate. It came in at **0.7702** — below the range,
making the regression *larger*, not smaller.

**Difference of means 0.1335, SE 0.0749, d/SE = 1.78.** Suggestive, not conclusive at n=3 vs n=2.

**The more interesting result: self-training is 2.7× noisier.** The t050 arm's `pred/label` sd is
**0.1013** against the teacher's **0.0382**, from a 0.1432 range across just two seeds. Both arms
trained on the *same fixed* pseudo-label set — generated once from the seed-42 teacher — so this
is not pseudo-label variability. **Training a student on noisy pseudo-labels amplifies its
sensitivity to initialisation.** Worth stating carefully at n=2, but the gap is large.

**Revised claim for R5.** "Self-training regresses `pred/label` from 0.9914 to 0.9134" was a
single-run comparison and overstated its precision. The supportable version is: **self-training
lowers counting accuracy (0.9753 → 0.8418 across seeds, d/SE 1.78) and makes it markedly less
reproducible.** R5's *decision* — reject self-training — was never in doubt, since no arm beat
the teacher on any seed.

### An error in every write-up, found while collecting this

`rooftop/train.py` defaults to **`--patience 15`**: training halts 15 epochs after the best
score. Every experiment in this repo describes itself as *"40 epochs"*; the true figure is
**up to 40, early-stopped**. Actual stopping points vary widely — this arm stopped at **epoch 23
with its best at epoch 8**, while the un-eroded run went to 40 with its best at 35.

This matters twice: the write-ups are inaccurate as stated, and **variable stopping is itself a
contributor to the seed spread** measured yesterday — a run that peaks early trains for
substantially less time. The seed-variance figures remain valid as a *total* run-to-run floor,
which is the quantity comparisons need, but the mechanism is not purely initialisation.

## Results — claim 1 (erosion)

*pending — un-eroded seed 43 still training*

## Threats to validity

- n=2 per configuration is still thin; it bounds rather than estimates. It is enough to move
  these from "one run vs one run" to a comparison of means, which is the actual gap.
- Agreement with Open Buildings, not accuracy (R12) — unchanged by any amount of replication.
