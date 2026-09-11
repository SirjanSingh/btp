# 2026-09-11-replication — settle the two claims sitting closest to the noise floor

| | |
|---|---|
| **Status** | running |
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

*pending*

## Threats to validity

- n=2 per configuration is still thin; it bounds rather than estimates. It is enough to move
  these from "one run vs one run" to a comparison of means, which is the actual gap.
- Agreement with Open Buildings, not accuracy (R12) — unchanged by any amount of replication.
