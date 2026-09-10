# 2026-09-10-solar-selftrain-conf — is it CBST's ratio policy, or self-training itself?

| | |
|---|---|
| **Status** | running |
| **Date** | 2026-09-10 |

## Question

Repeat [S2](../2026-09-10-solar-self-training/) with a **fixed confidence threshold** instead
of class-ratio matching, and see whether the collapse recovers.

**Why:** S2 cost **18.6 points** of target IoU (0.5611 → 0.3752) with precision falling
0.741 → 0.391. The suspected cause is CBST's ratio policy: forcing the source's 1.83 %
foreground on a target whose confidence distribution is crushed drove the threshold to
**0.0004**, so pseudo-labels were largely noise.

This arm selects **0.59 %** of pixels — what the model's own 0.5 threshold picks — which puts
the cut at **0.4563** instead of 0.0004. Same pipeline, same data, one variable changed.

**It decides between two very different conclusions:**
- **Recovers** → the failure is *CBST's ratio policy specifically*, and plain confidence
  self-training remains viable. The plan needs a caveat, not a rewrite.
- **Also degrades** → *self-training itself* is unsafe at this gap width, and the whole Tier-3
  branch of `plan/03` needs re-thinking before anyone applies it to Jaipur.

**Prediction:** **0.50–0.58** — recovering most of the loss but landing at or slightly below
the 0.5611 source-only baseline. Reasoning: a sane threshold stops the noise-labelling, but
self-training can still only reinforce what the model already believes, and the source-only
model is wrong about 44 % of the target. Precision should recover to ~0.65–0.75.

## Setup

Identical to S2 except `--class_ratio 0.0059` (→ threshold 0.4563) rather than 0.0183
(→ 0.0004). Train on google-real + ign-pseudo, select on source val, report on `ign_val`.
IGN labels never read during training.

## Results

*pending*

## Threats to validity

- Same C1 caveat: every BDAPPV crop contains a panel, so precision is not a deployment number.
- Single round, as in S2.
- 0.59 % is itself a choice — it is the model's own confident fraction, not a principled
  target. A sweep between 0.0059 and 0.0183 would locate the boundary.
