# 2026-09-10-solar-selftrain-conf — is it CBST's ratio policy, or self-training itself?

| | |
|---|---|
| **Status** | ✅ done — **self-training works; CBST was the problem** |
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

Raw: [`diagnostics/s4_crossdomain.json`](../../diagnostics/s4_crossdomain.json)

| | pseudo-label threshold | source | **target (ign_val)** | precision | recall |
|---|---|---|---|---|---|
| S1 source-only | — | 0.8723 | 0.5611 | 0.741 | 0.698 |
| S2 CBST ratio-matched | **0.0004** | 0.8678 | **0.3752** | 0.391 | 0.902 |
| **S4 confidence** | **0.4563** | 0.8747 | **0.6165** | **0.789** | **0.738** |

**+0.0554 over the source-only baseline — 17.8 % of the 31-point domain gap closed in a
single round.** Predicted 0.50–0.58, "recovering most of the loss but at or slightly below
baseline". **Wrong, in the good direction:** it landed above the band and above the baseline.

## Interpretation

**★ The answer is unambiguous: the failure was CBST's ratio policy, not self-training.**
Same pipeline, same data, same number of rounds — only the threshold policy changed, and the
target score moved from **0.3752 to 0.6165**. Self-training on this domain gap works. The
prescribed "fix" for foreground collapse was the entire cause of the collapse.

**Both precision and recall improved over the source-only model** — 0.741 → 0.789 and
0.698 → 0.738. That is a real gain, not a trade: the model became better at the target
without giving anything back. S2, by contrast, bought recall (0.902) by destroying precision
(0.391).

**Why ratio-matching fails here and the literature does not say so.** CBST assumes the target
confidence distribution is broadly comparable to the source's, so demanding the source's
class ratio picks out a sensible operating point. On a 31-point domain gap that assumption
breaks: the target distribution is crushed toward zero, so the ratio can only be satisfied by
dropping the threshold into the noise floor (**0.0004**). The method has a precondition that
nobody states — **it is safe only while the model retains calibrated confidence on the
target**, and that is exactly what a large domain gap destroys.

**The threshold evidence predicted both outcomes before either ran.** Recorded in S2 before
launch: "at 0.0004 nearly any activation counts as a panel". That single measurement — taken
in seconds, before any GPU time — correctly forecast S2's collapse *and* motivated S4.

## Decision

- [x] **Adopt confidence-threshold self-training. Drop CBST ratio-matching.**
- [x] `plan/03` Tier 3 and `MASTER_CONTEXT` both need a correction: CBST is not the safe
      default they present it as, and has a precondition neither states.
- [ ] **Now worth trying on Jaipur** — with a confidence threshold, not a class ratio. This is
      the first UDA method with measured evidence behind it rather than a citation.
- [ ] Run 2–3 rounds; CBST literature reports gains compound, and this was one round.
- [ ] Sweep the threshold between 0.4563 and 0.0004 to find where the cliff is — the boundary
      between working and catastrophic is currently unmapped.

## Threats to validity

- ⚠ **C1 unfixed** — every BDAPPV crop contains a panel, so precision is not a deployment
  number. Both arms share the defect, so the comparison holds.
- **The target's best threshold moved to 0.1** (from 0.5 for S1), so the improved model is now
  *under*-confident on IGN. At a fixed 0.5 it still scores 0.6135, above baseline, so the gain
  is not a threshold artifact — but it does mean the model is differently miscalibrated.
- One round, one seed, one threshold value.
- Selected on source val (correct), so the target number is honest but unoptimised.

## Threats to validity

- Same C1 caveat: every BDAPPV crop contains a panel, so precision is not a deployment number.
- Single round, as in S2.
- 0.59 % is itself a choice — it is the model's own confident fraction, not a principled
  target. A sweep between 0.0059 and 0.0183 would locate the boundary.
