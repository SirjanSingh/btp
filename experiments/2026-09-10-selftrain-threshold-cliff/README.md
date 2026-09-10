# 2026-09-10-selftrain-threshold-cliff — where does self-training stop working?

| | |
|---|---|
| **Status** | ✅ done — cliff located; stated mechanism refuted |
| **Date** | 2026-09-10 |

## Question

S4 (pseudo-label threshold **0.4563**) reached **0.6165**, beating the source-only baseline.
S2 (**0.0004**) collapsed to **0.3752**. Somewhere between them the method flips from helping
to destroying. Two intermediate points: **0.0100** and **0.0015**.

**Why:** a Jaipur transfer has no labels, so the threshold must be chosen blind. Knowing
*where* the cliff is — and how sharp — determines how much margin to leave.

## ★ The cliff is in the confidence distribution, before any training

> **REFUTED by this experiment's own results — kept verbatim as the pre-registered
> reasoning.** The distribution claim may hold, but the performance consequence drawn
> from it does not: the 0.01–0.45 interval is a *plateau*, not a fall-through. See
> Interpretation. The conclusion (threshold safer than ratio) survives for a different
> reason.

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

Both arms trained 30 epochs, checkpoint selected on **source** val. Target numbers are
`ign_val`, reported at a fixed **0.5** operating point (the honest blind choice) with the
target-optimal threshold shown separately.

| arm | pseudo-label thr | ratio | source @0.5 | **target @0.5** | target P | target R | target best |
|---|---|---|---|---|---|---|---|
| S1 source-only | — | — | 0.8723 | **0.5611** | 0.741 | 0.698 | 0.5611 |
| S4 confidence | 0.4563 | 0.0059 | 0.8745 | **0.6135** | 0.819 | 0.710 | 0.6165 |
| **S6a** | **0.0100** | 0.008 | 0.8717 | **0.5904** | 0.695 | 0.797 | 0.5982 |
| **S6b** | **0.00155** | 0.012 | 0.8708 | **0.4884** | 0.525 | 0.875 | 0.5056 |
| S2 CBST | 0.0004 | 0.0183 | 0.8678 | **0.3576** | 0.371 | 0.908 | 0.3752 |

**Prediction scorecard — both missed, both in the same direction:**
S6a predicted 0.42–0.52 → **0.5904** ❌ high. S6b predicted 0.37–0.45 → **0.4884** ❌ high.
I over-estimated the damage in both arms. The pre-registered escape clause fires:
*"if 0.0100 lands near 0.6 instead, the cliff is sharper and further down than the
distribution suggests, and threshold choice is safer than I think."* It landed at 0.59.

## Interpretation

**There is no cliff. The section above this one is wrong about the mechanism.** I argued from
the bimodal probability distribution that the interval between ~0.01 and ~0.45 was empty, so
ratio-matching would "fall straight through to the noise floor". The opposite is true of
performance: that interval is a **plateau**. The threshold moves **45×** from 0.4563 to 0.0100
and target IoU falls only **0.023**. The collapse happens below 0.01, not within the gap.

**The conclusion survives; the reason changes.** Threshold is still the safe control variable
and ratio still the dangerous one — but because of plateau width, not distribution shape:

- **Threshold:** anything in **0.01 – 0.46** lands within 0.023 IoU of the best. A blind pick
  anywhere in that 45× range beats the source-only baseline.
- **Ratio:** performance falls roughly **linearly** in ratio past 0.008. Mis-estimating the
  target's foreground ratio by 2× (0.006 → 0.012) costs **0.125 IoU** and drops below baseline.

Ratio is the smoothly-behaved variable and threshold the wildly non-linear one — which is
exactly why threshold is safer to guess. You cannot land badly on a plateau.

**Where it stops helping.** Self-training beats source-only at ratio 0.008 (0.5904 > 0.5611)
and loses at 0.012 (0.4884). The crossover sits between them, i.e. at roughly **1.4–2× the
source foreground prior** — not at the prior itself, which is what CBST prescribes.

**The mechanism is a clean precision/recall trade, monotonic in both.** As the threshold falls,
target precision drops **0.819 → 0.695 → 0.525 → 0.371** and recall rises
**0.710 → 0.797 → 0.875 → 0.908**. Lower thresholds pseudo-label more true panels *and* far
more noise; past ratio ~0.008 the noise dominates. Nothing here is a discontinuity.

**The strongest result is the one the arms share.** Source IoU across all five arms spans
**0.0067** (0.8678–0.8745) while target IoU spans **0.256** (0.3576–0.6135) — a **38×**
difference in sensitivity. A run that destroys a third of its target performance is invisible
on source val. On Jaipur, where only the source-side number exists, every one of these arms
would look identically healthy.

## Decision

- [x] **Threshold, not ratio.** Prescribe a fixed confidence threshold for the Jaipur transfer.
- [x] **Use ~0.45**, the S4 value: it is the best measured point and sits mid-plateau, so it is
      also the most forgiving of a wrong guess.
- [x] **CBST ratio-matching is rejected for this project**, now with the crossover located
      rather than inferred from a single catastrophic point.
- [ ] S7 multi-round self-training should start from S4's threshold, not a ratio policy.

## Threats to validity

- Same C1 caveat: every BDAPPV crop contains a panel, so target precision is measured on a
  task with no true negatives and is optimistic in absolute terms. Relative ordering across
  the five arms is unaffected — they share the caveat exactly.
- **The crossover is bracketed between ratio 0.008 and 0.012, not located.** A point at 0.010
  would pin it; the plateau's lower edge (between thr 0.0100 and 0.00155) is likewise only
  bracketed.
- One round, one seed per arm. The S4-vs-S6a gap (0.6135 vs 0.5904) is small enough that seed
  noise could reorder them; the S6b and S2 drops are far too large to be noise.
- Target-optimal thresholds (0.8 for both S6 arms) are reported but **not used for the
  decision** — picking an operating point on target val is target leakage. All comparisons
  above are at a fixed 0.5.
- `ign_val` is a sensor shift within France. Jaipur is a much larger shift, so the plateau
  width measured here is an upper bound on what to expect there.

---

## Follow-up arm: ratio 0.010 (launched 2026-09-10 ~19:55)

The threats section above says the crossover is *bracketed* between ratio 0.008 (0.5904, above
the 0.5611 baseline) and 0.012 (0.4884, below it), not located. This arm splits the bracket at
**ratio 0.010 → threshold 0.0031**.

**Prediction (before running): target IoU @0.5 of 0.53 – 0.57**, i.e. slightly *below* the
0.5611 baseline. Linear interpolation between the two bracketing arms gives 0.539; the decline
looked close to linear in ratio, so a big departure from that would mean the crossover is a
knee rather than a slope. Precision ~0.60–0.65, recall ~0.83–0.85 by the same interpolation.

If it lands **above 0.5611**, the usable ratio range extends further than the two-point
bracket suggested and CBST is less dangerous than S6 concluded — the honest test of that
conclusion.

### Result — ratio 0.010

**Target @0.5 = 0.5377** (best 0.5523 @thr 0.8), P **0.622**, R **0.832**; source 0.8727.

**Prediction hit on all three counts** — 0.53–0.57 → 0.5377; P 0.60–0.65 → 0.622;
R 0.83–0.85 → 0.832. Linear interpolation had said 0.539 and the arm returned 0.5377.

**The decline is linear in ratio, with a knee at ~0.008.** Slope per unit ratio:

| interval | ΔIoU / Δratio |
|---|---|
| 0.0059 → 0.008 | −11.0 |
| 0.008 → 0.010 | −26.4 |
| 0.010 → 0.012 | −24.7 |
| 0.012 → 0.0183 | −20.8 |

Flat-ish below 0.008, then a steady ~−25 per unit. Full curve, target @0.5:

| ratio | 0.0059 | 0.008 | **0.010** | 0.012 | 0.0183 |
|---|---|---|---|---|---|
| target IoU | 0.6135 | 0.5904 | **0.5377** | 0.4884 | 0.3576 |
| precision | 0.819 | 0.695 | **0.622** | 0.525 | 0.371 |
| recall | 0.710 | 0.797 | **0.832** | 0.875 | 0.908 |

**Crossover located: ratio ≈ 0.0091.** Between 0.008 (0.5904, above the 0.5611 baseline) and
0.010 (0.5377, below it); interpolating on the measured −26.4 slope gives **0.0091**. The
earlier 0.008–0.012 bracket is now 0.008–0.010.

**The headline this buys.** The source foreground prior is **0.0183** — and CBST prescribes
matching it. That is **2× past the ratio where self-training stops helping at all**. The
prescribed setting is not merely suboptimal; it sits at double the break-even point, which is
why S2 landed at 0.3576 rather than somewhere near baseline.

