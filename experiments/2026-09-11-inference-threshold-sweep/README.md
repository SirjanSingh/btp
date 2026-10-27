# 2026-09-11-inference-threshold-sweep — can merging be fixed at inference, for free?

| | |
|---|---|
| **Status** | ✅ done — complements, not substitutes; follow-up launched |
| **Date** | 2026-09-11 |

> ⚠️ **Noise-floor caveat (added 2026-09-11).** A same-config seed replicate measured
> run-to-run spread of **0.0597 `pred/label`** and **0.0495 merge** — see
> [`../2026-09-11-seed-variance/`](../2026-09-11-seed-variance/). Differences below
> that scale in this write-up are **not supported by a single pair of runs**.
> Specifically: the E04-curve-vs-teacher-curve gap (~0.07) is 1.2x the floor. The
> eroded-vs-un-eroded comparison is stronger (1.5-2.5x) and is further supported by three
> matched points all moving the same way, which a pairwise noise estimate does not capture.


## Question

Every instance metric in this repo is computed at a **fixed inference threshold of 0.5**, and
that choice has never been examined. Label erosion fixes merging by shrinking the *training
target*, which costs recall and a retrain. Raising the *inference* threshold shrinks predicted
blobs directly — it should separate touching buildings by the same geometry, but it costs
nothing, requires no retraining, and is reversible per-application.

**Why it matters:** the teacher's merge rate is **0.3155** — a third of buildings still fused,
and after tonight's work that is the largest remaining weakness in Stage 1. If a threshold
change buys a meaningful part of what erosion buys, it is strictly cheaper. If it does not,
that is itself informative: it would mean fusion is happening in the model's *confident*
interior rather than at soft edges, which erosion can fix and thresholding cannot.

Nothing in the repo distinguishes those two cases today.

## Predictions (before running)

Sweeping the teacher (MiT-B2, 0.4 m eroded labels) over thresholds 0.3 → 0.9:

1. **Merge rate falls monotonically with threshold**, because higher thresholds erode the
   predicted mask at its soft boundary — exactly where two adjacent buildings touch.
2. **Missed rate rises monotonically**, the same recall cost erosion pays.
3. **`pred/label` rises through 1.0 and overshoots**, mirroring the erosion sweep
   (0.4 m → 0.9914, 0.8 m → 1.4743). I expect the crossing near **0.6–0.7**.
4. **The trade will be *worse* than erosion's.** Erosion shrinks the target at *training* time,
   so the model learns to place gaps; thresholding only shaves probability mass at test time
   and cannot invent a gap the model did not represent. Concretely I expect that at the
   threshold where merge reaches erosion's 0.2666 (E04's value), the missed rate will be
   **higher** than E04's 0.3974.

**If prediction 4 is wrong** — if thresholding reaches merge 0.2666 at a missed rate below
0.3974 — then the whole erosion workstream is partly redundant with a one-line inference change,
which would be the most consequential finding of the night and would need saying plainly.

**Operating-point honesty:** the teacher was *selected* at 0.5 on this val set, so comparing
other thresholds on the same set mildly favours 0.5. The effect is small relative to the
metric swings expected here, and is noted rather than corrected.

## Setup

`scripts/merge_split_rate.py --threshold {0.3,0.4,0.5,0.6,0.7,0.8,0.9}` on the teacher
checkpoint, same 1,701-crop held-out val tiles, no retraining. Pure inference; ~2 min per point.

## Results

Teacher (MiT-B2, 0.4 m eroded labels), inference only, no retraining. The 0.5 row is the
already-published operating point.

| thr | merge | split | missed | **pred/label** |
|---|---|---|---|---|
| 0.3 | 0.4514 | 0.0465 | 0.2402 | 0.8463 |
| 0.4 | 0.3851 | 0.0640 | 0.2804 | 0.9157 |
| **0.5** | **0.3155** | 0.0840 | **0.3257** | **0.9914** |
| 0.6 | 0.2401 | 0.1086 | 0.3857 | 1.0818 |
| 0.7 | 0.1616 | 0.1347 | 0.4627 | 1.1941 |
| 0.8 | 0.0838 | 0.1572 | 0.5771 | 1.3664 |
| 0.9 | 0.0182 | 0.1469 | 0.7629 | 1.5456 |

**Prediction scorecard: 1 ✅, 2 ✅, 3 ❌, 4 ❌.**

- Merge falls monotonically ✅; missed rises monotonically ✅.
- `pred/label` crosses 1.0 at **≈0.51**, not the predicted 0.6–0.7 ❌. The default 0.5 is
  almost exactly the counting-optimal threshold — that is luck, not design, since 0.5 was never
  chosen for this.
- **Prediction 4 was wrong.** I expected thresholding to trace a *worse* merge/missed trade than
  erosion. Matched at equal merge rate, the two curves are nearly the same:

| merge rate | erosion → missed | threshold → missed |
|---|---|---|
| 0.4118 | 0.2688 (0.2 m) | **0.2642** |
| 0.3155 | 0.3257 (0.4 m) | 0.3257 *(same point)* |
| 0.1339 | **0.4689** (0.8 m) | 0.5034 |

## Interpretation

**The merge/missed trade is a property of the probability field, not of the training target.**
Erosion does not teach the model to place gaps between buildings in some way thresholding
cannot reach — it largely moves where 0.5 sits on a curve the model already has. That is a
genuinely deflationary result about the erosion workstream, and it needs stating plainly.

**But erosion is not redundant, and the evidence is in the third metric.** Compare the two
models at the *same* merge and missed rate:

| | merge | missed | **pred/label** |
|---|---|---|---|
| un-eroded model @ thr 0.5 | 0.4615 | 0.2423 | **0.7600** |
| **0.4 m eroded model @ thr 0.3** | 0.4514 | 0.2402 | **0.8463** |

Merge and missed match to within 0.01 — and `pred/label` differs by **0.086**. At an identical
fusion/recall operating point the eroded model **counts buildings substantially better**. So
erosion buys something thresholding cannot: it changes how the model *partitions* foreground
into instances, not just how much foreground it emits.

**Practical consequence: they are complements, not substitutes.** Threshold is a free,
per-application dial along the fusion/recall curve; erosion shifts the whole curve to a better
`pred/label`. Anyone deploying this should tune the threshold to the application (a
conservative rooftop-area estimate wants a different point than a building count) and keep the
erosion.

**A caveat that limits all of the above.** This compares a *curve* (thresholds on the eroded
model) against a *single point* (the un-eroded model at 0.5), because the un-eroded checkpoint
was destroyed by `filter-branch` (PITFALLS 3.14) and only its 0.5 row survives in the ledger.
One point cannot establish that the un-eroded model's whole curve sits below. **Retraining the
un-eroded model to sweep it properly is launched as the follow-up** — that is the experiment
that would actually settle whether erosion is necessary.

## Decision

- [x] **Keep reporting at 0.5** — it is within 0.01 of the `pred/label`-optimal threshold, so
      the existing numbers need no revision.
- [x] **Report the threshold sweep alongside single-point metrics from now on.** A single
      operating point hides that merge can be halved for a known recall cost.
- [x] **Erosion stays**, now defended by a matched-operating-point comparison rather than by
      the fixed-threshold comparison that motivated it.
- [ ] **Un-eroded model retrain launched** to replace the destroyed checkpoint and sweep its
      full curve. Until it lands, "erosion improves `pred/label` at matched merge" rests on one
      point and is stated as provisional.

## Threats to validity

- Agreement with Open Buildings, not accuracy (R12).
- The teacher was selected at threshold 0.5 on this val set, which mildly favours the 0.5 row.
- Single global threshold; a density-adaptive one is untested.
- The matched-operating-point comparison uses one surviving row for the un-eroded model.

## Threats to validity

- Agreement with Open Buildings, not accuracy (R12).
- A single global threshold; a per-tile or density-adaptive threshold might do better and is
  not tested.
- Merge/split are computed on connected components at the swept threshold, so the metric and
  the knob move together by construction — that is the point, but it means "merge fell" must
  always be read beside "missed rose".

---

## Follow-up: does self-training shift the curve, or only move along it?

The matched-operating-point comparison above showed **erosion shifts the fusion/recall curve**
to a better `pred/label`, while thresholding only slides along it. R5b's eroded-pseudo arms
were compared to the teacher at *single points*, which cannot tell those two cases apart.

Sweeping E04's own inference threshold answers it directly.

**Prediction (before running):** E04's curve will **coincide with the teacher's** — self-training
moves along the curve rather than shifting it. Reasoning: the teacher's probability field is
what generated E04's training target, so E04 should inherit the same instance structure, just
re-centred. Concretely, at matched merge rate I expect E04's `pred/label` within **±0.03** of
the teacher's curve.

**If E04's curve sits at better `pred/label` than the teacher's at matched merge**, then
self-training *does* shift the curve, and R5b's rejection was measuring the wrong thing — the
arms would have been compared at the wrong operating point rather than being genuinely worse.
That would reopen R5b.

### Result — the curves do not coincide, and the criterion I wrote was wrong again

E04's threshold sweep (0.5 row from R5b):

| thr | merge | missed | pred/label |
|---|---|---|---|
| 0.3 | 0.2903 | 0.3717 | 0.9584 |
| 0.4 | 0.2770 | 0.3852 | 0.9645 |
| 0.5 | 0.2666 | 0.3974 | 0.9672 |
| 0.6 | 0.2555 | 0.4093 | 0.9736 |
| 0.7 | 0.2427 | 0.4231 | 0.9796 |

**Prediction wrong.** I said E04's curve would coincide with the teacher's, `pred/label` within
±0.03 at matched merge. The gaps are **0.06–0.09** — the curves are genuinely different.

**E04's curve is compressed.** Across thresholds 0.3–0.7 its merge moves only
**0.2427–0.2903** and missed only **0.3717–0.4231**, where the teacher spans merge
**0.1616–0.4514** and missed **0.2402–0.4627**. Training on eroded pseudo-labels produced a
much more saturated probability field: the threshold barely moves it. That is a real, and
somewhat unwelcome, property — **it costs the free dial.** The teacher can be tuned across a
wide operating range at inference; E04 essentially cannot.

**Now the criterion I pre-registered.** I wrote: *"if E04's curve sits at better `pred/label`
than the teacher's at matched merge, self-training does shift the curve … that would reopen
R5b."* By that test E04 **passes at 4 of 5 matched points** (0.9645 vs 1.0376, 0.9672 vs
1.0500, 0.9736 vs 1.0633, 0.9796 vs 1.0787).

**I am not reopening R5b, because I wrote the same defective criterion as PITFALLS 3.22 —
single-metric, no comparison to the actual deliverable.** Checking the full set:

- **The teacher wins recall at every single matched merge point** (0.3458 vs 0.3717, 0.3563 vs
  0.3852, 0.3646 vs 0.3974, 0.3734 vs 0.4093, 0.3836 vs 0.4231).
- **E04's best `pred/label` at any threshold is 0.9796, at missed 0.4231.** The teacher at its
  default 0.5 gives **0.9914 at missed 0.3257** — better on both, simultaneously.

So E04 does shift the curve, and shifts it to a place that is worse on recall everywhere and
never reaches the teacher's counting. **R5b's rejection stands, now tested against E04's whole
curve rather than one point.**

**Two lessons, one of them about me.** (1) Self-training does change the model's instance
structure, not just its operating point — so "it only moves along the curve" was too glib.
(2) I wrote a single-metric pre-registered criterion **again**, hours after logging PITFALLS
3.22 for exactly that. Writing the rule down does not make it fire correctly; the discipline
that actually caught it was checking the full metric set before acting on the verdict. The
criterion is the artefact, the check is the practice.

---

## The un-eroded curve — erosion is necessary, confirmed on the full sweep

The provisional claim above rested on a **single surviving row** for the un-eroded model. That
checkpoint has now been retrained (val IoU **0.6580**, reproducing the destroyed model's 0.6569
to within 0.0011) and swept.

| thr | merge | missed | pred/label |
|---|---|---|---|
| 0.3 | 0.5567 | 0.1827 | 0.6803 |
| 0.4 | 0.5038 | 0.2173 | 0.7146 |
| 0.5 | 0.4559 | 0.2498 | 0.7507 |
| 0.6 | 0.4019 | 0.2892 | 0.7879 |
| 0.7 | 0.3335 | 0.3426 | 0.8285 |
| 0.8 | 0.2402 | 0.4304 | 0.9119 |

**Matched at equal merge rate, the eroded model wins on recall *and* counting at every point:**

| merge | eroded missed | un-eroded missed | eroded pred/label | un-eroded pred/label |
|---|---|---|---|---|
| 0.4514 | **0.2402** | 0.2531 | **0.8463** | 0.7538 |
| 0.3851 | **0.2804** | 0.3023 | **0.9157** | 0.7979 |
| 0.3155 | **0.3257** | 0.3595 | **0.9914** | 0.8446 |

**The strongest single statement: no inference threshold makes the un-eroded model count
correctly.** Its `pred/label` tops out at **0.9119** (thr 0.8) — and pays **0.4304** missed to
get there. The eroded model reaches **0.9914 at 0.3257 missed**. Counting is not a dial the
un-eroded model has; erosion is what creates it.

**So the deflationary reading earlier in this document was too strong, and is now corrected.**
Thresholding and erosion do trace similarly-*shaped* fusion/recall curves — that part holds.
But they are not interchangeable: erosion moves the whole curve to a strictly better place,
and the gap widens as merge falls (`pred/label` gap 0.093 → 0.118 → 0.147). The earlier
one-point comparison understated the effect.

**Final position on the three knobs, all now measured on full curves:**

| knob | what it does | cost |
|---|---|---|
| **inference threshold** | slides along the model's own curve | free, reversible |
| **label erosion** | shifts the curve to strictly better recall *and* counting | 0.018 IoU, one retrain |
| **self-training** | shifts the curve to a *worse*, compressed place | 2 h/arm, and forfeits the threshold dial |

Erosion is the only one of the three that buys a genuinely better model, and it is confirmed as
the right default.
