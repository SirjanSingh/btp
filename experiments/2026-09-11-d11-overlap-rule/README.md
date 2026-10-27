# 2026-09-11-d11-overlap-rule — how much of the miss rate is the rule, not the model?

| | |
|---|---|
| **Status** | ★ done — rule inflates miss rate 1.77×; large-building misses are artefact |
| **Date** | 2026-09-11 |

## Question

Every instance metric in this project — merge, split, missed, `pred/label` — depends on one
inherited constant: a label counts as **found** when a prediction covers **≥ 50 %** of it. That
threshold came from `merge_split_rate.py` on day one and has never been examined.

D10 showed why that matters. Across 20 inspected missed buildings, coverage ran **0.05–0.50**,
with only two at exactly zero — and both of those were Open Buildings false positives over
roofless ground. **Every genuine "miss" was a partial detection.** And because OB systematically
over-covers (compound walls, plot boundaries drawn as buildings), a model that correctly outlines
only the roof inside a compound lands well under 50 % and is recorded as having missed it
entirely.

So: **how much of the headline 33 % miss rate is the model failing, and how much is the rule?**

**Why it matters:** if miss rate collapses as the threshold relaxes, the 33 % figure is largely
an artefact of a constant nobody chose deliberately, and the "largest remaining error in Stage 1"
framing that has driven the last several experiments is wrong. That would be worth knowing before
anyone spends a 3-hour arm on it.

## Predictions (before running)

1. **Miss rate falls steeply between 0.5 and 0.2.** From D10's coverage distribution I expect
   roughly **0.28 → 0.12–0.18** at `min_overlap 0.2`, and **0.08–0.13** at `0.1`.
2. **The fall is proportionally larger for big buildings.** Compound over-coverage is a
   large-polygon phenomenon, so the 2000+ bin should lose a larger *fraction* of its misses than
   the 50–200 bin, whose misses are more likely genuine non-detections.
3. **A residual floor remains** — buildings with genuinely zero coverage stay missed at any
   threshold. I expect **≥ 5 %** overall even at `min_overlap 0.05`.

**What would change the conclusion:** if miss rate is roughly flat from 0.5 down to 0.1, then
misses really are non-detections, the rule is innocent, and D10's coverage sample was
unrepresentative of the 7,904 misses overall.

## Setup

`scripts/d8_missed_by_size.py` at `--min_overlap` ∈ {0.5, 0.3, 0.2, 0.1, 0.05} on the default
teacher (MiT-B2, 0.4 m eroded labels), same 1,701-crop val tiles, threshold 0.5. Only the
association rule changes; the model and predictions are identical throughout.

## Results

Overall miss rate, and by size bin, as the association threshold relaxes. **Model and
predictions are identical in every row — only the rule changes.**

| min_overlap | **ALL** | 50–200 | 200–400 | 400–900 | 900–2000 | 2000+ |
|---|---|---|---|---|---|---|
| **0.5** (inherited) | **0.2831** | 0.7479 | 0.7046 | 0.4757 | 0.2372 | 0.0714 |
| 0.3 | 0.2142 | 0.6957 | 0.6238 | 0.3624 | 0.1451 | 0.0322 |
| 0.2 | 0.1918 | 0.6693 | 0.5862 | 0.3235 | 0.1166 | 0.0234 |
| 0.1 | 0.1721 | 0.6429 | 0.5448 | 0.2895 | 0.0934 | 0.0169 |
| **0.05** | **0.1598** | **0.6215** | 0.5227 | 0.2630 | 0.0812 | **0.0137** |
| *reduction 0.5 → 0.05* | **−44 %** | **−17 %** | −26 % | −45 % | −66 % | **−81 %** |

**Prediction scorecard.**
1. Predicted 0.12–0.18 at `0.2` and 0.08–0.13 at `0.1`; actual **0.1918** and **0.1721** ❌ —
   the fall is real but **less steep than I predicted in both cases**. I over-estimated the
   artefact.
2. "Fall proportionally larger for big buildings" ✅ — and decisively: **−81 %** for 2000+ against
   **−17 %** for 50–200.
3. "Residual floor ≥ 5 %" ✅ — **16.0 %** remains even at 5 % overlap.

## Interpretation

**The inherited rule inflates the headline miss rate by 1.77×** (0.2831 vs 0.1598). Roughly
**44 % of all "misses" are partial detections**, not failures to detect. That constant was never
chosen deliberately, and it has been the denominator of the "largest remaining error in Stage 1"
framing driving the last several experiments.

**The two size regimes are different phenomena, and this separates them cleanly:**

- **Large buildings (2000+): the miss rate is almost entirely a rule artefact.** 0.0714 → 0.0137
  once partial credit is allowed — **81 % of it disappears**. This is exactly what D10 showed
  visually: OB draws compounds and plot boundaries, the model correctly finds the roofs inside,
  and a label-centred ≥50 % rule scores that as a miss. **Large-building detection is not a
  real problem.**
- **Small buildings (50–200): the miss rate is real.** Only 17 % of it dissolves. **62 % of small
  labelled buildings receive essentially no prediction at all** — under 5 % coverage. These are
  total non-detections, not clipped edges.

**That sharpens D8/D9's open question rather than answering it.** The remaining ambiguity is now
much better specified: it is not "does the model partly miss small buildings" but "**does the
model see nothing at all where OB claims a small building — and is anything there?**" D9
established OB's metadata cannot answer that. R12 still can.

**Nothing already decided changes.** Every comparative conclusion (erosion, self-training,
encoder, threshold) held the rule fixed across arms, so a shared inflation cancels. What changes
is the **absolute** framing of model quality — which is the thing this project has repeatedly had
to walk back, and for the same underlying reason each time.

## Decision

- [x] **Report miss rate with its association threshold stated**, never bare. "33 % missed" is
      meaningless without "at ≥50 % overlap".
- [x] **Stop treating large-building misses as a problem.** At 1.4 % under a permissive rule they
      are noise, and D10 shows the residue is largely OB error.
- [x] **The small-building non-detection rate (62 % at any threshold) is the one real,
      unexplained Stage-1 error.** Everything else has now been attributed to rules, labels, or
      trade-offs.
- [ ] **Do not re-tune the rule to flatter the model.** 0.5 is defensible for *instance counting*,
      where covering half a building is the minimum for claiming you found it. The fix is to
      report both, not to switch to whichever number looks better.

## Threats to validity

- Relaxing the threshold cannot distinguish "correctly found the roof inside an over-drawn OB
  polygon" from "clipped a corner of a building it mostly missed". Both become hits, so this
  **bounds** the artefact rather than isolating it.
- Merge and split rates also depend on this constant and were not swept; only the miss rate is
  isolated here.
- Agreement with Open Buildings throughout (R12).

## Threats to validity

- Relaxing the threshold cannot distinguish "correctly found a roof inside an over-drawn OB
  polygon" from "clipped a corner of a building it mostly missed". Both become hits. The sweep
  bounds the artefact, it does not separate those two.
- Agreement with Open Buildings throughout (R12).
- A lower threshold would also change merge and split rates, which are not swept here — this
  isolates the miss rate deliberately.
