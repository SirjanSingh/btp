# 2026-09-11-selftrain-eroded-pseudo — does eroding the pseudo-labels repair R5?

| | |
|---|---|
| **Status** | E04 done · E02 running |
| **Date** | 2026-09-11 |

## Question

R5 rejected self-training for Stage 1 with a **diagnosed** cause rather than a vague one:
pseudo-labels are the teacher's **raw** output, which contains fused blobs, while the teacher
itself was trained on **0.4 m eroded** Open Buildings labels, which contain deliberate gaps.
Self-training therefore trains round 2 on a target that has *lost the erosion* — `pred/label`
fell 0.9914 → 0.9134 and merge rose 0.3155 → 0.3371 even as IoU nudged up.

If that diagnosis is right, applying the same erosion to the pseudo-masks should restore both.

**Why it matters:** it decides whether "self-training fails on Stage 1" is a fact about the
method or an artefact of one fixable implementation detail. R5's conclusion is currently the
stronger claim and this is the test that could weaken it.

## Setup

Pseudo-masks from R5 arm B (teacher at threshold 0.50, 23.30 % foreground) eroded in raster
space by `scripts/erode_masks.py`, then trained identically to R5 (MiT-B2, 40 epochs, val on
the un-eroded OB val tiles).

| arm | erosion | kernel | pseudo fg | fg retained | masks emptied |
|---|---|---|---|---|---|
| **E04** | 0.4 m | 5×5 (~2.0 px/side) | **20.26 %** | 0.870 | 11 |
| **E02** | 0.2 m | 3×3 (~1.0 px/side) | **21.91 %** | 0.940 | 5 |

**E04 reproduces the teacher's own label statistics almost exactly** — Open Buildings eroded
0.4 m is **20.17 %** foreground, against E04's 20.26 %. That is the point of the arm.

**Two caveats stated up front.** (1) This is **raster** morphological erosion, not the polygon
buffer the label pipeline uses; it cannot separate components that the teacher already fused,
so it can only prevent *new* merging, not undo existing merging. (2) 0.4 m is 1.50 px at
26.6 cm/px, but an odd-sized kernel rounds to 5×5, removing ~2.0 px per side — E04 erodes
somewhat harder than the labels did.

## Predictions (before running)

**E04 — `pred/label` 0.95–1.02, merge 0.30–0.34, IoU 0.625–0.645.** If the diagnosis holds,
this should land near the teacher (0.9914 / 0.3155 / 0.6393) rather than near R5 arm B
(0.9134 / 0.3371 / 0.6432).

**E02 between arm B and E04 on every metric**, since it is the same operation at half strength:
`pred/label` ~0.93–0.97, merge ~0.33–0.35.

**What each outcome means:**

- **E04 reaches `pred/label` ≥ 0.95 and merge ≤ 0.3155** → the diagnosis was right, R5's
  rejection was premature, and the recipe is "self-train, but erode the pseudo-labels".
- **E04 restores `pred/label` but merge stays above the teacher's 0.3155** → erosion fixes
  *counting* but not *fusion*, consistent with caveat (1): raster erosion cannot split what is
  already joined. Self-training would still be losing ground on the merging problem.
- **E04 changes little (`pred/label` < 0.93)** → the diagnosis was wrong and R5's rejection
  stands on its own terms. This is the outcome that would most surprise me.

**Honest expectation:** I give the middle outcome the best odds. Counting should recover
because the label statistics now match; merging probably will not, because the fused blobs were
baked into the pseudo-masks before erosion ever ran.

## Results

**E04 complete; E02 still training.** Scored on the same 1,701-crop held-out OB val tiles.

| | val IoU | merge | split | frag/label | missed | **pred/label** |
|---|---|---|---|---|---|---|
| **teacher** (no self-training) | **0.6393** | 0.3155 | 0.0840 | 0.9203 | **0.3257** | **0.9914** |
| R5 arm B (raw pseudo) | 0.6432 | 0.3371 | 0.0787 | 0.9159 | 0.3156 | 0.9134 |
| **R5b E04 (0.4 m eroded pseudo)** | 0.6068 | **0.2666** | 0.0885 | 0.9030 | 0.3974 | **0.9672** |
| R5b E02 (0.2 m eroded pseudo) | *pending* | | | | | |

**Prediction scorecard for E04 — one hit, two misses, both misses in the same direction.**

| prediction | outcome |
|---|---|
| `pred/label` 0.95–1.02 | ✅ **0.9672** |
| merge 0.30–0.34 | ❌ **0.2666** — better than the range |
| IoU 0.625–0.645 | ❌ **0.6068** — worse than the range |

## Interpretation

**The diagnosis was right.** Eroding the pseudo-labels recovers most of what raw self-training
threw away: `pred/label` **0.9134 → 0.9672** (under-counting 8.7 % → 3.3 %) and merge
**0.3371 → 0.2666**, now *better* than the teacher's 0.3155. The mechanism proposed in R5 —
that self-training discards the label erosion — is confirmed by repairing exactly that and
watching both metrics move back.

**But my pre-registered criterion was badly written, and I am not going to hide behind it.**
I said *"E04 reaches `pred/label` ≥ 0.95 and merge ≤ 0.3155 → the diagnosis was right, R5's
rejection was premature."* Both thresholds are met. The second half does not follow, because I
bundled two different claims into one test and included no condition on **recall**:

- missed rate **0.3257 → 0.3974** — E04 misses **40 %** of buildings against the teacher's 33 %.
- `pred/label` 0.9672 is still short of the teacher's **0.9914**.
- IoU is down 0.033.

**E04 beats the teacher on merging alone and loses on everything else.** Erosion shrinks the
training target, so the model learns systematically smaller footprints: fewer fusions, more
buildings missed entirely. That is the same recall/fusion trade the original erosion sweep
measured on the labels — and here it is being paid *twice*, once in the teacher's labels and
again in the pseudo-labels. Double erosion is over-erosion.

**So R5's rejection stands, on better evidence than before.** Self-training with eroded
pseudo-labels is a real improvement over self-training with raw ones, and still not an
improvement over not self-training. The teacher remains the deliverable.

**A lesson about pre-registration itself.** A criterion that names two metrics can be satisfied
while the model gets worse on a third. The fix is not to abandon pre-registration but to state
the *decision rule* over the full metric set — here it should have been "beats the teacher on
`pred/label` **and** does not lose recall", which E04 fails. Recorded in PITFALLS.

## Decision

- [x] **R5's rejection of self-training for Stage 1 stands.** The teacher (0.9914, 0.3155,
      0.6393) is still the best model on the deciding metric.
- [x] **The R5 diagnosis is confirmed** — pseudo-label erosion recovers 0.9134 → 0.9672. Worth
      reporting as mechanism even though the method is rejected.
- [x] **Double erosion is over-erosion.** If self-training is ever revisited, erode the
      pseudo-labels *less* than the labels, not equally — E02 (0.2 m) is the running test of
      exactly that.
- [ ] Await E02. If it beats E04 on missed rate while holding `pred/label` ≥ 0.95, the
      "erode less on the second pass" rule is confirmed rather than merely argued.

## Threats to validity

- Agreement with Open Buildings, not accuracy (R12) — as with every Jaipur number.
- Single seed per arm; R5 arm B's IoU edge over the teacher (+0.0039) was already inside noise.
- Only arm B's pseudo-labels are used as the base. The thr-0.80 arm is not retried, since its
  failure was foreground starvation rather than missing erosion.
- Erosion strength is confounded with kernel rounding (E04 removes ~2.0 px where 0.4 m is
  1.5 px), so E04 is slightly stronger than the label pipeline's 0.4 m.
