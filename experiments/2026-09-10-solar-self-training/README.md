# 2026-09-10-solar-self-training — CBST on the one gap we can measure

| | |
|---|---|
| **Status** | ✅ done — **strong negative result** |
| **Date** | 2026-09-10 |

## Question

Take [S1](../2026-09-10-solar-google-to-ign/)'s source-only model (IGN **0.5611**, in-domain
ceiling **0.8723**), pseudo-label the IGN training images with it, and retrain on
google-real + ign-pseudo. **How much of the 31-point gap closes?**

**Why here rather than Jaipur:** Jaipur has zero labels, so a self-training run there can
never be scored, only argued about. Both BDAPPV domains are labelled, so every knob —
threshold policy, class ratio, number of rounds — can be tuned against a real number, then
frozen and transferred. That is `MASTER_CONTEXT`'s ordering principle, and this is the first
run to actually exercise it.

**The IGN labels are never read during training** — only at evaluation. Treating a labelled
set as unlabelled is what makes this a valid UDA rehearsal instead of a supervised run in
disguise.

## ★ A finding that arrived before training started

CBST sets the threshold so the selected foreground fraction matches the source prior, rather
than using a fixed 0.95 — the mechanism `MASTER_CONTEXT` names as the fix for foreground
collapse. Measured source (google) foreground: **1.83 %**. To select 1.83 % of IGN pixels the
threshold has to drop to **0.0004**.

| | |
|---|---|
| threshold needed for source-matched ratio | **0.0004** |
| foreground a fixed 0.5 threshold would select | **0.59 %** (a third of source) |

**The model's probability distribution on IGN is crushed toward zero.** Ratio-matching
therefore drags the threshold into the noise floor: at 0.0004 nearly any activation counts as
a panel. This is foreground collapse visible *in the confidence distribution itself*, before a
single pseudo-label is used — and it is a caution about CBST that the plan does not mention.
The prescribed fix has a failure mode of its own on a gap this wide.

**Prediction (before running):** I expect this to **fail or barely move** — IGN IoU
**0.52–0.60**, i.e. plausibly *below* the 0.5611 source-only baseline. Pseudo-labels drawn at
a 0.0004 threshold are close to noise-shaped, and training on them should teach
over-prediction. If it lands above 0.60 I will have badly misread the threshold evidence.

Secondary: **precision falls, recall rises** — the signature of learning from over-inclusive
labels.

## Setup

| | |
|---|---|
| Train | `data/bdappv_st/train` — **16,763** pairs: 10,665 google (real labels) + 6,098 ign (pseudo) |
| Pseudo-labels | `data/bdappv_pseudo/ign_train`, fg fraction 0.0183 by construction |
| Model selection | `google_val` (source) — never the target |
| Report on | `ign_val`, 771 pairs, untouched |

Symlink trees, 25 MB of masks, nothing duplicated.

## Results

Raw: [`diagnostics/s2_crossdomain.json`](../../diagnostics/s2_crossdomain.json)

| | source (google_val) | **target (ign_val)** | best thr on target | precision | recall |
|---|---|---|---|---|---|
| S1 source-only | 0.8723 | **0.5611** | 0.5 | 0.741 | 0.698 |
| **S2 self-trained** | 0.8678 | **0.3752** | **0.8** | **0.391** | 0.902 |

**Self-training cost 18.6 points of target IoU — a 33 % relative degradation.**

Predicted 0.52–0.60, "plausibly below the baseline", with precision falling and recall
rising. Direction exactly right; **severity badly underestimated**.

## Interpretation

**★ The prescribed fix caused the failure it was prescribed to prevent — inverted.**
`MASTER_CONTEXT` recommends CBST class-ratio thresholding because a fixed high threshold
causes *foreground collapse*: sparse predictions → sparser pseudo-labels → sparser teacher.
Ratio-matching does prevent that. But on a domain gap this wide it prevents it **by force**:
to select the source's 1.83 % of pixels it drove the threshold to **0.0004**, so the
pseudo-labels were largely noise labelled as panel. The model dutifully learned to
over-predict.

The evidence is in the precision/recall split: **precision collapsed 0.741 → 0.391** while
recall rose 0.698 → 0.902. It is not confused — it is confidently painting panels everywhere.

**The threshold pathology predicted this before a single epoch ran.** That finding was
recorded in this file before launch, and it was the right read.

**The target's optimal threshold moved 0.5 → 0.8**, further confirming systematic
over-prediction: the only way to use this model is to demand much more confidence than before.

**This is exactly why the bench exists.** Had this been run on Jaipur, where no target label
exists, there would have been no way to know it had made things 33 % worse — the in-domain
score barely moved (0.8723 → 0.8678) and would have looked like a healthy run.
`MASTER_CONTEXT`'s ordering principle just paid for itself.

## Decision

- [x] **Do not run naive CBST self-training on Jaipur.** It would have silently degraded the
      target model with no way to detect it.
- [x] Record that class-ratio thresholding has a failure mode of its own, not mentioned in
      `plan/03` or `MASTER_CONTEXT`: it is safe only when the target confidence distribution
      is not crushed.
- [ ] **Next: isolate the cause.** Repeat with a fixed 0.5 confidence threshold instead of
      ratio-matching. If that recovers, the problem is CBST's ratio policy specifically; if it
      also degrades, self-training itself is unsafe at this gap width.
- [ ] A ratio *between* source prior and model confidence (e.g. 0.006, what 0.5 selects) is
      the obvious middle ground and untested.

## Threats to validity

- ⚠ **C1 unfixed** — every BDAPPV crop contains a panel, so precision here is not a
  deployment number. The *relative* collapse is still real: both arms share the defect.
- One round of self-training. CBST normally runs 2–3 rounds with a growing ratio; a single
  aggressive round may be the worst case rather than a fair test of the method.
- Model selected on source val (correctly — selecting on target would leak), so the target
  number is honest but the run was never optimised for it.

## Threats to validity

- ⚠ **C1 unfixed** — every BDAPPV crop contains a panel, so no precision number here is a
  deployment figure. The *relative* movement against S1 is still informative; both share the
  defect.
- One round of self-training only. CBST is usually run 2–3 rounds with a ratio that grows
  each round; a single round at a fixed ratio is the simplest version and may understate it.
- Class ratio taken from a 600-image sample of google masks (1.83 %), not the full set.
- Ambiguous pixels are assigned to background in a binary mask. A three-state
  foreground/ignore/background target would be the better design and is not implemented.
