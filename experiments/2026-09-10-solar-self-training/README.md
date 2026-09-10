# 2026-09-10-solar-self-training — CBST on the one gap we can measure

| | |
|---|---|
| **Status** | running |
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

*pending*

## Threats to validity

- ⚠ **C1 unfixed** — every BDAPPV crop contains a panel, so no precision number here is a
  deployment figure. The *relative* movement against S1 is still informative; both share the
  defect.
- One round of self-training only. CBST is usually run 2–3 rounds with a ratio that grows
  each round; a single round at a fixed ratio is the simplest version and may understate it.
- Class ratio taken from a 600-image sample of google masks (1.83 %), not the full set.
- Ambiguous pixels are assigned to background in a binary mask. A three-state
  foreground/ignore/background target would be the better design and is not implemented.
