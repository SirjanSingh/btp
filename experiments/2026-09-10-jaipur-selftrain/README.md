# 2026-09-10-jaipur-selftrain — does the solar-bench recipe transfer to Jaipur rooftops?

| | |
|---|---|
| **Status** | running |
| **Date** | 2026-09-10 |

## Question

S4/S6/S7 settled the self-training recipe on the one place it could be *measured* —
BDAPPV google→ign: **one round, fixed confidence threshold, never a class ratio**. This is the
transfer that whole bench existed to inform. Apply it to the Jaipur rooftop model and ask
whether it helps.

**Why it matters:** this is the first UDA method in the project applied to the actual target
with evidence behind its hyperparameters rather than a guess. `MASTER_CONTEXT`'s ordering
principle in full: settle where measurable, freeze, transfer blind.

## The backlog recipe is wrong and is being overridden

`BACKLOG.md` R5 says: *"keep confident pixels using the measured 23 % class ratio (not a fixed
0.95)"*. That is **exactly the CBST ratio policy S6 measured as harmful** — on the solar bench
the prescribed ratio sat at 2× the break-even point and cost 0.20 IoU. Following it here would
repeat a mistake already paid for. This arm uses a **fixed threshold**.

## But the solar result may not transfer, and that is the real question

The S6 danger has a stated mechanism: ratio→threshold is steep when the target ratio lands in
the **tail** of the confidence distribution. Solar foreground is ~0.6–1.8 % of pixels — deep in
the tail, where a 1.36× ratio change moved the threshold 45×.

**Jaipur rooftops are not sparse.** The measured target prior is **23.1 %** at confidence
≥ 0.75 (28.2 % raw) — squarely in the *bulk* of the distribution, where the quantile map should
be gentle. So the mechanism that made ratio dangerous for solar may be largely absent here.

**Prediction (before running):**

1. **The ratio→threshold map will be far flatter than solar's.** Selecting 23 % of pixels
   should land at a threshold of order **0.2–0.5**, not 0.0004. If it lands below 0.05, the
   rooftop model is as poorly calibrated on Jaipur as the solar model was on IGN, and the
   S6 warning transfers intact.
2. **Self-training will help, modestly: `pred/label` stays within 0.95–1.05 and merge rate
   improves by 0.00–0.04.** Confidence self-training raised solar target IoU 0.052; the
   rooftop model starts from a much stronger base (it was trained on target-domain imagery
   already, just with noisy labels), so there is less headroom.
3. **The risk is the opposite of solar's.** With a 23 % prior, a fixed 0.45 threshold on a
   model that already predicts ~23 % foreground barely changes anything — the failure mode
   here is a **null result**, not collapse.

**What would change my mind:** if merge rate *worsens*, pseudo-labelling is reinforcing the
fused blobs the erosion work exists to prevent, and self-training is actively wrong for this
stage.


## Probe result (before training) — the S6 warning does NOT transfer, and reverses

Ran the ratio policy on 600 crops purely to read off the threshold it implies:

```
class-ratio 0.231: threshold = 0.8138   (a fixed 0.5 would select 0.3120)
```

**Prediction 1 half-right.** I said the map would be gentle rather than tail-steep — correct,
0.8138 is squarely in the bulk, nothing like solar's 0.0004. I also said "order 0.2–0.5", and
it is **0.81**. The direction of the error is the informative part.

**The rooftop model OVER-predicts foreground; the solar model UNDER-predicted it.** At a fixed
0.5 the rooftop model calls **31.2 %** of pixels building, against Open Buildings' **23.1 %**.
So ratio-matching here *raises* the threshold to 0.81 to become more selective, whereas on
solar it *lowered* the threshold to 0.0004 to become less so.

That inverts the risk. CBST is dangerous when it forces the threshold **down** into noise to
manufacture foreground the model does not believe in. On Jaipur rooftops it would force the
threshold **up**, discarding the model's least-confident predictions — which is the
conservative direction. **The S6 prohibition is specific to sparse-foreground targets and does
not generalise to this stage.** Worth stating plainly, because the tidy lesson "never use a
ratio policy" would have been the wrong thing to carry over.

Note also this is the mirror image of D6: the *AIRS seed* predicted only 5.69 % foreground
against a 23–28 % truth, a 5× under-prediction. The weakly-supervised model has overshot to
31.2 %. Training on Open Buildings did not just fix the under-prediction — it overcorrected.

## Two arms

Because the threshold is no longer obviously choosable, this brackets it:

| arm | threshold | pseudo-label fg | rationale |
|---|---|---|---|
| **A** | **0.80** | ≈ 23 % | high-confidence; coincides with the measured OB prior |
| **B** | **0.50** | ≈ 31 % | the model's own default operating point, more inclusive |

**Prediction:** arm A ≥ arm B on `pred/label` and merge rate, because B's extra 8 % of
foreground is exactly the low-confidence margin where buildings fuse. If B wins, the model's
uncertain band contains real buildings OB is missing, which would be evidence *about OB* and a
reason to accelerate R12.

## Setup

Teacher: the current default rooftop model (**MiT-B2, 0.4 m eroded labels**, val IoU 0.6393,
`pred/label` 0.9914). Pseudo-label the 7,371 Jaipur **train** crops at a fixed threshold,
retrain from ImageNet init, evaluate against the held-out Open Buildings val tiles with
IoU + merge/split + `fragments_per_label` + `pred/label`.

**This measures agreement with Open Buildings, not accuracy** — as does every Jaipur number
until R12's hand labels exist. A gain here means "agrees more with OB", which is the right
target only insofar as OB is right.

## Results

*pending*

## Threats to validity

- Agreement, not accuracy (R12). If self-training makes the model imitate OB's *errors* more
  faithfully, every metric here improves while the model gets worse.
- The teacher was selected on the same OB-derived val set the student is scored on.
- One round, one seed, one threshold — S6 says the threshold plateau is wide, but that was
  measured on a sparse-foreground task and may not hold at 23 %.
