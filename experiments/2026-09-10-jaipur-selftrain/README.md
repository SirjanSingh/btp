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


## Probe result — and the correction that followed

**First, what I reported (wrong).** A 600-crop probe returned:

```
class-ratio 0.231: threshold = 0.8138   (a fixed 0.5 would select 0.3120)
```

From this I concluded the rooftop model **over-predicts** foreground — 31.2 % against Open
Buildings' 23.1 % — and wrote up a tidy story that the S6 ratio-warning "reverses" here.

**That was wrong, and the full-set numbers refute it.** Running both thresholds over all 7,371
crops:

| threshold | pseudo-label foreground | vs OB prior (23.1 %) |
|---|---|---|
| 0.50 | **23.30 %** | +0.2 pts — essentially exact |
| 0.80 | **17.17 %** | −6 pts |

At its default operating point the model selects **23.3 %** against a 23.1 % prior. It is
**well calibrated on Jaipur, not over-predicting.** The 31.2 % figure was an artefact.

**Root cause.** `--limit 600` took `sorted(names)[:600]`. Crop filenames sort by parent tile,
so all 600 came from **2 of 16 tiles** — `map67_1-2` (0.351 density) and `map67_1-3` (0.242),
both above the 0.282 median. The probe measured the densest corner of the city and called it
Jaipur. Fixed in `self_train_pseudolabel.py` to stride across the full list; logged as
PITFALLS 3.21.

**What survives.** The claim the probe was actually run to test — *is the ratio→threshold map
tail-steep, as it was for solar?* — is answered, and more cleanly than before: matching the
23.1 % prior lands at a threshold of **~0.50**, right in the bulk. Solar's equivalent was
0.0004. So the S6 prohibition on ratio policies **does not transfer to this stage**, because
its mechanism (a ratio that forces the threshold into the noise floor) requires a
sparse-foreground target. What does *not* survive is the "model over-predicts, so ratio-matching
becomes conservative" story — the model does neither. Ratio-matching here would simply pick
≈0.5 and change almost nothing.

**A separate correction, to D6's successor claim.** I wrote earlier today that weak supervision
"overcorrected" the AIRS seed's 5.69 % under-prediction to 31.2 %. It did not overcorrect: it
landed at 23.3 % against a 23.1 % truth. The weakly-supervised model reproduces the target
prior almost exactly, which is a better result than the one I reported.

## Two arms

| arm | threshold | pseudo-label fg (full set) | what it tests |
|---|---|---|---|
| **A** | **0.80** | **17.2 %** | high-confidence only — discards a quarter of OB's foreground |
| **B** | **0.50** | **23.3 %** | the model's own operating point, ≈ the OB prior |

**Prediction (arm A vs arm B):** arm B should win. A's threshold throws away 6 points of
foreground the labels claim exists, so it should under-segment — `pred/label` below 1 and
missed rate up. Arm B is near-identical to the teacher's own decision boundary, so its main
risk is a **null result**: pseudo-labels that reproduce what the model already does teach it
nothing (exactly the S7 saturation mechanism, which is the closest measured analogue).

**Concretely:** arm B within ±0.01 IoU of the 0.6393 teacher and `pred/label` 0.95–1.05;
arm A down 0.01–0.04 IoU with `pred/label` 0.80–0.95. If arm A *wins*, the low-confidence
band the teacher emits is mostly noise, and raising the operating point is a free improvement.

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
