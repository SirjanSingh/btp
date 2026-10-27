# 2026-09-10-jaipur-selftrain — does the solar-bench recipe transfer to Jaipur rooftops?

| | |
|---|---|
| **Status** | ❌ negative — teacher wins; kill criterion fired |
| **Date** | 2026-09-10 |

> ⚠️ **Noise-floor caveat (added 2026-09-11).** A same-config seed replicate measured
> run-to-run spread of **0.0597 `pred/label`** and **0.0495 merge** — see
> [`../2026-09-11-seed-variance/`](../2026-09-11-seed-variance/). Differences below
> that scale in this write-up are **not supported by a single pair of runs**.
> Specifically: arm B's `pred/label` drop (0.078) is 1.3x the floor. The direction is
> corroborated by R5b's eroded arms moving it back, which is independent evidence, but the
> single comparison alone would not carry it.


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

Teacher = the current default (MiT-B2, 0.4 m eroded OB labels). All three scored on the same
1,701-crop held-out val tiles.

| | val IoU | merge | split | frag/label | missed | **pred/label** |
|---|---|---|---|---|---|---|
| **teacher** (no self-training) | 0.6393 | 0.3155 | 0.0840 | 0.9203 | 0.3257 | **0.9914** |
| **arm B** (thr 0.50) | **0.6432** | 0.3371 | 0.0787 | 0.9159 | 0.3156 | **0.9134** |
| **arm A** (thr 0.80) | 0.5673 | **0.0998** | 0.1541 | 0.9221 | 0.5474 | **1.2282** |

**Self-training does not help. The teacher stays the best model.**

Arm B gains **+0.0039 IoU** — and moves `pred/label` from 0.9914 to **0.9134**, i.e.
under-counting rises from 0.9 % to 8.7 %. Merge rate worsens too (0.3155 → 0.3371). On the
metric that decides a per-building kW estimate, the arm with the better IoU is the worse model.
This is the project's own thesis landing on its own experiment.

**Prediction scorecard — one hit, three misses.**

| prediction | outcome |
|---|---|
| arm B beats arm A | ✅ clearly |
| arm B within ±0.01 IoU of teacher | ✅ +0.0039 |
| arm B `pred/label` 0.95–1.05 | ❌ **0.9134** |
| arm A down 0.01–0.04 IoU | ❌ down **0.0720**, far worse |
| arm A `pred/label` 0.80–0.95 (under-segments) | ❌ **1.2282** — it *over*-counts |

**My pre-registered kill criterion fired.** I wrote: *"if merge rate worsens, pseudo-labelling
is reinforcing the fused blobs the erosion work exists to prevent, and self-training is
actively wrong for this stage."* Merge went **0.3155 → 0.3371**. By the standard set before
the run, self-training is wrong for Stage 1.

## Interpretation

**Arm A does not under-segment — it hallucinates.** I expected a high threshold to produce
conservative, sparse masks that miss buildings: `pred/label` below 1. Instead it misses
**54.7 %** of labels entirely *while emitting 1.23 predictions per label*. Reconciling the two:
`frag/label` 0.9221 × 27,918 labels ≈ 25,700 components touch a label, against **34,288**
predicted — so **~8,500 components (25 %) touch no label at all.** Training on high-confidence
pseudo-labels that cover only 17.2 % of pixels teaches the model that buildings are small and
rare, and it scatters fragments into the gaps.

This is the **same signature as 0.8 m over-erosion** (`pred/label` 1.47 with `frag/label` only
1.04): starve the model of foreground and it invents buildings rather than shrinking them. Two
different knobs — label erosion and pseudo-label threshold — with one failure mode. Worth
naming: **foreground starvation produces hallucination, not conservatism.**

**Arm B fails for the opposite reason, and it is the more interesting failure.** Its
pseudo-labels reproduce the teacher's own decision boundary almost exactly (23.3 % foreground
against the teacher's own operating point). Nearly all its components land on real labels —
`frag/label` 0.9159 against 25,501 predictions is essentially zero hallucination. But it merges
*more* than the teacher. Pseudo-labels are the teacher's **raw** output, which contains the
fused blobs; the teacher's real training signal was **eroded** Open Buildings labels, which
contain the gaps. Round 2 therefore trains on a target *without* the erosion that made round 1
good. **Self-training silently discards the single most valuable property of the label set.**

That is the general lesson, and it is not specific to Jaipur: when the labelling pipeline
contains a deliberate correction the model has not fully learned, pseudo-labelling from that
model **throws the correction away**. Erosion cost 0.018 IoU and bought `pred/label` 0.76 →
0.99; self-training hands back a third of that for +0.004 IoU.

**Why the solar bench did not predict this.** S4 worked (0.5611 → 0.6135) because BDAPPV's
labels carry no analogous correction — a panel mask is just a panel mask. The rooftop labels
are engineered. **The bench transfers hyperparameters, not the decision to use the method.**

## Decision

- [x] **Reject self-training for Stage 1.** The teacher is the deliverable; neither arm improves
      it on `pred/label` or merge rate.
- [x] **Keep IoU out of the headline.** Arm B is the concrete case where higher IoU means a
      worse model, measured on this project's own target.
- [ ] **If revisited: erode the pseudo-labels before training.** The obvious repair — apply the
      same 0.4 m erosion to the teacher's output. Cheap (one flag) and directly targets the
      diagnosed cause. Queued, not run.
- [x] Record "foreground starvation → hallucination" as the shared signature of arm A and 0.8 m
      over-erosion.

## Threats to validity

- **Agreement, not accuracy (R12).** Every number is against Open Buildings. If OB merges two
  structures, `pred/label` cannot see it, and a model that copies OB's errors scores well.
- Single seed per arm; arm B's +0.0039 IoU is inside seed noise, though its −0.078
  `pred/label` is not.
- The teacher was selected on the same OB-derived val set the students are scored on, which
  favours the teacher. The instance-metric gaps are large enough that this does not explain them.
- Two thresholds only. The repair (eroded pseudo-labels) is untested, so "self-training fails
  here" is really "self-training *from raw teacher output* fails here".

## Threats to validity

- Agreement, not accuracy (R12). If self-training makes the model imitate OB's *errors* more
  faithfully, every metric here improves while the model gets worse.
- The teacher was selected on the same OB-derived val set the student is scored on.
- One round, one seed, one threshold — S6 says the threshold plateau is wide, but that was
  measured on a sparse-foreground task and may not hold at 23 %.
