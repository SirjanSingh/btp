# 2026-09-10-solar-google-to-ign — the one domain gap we can actually measure

| | |
|---|---|
| **Status** | ✅ done |
| **Date** | 2026-09-10 |

## Question

Train the Stage-2 solar model on **Google** imagery only and report on **IGN** imagery.
Both are BDAPPV, both are fully labelled, and they are different sensors at different GSD
(Google ~10 cm, IGN ~20 cm).

**Why this is the most valuable rung in the project.** Jaipur has **zero labels**, so no
adaptation method can ever be *measured* there — only argued about. google→ign is the only
setting where a real target IoU exists. `MASTER_CONTEXT`'s ordering principle is explicit:
tune λ_st, CBST thresholds and class-mix here where the score is visible, freeze them, then
transfer to Jaipur blind. Doing it in the other order is guessing with extra steps.

This run establishes the **source-only baseline** — the drop that every later adaptation
method has to close.

**Prediction (before running):** in-domain google val IoU **0.82–0.86** (prior solar runs
reached ~0.85). On IGN, a drop to **0.55–0.70**. The GSD ratio here is only 2× versus
Stage 1's 3.55×, and panels are far more visually distinctive than roofs, so I expect a
smaller relative drop than the rooftop domain gap — but a clear one.

## Setup

| | |
|---|---|
| Train | `google_` crops only — **10,665** pairs |
| Model selection | `google_val` — **1,323** pairs (**in-domain**) |
| Report on | `ign_val` — **771** pairs (**target, never used for selection**) |
| Also available | `ign_train` 6,098 pairs — held back, this is the *source-only* arm |

**Protocol note, and it matters:** the checkpoint is selected on the **source** validation
set, not the target. Selecting on `ign_val` would leak the target into model choice and
inflate the reported gap-closure of every later method — the exact failure `MASTER_CONTEXT`
E1/E2 warn about. The honest source-only number requires never looking at IGN during
training.

Splits are symlink trees under `data/bdappv_split/` (2 MB, no data duplicated).

```bash
./run_docker.sh 3 "python -u solar_panel/train_solar.py \
    --train_dir data/bdappv_split/google_train \
    --val_dir   data/bdappv_split/google_val \
    --epochs 40 --batch_size 16 --workers 3 --save_every 999 \
    --ckpt_dir experiments/2026-09-10-solar-google-to-ign/checkpoints \
    --log_dir  experiments/2026-09-10-solar-google-to-ign/outputs"
# then evaluate that checkpoint on data/bdappv_split/ign_val
```

## Results

Raw: [`diagnostics/s1_crossdomain.json`](../../diagnostics/s1_crossdomain.json)

| | best IoU | at threshold | precision | recall |
|---|---|---|---|---|
| **google_val** (source, in-domain) | **0.8723** | 0.5 | 0.931 | 0.933 |
| **ign_val** (TARGET, never seen) | **0.5611** | 0.5 | 0.741 | 0.698 |

**Domain gap: −0.3112 absolute, −35.7 % relative.**

Predicted 0.82–0.86 in-domain (got 0.8723, slightly above) and 0.55–0.70 on IGN (got 0.5611,
at the bottom of the band). Both calls landed.

## Interpretation

**★ This is a real capability drop, not miscalibration — and the threshold sweep is what
proves it.** The optimum sits at **0.5 on both domains**. Contrast the rooftop case, where
the AIRS seed's best threshold on Jaipur was **0.01** and simply turning that dial recovered
0.142 → 0.333 IoU. There, most of the apparent domain gap was calibration. Here there is no
free lunch: the model is genuinely worse on IGN, and closing the gap requires learning, not
rescaling.

That distinction matters because the two need different fixes, and reporting a single fixed
threshold would have conflated them.

**Precision and recall fall together** (0.93/0.93 → 0.74/0.70), rather than one collapsing.
The model is not systematically over- or under-predicting on IGN; it is simply less accurate
— consistent with a genuine appearance shift rather than a prior shift.

**The project now has a measurable adaptation testbed with explicit headroom: 0.5611 → 0.8723.**
Every UDA method — self-training, CBST, DAFormer/HRDA/MIC — can be scored on how much of
those 31 points it closes. On Jaipur none of them can ever be scored at all. Per
`MASTER_CONTEXT`'s ordering principle, hyperparameters should be tuned here and transferred
frozen.

## Decision

- [x] **Source-only baseline established: 0.5611 on IGN.** This is the number to beat.
- [ ] S2: run self-training / CBST here and report gap closed, before touching Jaipur.
- [ ] Re-run after C1 is fixed — see below; the absolute values are not deployment numbers.

## Threats to validity

- ⚠ **C1 (zero negatives) is unfixed, and now known to be unfixable from disk**:
  `solar_panel/bdappv/` is empty, so a re-prep needs BDAPPV re-downloaded from source. Every
  crop on both sides contains a panel, so **the model has never seen a panel-free roof and
  these precision figures are not deployment numbers.** The *relative* drop remains
  informative because both domains share the defect — but the 0.8723 in particular would fall
  sharply against realistic negatives.
- Google and IGN differ in GSD, sensor **and** geography within France. The 31-point drop
  bundles all three; it is not a pure resolution effect.
- IGN crops were resized to 512² to match the model input, which adds a resampling step the
  source domain did not undergo.
- Single seed.

## Threats to validity

- ⚠ **`MASTER_CONTEXT` C1 is unfixed: BDAPPV crops contain ZERO negatives.**
  `prep_bdappv.py:85` drops mask-less images, so every crop contains a panel and the model
  is never shown a panel-free roof. **Any precision figure from this run is measuring the
  wrong task** and must not be quoted as a deployment number. The *relative* google→ign drop
  is still informative — both sides share the defect — which is why the run is worth doing
  before S3 fixes it. Re-run after C1.
- Google vs IGN differ in GSD *and* sensor *and* geography within France. The measured drop
  bundles all three; it is not a pure resolution effect.
- Single seed.
