# 2026-09-10-solar-google-to-ign — the one domain gap we can actually measure

| | |
|---|---|
| **Status** | running |
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

*pending*

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
