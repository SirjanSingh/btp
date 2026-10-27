# 2026-09-09-does-the-airs-seed-help — is the AIRS pretraining worth anything?

| | |
|---|---|
| **Status** | ✅ done |
| **Date** | 2026-09-09 |
| **Commit** | `HEAD` on `feat/init-project-setup` |

## Question

Repeat the weak-supervision run **from ImageNet initialisation** instead of the AIRS
checkpoint. Everything else identical.

**Why it matters — this is the most plan-changing single result available.** If ImageNet
init matches 0.6475, then the AIRS seed contributes *nothing*, and:

- the missing AIRS dataset stops being a blocker,
- the source-domain question (AIRS vs Inria vs SpaceNet-Khartoum) becomes moot,
- the entire domain-adaptation framing loses its premise — there is no source to adapt
  *from*, only a target to train *on*.

Nothing else in the backlog can retire that much of the plan in one run.

**Prediction (before running):** **0.60–0.64**, a little below the seeded 0.6475. The AIRS
seed should be worth something — generic aerial-imagery features, roof-shaped priors — but
much less than its 0.8784 source score suggests, because 7,371 target crops is plenty to
learn from directly. If the gap is under 0.01 I would call the seed worthless here.

## Setup

Identical to [`2026-09-09-weak-supervision-jaipur`](../2026-09-09-weak-supervision-jaipur/)
except `--init_weights` is omitted (encoder falls back to `encoder_weights=imagenet`).

```bash
./run_docker.sh 7 "python -u rooftop/train.py \
    --train_dir data/jaipur_weak/train --val_dir data/jaipur_weak/val \
    --epochs 40 --batch_size 16 --workers 4 --save_every 999 \
    --ckpt_dir experiments/2026-09-09-does-the-airs-seed-help/checkpoints \
    --log_dir  experiments/2026-09-09-does-the-airs-seed-help/outputs"
```

## Results

Raw: `outputs/*.json`

| | AIRS-seeded | **ImageNet init** |
|---|---|---|
| best val IoU | 0.6475 | **0.6483** |
| best epoch | 33 | **33** |
| precision @ best | 0.7361 | 0.7363 |
| recall @ best | 0.8434 | 0.8443 |
| F1 @ best | 0.7861 | 0.7866 |
| first epoch ≥ 0.63 | 9 | **9** |

**Prediction was 0.60–0.64, below the seeded run. Measured 0.6483 — above it, and outside
the predicted band.** I expected the AIRS features to be worth *something*; they are worth
nothing measurable.

## Interpretation

The two runs are not merely close, they are **indistinguishable in every respect that was
recorded**: same best epoch, same convergence speed to 0.63, precision and recall matching to
three decimal places. A +0.0008 IoU difference on a single seed is noise. If the AIRS
checkpoint carried useful features, the seeded run should at minimum have converged *faster* —
it did not, reaching 0.63 at epoch 9 exactly like the ImageNet run.

**So the AIRS checkpoint — 0.8784 IoU on its own domain — contributes nothing to Jaipur
beyond what generic ImageNet weights already provide.**

D2/D3 from the same night explain why. AIRS is **7.69 %** foreground with a median tile at
2.06 % — mostly empty. Its buildings occupy **21,084 px** against Jaipur's **913 px**, a 23×
difference in scale. The source is not a harder version of the target; it is a different
visual problem, and features learned on 145×145 px suburban roofs do not transfer to 30×30 px
dense ones.

## Decision

- [x] **The source-domain question is closed.** AIRS vs Inria vs SpaceNet-Khartoum stops
      being the blocking decision it has been for two sessions. Do not spend the ~14 GB and
      the network time on AIRS imagery for pretraining purposes.
- [x] **The missing-AIRS "blocker" was never a blocker.** `MASTER_CONTEXT` C5 should be
      downgraded: it blocks D5 and any co-training experiment, nothing else.
- [ ] Effort moves to the target side — label quality (SAM2 refinement, shift correction),
      architecture (R4), and crop/scale choices (D3's 23× finding).
- [ ] **Caveat before generalising:** this tests *pretrain-then-finetune*. It does **not**
      test co-training on a GSD-matched source (R7), where a source at 30 cm might still help.

## Threats to validity

- **Single seed, single run.** A 0.0008 difference is meaningless; the claim rests on the
  *whole trajectory* matching, not the final number.
- Measured against Open Buildings footprints, not roof ground truth.
- 7,371 target crops is a lot. With far fewer target labels, pretraining would likely matter
  more — this result is specific to the data-rich weak-supervision regime.
- Only tests the ResNet-34 encoder. A different architecture might exploit source features
  differently.

## Threats to validity

- Same footprint-label caveat as the parent experiment: this measures agreement with Open
  Buildings ground footprints, not roof accuracy.
- One seed, one run. A 0.01–0.02 difference is within run-to-run noise and should not be
  read as a real gap.
