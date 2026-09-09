# 2026-09-09-does-the-airs-seed-help — is the AIRS pretraining worth anything?

| | |
|---|---|
| **Status** | running |
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

*pending*

## Threats to validity

- Same footprint-label caveat as the parent experiment: this measures agreement with Open
  Buildings ground footprints, not roof accuracy.
- One seed, one run. A 0.01–0.02 difference is within run-to-run noise and should not be
  read as a real gap.
