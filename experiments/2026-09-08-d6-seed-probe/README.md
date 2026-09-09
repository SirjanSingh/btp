# 2026-09-08-d6-seed-probe — the AIRS-trained seed predicts 5.7 % foreground on Jaipur

| | |
|---|---|
| **Status** | ✅ done |
| **Date** | 2026-09-08 |
| **Commit** | `02852a0` on `feat/init-project-setup` |
| **Supersedes** | — |

> Backfilled 2026-09-09 from `diagnostics/d6/d6_summary.json`. The run itself is from
> 2026-09-08; the Prediction section is reconstructed from `MASTER_CONTEXT` and is marked
> as such, because it was not written down beforehand.

## Question

Diagnostic **D6** (`MASTER_CONTEXT` §6.3). Run the existing AIRS-trained checkpoint over
random crops of the unlabelled Jaipur mosaic and record what it *predicts*. There are no
target labels, so this cannot measure accuracy — that is the point. It measures predicted
foreground fraction and the confidence distribution.

**Why it matters:** it decides whether this checkpoint is usable as a self-training teacher
at all. A teacher that predicts far below the true prior collapses under self-training —
sparse predictions → sparse pseudo-labels → sparser teacher. Paired with D1 it is the
project's motivating figure.

**Prediction:** *(not recorded in advance — reconstructed)* `MASTER_CONTEXT` §3.1 assumed
AIRS ~15 % fg and Jaipur ~50 %, and predicted the seed would under-predict substantially.
The direction was expected; the magnitude was not quantified.

## Setup

| | |
|---|---|
| Data | `data/jaipur/` — 16 GeoTIFF tiles, EPSG:3857, **zero labels**. 640 random 512² crops, 40 per tile, seed 0 |
| Model / ckpt | U-Net + ResNet-34, `rooftop/checkpoints/unet_resnet34_best.pth` (AIRS-trained, 2026-03-30) |
| Hardware | 1× V100 |
| Container | `nvcr.io/nvidia/pytorch:24.05-py3` via `run_docker.sh` |
| Runtime | a few minutes |

```bash
# 96-crop smoke test first, then the real run
./run_docker.sh "" "python -u scripts/d6_seed_probe.py --n_crops 96  --out_dir diagnostics/d6_smoke"
./run_docker.sh "" "python -u scripts/d6_seed_probe.py --n_crops 640 --out_dir diagnostics/d6"
```

## Results

Raw output: [`diagnostics/d6/d6_summary.json`](../../diagnostics/d6/d6_summary.json),
`d6_confidence.csv`. Smoke run retained at `diagnostics/d6_smoke/`.

Measured GSD **0.26618 m/px** — a **3.55×** gap against AIRS's 7.5 cm.

Predicted foreground fraction, threshold sweep (640 crops):

| threshold | mean fg | median | p90 | crops fully empty |
|---|---|---|---|---|
| 0.05 | 10.19 % | 8.70 % | 21.83 % | 8.9 % |
| 0.20 | 6.65 % | 5.15 % | 14.88 % | 10.5 % |
| **0.35** (ref) | **5.69 %** | **4.25 %** | 12.68 % | 11.6 % |
| 0.50 | 5.08 % | 3.65 % | 11.57 % | 12.0 % |
| 0.70 | 4.31 % | 2.90 % | 10.14 % | 13.3 % |

Per-tile mean fg spans **0.95 % – 11.01 %** (`map67_4-1` lowest, `map67_1-1` highest).

## Interpretation

The prediction is **remarkably insensitive to threshold** — 14× of threshold range moves the
mean only from 10.2 % to 4.3 %. The model is not uncertain-but-correct, it is confidently
predicting very little. Lowering the threshold will not rescue it.

The 96-crop smoke run gave 6.99 % against the full run's 5.69 % at the same threshold. A
23 % relative difference from sampling alone is a useful warning: **do not quote D6-style
numbers from small samples**, and treat per-tile figures (40 crops each) as indicative only.

## Decision

- [x] Ran D1 to get the true prior for comparison → [`2026-09-09-d1-target-prior`](../2026-09-09-d1-target-prior/).
- [x] Keep 0.35 as the reference threshold; the sweep shows the choice is not load-bearing.
- [ ] **The seed is not usable as a naive self-training teacher.** Any self-training run needs
      the CBST class-ratio correction from D1, not a fixed confidence threshold.

## Threats to validity

- **No target labels exist**, so this is a measurement of model behaviour, not of error. It
  cannot be turned into an IoU.
- 640 crops over 175 km² is a thin sample; see the smoke-vs-full gap above.
- Random crops sample *area*, not buildings, so dense and sparse wards are weighted by
  footprint rather than by importance.
- The checkpoint is the Stage-1 baseline, not a tuned model. It is the right seed to probe,
  but "the seed under-predicts" is not the same claim as "U-Net/ResNet-34 cannot do this".

## Reproduce

Needs `data/jaipur/` (8 GB, restored 2026-09-08) and the checkpoint, which is **gitignored** —
`rooftop/checkpoints/unet_resnet34_best.pth`, 293 MB, from the 2026-03-30 100-epoch run.
Seed 0 is fixed, so crop selection is deterministic given the same tile list.
