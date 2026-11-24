# 2026-10-30-duda — arm E: DUDA distillation, AIRS -> Jaipur, no target labels

| | |
|---|---|
| **Status** | ❌ done, no gain (launched 2026-10-30 16:05 IST, GPU 2; `DUDA_DONE` written 16:58 DGX clock) |
| **Question** | Does a large teacher distilled into the small UNet avoid the collapse that sank plain self-training (arms A-D)? |
| **Compare with** | `../2026-10-30-annotation-free-selftrain/` arms A-D, same data and eval crops |
| **Method** | DUDA, Kang et al., WACV 2026, [arXiv 2504.09814](https://arxiv.org/abs/2504.09814) |

## Why this arm

Arms A-D all start from the AIRS resnet34 seed and let the small network label Jaipur for
itself. All four collapsed: by iteration 3000 the predicted roof fraction swings to near 0
(or near 1) and IoU@0.5 ends at 0.03-0.09, below AdaBN alone. DUDA attacks the obvious
suspect: a small network's own pseudo-labels. Here the pseudo-labels come from a separate
large network, and the small one is trained to match them.

## What runs

| Network | Architecture | Start | Updated by |
|---|---|---|---|
| LT large teacher | UNet / mit_b5 | ImageNet | EMA of LS |
| LS large student | UNet / mit_b5 | ImageNet | gradients |
| SS small student | UNet / resnet34 | AIRS seed + AdaBN (40 batches) | gradients; **this is the deployed model** |

1. **Source warm-up** (iters 1-1000): LS trains on AIRS crops only; LT follows by fast EMA (0.99).
2. **Pre-adaptation** (1001-3000): LS on source + LT's hard pseudo-labels; SS on source +
   the same pseudo-labels + KL to LT's soft output; LT = EMA(LS) at 0.999.
3. **Inconsistency**: per class, 1 - IoU(LT, SS) averaged over 40 target batches, then
   normalised so the two class weights sum to 2.
4. **Fine-tuning** (1500 iters): LT and LS frozen; SS trains against LS, with each class's
   pseudo-label loss weighted 2 - I' and its KL weighted I'.

Data (read-only, identical to arms B-D): `data/st_run/airs_crops/train` (125 AIRS crops),
`data/st_run/target_tiles` (1,560 unlabelled tiles from the 13 non-eval tiles),
`data/st_run/eval/val` (1,701 crops of tiles 1-1, 2-2, 4-3; measure only).

## Adapting the paper to this problem

- **Binary.** C = 2 (background, roof). Pseudo-label = argmax (p > 0.5). The paper uses no
  confidence threshold, so neither does this. The per-class losses L^(c) are read as the
  loss over pixels pseudo-labelled c, normalised by all pixels.
- **KL** is the two-class KL between sigmoid outputs, with no temperature (the paper gives none).
- **Source warm-up** is ours, not the paper's. The paper's benchmarks have thousands of
  labelled source images; we have 125 AIRS crops and an ImageNet-initialised large network,
  so without it LT's first pseudo-labels are noise.
- **Iterations scaled** from 40k + 80k to 3k + 1.5k for a shared node.
- **Source loss** is the repo's `combined_loss` (BCE + Dice), matching arms A-D.
- **Helpers** (`TargetDataset`, `adabn`, `ema_update`, `evaluate`, `unlabelled_stats`) are imported
  from `st_common.py`, a frozen copy of arms A-D's `self_train.py` (hash in `logs/st_common.sha1`).

## Reporting rule (fixed before the run)

Primary = the **last** small-student evaluation. Best-over-evaluations is selected on the eval
set, so it is reported only as an oracle upper bound. Pre-registered stop rule: if LT's
predicted roof fraction is still about 0 at iteration 1000, stop and lengthen the warm-up.

Every number below is **agreement with Open Buildings footprints**, not accuracy against roof
outlines. Hand labels (`labelling/r12_batch2`) will replace that once drawn.

## Results

Source: `logs/duda_20261030_103014.json` on the DGX, read and relayed by the DGX Claude session on
2026-11-07 (not re-read from the file here). Single seed (42); noise floor about 0.004 IoU.

| Row | IoU@0.5 (primary) | Note |
|---|---|---|
| AIRS seed, no adaptation | 0.1419 | `diagnostics/method_comparison.json` |
| AdaBN alone (Sept / arms B-D start) | 0.2523 / 0.306 | different BN sample, see arms README |
| Arms A / B / C / D, final (it 3000) | 0.0720 / 0.0548 / 0.0405 / 0.1643 | all collapsed (roof fraction 0.02-0.06) |
| E: large teacher alone, end of pre-adaptation (it 3000) | 0.0633 | roof fraction 0.0235; peaked 0.1434 at it 500, 0.0156 at it 1500 |
| E: small student, end of pre-adaptation (it 3000) | 0.2747 | start (AdaBN) was 0.3068 |
| **E: DUDA small student, last (it 4500)** | **0.2673** | primary; best-threshold 0.2900 at thr 0.2 |

Small student by evaluation: 0.3068 (it 0, 500, 1000), 0.2551 (1500), 0.2666 (2000), 0.2748 (2500),
0.2747 (3000), 0.2838 (3500), 0.2613 (4000), 0.2673 (4500). The oracle best is the iteration-0 value.

## Interpretation

DUDA did not help. The primary number (0.2673) is below its own starting point (0.3068), so it is
worse than AdaBN alone. It did avoid the collapse of arms A-D (final roof fraction 0.136 against
0.02-0.06) but ended far from the 0.648 weak-supervision model.

- **Stop rule.** The large teacher's roof fraction at iteration 1000 was 0.0334 (IoU 0.0648),
  borderline against "still about 0". The run was not stopped or lengthened.
- **The large teacher collapsed** during pre-adaptation (IoU 0.1434 to 0.0156 by it 1500, pseudo-label
  roof fraction 0.003-0.03), so the small student was distilled from a poor teacher.
- **`avg_tgt_kl` is NaN at every fine-tuning evaluation** (3500, 4000, 4500). The KL term may not have
  worked during fine-tuning. This is a possible bug, not investigated; treat the fine-tuning stage as
  unverified.
- Inconsistency weights: I = [0.1258, 0.8492], normalised [0.258, 1.742].

## Files

`duda.py` (method), `run_duda.sh` (launcher), `st_common.py` (frozen helpers),
`logs/duda_*.json` (every evaluation), `logs/DUDA_DONE`, `ckpt/*/last_small_student.pth`.
