# 2026-11-20-toward-higher-iou: steps 1-3 of the plan to move past 0.70

| | |
|---|---|
| **Status** | steps 1-3 done; step 4 (label refinement) and step 6 (fine-tune on calib crops) next |
| **Date** | 2026-11-20 |
| **Code** | `scripts/estimate_offset.py`, `scripts/hand_ensemble_budget.py` |
| **Results** | `offset_{calib,test,all}.json` (this folder), `diagnostics/hand_labels/ensemble_budget.json` |

## Why
The best model (ResNet34 on shifted Open Buildings labels) scores 0.697 test IoU on the hand
crops. The mentor asked for about 0.90. We agreed 0.75-0.80 is the realistic target, because the
best fully supervised results on the Inria aerial benchmark (0.3 m, clean labels) are about 0.83.
These three cheap steps run before any new training.

## Step 1: is the 7 px / 3 px shift leaking test information?
The shift was measured on all 152 crops, test half included. Measured again on each half alone
(`scripts/estimate_offset.py`, best-IoU shift per crop, median):

| Half | Crops used | Median dy (south) | Median dx | IQR dy | IQR dx |
|---|---|---|---|---|---|
| calib | 58 / 76 | 7 | -3 | 5-10 | -5 to -1.25 |
| test | 62 / 76 | 7.5 | -3 | 5-9 | -4 to -0.25 |
| all | 120 / 152 | 7 | -3 | 5-10 | -5 to -1 |

The calib half alone gives exactly 7 / -3. No leak; no retrain needed.

## Step 3: test-time augmentation and ensembles (no training)
Each model is run plain and with 8-way dihedral TTA (4 rotations x flip, mapped back, averaged).
Ensembles average probabilities. Threshold from a 0.15-0.70 sweep on calib, IoU on test, paired
bootstrap against plain r34_c0_shift (0.697).

| Config | Thr | Test IoU | Area ratio | Gain (95% CI) |
|---|---|---|---|---|
| r34_c0_shift + TTA (**chosen on calib**) | 0.40 | **0.706** | 1.08 | +0.010 (+0.007 to +0.013) |
| best on test (oracle): r34 seed 44 + MiT-B2, TTA | 0.35 | 0.707 | 1.09 | +0.010 |
| all 4 models + TTA | 0.40 | 0.705 | 1.08 | +0.009 |
| all 4 models, no TTA | 0.40 | 0.698 | 1.07 | +0.002 |
| MiT-B2 alone + TTA | 0.35 | 0.704 | 1.08 | +0.007 |

TTA is worth +0.010 and is real. Averaging models adds nothing on top: the seeds make the same
mistakes, so the errors come from the labels, not from training noise.

## Step 2: error budget (test half, r34_c0_shift + TTA)
| Measure | Value |
|---|---|
| IoU | 0.706 |
| IoU ignoring a 1 / 2 / 3 px band round each hand roof edge | 0.736 / 0.759 / 0.776 |
| wrong pixels inside missed buildings (no predicted pixel) | 4% |
| wrong pixels inside invented buildings (no hand pixel) | 3% |
| wrong pixels on edges and partly covered roofs | 93% |

- Almost all error is shape error on buildings the model did find, not missed or invented buildings.
- Even forgiving 80 cm (3 px) round every edge only reaches 0.776. So the predicted shapes are
  off by more than an edge: footprint-vs-roof differences, merged neighbours, and annotator style
  (hand/OB area ratio 1.02 for one annotator, 1.24-1.37 for the others).
- That points at the labels: step 4 (refine footprints into roof outlines) and step 6 (fine-tune
  on the 76 calib hand crops) target exactly this; more models or bigger encoders do not.

## Step 6: fine-tune on hand crops (2-fold, separate from the zero-label result)
`step6_finetune.sh`: start from r34_c0_shift, train 30 epochs (lr 1e-4, batch 8) on one half's 76 hand
crops, use the epoch-30 checkpoint (no model selection on the held-out half), pick the threshold on the
training half, report on the other half. Both halves have a similar annotator mix (Khushi 51/44, Priyal 25/25, Sirjan 0/7).

| Fold | Trained on | Scored on | Zero-label r34 (+TTA) | Fine-tuned (+TTA) | Diff vs plain r34, 95% CI |
|---|---|---|---|---|---|
| A | calib | test | 0.697 (0.706) | 0.675 (0.684) | -0.021 (-0.062 to +0.011) |
| B | test | calib | 0.656 (0.666) | 0.682 (0.688) | +0.026 (+0.015 to +0.037) |

Verdict: **no reliable gain.** One fold improves, the other gets worse, and the average is about +0.003.
76 crops are too few: the model fits the training half (its IoU there reaches 0.75-0.77) and does not
carry that over. It does not pass the stop rule, so the zero-label r34 + TTA (0.706 on test) stays the
headline. A gentler fine-tune (fewer epochs, frozen encoder) could be tried, but choosing those settings
would need a third split we don't have.

## Round 1: five one-change experiments in parallel (started 2026-11-20 evening)
Each run changes one thing against r34_c0_shift (seed 42, 40 ep, bs 16, model pick on OB conf>=0.75
val tiles). Launcher `round1.sh TAG GPU`; `round1_after.sh` waits for the labels, trains the label
variants and scores everything into `round1_scores.json`. Big files (SAM weights, extra libraries,
refined labels, checkpoints) live in host `/tmp/btp_data/iou`, mounted as `data/iou`, because the
home quota had only about 2 GB free. `/tmp` is wiped on reboot: copy any winner's `best.pth` home.

| Tag | Change | Source |
|---|---|---|
| edge | BCE weighted by 1 + 200 x Gaussian(sigma 5.6 px) of the label boundary (`--edge_weight 200`) | Sirko et al. 2021 |
| dino | encoder `tu-convnext_tiny.dinov3_lvd1689m` (31.9 M params; timm 1.0.20 + smp 0.5.0 from `data/iou/pylib_dino`) | DINOv3 2025; Align and Segment 2026 |
| lab_align | each OB polygon moved separately (±14 px) to best overlap the r34_c0_shift+TTA prediction | OMAF 2026 / AnS 2026 idea |
| lab_sam | SAM ViT-H box prompt per polygon (global shift, box +10 px); SAM outline kept if IoU with the footprint >= 0.3 and area 0.5-2x | Kirillov et al. 2023 |
| lab_align_sam | SAM prompted with the per-building-shifted box | both |

**Label gate before training** (`scripts/refine_labels.py --mode gate`, labels scored against the
hand roofs, pooled IoU, 24 px border ignored; settings chosen on calib):

| Labels | calib | test |
|---|---|---|
| global shift (current) | 0.589 | 0.606 |
| per-building shift | 0.598 | 0.610 |
| SAM, box pad 6 / 10 / 14 | 0.611 / **0.615** / 0.611 | 0.633 / 0.642 / 0.636 |
| shift + SAM, box pad 6 / 10 / 14 | 0.622 / **0.629** / 0.626 | 0.641 / 0.650 / 0.647 |
| the r34_c0_shift+TTA prediction itself | 0.689 | 0.723 |

All three refined sets beat the current labels, so all three are trained (pad 10, tau 0.3, chosen on
calib). Note that the model's own prediction is still a better label than any refined footprint:
soft-label self-training (Sirko 2021) is the obvious next candidate.

**Round 1 results** (scored 2026-11-21 05:00 IST by `scripts/hand_ensemble_budget.py`, file
`round1_scores.json` on the DGX). Threshold chosen on calib, IoU on test, gain against plain
r34_c0_shift (0.697) with a 95% paired-bootstrap interval. One seed each.

| Model | calib | test | gain vs 0.697 | Verdict |
|---|---|---|---|---|
| r34_c0_shift + TTA (baseline) | 0.670 | 0.706 | +0.010 | reference |
| edge + TTA | 0.666 | 0.695 | -0.001 (-0.007 to +0.005) | no gain |
| dino + TTA | 0.643 | 0.685 | -0.011 (-0.019 to -0.005) | worse |
| lab_align + TTA | 0.668 | 0.698 | +0.001 (-0.004 to +0.007) | no gain |
| lab_sam + TTA | 0.667 | 0.705 | +0.009 (+0.002 to +0.015) | small, needs seed 2 |
| lab_align_sam + TTA | **0.680** | 0.704 | +0.007 (-0.002 to +0.017) | best single model on calib |
| ens(lab_align, lab_align_sam) + TTA, chosen on calib | 0.682 | **0.710** | +0.014 (+0.008 to +0.020) | gain |
| ens(r34_c0_shift, lab_align_sam) + TTA (test oracle, not a claim) | 0.682 | 0.713 | +0.016 | |

Reading it: the label gate ranked shift+SAM labels best, and the shift+SAM model is also the best
single model on calib, with the area ratio closest to 1 (1.01). But the training gain is less than
half the label gain (the gate went 0.606 to 0.650 on test; the model moved about +0.007). The
network already averages out much of the footprint noise. Second seeds (`SEED=43`, tags
`lab_align_sam_s43`, `lab_sam_s43`) were started at 05:25 on GPUs 5 and 1.

## Round 2: soft-label self-training (started 2026-11-21 ~03:00 IST)
Teacher = r34_c0_shift + TTA. `scripts/refine_labels.py --mode pseudo` writes the teacher
probability as a 0-255 PNG mask per crop (`data/iou/lab_soft_pred`) and fused masks
`w * teacher + (1 - w) * lab_align_sam` (`data/iou/lab_soft_fuse{w}`). `rooftop/train.py --soft_masks`
reads masks as probabilities (mask / 255) instead of thresholding them. Launcher: `round2.sh TAG GPU
"ARGS"` (`SEED=43` env for a second seed); `round2_labels.sh` runs the fusion gate and the pseudo
labels; `round2_after.sh` trains and scores everything into `round12_scores.json`.

Fusion gate (labels vs hand roofs, calib / test): teacher 0.689 / 0.723, fuse 0.9 0.688 / 0.715,
fuse 0.8 0.688 / 0.714, fuse 0.7 0.686 / 0.710, fuse 0.6 0.678 / 0.699, shift+SAM 0.629 / 0.650.
Fusion never beat the teacher on calib, so we trained on the teacher (st_soft_pred) and fuse 0.8.

| Model | 1 pass test | + TTA test | gain of + TTA vs 0.697 |
|---|---|---|---|
| st_soft_pred | 0.703 | 0.706 | +0.009 (+0.006 to +0.014) |
| st_soft_fuse08 | 0.706 | 0.708 | +0.011 (+0.007 to +0.016) |
| ens(lab_align_sam, st_soft_pred) + TTA, best of 510 on calib | | **0.712** | +0.015 (+0.009 to +0.022) |

The student reproduces teacher + TTA in a single forward pass (8x cheaper for the city run) but
does not clearly beat it, so a third round is not worth it. Overall the overnight ceiling is about
0.71; no single change moved the model by more than seed noise.

**Second seeds** (`seeds_scores.json` on the DGX, seed 43): lab_align_sam 0.706 single pass /
0.708 + TTA (seed 42: 0.705 / 0.704), so its +0.008 single-pass gain reproduces, but with TTA it
equals the baseline + TTA (0.706); the two fix the same errors. lab_sam seed 43: 0.699 / 0.704, gain
not reproduced.

The best ensemble also reproduces: ens(lab_align_sam_s43, st_soft_pred) + TTA = 0.712 (+0.015,
+0.010 to +0.021).

Label gate vs trained model (calib, six label sets): Spearman 0.77 single-pass, 0.54 with TTA.
With n = 6 neither is significant.

Checkpoints kept in DGX home (`ckpt_keep/`, 1.1 GB): st_soft_pred, st_soft_fuse08 (the 0.706
single-pass student for the city run), lab_align_sam, lab_align_sam_s43. Home has about 1 GB left.

Disk note: `hand_ensemble_budget.py` writes a `<out>_probs.npz` cache (0.4-1 GB) next to the output
JSON. Four of them filled the 90 GB home quota at 07:20; they were moved to
`/tmp/btp_data/iou/probs/` (wiped on reboot, regenerable with `--rerun`).

Instance labels: `scripts/make_instance_labels.py` burns every polygon with its own ID (16-bit PNG
per crop, `data/iou/inst`), so touching buildings stay separate for per-building steps.

## Step 0: sources, checked on dblp (2026-11-20)
Checked by hand in a browser (dblp blocks scripts). Record links and notes are in `dblp_check.md`.

Method:
- Ronneberger, Fischer, Brox. U-Net. MICCAI 2015. https://dblp.org/rec/conf/miccai/RonnebergerFB15
- He, Zhang, Ren, Sun. Deep Residual Learning for Image Recognition. CVPR 2016. https://dblp.org/rec/conf/cvpr/HeZRS16
- Sirko et al. Continental-Scale Building Detection from High Resolution Satellite Imagery. CoRR abs/2107.12283, 2021. https://dblp.org/rec/journals/corr/abs-2107-12283
- Kaiser et al. Learning Aerial Image Segmentation From Online Maps. IEEE TGRS 55(11), 2017. https://dblp.org/rec/journals/tgrs/KaiserWLJHS17
- Mnih, Hinton. Learning to Label Aerial Images from Noisy Data. ICML 2012. https://dblp.org/rec/conf/icml/MnihH12
- Girard, Charpiat, Tarabalka. Noisy Supervision for Correcting Misaligned Cadaster Maps Without Perfect Ground Truth Data. IGARSS 2019. https://dblp.org/rec/conf/igarss/GirardCT19
- Loshchilov, Hutter. Decoupled Weight Decay Regularization (AdamW). ICLR 2019. https://dblp.org/rec/conf/iclr/LoshchilovH19
- Xie et al. SegFormer. NeurIPS 2021. https://dblp.org/rec/conf/nips/XieWYAAL21

The UDA comparison that failed:
- DUDA (Kang et al.): dblp has only the preprint, CoRR abs/2504.09814 (2025), and no WACV 2026 record.
  Cite it as arXiv 2025. https://dblp.org/rec/journals/corr/abs-2504-09814
- Kasmi et al. Space-scale Exploration of the Poor Reliability of Deep Learning Models: the Case of the
  Remote Sensing of Rooftop Photovoltaic Systems. dblp: CoRR abs/2408.07828, **2024** (earlier notes said
  2025). https://dblp.org/rec/journals/corr/abs-2408-07828
