# 2026-11-22-crossearth-hand-scoring: CrossEarth zero-shot on the 152 hand crops

| | |
|---|---|
| **Status** | done 2026-11-22 (DGX, GPUs 1 and 3, env `crossearth` in `~/crossearth_work`) |
| **Date** | 2026-11-22 |
| **Code** | `scripts/score_crossearth_hand.py` |
| **Method** | CrossEarth, Gong et al., TPAMI 2026, [arXiv 2410.22629](https://arxiv.org/abs/2410.22629), code [VisionXLab/CrossEarth](https://github.com/VisionXLab/CrossEarth) (MIT) |

## Question
Does a remote-sensing domain-generalisation foundation model, with no Jaipur data and no rescaling
of Jaipur, beat the annotation-free arms (best 0.317 hand IoU, DUDA 0.275, AIRS seed 0.154)?
Region-agnostic by design: nothing is resized to match Jaipur.

## Protocol (same as `hand_ensemble_budget.py`)
152 crops, tiles split calib/test (76/76). Threshold chosen on calib from a 0.05-0.95 sweep, IoU
reported on test, crop bootstrap 95% CI. Reference rows: r34_c0_shift 0.697 (0.706 + TTA), arm R 0.317.

## Setup on the DGX (untested)
CrossEarth needs mmseg 1.x, mmcv 2.x, torch 2.0.1, xformers 0.0.20, so it does not run in the rooftop
image. Build its own environment under `/home` (the DGX `/` is full; see `run_docker.sh`).
1. `git clone https://github.com/VisionXLab/CrossEarth` and follow its README (conda or a venv on /home).
2. Download `dinov2_converted.pth` (and `dinov2_converted_1024x1024.pth`) plus a checkpoint from its
   Hugging Face badges into `CrossEarth/checkpoints/`.
3. Edit the dataset paths in the chosen `configs/...` file (inference only needs the model, but the
   config must load).
4. **Find the building class.** Run the script once; it prints the checkpoint's class names. Pass the
   id(s) meaning building/roof as `--roof_class_ids`. Do not guess.

## Run
```bash
python scripts/score_crossearth_hand.py \
  --index data/hand_r12b2/index.csv \
  --config CrossEarth/configs/CrossEarth_dinov2/<config>.py \
  --checkpoint CrossEarth/checkpoints/<checkpoint>.pth \
  --roof_class_ids <ID> --tag crossearth_<name> --tta --gpu 4 \
  --out diagnostics/hand_labels/crossearth_<name>.json
```
For Mask2Former heads add `--logit_mode raw` if softmax gives odd maps (check a few predictions by eye).
Try one checkpoint per benchmark family that contains a building class; report all, not just the best.

## Known unknowns
- Which released checkpoint and class map suit "roof" is unverified. The paper's benchmarks are
  multi-class land cover plus some building tasks; I did not read the paper beyond the abstract.
- Input size: the README example uses 512x512 inference; our crops are 512x512, so no resize is
  expected, but check the config's test pipeline.
- Hand IoU is not comparable to the paper's numbers (different labels, metric and classes).

## Results (zero-shot, no Jaipur image or label used in training; threshold chosen on calib)
Config: `jaipur_building_512.py` (this folder; the base config's 5 classes changed to 2). Weights from
Hugging Face `Cusyoung/CrossEarth`. Class order in the checkpoint is ['Building', 'Background'], so the
roof class id is **0**. A first run with id 1 scored the inverted map (test 0.278, area 3.59), which is
a class-id mistake and not a result. JSONs: `diagnostics/hand_labels/crossearth_whu_{a2s,s2a}_c0.json`.

| Model | Thr | Calib IoU | **Test IoU** (95% CI) | Test area ratio |
|---|---|---|---|---|
| WHU-Building-A2S (aerial source) | 0.35 | 0.616 | 0.647 (0.616-0.677) | 1.07 |
| **WHU-Building-A2S + 8-way TTA** | 0.50 | 0.666 | **0.677** (0.650-0.704) | 0.95 |
| WHU-Building-S2A (satellite source) | 0.30 | 0.617 | 0.648 (0.625-0.670) | 1.13 |
| WHU-Building-S2A + TTA | 0.30 | 0.632 | 0.664 (0.637-0.691) | 1.21 |

For comparison (same 152 crops): AIRS seed 0.154, best annotation-free arm R 0.317 (0.262 on the
152-crop test half), DUDA 0.275; OB-trained ResNet34 on shifted labels 0.697 (0.706 + TTA).

Caveats: single run per checkpoint; the only Jaipur-derived choice is the threshold (same as every other
row). CrossEarth's source model was trained with labels on WHU Building, so this is domain
generalisation from a labelled source, the same status as AIRS. Not compared against Jaipur
weak-label training at equal cost. No rescaling of Jaipur was done.

## UDA step: soft-label self-training (2026-11-22)
Teacher = CrossEarth WHU-Building-A2S with 8-way TTA, run on 8,920 unlabelled Jaipur tiles
(`scripts/crossearth_teacher_maps.py`; the 152 hand crops are excluded). Student = U-Net/ResNet34
initialised from the AIRS checkpoint `rooftop/checkpoints/unet_resnet34_best.pth`, trained 6 epochs
on the teacher's soft probabilities (soft BCE + soft Dice, no Jaipur label of any kind;
`scripts/soft_selftrain_student.py`). Last epoch reported, fixed in advance; threshold from calib.
Result JSON: `diagnostics/hand_labels/student_a2s_soft.json`.

| Model | Thr | Calib IoU | **Test IoU** (95% CI) | Area |
|---|---|---|---|---|
| Teacher alone (A2S + TTA) | 0.50 | 0.666 | 0.677 (0.650-0.704) | 0.95 |
| Student, plain | 0.55 | 0.672 | 0.702 (0.675-0.729) | 1.02 |
| **Student + 8-way TTA** | 0.55 | 0.674 | **0.708** (0.680-0.735) | 1.02 |

Per-epoch plain test IoU: 0.678, 0.697, 0.703, 0.691, 0.696, 0.702 (epoch-to-epoch noise ~0.01).
Caveats: single run, one seed; the student trained on unlabelled images from the same city as the
hand crops (transductive UDA; crops excluded by name only, windows could overlap); the student gain
over the teacher (+0.03) is about the CI width, so it needs a repeat. Matches the OB-trained
ResNet34 (0.697 / 0.706 with TTA) without any OB or hand labels.

### Seed repeat (seeds 0, 1, 2; same teacher maps, only student init order/augmentation differ)
| Seed | Plain test IoU | TTA8 test IoU |
|---|---|---|
| 0 | 0.7017 | 0.7076 |
| 1 | 0.6914 | 0.7098 |
| 2 | 0.6940 | 0.6995 |
| **mean ± sd** | **0.696 ± 0.005** | **0.706 ± 0.005** |

The teacher maps are fixed across seeds, so this measures student-training noise only, not teacher
variance. Student mean (0.706) beats the teacher (0.677) by ~0.03, about 5 sd of student noise.
Calib IoU is stable at 0.67 for every seed.

### Longer training and teacher ensemble (test IoU, thr from calib; all one-off unless stated)
| Teacher | Epochs | Seeds | Plain | TTA8 |
|---|---|---|---|---|
| A2S | 6 | 0,1,2 | 0.696 ± 0.005 | 0.706 ± 0.005 |
| A2S | 15 | 0 | 0.708 | 0.711 |
| A2S+S2A mean | 6 | 0 | 0.690 | 0.699 |
| A2S+S2A mean | 15 | 0,1 | 0.700 / 0.694 | 0.704 / 0.703 |

The averaged teacher alone scores 0.701 on the hand crops (A2S alone 0.677), yet the students trained on
it (0.69-0.70 plain, 0.70-0.71 TTA) are no better than A2S-only students. Student quality plateaus
at ~0.70-0.71 whichever teacher is used; the 95% CI is about +-0.027, so none of these differences
(<= 0.01) is resolved. Test IoU was flat from epoch ~5 of the 15-epoch runs. S2A maps:
`data/jaipur_teacher_s2a` on the DGX. JSONs: `diagnostics/hand_labels/student_a2s_soft{,_s1,_s2}.json`.
