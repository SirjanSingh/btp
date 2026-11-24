# 2026-11-19-hand-label-model-scoring: how good are the models on hand-drawn roofs?

| | |
|---|---|
| **Status** | 152-crop set frozen; shifted-label runs scored (new best 0.697 test IoU); phase B running |
| **Date** | 2026-11-19 |
| **Code** | `scripts/score_models_hand.py` (run in the container: `./run_docker.sh 4 "python scripts/score_models_hand.py ..."`) |
| **Results** | `diagnostics/hand_labels/models_r12b2.json` and `.md` |

## Why
Every Jaipur number before this was agreement with Open Buildings (OB) footprints, which are also
the training labels. This is the first measurement against hand-drawn roofs.

## Setup
- **Eval set:** 136 hand-labelled crops from `labelling/r12_batch2` (512 x 512, 0.266 m/px, all
  16 tiles, all five density bins). Annotators: Khushi 79, Priyal 50, Sirjan 7.
  - 30 crops come from the held-out tiles (1-1, 2-2, 4-3).
  - 106 come from tiles the OB-trained models were trained on, with OB labels. No hand label was
    ever used in training.
- **Models:** all 29 rooftop checkpoints still on the DGX:
  - the AIRS seed
  - 13 weak-supervision runs
  - 5 September Jaipur self-training runs
  - 10 annotation-free runs (arms A-R plus DUDA; `last_teacher.pth`, which is the label-free
    reporting rule)
- **Preprocessing and threshold:** identical to `rooftop/train.py`. The main threshold is 0.5;
  the threshold sweep in the table is tuned on this eval set, so it is an oracle.
- **Metrics:** pooled pixel IoU, precision and recall, plus building counts (4-connected
  components of at least 50 px).
- **Error bars:** a crop-level bootstrap (2,000 resamples, paired across models).

## Results (threshold 0.5)

| Model | Hand IoU (95% CI) | Held-out 30 | P | R | Area pred/hand | Buildings pred/hand | OB IoU |
|---|---|---|---|---|---|---|---|
| ResNet34, OB all confidences (`label-quantity-vs-quality`) | **0.632** (0.607-0.657) | 0.661 | 0.74 | 0.81 | 1.09 | 1344/2191 | 0.636 |
| ResNet34, OB conf>=0.75, ImageNet init | 0.605 (0.582-0.628) | 0.616 | 0.77 | 0.74 | 0.96 | 1614/2191 | 0.655 |
| MiT-B2, uneroded | 0.593 (0.568-0.617) | 0.602 | 0.77 | 0.72 | 0.93 | 1673/2191 | 0.671 |
| MiT-B2, eroded 0.4 m (**current default**) | 0.564 (0.541-0.586) | 0.571 | 0.79 | 0.66 | 0.83 | 2278/2191 | 0.647 |
| MiT-B2, eroded 0.4 m, seeds 43 / 44 | 0.578 / 0.568 | | | | 0.87 / 0.85 | | |
| MiT-B5, eroded 0.4 m | 0.571 | 0.575 | 0.79 | 0.67 | 0.85 | 2373/2191 | 0.663 |
| MiT-B2, eroded 0.8 m | 0.526 | 0.527 | 0.82 | 0.60 | 0.73 | 3468/2191 | 0.592 |
| **Open Buildings itself** (conf>=0.75 reference masks) | 0.504 (0.482-0.526) | 0.505 | 0.74 | 0.61 | 0.82 | 2701/2191 | 1.000 |
| Sept self-training t050 / t080 | 0.570 / 0.474 | | | | 0.85 / 0.64 | | |
| Annotation-free, best (arm R) | 0.317 | 0.247 | 0.77 | 0.35 | 0.46 | 1776/2191 | 0.286 |
| Annotation-free, DUDA | 0.275 | 0.223 | 0.71 | 0.31 | 0.43 | 1167/2191 | 0.255 |
| AIRS seed (no Jaipur data) | 0.154 | 0.113 | 0.80 | 0.16 | 0.20 | 942/2191 | 0.139 |
| Annotation-free arms A, B, C, C2, D, D2, G, M | 0.04-0.13 | | | | | | |

Paired differences against the default model (crop bootstrap, 95%):
- all-confidence ResNet34: +0.068 (+0.050 to +0.087)
- ResNet34 at conf>=0.75: +0.041
- MiT-B2 uneroded: +0.029
- OB itself: -0.060

Seed-to-seed spread on the same config is about 0.015. That noise is not in the bootstrap CI, so
treat any gap under about 0.02 as noise.

## Update: final 152-crop set, threshold chosen on one half and reported on the other
After the first run, 16 blank crops were added as empty: crops the labellers left with no
polygons where Open Buildings is also empty. Five blank crops where OB has buildings were dropped
as probably unfinished. That gives 152 crops.

Tiles are split into two halves of 76 crops each, with matching density mixes:
- **test:** 1-1, 2-2, 4-3 (the held-out tiles), plus 1-3, 2-4, 3-1, 3-3, 4-1
- **calib:** the other 8 tiles

The threshold is picked on calib (0.3 won for every model; the sweep's floor is 0.3, so lower may
be better) and reported on test. `diagnostics/hand_labels/models_r12b2_152.json` stores the counts
for every crop and threshold.

| Model | Test IoU @0.5 | Area @0.5 | **Test IoU @0.3** | **Area @0.3** |
|---|---|---|---|---|
| ResNet34, OB all confidences | 0.647 | 1.03 | **0.659** | 1.12 |
| ResNet34, OB conf>=0.75 | 0.600 | 0.89 | 0.632 | **0.99** |
| MiT-B2 uneroded | 0.589 | 0.87 | 0.623 | 0.97 |
| MiT-B2 eroded 0.4 m (default) | 0.553 | 0.76 | 0.597 | 0.87 |
| Open Buildings | 0.499 | | | |
| Annotation-free arm R | 0.262 | 0.36 | 0.326 | 0.48 |

## Energy chain rerun (`scripts/d16_capacity_estimate.py`, k_usable 0.60, PVOUT 1750 unchanged)
| Model, threshold | Predicted roof area | Capacity | Energy | Area / hand (calib / test) | Capacity / calib area ratio |
|---|---|---|---|---|---|
| Old default, 0.5 (reproduces Sept) | 38.2 km² | 4.59 GWp | 7.62 TWh/yr | 0.91 / 0.76 | 5.03 GWp |
| ResNet34 conf>=0.75, 0.3 | 49.3 km² | 5.91 GWp | 9.83 TWh/yr | 1.15 / 0.99 | 5.13 GWp |
| ResNet34 all-conf, 0.3 | 55.6 km² | 6.67 GWp | 11.08 TWh/yr | 1.27 / 1.12 | 5.27 GWp |

After dividing each model by its own hand-measured area bias, the three models agree within
about 5%: **about 5.0-5.3 GWp using the calibration half, about 6.0 using the test half.** The two
halves disagree on area bias by about 15%, which is a real uncertainty from tile and annotator
differences. The old budget's "segmentation area 2.2%" only measured seed noise against OB and
understated this. Calibrated estimate: **about 5-6 GWp (8.3-10 TWh/yr)**, up from 4.59, before
k_usable and PVOUT are sourced.

## Offset per tile
The median OB-to-hand offset is 7 px south / 3 px west (the shift that has to be applied to OB),
with IQR dy 5-10, dx -4 to -1. Every tile falls in dy 5-11, dx -5 to 0, so a single global
correction is justified. `scripts/make_weak_labels.py --shift_px 7 -3` applies it.

## Update 2026-11-20: training on shifted labels (`prep_and_train.sh`)
Both ResNet34 runs use the same config as the Sept runs, with OB labels moved 7 px south and 3 px
west. Checkpoints are still selected on the unshifted OB val tiles, so no hand label touches
training. Scored on the 152 crops (`diagnostics/hand_labels/models_shift_152.json`); the
threshold is chosen on calib and reported on test.

| Model | All-152 IoU @0.5 | Thr (calib) | **Test IoU** | Test area | Test vs old best (paired, 95%) |
|---|---|---|---|---|---|
| ResNet34 all-conf, **shifted** | **0.675** | 0.4 | **0.697** | 1.07 | **+0.038 (+0.031 to +0.045)** |
| ResNet34 conf>=0.75, shifted | 0.658 | 0.3 | 0.681 | 1.01 | +0.022 (+0.004 to +0.039) |
| ResNet34 all-conf (old best) | 0.632 | 0.3 | 0.659 | 1.12 | - |
| ResNet34 conf>=0.75 | 0.605 | 0.3 | 0.632 | 0.99 | -0.027 |

Correcting one global offset in the weak labels is worth about +0.04 IoU on real roofs, which
is more than 2x the seed noise. OB-agreement IoU drops at the same time (0.636 to 0.591),
another case where scoring against OB would rank the models the wrong way round.

**Energy rerun with the new best model** (`energy_r34_c0_shift_t0.4.json`):
- At threshold 0.4 the model predicts 53.3 km² of roof, giving 6.40 GWp / 10.64 TWh (k_usable
  0.60, PVOUT 1750).
- Its hand-measured area bias is 1.21 on calib and 1.07 on test. Dividing by it gives
  **5.29-5.98 GWp**, the same range as the earlier models' calibrated figures (5.0-6.0). The
  estimate holds steady while the segmentation improves.

**PVOUT is now sourced.** Global Solar Atlas (Solargis data v2.2.68, period to 2025, queried
2026-11-20, `diagnostics/gsa_jaipur_lta.json`):
- PVOUT is **1,627-1,641 kWh/kWp/yr** at 5 points across Jaipur. GHI is 1,883-1,897; the
  optimal tilt is 28-29°.
- This replaces the uncited 1,750, a 7% cut.
- With PVOUT 1,630 x rooftop derate 0.95: **5.29-5.98 GWp, about 8.2-9.3 TWh/yr** (k_usable
  0.60, still assumed).

**k_usable sensitivity (web search, 2026-11-20; primary papers not yet opened, so treat these as
leads):**
- Mumbai: 28% of built-up area (Singh & Banerjee 2015) and 33% of roof area in another study.
- Nagpur: modelled at 50% and 70%.
- One Indian city study: a "solar utilisation roof factor" of 25-50% for most buildings.
- Kathmandu (Nepal): about 21% (arXiv 2009.02524).

Our 0.60 sits at the high end of what Indian studies use.

Checked against the primary text (2026-11-20):
- **Bhopal** (Saha & Mandal 2016, Current Urban Studies 4(3)) uses **0.30** of total roof
  area: "The lowest fraction obtained from literature review is 0.30." It cites the MNRE
  standard, but does not derive the number itself.
- **Kathmandu / Pokhara / Biratnagar** (arXiv 2009.02524) uses **30% of the shadow-free
  rooftop area**, chosen from household survey answers. That is about 21% of gross roof area
  (15.4 of 72.7 m² per building).
- **Not verified:** Mumbai 28-33%, Nagpur 50/70% and the 25-50% study. The MDPI page returned
  403. Do not cite these until they are read.

| k_usable | Capacity (calib / test area bias) | Energy, TWh/yr |
|---|---|---|
| 0.60 (current) | 5.29-5.98 GWp | 8.2-9.3 |
| 0.50 | 4.41-4.98 GWp | 6.8-7.7 |
| 0.33 (Mumbai-like) | 2.91-3.29 GWp | 4.5-5.1 |

**MiT-B2 on shifted labels** (`models_mitb2_shift_152.json`; early-stopped at epoch 26):
- Test IoU **0.700** @0.3 (calib-chosen), area 1.10, against 0.697 for ResNet34 shifted. That
  is a tie, inside seed noise.
- Unshifted MiT-B2 scored 0.623, so the shift gives MiT-B2 **+0.077**.
- Label quality matters; encoder choice does not. On correct labels both encoders reach about
  0.70.

**Seeds** (`models_seeds_152.json`):
- ResNet34 shifted, seeds 42 / 43 / 44: test IoU **0.697 / 0.693 / 0.693**.
- Calib picks threshold 0.4 for every seed, and every seed has a test area ratio of 1.06-1.07.
- Each seed's gain over the unshifted model has a 95% CI that excludes zero: +0.038 / +0.035
  / +0.034.
- The shift gain is robust across seeds; the seed spread is 0.004.

Phase B (`phase_b.sh`) finished 2026-11-20 06:30. It ran:
- seeds 43 and 44 of the shifted all-conf model
- MiT-B2 on the shifted labels
- the energy rerun with the new best model at 0.4 and 0.5

## What it means
1. **Every OB-trained model beats OB itself on real roofs** (0.53-0.63 vs 0.50). The network
   learns the roof that is visible in the image rather than reproducing the label's errors (offset,
   footprint-vs-roof, missed buildings). Weak supervision works better than the OB-only numbers
   suggested.
2. **The model ranking changes.** Under OB scoring, MiT-B2 uneroded was best (0.671) and the
   all-confidence model looked worse (0.636). Under hand labels, the all-confidence ResNet34 is
   best and MiT-B2/B5 do not beat ResNet34. The September "confounded" verdict on
   label-quantity-vs-quality came from scoring against conf>=0.75 footprints.
3. **Erosion costs area.** The 0.4 m default gets building counts right (2278 vs 2191) but
   predicts only 83% of the true roof area. The first energy figure (4.59 GWp) used this model,
   so it is likely about 15-20% low on area. This needs a proper rerun, not a scale factor
   (precision is 0.79, so some predicted area is not roof).
4. **All models are under-confident.** Threshold 0.3 beats 0.5 for every model (oracle sweep).
5. **Annotation-free self-training does not get close.** The best arm (R, 0.317) doubles the AIRS
   seed but is half the OB-trained models. On held-out tiles it is lower still (0.247).
6. **OB vs hand:** OB sits about 6-8 px (1.6-2 m) north and 2-4 px east of the hand-drawn roofs,
   and this holds for all three annotators. The OB reference masks are conf>=0.75, which explains
   part of its 0.61 recall.

## Caveats
- 136 crops, single training seed for most models.
- Three annotators with measurably different styles: hand area / OB area is 1.02 for Khushi's own
  batch and 1.24-1.37 for the others. No inter-annotator overlap yet.
- The threshold sweep is tuned on this set. Before any tuned number is reported, the hand labels
  must be split into calibration and test sets (by tile).
- The imagery licence is still unresolved (`plan/07`). These are internal measurements.
