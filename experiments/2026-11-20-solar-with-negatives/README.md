# 2026-11-20-solar-with-negatives: retrain the solar model with "no panel" images

| | |
|---|---|
| **Status** | done: model trained (val IoU 0.823), existing PV = 0.44 km² / ~89 MWp (lower bound) |
| **Date** | 2026-11-20 |
| **Code** | `build_solar_neg.py`, `run.sh` (this folder); `scripts/solar_on_hand_crops.py` |
| **Results** | `logs/`, `diagnostics/solar_on_hand_neg/` on the DGX |

## Why
To subtract roof area that already has solar panels, we need a panel detector that works on
Jaipur. All 8 solar checkpoints on the DGX were tested on the 152 hand-labelled Jaipur crops
(`scripts/solar_on_hand_crops.py`). None is usable:

| Checkpoint | Roof flagged as panel (1x / 2.6x) | Flagged area off-roof (2.6x) | Crops with a detection (2.6x) |
|---|---|---|---|
| April BDAPPV runs (5) | 0.45-0.84% / 3.0-4.0% | 39-54% | 143-151 / 152 |
| Sept google-to-ign | 0.27% / 4.60% | 31% | 138 / 152 |
| Sept self-training (2) | 0.19-1.57% / 2.5-7.7% | 36-47% | 135-144 / 152 |

From the gallery:
- At 1x, the top detection is a canal.
- At 2.6x, the model flags metal-sheet factory roofs, white roofs with grid lines and blue tarps.
- One clear panel array was missed.

The answer swings 10x with input scale.

Cause: `solar_panel/prep_bdappv.py` skipped every BDAPPV image without a mask, so every
training crop contained a panel. The model never learned what "no panel" looks like.

## What changes
The training set (`/tmp/btp_data/solar_neg`, mounted at `data/solar_neg`) has three sources:
1. **Positives:** the same BDAPPV Google positives and split as before.
2. **BDAPPV negatives:** every BDAPPV Google image with no mask, about 15k. It is re-downloaded
   from Zenodo record 7358126 (8.2 GB) and split 90/10 by name hash.
3. **Jaipur negatives:** Jaipur patches with no Open Buildings polygon at any confidence
   (fields, canals, roads, open ground). These are 154 px patches resized to 400 px, i.e.
   upsampled 2.6x to the BDAPPV pixel size.
   - Capped at 4,000 train / 600 val.
   - Taken from the existing train/val tiles.
   - None of the 152 eval crops is used.

No Jaipur panel labels are used, and no new labelling was needed.

Training uses the same recipe as `2026-09-10-solar-google-to-ign`: ResNet34 U-Net, ImageNet
init, batch 16, 30 epochs, seed 42. The check afterwards is the same Jaipur crop test at scales
1 / 2.6 / 3.4, plus the gallery checked by eye. The trainer resizes 400 px to 512 px, so 3.4 is
the scale that matches what the model saw.

## Progress
- 03:13 IST: zip downloaded (8,165,112,873 bytes, 67,128 entries). It holds 28,807 Google
  images and 13,303 masks, so 15,504 images have no mask. Build started.

- 03:50 IST: training started. 28,598 train crops (10,665 positive, 13,978 BDAPPV negatives,
  3,955 Jaipur negatives) and 3,452 val crops. About 6.5 min per epoch.
- Val IoU (val includes negatives, so false positives count): 0.738, 0.763, 0.787 and 0.797
  after epochs 1-4.
- **Early Jaipur check, epoch-4 checkpoint** (`diagnostics/solar_on_hand_neg_ep4`):

  | Scale | Roof flagged as panel | Flagged area off-roof | Crops with a detection |
  |---|---|---|---|
  | 1 | 0.02% | 0.4% | 5 / 152 |
  | 2.6 | 0.99% | 6% | 77 / 152 |
  | 3.4 | 0.87% | 6% | 79 / 152 |

  The old models flagged 3-8% of roof area, 31-54% of it off-roof, in 135-151 crops.
  - In the gallery, the top detections are real panel arrays: institutional blocks in 2-1, a
    round building and a hall in 3-1, rooftop rows in 4-2. Canals, metal sheets and tarps are
    no longer flagged.
  - Recall is incomplete: some arrays are only partly covered and some are missed. For now,
    treat the detected area as a **lower bound** on existing PV.
  - Scale 1 detects almost nothing, so inference must upsample (2.6-3.4x).

- **First whole-area estimate** (`scripts/existing_pv_area.py`, snapshot from about epoch 10,
  scale 3.4, roof model r34_c0_shift @0.4, all 9,072 crops; `existing_pv_early.json`):
  - **0.40 km² of panels on roofs, 0.74% of predicted roof area**, which is about **79 MWp**
    at 20% efficiency.
  - Per tile it ranges from 0.21% (4-1) to 1.13% (3-2).
  - Effect on the potential: usable area 0.60 x 53.3 km² minus 0.40 km² is a 1.2% cut, so
    5.29-5.98 GWp becomes about 5.22-5.91 GWp.
  - This is a lower bound, because the model misses part of some arrays.
  - Sanity check (news reports, Business Standard 2026): the JVVNL (Jaipur discom) area had
    about 41.6k PM Surya Ghar rooftop systems. JVVNL covers many districts beyond our
    175 km², so a few tens to ~100 MWp inside the AOI is the right order of magnitude. This
    is not a validation.

- **Final model** (epoch 30, best val IoU 0.823 on a val set that includes negatives),
  Jaipur 152 crops:

  | Scale | Roof flagged as panel | Flagged area off-roof | Crops with a detection |
  |---|---|---|---|
  | 1 | 0.06% | 13% | 5 |
  | **2.6** | **0.84%** | **5%** | 54 |
  | 3.4 | 0.51% | 5% | 52 |

  In the scale-2.6 gallery (`labelling/r12_batch2/solar_gallery_final/`, local only), the top
  detections are all real arrays and cover them more fully than the epoch-4 snapshot did:
  - institutional rows in 2-1
  - a carport/shed array and a round building in 3-1
  - a commercial rooftop array in 4-4

  **Scale 2.6 is the main setting.** It matches BDAPPV's native 0.1 m/px and finds more panel
  area than 3.4 with the same precision. The whole-area run uses both scales
  (`existing_pv_final_s{2.6,3.4}.json`).

## Result: existing PV over the whole study area (final model, scale 2.6)
`existing_pv_final_s2.6.json`, 9,072 crops, 168 km²:
- **0.444 km² of panels on roofs, 0.83% of predicted roof area, about 89 MWp** at 20%
  efficiency. Panels off-roof: 0.031 km².
- Per tile it ranges from 0.31% (4-1) to 1.79% (4-3). The highest tiles are 4-3, 3-2 and 3-4.

Effect on the potential, with roof area divided by its hand-measured bias (1.21 calib / 1.07
test), k_usable 0.60, eta 0.20 and PVOUT 1630 x 0.95:

| | Roof (calibrated) | Usable minus existing panels | Remaining capacity | Energy |
|---|---|---|---|---|
| calib bias | 44.1 km² | 26.4 - 0.44 = 26.0 km² | **5.20 GWp** | **8.0 TWh/yr** |
| test bias | 49.8 km² | 29.9 - 0.44 = 29.5 km² | **5.89 GWp** | **9.1 TWh/yr** |

Cross-check at scale 3.4 (`existing_pv_final_s3.4.json`): 0.282 km², 0.53% of roof, about 56 MWp.
The existing-PV range is therefore **56-89 MWp** (a lower bound; 2.6 is the main setting).

Existing rooftop PV is about 1.5-1.7% of the potential. Subtracting it is correct but changes
the headline by under 2%. k_usable (0.33-0.60 in the literature) matters about 20x more.

## Caveats
- A BDAPPV image with "no mask" means the crowd annotators drew no panel. Most such images
  should have no panel, but some may hold a missed one. That makes them noisy negatives, not
  clean ones.
- No Jaipur panel labels exist, so there is no Jaipur accuracy number, only false-positive
  checks and a visual check.
- The Jaipur negatives cover ground, not roofs. Metal-sheet roofs and tarps are covered only
  by the French roofs without panels.
- `/tmp` on the DGX is shared and can be wiped. The dataset can be regenerated by `run.sh`.
- BDAPPV Google images fall under Google's terms (same unresolved licence as the Jaipur
  imagery). Nothing goes into the repo.
