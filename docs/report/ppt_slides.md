# Presentation slide content

Draft content for the project presentation. Numbers come from `docs/report/sota_standouts.md`
and the experiment READMEs. † = unverified (check against the paper PDF before final submission).

Status: Motivation, Problem Statement, Objective, SOTA, Novelty, Where We Stand done. Still to
write: Literature Review, Methodology, Experiments, Results, Conclusion, Future Plan, References.

Wording rule: say "no Jaipur labels used for training", not "zero labels".

Background used: last semester's deck ("BTP Even Semester 2026-2027", 22 Apr 2026, Canva) and the repo.

---

## Slide: Motivation

**Title:** Motivation: Why Estimate Rooftop Solar from Satellite Images?

- **India is pushing rooftop solar, but planning is blind**
  - Large national rooftop programmes (e.g. PM Surya Ghar, 2024, 1 crore homes target†)
  - Planners have no city-wide map of *where* rooftop potential is highest
- **Manual surveys don't scale**
  - Assessing millions of roofs one by one is too slow and too costly
- **The imagery already exists**
  - High-resolution aerial/satellite images cover Indian cities
  - But no open tool turns them into roof area, capacity (kWp) and energy (kWh/yr) for India
- **Global models don't transfer to India**
  - Existing models and datasets are Western (US, New Zealand, France)
  - Indian roofs are small, flat, concrete, densely packed, with water tanks and clutter

**Our journey (the chain):**

| Stage | When | What | Result |
|---|---|---|---|
| 1. Literature | Sem 1 | DeepSolar (US, 1.47M installations), U-Net, AIRS, BDAPPV, US PV dataset | Two-stage idea: roofs, then panels |
| 2. Rooftop model (M₁) | Sem 1 | U-Net + ResNet-34 on AIRS (Christchurch, 7.5 cm/px) | **0.9016 IoU** (210 tiles) > PSPNet 0.899; 0.8664 on full test |
| 3. Panel model (M₂) | Sem 1 | U-Net + ResNet-34 on BDAPPV (France, Google + IGN) | **0.848 IoU**, F1 0.918 |
| 4. Energy formula | Sem 1 | Roof area → capacity, P = A × η | Planned: India deployment |
| 5. Move to India | Sem 2 | Ran M₁ on Jaipur (26.6 cm/px) | **Collapsed: 0.154 IoU** on real roofs |
| 6. Fix the transfer | Sem 2 | Weak labels, domain adaptation, foundation models | **0.717 IoU**, no Jaipur labels used for training |
| 7. Energy for Jaipur | Sem 2 | Roof area → GWp → TWh/yr | ≈ 5.3–6.0 GWp, 8.2–9.3 TWh/yr |

**Speaker line:** "Last semester our models beat published baselines on Western data. When we moved to Jaipur, the rooftop model dropped from 0.90 to 0.15. Making it work for India became this semester's problem."

---

## Slide: Problem Statement

**Title:** Problem Statement

**One-line statement:**
> Estimate rooftop solar potential (roof area, capacity, energy) for an Indian city from aerial imagery, **when no labelled Indian training data exists**.

**Why it's hard: the domain gap from AIRS (NZ) to Jaipur (India)**

| Factor | AIRS (source) | Jaipur (target) |
|---|---|---|
| Resolution | 7.5 cm/px | **26.6 cm/px** (3.55× coarser) |
| Median building size | 21,084 px | **913 px** (23× smaller) |
| Roof share of image | 7.7% | **28.2%** (3.7× denser) |
| Buildings touching a neighbour | — | **78%** |
| Imagery | Orthorectified aerial | Off-nadir Google tiles (roofs shifted from footprints) |
| Labels | Full pixel masks | **None** |

**What goes wrong:**
- The AIRS model predicts only **5.7% roof** on Jaipur (true ≈ 28%) → **0.154 IoU**
- Image-style fixes (FDA, histogram matching) make it **worse**
- Classical domain adaptation (CBST, DUDA, 10 self-training variants) gets only **0.04–0.33**
- Free footprint data (Google Open Buildings, 523k polygons) is shifted and footprint-shaped: **0.50 IoU** against real roofs
- Hand-labelling a whole city is the expensive step we want to avoid

**Speaker line:** "Last semester's plan was to hand-label Indian roofs and fine-tune. We asked a harder question: can we do it without labelling any training data?"

---

## Slide: Objective

**Title:** Objectives

1. **Rooftop segmentation for Jaipur without Jaipur training labels**
   - Transfer from labelled foreign data using domain adaptation, weak labels and foundation models
   - Target: close the gap from 0.154 IoU toward supervised SOTA (0.83 on Inria)
2. **Honest evaluation on real Indian roofs**
   - Build a hand-labelled Jaipur test set (152 crops, calib/test split by tile)
   - Score every model and label source on it, instead of on footprint data
3. **Find the method that actually works for Indian cities**
   - Compare source-only, image translation, classical UDA, weak supervision (Open Buildings) and foundation-model self-training on the same test
4. **A light, deployable model for city scale**
   - Distil the large model into a small U-Net that runs over all 175 km² of Jaipur
5. **From pixels to energy**
   - Roof area → capacity (kWp) → energy (kWh/yr), using Global Solar Atlas PVOUT
   - Report an error budget, and estimate existing rooftop PV with the panel model (M₂)

**Speaker line:** "Our objective moved from 'segment roofs' to 'segment Indian roofs with no Indian training labels, prove it on real hand labels, and turn it into an energy number'."

---

## Slide: State of the Art in Our Domain

**Title:** State of the Art: Building and Rooftop Segmentation

- **Supervised, clean labels (upper bound)**
  - Inria (0.3 m, US and Austrian cities): UANet 2024 **0.831 IoU**, UAGLNet 0.837
  - WHU (Christchurch, NZ): DeepSwinLite 2025 **~0.92 IoU**
  - Both need thousands of hand labels from the test city
- **Zero-shot foundation models**
  - Grounded-SAM on WHU: **0.71 IoU**; SAM + CLIP: 0.49 (ISPRS 2025)
  - CrossEarth (arXiv 2410.22629): remote-sensing foundation model (frozen DINOv2 + Reins adapters), no target adaptation
- **Misaligned or weak footprint labels (2026)**
  - OMAF (CVPR 2026): per-building offset correction on Google Earth imagery, +9.22 mIoU from self-alignment
  - Align-and-Segment (2026): real Google Earth 0.54 → 0.62 IoU†
- **Domain adaptation**
  - DAFormer → HRDA → MIC, and DUDA (WACV 2026): SOTA on street scenes
  - Rarely tested on remote sensing with a large resolution gap
- **Indian work is scarce**
  - Rangdal 2025: rural India, supervised YOLOv11, 0.61 mIoU
  - Singla & Kalase 2024: Indian cities, needs elevation data (DSM)
  - Chandigarh 2025: had to combine foreign datasets for training
  - **No public hand-labelled urban Indian roof benchmark found**
  - **No published India accuracy for Google's Open Buildings**

| Setting | Best known | Labels from target city |
|---|---|---|
| Supervised (Inria / WHU) | 0.83 / 0.92 | Thousands |
| Zero-shot (Grounded-SAM, WHU) | 0.71 | None |
| Rural India (supervised) | 0.61 mIoU | Hand labels |
| **Ours (urban Jaipur)** | **0.717** | **None used for training** |

**Speaker line:** "Top scores come from clean labels in Western cities. Without labels, the field is around 0.7, and urban India has no benchmark at all."

---

## Slide: Novelty: What We Did Differently

**Title:** Our Contributions

- **First hand-labelled evaluation of rooftop segmentation in an Indian city**
  - Jaipur at 26.6 cm/px, 152 hand-drawn crops split by tile into calib/test (76/76)
  - Likely the **first city-scale check of Google Open Buildings in India**; it scores only **0.50 IoU** against real roofs
- **Self-trained a foundation model on an Indian city with no Jaipur labels used for training**
  - CrossEarth zero-shot 0.677 → **0.717** after self-training on 8,920 unlabelled Jaipur tiles
  - Only the adapters and the head are trained; the backbone stays frozen (one V100, about 65 min)
- **Calibrated pseudo-labels**
  - Roof cut-off at 82% precision, background cut-off at 98.6%, uncertain pixels ignored
  - CrossEarth's outputs only span [0.27, 0.73], so standard thresholds break
- **Light, deployable student**
  - ResNet34 student: 0.706 ± 0.005 over 3 seeds
  - Matches the teacher with TTA in a single pass, so a city-wide run is **8× cheaper**
- **Key finding: the choice of source model matters more than the adaptation method**
  - Classical UDA from a small Western model: 0.04–0.33
  - A foundation-model source plus self-training: 0.717
- **Full pipeline to energy:** Jaipur roof potential ≈ **5.3–6.0 GWp, 8.2–9.3 TWh/yr**

**Why it's publishable:**
- New setting and evaluation: an urban Indian roof benchmark, which doesn't exist yet
- New application: label-free foundation-model self-training on a Global South city
- An honest empirical lesson for Global South mapping: source choice beats adaptation method
- Pitch as an **empirical / application paper**, not a new algorithm
- Venues: CVPR EarthVision, IGARSS, NeurIPS Climate Change AI workshop, IEEE JSTARS

**Speaker line:** "Our method components are known. What's new is the setting, the benchmark, and the finding that a strong source with self-training is what works for Indian cities."

---

## Slide: Where We Stand

**Title:** Results vs State of the Art

**Same test set and protocol (fair comparison):**

| Method | Test IoU |
|---|---|
| Google Open Buildings masks | 0.499 |
| CrossEarth zero-shot, S2A | 0.664 |
| CrossEarth zero-shot, A2S | 0.677 |
| U-Net trained on Open Buildings | 0.706 |
| **Ours: self-trained CrossEarth** | **0.717** |

**Gains:**
- **+0.040 over CrossEarth** (+0.064 single pass)
- **+0.21 over Google's Open Buildings**
- 0.154 → 0.717 from the source-only model, about 4.6×
- Our runs of DUDA and other UDA baselines: 0.27–0.38, even with the best threshold picked after the fact

**Against clean-label SOTA (scale only, different datasets):**

| Reference | IoU | Ours as % |
|---|---|---|
| Inria SOTA (supervised) | 0.831 | **86%** |
| WHU SOTA (supervised) | ~0.92 | 78% |
| Grounded-SAM zero-shot (WHU) | 0.71 | ≈ equal, on harder data |

**Why we're a bit lower than supervised SOTA:**
- **No Jaipur training labels.** Fine-tuning on 76 hand crops added only about +0.003, so a few labels don't close the gap
- **Errors are on edges.** 93% of wrong pixels are on roof boundaries; ignoring a 3 px (80 cm) band gives **0.776**
- **Off-nadir Google imagery.** Roofs are displaced from their footprints; footprints match roofs at only 0.58 IoU even when perfectly aligned
- **Annotator style** differs, so part of the gap is label disagreement
- **Global South cities score lower everywhere.** SpaceNet, same winning method: F1 0.885 Las Vegas vs 0.544 Khartoum (a 0.34 drop)

**Speaker line:** "On the same Jaipur test we beat both the foundation model and Google's own building data. Against Western supervised SOTA we reach 86%, with no Jaipur training labels."

---

## Speaker notes (if examiners ask)

- 0.717 is a single run; its +0.011 lead over the student is within the confidence interval
- Threshold and pseudo-label cut-offs were set on the 76 calib crops
- CrossEarth's source model was trained on labelled WHU data
- Imagery licence unresolved: tiles look Google-sourced, and Google's terms forbid ML training
- Energy figure: this deck uses the November figure calibrated on hand labels (5.3–6.0 GWp);
  `sota_standouts.md` still quotes the September 4.59 GWp / 7.62 TWh/yr. Use one figure throughout
