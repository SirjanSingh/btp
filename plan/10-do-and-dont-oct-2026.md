# 10. What to do, what not to do (updated 19 Oct 2026)

This page answers four questions: **what should we work on next**, **which paper to use as
the baseline** (section 4), **can the new Jev 1.14 image model help** (section 3), and **how to use the campus image (`daraset/projs.tif`)**
(section 2). It builds on the measured results from September
(`experiments/` and `.planning/MASTER_CONTEXT.md` on `origin/feat/init-project-setup`) and on
`plan/01`–`09`.

> **Short version.** The project is now about estimating **solar energy (kWh per year)**,
> and making segmentation slightly better is not the goal any more. The biggest gap is
> that we have **zero real labels and zero real ground truth** for India. The campus image
> is the cheapest way to fix that, so use it as a **test site**, never as training data.
> The new **Jev 1.14** image model can help as a *checker* (PV or water heater? usable
> roof fraction?), but it cannot measure roof area and its answers are not ground truth.

---

## 1. Where things stand

| What | Number | Status |
|---|---|---|
| Rooftop model on Jaipur, trained on Open Buildings footprints (weak labels) | IoU **0.648** vs 0.142 for the AIRS-only model | measured |
| Domain-adaptation tricks (FDA, histogram matching) | made results **worse** | measured |
| Self-training on Jaipur | rejected after 4 tries | measured |
| First city-wide energy figure for Jaipur | **4.59 GWp, 7.62 TWh/yr, ±15.2%** | measured, but rests on assumptions |
| Biggest source of error in that figure | `k_usable` (assumed 0.60) = **68%** of the error bar; segmentation = 2% | measured |
| Hand labels on Indian imagery | **0** (30 crops sent out, none returned) | open |
| Solar panel stage on Jaipur | not started | open |
| Licence for the Google imagery | unresolved | open, see `plan/07` |

What this means: better segmentation barely moves the final number. What moves it is
**knowing how much of a roof is actually usable** (`k_usable`) and **having some ground
truth at all**.

---

## 2. The campus image: `daraset/projs.tif`

### What it is (checked from the file, 19 Oct 2026)

| Property | Value |
|---|---|
| Area shown | **LNMIIT campus** and the fields around it |
| Centre | 26.9354° N, 75.9237° E |
| Size | 3924 × 2092 px, about **1.05 km × 0.56 km (≈0.59 km²)** |
| Projection | EPSG:3857 (Web Mercator), same as the Jaipur tiles |
| Ground resolution | 0.30 Mercator m × cos(26.94°) = **0.267 m/px**, almost the same as Jaipur (0.266) |
| Overlap with the 16 Jaipur tiles | **None.** The Jaipur mosaic ends about 9 km to the west. |
| Likely source | Google satellite (same zoom-19 Web Mercator export as the Jaipur tiles) |

Because it matches the Jaipur tiles in resolution and looks, but lies outside them, it works
as a **held-out test site**: the model has never seen it.

### Yes, use it. Here is what it is good for

1. **The first real accuracy number on Indian imagery.** The campus has maybe 40 to 60
   buildings. Drawing every rooftop by hand in QGIS or CVAT takes a few hours. That gives a
   real rooftop IoU, which no amount of training on weak labels can provide.
2. **Measuring `k_usable` directly.** This is the biggest win. On each campus roof, mark
   what cannot hold panels: water tanks, stairwell boxes, AC units, parapets, shaded parts.
   Usable area divided by total roof area gives a **measured** `k_usable` from real Indian
   flat roofs, in place of the 0.60 guess that makes up 68% of the error bar.
3. **Real ground truth for energy, which nothing else gives us.** Ask the LNMIIT estate
   office: how many kWp of rooftop solar is installed on campus, on which buildings, and how
   many kWh it generated last year. The pipeline predicts capacity and kWh/yr for the same
   buildings, so for the first time we can compare a prediction with a real meter reading.
   These are LNMIIT's own numbers, not Google data, so they carry no licence problem.
4. **A first check of the solar panel stage.** Some campus roofs appear to have panel
   arrays. We can walk over and confirm which roofs really have PV and which have water
   heaters or tanks instead. This gives real positives and real hard negatives, which is
   exactly the precision test that `MASTER_CONTEXT` says matters most.
5. **A clean figure for the report.** Show input → rooftop mask → usable area → panels →
   kWh/yr on our own campus. Examiners will recognise the place, and every step can be
   checked by walking over.

### Things to watch out for

- **Do not train on it.** Keep it purely as a test set. Once it is used in training, the
  only independent check we have is gone.
- **The campus is not typical Jaipur.** It has large flat institutional roofs, lots of trees
  and wide spacing. It does not stand in for dense residential areas like the walled city.
  Report it as **one validation site** next to labelled Jaipur crops, not in place of them.
- **The licence problem still applies.** The image is most likely Google, and Google's
  terms forbid using Maps content to test or validate ML models, not only to train them
  (`plan/07`). Our position is better than with the Jaipur tiles, though: LNMIIT owns the
  campus, so a **drone survey of the campus** is easy to get permission for and produces a
  properly licensed image of the same place. Plan: develop on `projs.tif` now, and report
  final numbers on a drone image if one can be arranged.
- **Read the ground resolution from the file. Never hard-code it.** The value above
  (0.267 m/px) is already corrected by cos(lat).

### The other files in `daraset/mx/`

- `BAND.jpg` and `BAND111111.jpg` are **the same file**: a 514 × 516 grey preview of a whole
  satellite scene. At that size each pixel covers a large patch of ground. They cannot
  show rooftops, so **do not use them**.
- `C3-MX/` is **empty**. If real Cartosat-3 data turns up: its multispectral (MX) bands are
  about 1.1 m/px, too coarse for rooftops on their own. The panchromatic (PAN) band is
  about 0.25–0.28 m/px, similar to our Jaipur tiles
  ([NRSC Cartosat-3 brochure](https://www.nrsc.gov.in/sites/default/files/pdf/training/C3_BROCHURE_JAN2021_modified.pdf),
  [eoPortal](https://directory.eoportal.org/satellite-missions/cartosat-3)). Pan-sharpened
  Cartosat-3 from NRSC/Bhuvan is the most promising **licensed** replacement for the Google
  imagery (option 1 in `plan/07`). Worth asking the department whether it can get a scene
  covering Jaipur.

---

## 3. Jev 1.14, the new image model: can we use it for deciding things?

### What it is (checked 19 Oct 2026)

TypeSafe AI released **Jev 1.14** on **14 Oct 2026**. Jev is a *typed-decision* model: you
ask a question and get back a typed answer (a choice from a list, a score, a yes/no-style
answer) instead of free text. Version 1.14 added three things, and **TypeSafe says none of
them is production-ready**:

- **Image input.** PNG/JPEG only, 256–1568 px per side. Only on `jev-1.13.0`, `jev-1.14.0`
  and `jev-preview`, and only when the request sends the header
  `TypeSafe-Danger-Zone-Images: true`.
- **A `Box` output type.** **One** bounding box per question, with edges as fractions of the
  image plus a confidence score. **No masks, no polygons, no multiple boxes.**
- **Longer context** (opt-in).

Price: $0.042 per million input tokens (images count as input tokens), and output is free.
It runs only as a hosted API, so it cannot run offline on the DGX.

Sources: [MarkTechPost](https://www.marktechpost.com/2026/10/14/typesafe-ai-jev-1-14-images-box-long-context/),
[Hermes AI](https://hermes-ai.net/news/typesafe-ai-s-jev-1-14-brings-vision-and-256k-context-to-typed-decisions/),
[runtimewire](https://runtimewire.com/article/typesafe-jev-1-14-danger-zone-experiments),
[madewithjev.com (community page on Jev images)](https://madewithjev.com/jev-images).
TypeSafe's own docs could not be fetched, so check them before building on any detail above.

### The key point

**Jev decides. It does not measure.** Our energy number comes from **roof area in m²**,
which needs a pixel mask. Jev returns at most one box per question, so it **cannot replace
the U-Net** for rooftops or panels. It fits as a **checker/decider stage on top of** the
segmentation pipeline.

### Where it fits (best first)

1. **Telling PV apart from water heaters and tanks (Stage 2 precision).** For each
   panel-like blob the solar model finds, crop the roof (it can be upscaled to 256 px, the
   minimum) and ask Jev a Choice question: `PV panel / solar water heater / water tank /
   other`. Water heaters look almost the same as PV from above and are more common in
   India, so this is exactly the false-positive problem `MASTER_CONTEXT` flags as the
   biggest Stage 2 risk.
2. **Finding PV roofs to label.** PV is rare in Jaipur, so finding positive examples is the
   bottleneck for labelling. Ask "is there rooftop PV in this tile?" over the 11,140 Jaipur
   tiles, then hand-check only the tiles it flags. That makes the labelling batch much
   richer in positives. The model only **picks** which tiles to look at; a human still draws
   the labels.
3. **Rough `k_usable` per roof, as an experiment.** Ask for a Score: "what fraction of this
   roof is free of tanks, stair boxes, AC units and shade?". Compare with the hand-measured
   `k_usable` from the campus (section 2). If it agrees, it becomes a city-wide `k_usable`
   estimate, and that attacks the 68% error term directly. If it doesn't, drop it.
4. **Checking weak labels.** Overlay an Open Buildings outline on the tile and ask
   "does this outline match one roof / several merged roofs / no building?". This can flag
   bad weak labels before training.

### What not to do with it

- **Don't use Jev boxes as areas**, and don't count panels from them. One box per question
  is far too coarse.
- **Don't use its answers as ground truth, or as training labels without a human check.**
  TypeSafe itself says a valid box does not prove the object is there, and that the
  confidence comes from the answer distribution, not from looking. A widely shared post is
  titled ["Jev can't be calibrated"](https://www.alexmolas.com/2026/09/23/jev-cant-be-calibrated.html).
  So don't report its confidence as a probability.
- **Don't build the thesis on it.** It is an experimental preview feature behind a
  "Danger Zone" header, and the `jev-preview` / `jev-latest` aliases change version without
  notice. If we use it, pin **`jev-1.14.0`** and record the version and date of every run,
  or the results can't be reproduced.
- **Don't send it Google imagery until the licence question is settled.** Uploading Google
  tiles to a third-party API to make decisions is another use the Google terms may not
  allow (`plan/07`). Test on the campus first, ideally on a drone image.
- **Validate before trusting.** Every use above needs a check against hand labels (campus
  roofs plus the 30 Jaipur crops) **before** any Jev-based number enters the report. With
  no labels, nobody can say whether Jev is right.

**Verdict:** worth a **small, cheap pilot** (use 1, then use 3) **after** the campus labels
exist. It is a filter and decider on top of the pipeline, not a replacement for it, and not
a source of ground truth.

---

## 4. Which paper to use as the baseline (checked 19 Oct 2026)

Our claim is: **solar energy for a whole Indian city, with zero target labels.** No single
paper does all of that, so use **one main paper to build on** and **one or two reference
points** for the parts it doesn't cover.

### Main baseline to build on: Kasmi et al. 2025

**G. Kasmi, L. Dubus, Y.-M. Saint-Drenan, P. Blanc, "Space-scale exploration of the poor
reliability of deep learning models: the case of the remote sensing of rooftop
photovoltaic systems", *Environmental Data Science*, 2025**
([arXiv 2408.07828](https://arxiv.org/abs/2408.07828),
[journal](https://www.cambridge.org/core/product/3415C8A022E481B7C955FE284D4DC0E5)).

Why this one:
- **Same data.** It uses **BDAPPV**, our Stage 2 training set, from the same group that
  released BDAPPV ([Kasmi et al. 2023, *Scientific Data*](https://arxiv.org/abs/2209.03726)).
- **Same problem.** It is about models breaking when the imagery changes. Trained on Google
  images, their ResNet-50 gets F1 **0.98** on Google but **0.46** on IGN, because recall
  drops to 0.32. That is the same Google→IGN shift we already measured (our IoU went from
  0.561 to 0.617 with self-training), and the same failure we expect going from France to
  Jaipur.
- **Same conclusion as our data.** Their fix is **augmentation**, not heavy adaptation
  methods: blurring raises F1 on IGN from 0.44 to 0.74, though with many false positives.
  We also found simple methods beat FDA, histogram matching and CBST.
- **Everything is public**, so we can reproduce it, not just quote it:
  [code](https://github.com/gabrielkasmi/robust_pv_mapping),
  [model weights](https://zenodo.org/records/12179554),
  [BDAPPV data](https://zenodo.org/records/7358126).
- **It is a classifier** ("is there PV in this image?"), which is exactly the
  PV / water-heater decision stage from section 3.

How to work on it:
1. **Reproduce their Table 1 / Table 3 numbers** (Google→IGN F1) with their code and
   weights. This checks our setup against a published number.
2. **Run their model on Jaipur and on the campus**, with hand-checked PV and water-heater
   roofs as the test set. That gives the first published-method number on Indian imagery.
3. **Add what they don't have**: a pixel mask (our Stage 2 U-Net) for panel area, hard
   negatives such as water heaters and tanks, and the energy chain on top.
4. **Our extension** is then easy to state: their reliability study goes from France to an
   Indian city, and from detection to kWh/yr with a measured `k_usable`.

### Reference points for the other parts (compare against these, don't build on them)

| Part of the pipeline | Compare against | What it gives us |
|---|---|---|
| Rooftop segmentation | **Google Open Buildings itself**, turned into a mask ([Sirko et al. 2021, arXiv 2107.12283](https://arxiv.org/abs/2107.12283)) | Costs nothing. Our model is trained on these footprints, so it **must beat them** against the hand labels, or training it was pointless. |
| Rooftop segmentation, zero-shot | **SAM 3** (`plan/09`, Priyal) | The strongest no-training baseline |
| The final energy number | **Joshi et al. 2021, *Nature Communications*** ([paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8492708/)), a global rooftop PV potential estimate | An independent check of our 7.62 TWh/yr for Jaipur. Check whether their gridded data can be downloaded for the Jaipur area. |
| Old source-domain number | AIRS / PSPNet 0.899 (Chen et al. 2019) | Background only. It is a different city and resolution, so **not** our baseline any more. |

### Papers not to use as the baseline

| Paper | Why not |
|---|---|
| Singla, arXiv 2411.04610 (Indian cities, satellite + DEM) | Needs a 1 m DEM and IMD ground-station data we don't have. The abstract reports no segmentation metrics. Licence CC BY-NC-ND. Cite it in related work only. |
| "Rooftop Solar Potential of an Indian Metropolis" (IEEE 9553088, cited in `MASTER_CONTEXT`) | **Could not be found on 19 Oct 2026.** Don't cite it until someone opens the actual paper. |
| Reliance Industries, arXiv 2501.02840 (rooftop PV in Maharashtra) | Classification on internal, unreleased data, so it can't be reproduced. Related work only. |
| DAFormer / HRDA / MIC | Built for street scenes. Our measurements say heavy domain adaptation lost to weak labels. Keep at most one as a comparator if DGX time is spare. |
| AIRS (Chen et al. 2019) as "the number to beat" | That was the old framing. The project is now about energy in Jaipur, not IoU on Christchurch. |

---

## 5. DO: in this order

| # | Task | Why | Effort |
|---|---|---|---|
| 1 | **Merge the branches.** `origin/feat/init-project-setup` (all the September work) and `origin/master` (SAM 3 plan, reading list) have diverged in both directions. Merge them and work from one branch. | People keep reading the stale checkout and drawing wrong conclusions | 30 min |
| 2 | **Hand-label the campus image**: rooftop outlines plus no-panel zones on each roof | First real IoU, first measured `k_usable` | 3–5 h |
| 3 | **Ask the estate office for installed solar kWp per building and last year's kWh** | The only real energy ground truth available to us | an email |
| 4 | **Run the current pipeline on `projs.tif`** and compare with steps 2 and 3 | Turns "±15%, assumed" into "checked against a real site" | 1 h once labels exist |
| 5 | **Get the 30 staged Jaipur crops labelled** (`labelling/r12_batch1`) | Dense residential ground truth, which the campus cannot give | split between both of us |
| 6 | **Read PVOUT / GTI / OPTA for Jaipur from Global Solar Atlas** and fix the PVOUT × PR double count | Still marked TODO; currently the losses are counted twice | 1 h |
| 7 | **Fix the BDAPPV zero-negatives bug** (`prep_bdappv.py` drops panel-free images), then start the solar stage | Every solar metric so far measures the wrong task | half a day plus a retrain |
| 7b | **Reproduce Kasmi et al. 2025** (Google→IGN F1 with their public code and weights), then run their model on campus and Jaipur PV roofs (section 4) | Our main baseline: a published, reproducible number to beat | 1–2 days |
| 8 | **Run SAM 3 zero-shot (`plan/09`) on the campus and the labelled crops** | Strong, cheap baseline. Meaningful only once steps 2 and 5 exist. | 1 day |
| 9 | **Pilot Jev 1.14 as a PV vs water-heater checker** on campus roof crops, then try its `k_usable` score against the hand-measured values (section 3). Pin `jev-1.14.0`. | Cheap precision filter for Stage 2; might give city-wide `k_usable`. Meaningful only once steps 2–4 exist. | 1 day |
| 10 | **Decide the imagery licence question** (drone, Bhuvan/Cartosat, or written permission) and write an "imagery provenance and licence" paragraph for the report | First question any reviewer will ask; the repo is public | ask the supervisor |

---

## 6. DON'T

| Don't | Why |
|---|---|
| **Train on `projs.tif`**, or tune thresholds on it | It is the only independent test site. Once used for tuning, it no longer tests anything. |
| **Keep tuning segmentation** (new backbones, more epochs, chasing PSPNet 0.899) | Segmentation is 2% of the energy error bar. `k_usable` is 68%. |
| **Spend more time on FDA, histogram matching, or CBST thresholds** | Measured: they made things worse or collapsed |
| **Frame the paper as "our UDA method beats SAM 3"** (as `plan/09` §1.3 does) without new evidence | Our own measurements say weak supervision from Open Buildings won and the adaptation tricks lost. Build the story on what was measured: weak labels + measured `k_usable` + checked energy. |
| **Put predicted numbers into the report** (e.g. the SAM 3 IoU ranges "0.55–0.65" in `plan/09`) | Those are guesses. Only measured numbers go in. |
| **Hard-code ground resolution** (0.075, 0.10, 0.30) | Always read it from the GeoTIFF and multiply by cos(lat). Getting this wrong once already meant a 12.6× error in area. |
| **Quote any solar-panel metric** before the zero-negatives fix | The model has never seen a roof without a panel |
| **Combine PVOUT with PR**, or apply losses twice in any other way | Already caught once. Use GTI × PR, or PVOUT × roof derate, never both. |
| **Push model weights or derived masks trained on Google imagery to the public repo** | Licence problem. Keep them on the DGX/Drive until the licence question is settled. |
| **Use Jev 1.14 for areas, as ground truth, or as unchecked training labels**, or send it Google tiles before the licence is settled | It returns one box per question and its confidence isn't calibrated. It is an experimental preview, not production-ready. |
| **Use `BAND.jpg` or anything in `daraset/mx/`** for rooftops | It is a scene preview at a few hundred pixels across, so no rooftops are visible |
| **Rewrite git history** (`filter-branch`, force-push) | It already destroyed two DGX checkpoints in September |
| **Trust the local `D:` checkout without `git fetch`** | It has been stale twice and produced wrong "doesn't exist" claims both times |
