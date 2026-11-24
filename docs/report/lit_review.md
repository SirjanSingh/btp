# Related work and novelty

Written 2026-11-21. How each citation was checked: the arXiv export API (title, authors, dates, comments),
Crossref (DOIs, venues), OpenAlex, and web search. dblp was not queried automatically, but a hand
check from 2026-11-20 is in `experiments/2026-11-20-toward-higher-iou/dblp_check.md`. Search window
for the related-work threads: 2022 to Nov 2026, English only, arXiv plus Crossref-indexed journals and
proceedings. Older papers appear only as background and are labelled that way. "One-liners" summarise
abstracts or our earlier notes. Any number from a search snippet or a machine summary is tagged
*(unverified)*.

## 1. Citation verification

| # | Citation as requested | Status | Canonical link | Notes |
|---|---|---|---|---|
| 1 | Ronneberger, Fischer, Brox. U-Net. MICCAI 2015 | VERIFIED (Crossref, arXiv) | https://doi.org/10.1007/978-3-319-24574-4_28 · arXiv 1505.04597 | LNCS 9351, pp. 234-241 (dblp hand check) |
| 2 | He, Zhang, Ren, Sun. Deep Residual Learning. CVPR 2016 | VERIFIED (Crossref, arXiv) | https://doi.org/10.1109/CVPR.2016.90 · arXiv 1512.03385 | |
| 3 | Sirko et al. Continental-Scale Building Detection from High Resolution Satellite Imagery. 2021 | VERIFIED (arXiv API) | https://arxiv.org/abs/2107.12283 | arXiv only; no peer-reviewed version found |
| 4 | Kaiser, Wegner, Lucchi, Jaggi, Hofmann, Schindler. Learning Aerial Image Segmentation From Online Maps. IEEE TGRS 55(11), 2017 | VERIFIED (Crossref) | https://doi.org/10.1109/TGRS.2017.2719738 · arXiv 1707.06879 | |
| 5 | Mnih, Hinton. Learning to Label Aerial Images from Noisy Data. ICML 2012 | VERIFIED (OpenAlex; ICML PDF returns HTTP 200) | http://icml.cc/2012/papers/318.pdf | No DOI (ICML 2012 has none) |
| 6 | Girard, Charpiat, Tarabalka. Noisy Supervision for Correcting Misaligned Cadaster Maps Without Perfect Ground Truth Data. IGARSS 2019 | VERIFIED (Crossref, arXiv) | https://doi.org/10.1109/IGARSS.2019.8898071 · arXiv 1903.06529 | |
| 7 | Loshchilov, Hutter. Decoupled Weight Decay Regularization. ICLR 2019 | VERIFIED (arXiv comment "Published as a conference paper at ICLR 2019"; dblp hand check) | https://arxiv.org/abs/1711.05101 · https://openreview.net/forum?id=Bkg6RiCqY7 | The OpenReview id comes from recall and was not opened. The arXiv link is the safe one |
| 8 | Xie et al. SegFormer. NeurIPS 2021 | VERIFIED (NeurIPS proceedings page title matches; arXiv) | https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html · arXiv 2105.15203 | |
| 9 | Kirillov et al. Segment Anything. ICCV 2023 | VERIFIED (Crossref) | https://doi.org/10.1109/ICCV51070.2023.00371 · arXiv 2304.02643 | 12 authors |
| 10 | Siméoni et al. DINOv3. 2025 | VERIFIED (arXiv API) | https://arxiv.org/abs/2508.10104 | 26 authors. arXiv only; no venue found |
| 11 | Kang et al. DUDA: Distilled Unsupervised Domain Adaptation for Lightweight Semantic Segmentation | VERIFIED (Crossref, arXiv) | https://doi.org/10.1109/WACV61042.2026.00784 · arXiv 2504.09814 | **WACV 2026 is confirmed by the Crossref DOI.** The 2026-11-20 dblp note (CoRR only) is out of date |
| 12 | Kasmi, Dubus, Saint-Drenan, Blanc. Space-scale exploration of the poor reliability of deep learning models: the case of the remote sensing of rooftop photovoltaic systems | VERIFIED (Crossref, arXiv) | https://doi.org/10.1017/eds.2025.13 · arXiv 2408.07828 | arXiv preprint 2024; journal version in *Environmental Data Science* 4 (2025) e22. Cite the journal, 2025 |
| 13 | "Maggiolo" on the Inria aerial benchmark | VERIFIED under the correct name: **Maggiori**, Tarabalka, Charpiat, Alliez. Can semantic labeling methods generalize to any city? The Inria aerial image labeling benchmark. IGARSS 2017 | https://doi.org/10.1109/IGARSS.2017.8127684 | "Maggiolo" is a misspelling. The author list comes from recall; Crossref confirmed the first author and title only |
| 14a | OMAF: Xu, Liu, Zeng, Wu, Qu, Zhang. Revisiting the Necessity of Full Accuracy: Weakly Supervised Object-Level Offset Correction for Misaligned Building Labels. CVPR 2026 | VERIFIED (CVPR 2026 virtual poster page, read directly) | https://cvpr.thecvf.com/virtual/2026/poster/38531 · https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Revisiting_the_Necessity_of_Full_Accuracy_Weakly_Supervised_Object-Level_Offset_CVPR_2026_paper.html | The CVF open-access page returns 403 to automated fetches. Datasets: Islahiye and Antakya |
| 14b | Guthula et al. Align and Segment: Unsupervised Learning for Building Segmentation From Misaligned Labels | VERIFIED (arXiv API) | https://arxiv.org/abs/2607.10841 | arXiv July 2026; 8 authors; no venue yet |

Every requested citation exists. Two had wrong details: the author name in #13 is Maggiori, and the
venue in #11 is WACV 2026 (now confirmed). One correction to our own notes: arXiv 1612.01337 is
Marmanis et al., *not* Kaiser. Kaiser's arXiv ID is 1707.06879.

## 2. Closest prior work, 2022-2026

### (a) Training on noisy or misaligned map and footprint labels, with alignment correction

| Paper | What they did | How we differ |
|---|---|---|
| Xu et al., OMAF, CVPR 2026 (14a above) | Object-level offset pseudo-labels from a prior-regularised self-alignment, then an instance-level offset regressor, on non-orthorectified Google Earth imagery. They report up to +40.6% mIoU over baselines, and the self-alignment contributes 9.22 in ablation (poster page) | Their per-object correction subsumes ours. Our per-building template match (+0.004 label IoU over the global shift) is a weaker version of the same idea. We add evaluation against *roof* outlines in an Indian city and a comparison with UDA |
| Guthula et al., Align and Segment, arXiv 2607.10841 (2026) | A transformation net moves each label patch onto the segmenter's prediction during training, with no clean labels. The segmenter is a U-Net with a DINOv3 ConvNeXt-T encoder | Our "move the label onto the model prediction" step is their mechanism, done offline. Our DINOv3 ConvNeXt run copies their encoder choice |
| Li et al., DragOSM, arXiv 2509.17951 (2025) | Aligns historical OSM labels to roofs and footprints in off-nadir imagery. Releases ReBO, with hand-corrected roof and footprint polygons | Works on polygons. Shows that roof and footprint differ in off-nadir views, which is the core of our footprint-to-roof gap |
| Usmani, Bovolo, Napolitano. Remote Sensing 15(18) 4639, 2023. https://doi.org/10.3390/rs15184639 | CNN-based analysis of OSM noise: misalignment, missing and outdated buildings | Focuses on diagnosing OSM. We measure label quality against hand roofs and train on the result |
| Liu et al., AIO2, IEEE TGRS 2024. https://doi.org/10.1109/TGRS.2024.3373908 · arXiv 2403.01641 | Online correction of incomplete object labels with a mean teacher | Targets missing objects, not offset or shape. Could address OB's missing roofs |
| Song, Yang, Jung. Self-Filtered Learning. IEEE JSTARS 16, 2023. https://doi.org/10.1109/JSTARS.2022.3230625 | Multi-round, loss-based filtering of noisy OSM and Microsoft footprint labels | A label-cleaning baseline we have not run |
| Sirko et al. 2023, High-Resolution Building and Road Detection from Sentinel-2, arXiv 2310.11622 | Google's own follow-up: Sentinel-2 student models distilled from high-resolution teachers | Shows that Open Buildings labels are themselves model output, which matters for our weak-vs-UDA claim |
| *Background (pre-2022):* Kaiser 2017; Girard 2019; Fobi, Conlon, Taneja, arXiv 2005.13180 (ACM COMPASS 2020), an alignment-correction network using a few verified labels | | |

### (b) SAM and foundation-model label refinement for buildings

| Paper | What they did | How we differ |
|---|---|---|
| Wang et al., SAMRS, NeurIPS 2023 Datasets & Benchmarks, arXiv 2305.02034 | Converts existing detection boxes into segmentation labels with box-prompted SAM, at dataset scale | The closest precedent for our "OB polygon box to SAM mask" step. They do not gate label quality against hand masks |
| Kim & Kim, Remote Sensing 16(3) 526, 2024. https://doi.org/10.3390/rs16030526 | Unsupervised building segmentation: SAM pseudo-labels, then edge-driven weakly supervised refinement | No external footprint prior. We prompt SAM with existing footprints |
| Mayladan et al., Zero-Shot Refinement of Buildings' Segmentation Models using SAM, arXiv 2310.01845 | Uses a pretrained building model's outputs as SAM prompts to refine its masks | Similar spirit to our "per-building shift + SAM" (0.650). They refine predictions, not training labels |
| Ma et al., SAM-assisted RS segmentation with object and boundary constraints, arXiv 2312.02464 | Adds SAM-derived object and boundary losses to semantic segmentation training | An alternative to our Sirko edge weighting, aimed at the same boundary error |
| Chen et al., RSPrompter, IEEE TGRS 2024. https://doi.org/10.1109/TGRS.2024.3356074 | Learns prompts for SAM-based remote-sensing instance segmentation | Needs labelled training data |
| Yin et al., SAMST, IGARSS 2025, arXiv 2507.11994 | Filters and refines pseudo-labels with SAM for semi-supervised remote-sensing segmentation | Semi-supervised with clean seed labels; we have none |
| Dabaja & Celik, SAM 3 in remote sensing, arXiv 2607.09583 (2026) | Zero-shot text and visual prompting of SAM 3 on aerial benchmarks | Supports box prompts over text prompts. Our earlier note's numbers from this paper are *(unverified)* |

### (c) Rooftop solar potential from imagery, India and the Global South

| Paper | What they did | How we differ |
|---|---|---|
| Batchu et al., Satellite Sunroof, arXiv 2408.14400 (2024) | Google predicts DSMs and roof segments from satellite imagery to extend the Solar API to the Global South | The strongest competitor for an energy-estimate claim. It is industrial-scale with proprietary training data; ours uses only open labels and a published, held-out hand-label test set |
| Singla & Kalase, arXiv 2411.04610 (2024; also an EarthArXiv preprint, not peer reviewed) | Rooftop solar potential for Indian cities from 50 cm imagery plus a 1 m DSM | Has a DSM and models shading; we have 2D roofs only. Cities and totals not read *(unverified)* |
| Rangdal et al., ISPRS Annals X-5/W2-2025, 537-542. https://doi.org/10.5194/isprs-annals-X-5-W2-2025-537-2025 | Rooftop segmentation in rural India (YOLOv11 per search snippet, *unverified*) | Rural setting, supervised labels, no energy estimate |
| Zhang et al., Worldwide rooftop PV may mitigate global warming, Nature Climate Change 2025. https://doi.org/10.1038/s41558-025-02276-3 | Global rooftop area from a ViT on 1.2 m imagery, extrapolated with random forests | Global and coarse. We are city-scale at 26.6 cm |
| Li, Wang et al., SolarSAM, arXiv 2407.00296 (2024) | Text-prompted SAM roof segmentation plus PV modelling for a Chinese city | The arXiv record lists one author (G. Wang); the journal version (Renewable Energy 237, 2024, per repository listings) lists more. Journal DOI *not verified* |
| Krapf et al., RID roof dataset, Remote Sensing 14(10) 2299, 2022. https://doi.org/10.3390/rs14102299 | A roof and superstructure dataset for PV potential, with an annotator-agreement experiment (mean IoU 0.15-0.70 by class, per search snippet, *unverified*) | The only annotator-agreement study for roofs we found. It is relevant to our annotator-style finding |
| *Background (pre-2022):* Joshi et al., Nature Communications 2021, https://doi.org/10.1038/s41467-021-25720-2 | Global rooftop PV potential; names footprint data as a key error source | |

**Gap:** we found no peer-reviewed paper that estimates city-scale rooftop PV for an Indian city from
open footprint labels and also validates roofs against held-out hand labels. Two lower-quality items
from search, Sharma & Maheshwari (ISES SWC 2023) and Rastogi (UN-GGIM slides), were seen only as
snippets and are not cited.

### (d) Self-training and pseudo-label refinement for weakly supervised remote-sensing segmentation

| Paper | What they did | How we differ |
|---|---|---|
| Sirko et al. 2021 (background) | Soft teacher labels with a KL loss beat hard pseudo-labels | Our failed self-training arms used hard labels from an out-of-domain (AIRS) source. Soft in-domain self-training is untested |
| Liu et al., AIO2, TGRS 2024 | A mean teacher supplies pseudo-labels to correct incomplete labels | A direct candidate for our "predictions are better labels" iteration |
| Song et al., SFL, JSTARS 2023 | Multi-round sample filtering on OSM and Microsoft labels | Loss-based cleaning, not prediction replacement |
| Yin et al., SAMST, 2025 | SAM-filtered pseudo-labels | Needs labelled seeds |
| Girard et al. 2019 (background) | Multi-round correction; correcting the *original* labels each round works better than correcting the previous output | Bears directly on how to iterate our label gate |
| Dixit et al., Cross Pseudo Supervision for sparsely labelled geospatial images, arXiv 2408.02382 (2024) | Semi-supervised land-use and land-cover mapping on Indian cities; Google and Microsoft footprints used only for evaluation | India, but land-use classes and partial human labels |

### (e) Remote-sensing foundation models relevant to sub-metre buildings

| Paper | What they did | Relevance |
|---|---|---|
| Siméoni et al., DINOv3, arXiv 2508.10104 (2025) | Self-supervised backbone, including a variant pretrained on Maxar satellite imagery | Our running ConvNeXt-T experiment. A CVPR 2026 MORSE workshop paper (Faulkenberry & Prasad, arXiv 2605.03175) reports that the satellite variant does not beat the web-image variant on GEO-Bench *(per search summary, unverified)* |
| Jakubik et al., Prithvi, arXiv 2310.18660 (2023) | HLS (30 m multispectral) foundation model | Wrong resolution for 26.6 cm roofs |
| Cong et al., SatMAE, NeurIPS 2022, arXiv 2207.08051 | Temporal and multispectral MAE | Same resolution mismatch |
| Reed et al., Scale-MAE, ICCV 2023, arXiv 2212.14532 | Scale-aware MAE, including sub-metre imagery | The most resolution-relevant MAE. Not tried |
| Guo et al., SkySense, CVPR 2024, arXiv 2312.10115 | Multimodal RS foundation model | Weights are restricted *(unverified)* |
| Marsocci et al., PANGAEA, arXiv 2412.04204; Lacoste et al., GEO-Bench, arXiv 2306.03831; Simumba et al., GEO-Bench-2, arXiv 2511.15658 | Benchmarks of geospatial foundation models. GEO-Bench-2 includes SpaceNet building tasks | Show that geospatial foundation models do not reliably beat general backbones. This supports our "encoder size doesn't matter" (r34 0.697 vs MiT-B2 0.700) |
| Clay | **No paper located.** Model cards and documentation only | Do not cite as a paper |

## 3. Novelty map

The brief's numbers are used as given. Each label-gate number is the IoU of a label source against the
hand roofs on the test half. A note on "zero labels": the calib hand crops set both the global offset
and the threshold. The honest phrasing is "zero target-domain *training* labels, plus 76 calibration
crops".

| Claim | Rating | Reasoning |
|---|---|---|
| U-Net/ResNet34, BCE+Dice, AdamW, dihedral TTA, threshold chosen on calib | (i) Standard | Textbook recipe |
| Training on Open Buildings footprints as weak labels | (i) Standard | The Kaiser 2017 line of work. Also partly distilling Google's own detector (Sirko 2021, 2023) |
| Single global footprint-to-roof offset (7 S / 3 W), +0.038 test IoU | (ii) Incremental | It is the simplest case of Girard 2019 and OMAF. It was **measured against hand roofs** (calib only; re-measuring on calib reproduces it), so it is *not* label-free. Only the per-building variant is label-free |
| Per-building template-match alignment to the model prediction (label 0.606 to 0.610) | (i)/(ii) | This is AnS's mechanism, done offline. OMAF does object-level offsets at CVPR 2026. The +0.004 gain is negligible. It is a useful negative result: once the global offset is applied, the remaining position error is not the bottleneck; shape is |
| Box-prompted SAM refinement of footprints, with a label gate against hand roofs *before* training (0.642; with per-building shift 0.650) | (ii) Incremental, but the gate is the defensible part | SAMRS and Mayladan et al. already do box- or prediction-prompted SAM. What we found no precedent for is a pre-training *label-quality gate* that ranks label sources by IoU against held-out hand roofs, as a cheap proxy for which labels to train on. Whether the gate predicts downstream IoU is shown only once the three refined-label runs finish |
| The model's TTA prediction (0.723) beats every refined footprint label (best 0.650) | (ii) Diagnostic, not yet a result | Expected from Sirko's self-training and Girard's multi-round work. It becomes a claim only if retraining on predictions (soft labels, refreshed from the original OB labels per Girard) beats 0.706 |
| Weak supervision (0.697-0.706) far above seven UDA methods (0.04-0.33) on the same hand-label benchmark | (iii) Plausibly publishable, **but confounded** | The comparison is not like-for-like. OB labels are in-domain output of a detector trained on huge labelled data; the UDA arms adapt from an out-of-domain source (AIRS). UDA at 0.04-0.33 will look like broken baselines unless we report (1) AIRS source-only on Jaipur, (2) AIRS+OB combined, and (3) an OB-supervised UDA oracle. With those, "for Global South cities, open footprint products beat UDA from Western benchmarks" is a useful practical finding. Iqbal & Ali (arXiv 2007.02277, 2020) is the nearest earlier comparison, for built-up areas |
| Annotator-style bias (hand/OB area ratio 1.02 vs 1.24-1.37); 93% of error on edges; a 3 px edge band caps IoU at 0.776 | (ii)/(iii) | RID (2022) measured annotator agreement for roof superstructures, and BuildMapper assumes a 2-3 px edge error. We found no study that quantifies how annotator *style* changes the apparent ranking of building models. Needs a double-annotated subset (the same crops by two or three annotators) to turn into a claim |
| End-to-end GWp/TWh estimate (4.59 GWp / 7.62 TWh/yr ±15%) for Jaipur from open labels only | (ii) Incremental as research; good as an application | Satellite Sunroof covers the Global South at scale. Singla & Kalase cover Indian cities with a DSM. Our differentiator is an open-label pipeline with a held-out roof test set and propagated uncertainty. The ±15% needs a component breakdown (roof IoU and bias, the usable-area factor, irradiance) |

## 4. Paper framings and venues

**Framing A: "Open footprints vs domain adaptation for roof segmentation in a Global South city."**
Venues: CVPR EarthVision workshop, IGARSS, or the NeurIPS Climate Change AI workshop (fits the energy
angle). Extra experiments: AIRS source-only, AIRS+OB, and an OB-supervised oracle for UDA, so the
0.04-0.33 numbers are interpretable. A second city or tile set, if possible. Report paired bootstrap CIs
throughout (already done).

**Framing B: "Which weak label should you train on? A label-quality gate for footprint-derived roof
labels."** Venues: ISPRS Annals/Archives (Geospatial Week) or IGARSS; extended to IEEE JSTARS.
Extra experiments: finish the three refined-label training runs and show the gate's ranking predicts
downstream test IoU. Add iterative soft-label self-training from the 0.723 predictions (refreshed from
original OB each round) against OMAF/AnS-style per-object alignment. Add annotator agreement on a
double-labelled subset.

**Framing C: "City-scale rooftop PV potential for Jaipur from open labels with quantified
uncertainty."** Venues: NeurIPS CCAI workshop, *Remote Sensing* (MDPI), or an energy journal.
Extra experiments: a sensitivity analysis of the ±15%; a comparison with Satellite Sunroof or Google
Solar API figures where available; optionally Open Buildings 2.5D heights for shading.

**Publication risks.** (1) **Imagery licensing:** Google Maps tiles were used for ML training. The Terms
of Service issue is unresolved and the repo is public. A venue may refuse the paper, and the dataset
cannot be released. Resolve this before submitting, or re-run on licensed imagery. (2) Open Buildings is
model output, so reviewers will ask whether we are distilling Google's detector. (3) The test set is
76 crops from three annotators, so CIs are wide.

## References

1. Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net. MICCAI. https://doi.org/10.1007/978-3-319-24574-4_28
2. He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR. https://doi.org/10.1109/CVPR.2016.90
3. Sirko, W., et al. (2021). Continental-Scale Building Detection from High Resolution Satellite Imagery. arXiv:2107.12283
4. Kaiser, P., Wegner, J. D., Lucchi, A., Jaggi, M., Hofmann, T., Schindler, K. (2017). Learning Aerial Image Segmentation From Online Maps. IEEE TGRS 55(11). https://doi.org/10.1109/TGRS.2017.2719738
5. Mnih, V., Hinton, G. (2012). Learning to Label Aerial Images from Noisy Data. ICML. http://icml.cc/2012/papers/318.pdf
6. Girard, N., Charpiat, G., Tarabalka, Y. (2019). Noisy Supervision for Correcting Misaligned Cadaster Maps Without Perfect Ground Truth Data. IGARSS. https://doi.org/10.1109/IGARSS.2019.8898071
7. Loshchilov, I., Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR. arXiv:1711.05101
8. Xie, E., et al. (2021). SegFormer. NeurIPS. arXiv:2105.15203
9. Kirillov, A., et al. (2023). Segment Anything. ICCV. https://doi.org/10.1109/ICCV51070.2023.00371
10. Siméoni, O., et al. (2025). DINOv3. arXiv:2508.10104
11. Kang, B., et al. (2026). DUDA. WACV. https://doi.org/10.1109/WACV61042.2026.00784
12. Kasmi, G., Dubus, L., Saint-Drenan, Y.-M., Blanc, P. (2025). Space-scale exploration of the poor reliability of deep learning models... Environmental Data Science 4, e22. https://doi.org/10.1017/eds.2025.13
13. Maggiori, E., et al. (2017). Can semantic labeling methods generalize to any city? The Inria aerial image labeling benchmark. IGARSS. https://doi.org/10.1109/IGARSS.2017.8127684
14. Xu, J., Liu, Y., Zeng, X., Wu, J., Qu, Y., Zhang, L. (2026). Revisiting the Necessity of Full Accuracy (OMAF). CVPR. https://cvpr.thecvf.com/virtual/2026/poster/38531
15. Guthula, V. B., et al. (2026). Align and Segment. arXiv:2607.10841
16. Li, K., et al. (2025). DragOSM. arXiv:2509.17951
17. Usmani, M., Bovolo, F., Napolitano, M. (2023). Remote Sensing and Deep Learning to Understand Noisy OpenStreetMap. Remote Sensing 15(18). https://doi.org/10.3390/rs15184639
18. Liu, C., et al. (2024). AIO2. IEEE TGRS. https://doi.org/10.1109/TGRS.2024.3373908
19. Song, et al. (2023). Self-Filtered Learning for Semantic Segmentation of Buildings... IEEE JSTARS 16. https://doi.org/10.1109/JSTARS.2022.3230625
20. Sirko, W., et al. (2023). High-Resolution Building and Road Detection from Sentinel-2. arXiv:2310.11622
21. Fobi, S., Conlon, T., Taneja, J., et al. (2020). Learning to segment from misaligned and partial labels. arXiv:2005.13180
22. Wang, D., et al. (2023). SAMRS. NeurIPS Datasets & Benchmarks. arXiv:2305.02034
23. Kim, Kim (2024). Integrated Framework for Unsupervised Building Segmentation with SAM-Based Pseudo-Labeling... Remote Sensing 16(3). https://doi.org/10.3390/rs16030526
24. Mayladan, A., et al. (2023). Zero-Shot Refinement of Buildings' Segmentation Models using SAM. arXiv:2310.01845
25. Ma, X., et al. (2023). SAM-Assisted Remote Sensing Imagery Semantic Segmentation with Object and Boundary Constraints. arXiv:2312.02464
26. Chen, K., et al. (2024). RSPrompter. IEEE TGRS. https://doi.org/10.1109/TGRS.2024.3356074
27. Yin, J., et al. (2025). SAMST. IGARSS. arXiv:2507.11994
28. Dabaja, M., Celik, T. (2026). Promptable Concept Segmentation from Above (SAM 3). arXiv:2607.09583
29. Batchu, V., et al. (2024). Satellite Sunroof. arXiv:2408.14400
30. Singla, J. G., Kalase, A. (2024). Solar potential analysis over Indian cities using high-resolution satellite imagery and DEM. arXiv:2411.04610
31. Rangdal, et al. (2025). Rooftop Segmentation using Geospatial Artificial Intelligence for Rural India. ISPRS Annals X-5/W2-2025. https://doi.org/10.5194/isprs-annals-X-5-W2-2025-537-2025
32. Zhang, et al. (2025). Worldwide rooftop photovoltaic electricity generation may mitigate global warming. Nature Climate Change. https://doi.org/10.1038/s41558-025-02276-3
33. Wang, G. (arXiv) / Li, G., Wang, G., et al. (journal) (2024). SolarSAM. arXiv:2407.00296
34. Krapf, S., et al. (2022). RID: Roof Information Dataset. Remote Sensing 14(10). https://doi.org/10.3390/rs14102299
35. Joshi, S., et al. (2021). High resolution global spatiotemporal assessment of rooftop solar PV potential. Nature Communications. https://doi.org/10.1038/s41467-021-25720-2
36. Dixit, Y., et al. (2024). Cross Pseudo Supervision Framework for Sparsely Labelled Geospatial Images. arXiv:2408.02382
37. Jakubik, J., et al. (2023). Foundation Models for Generalist Geospatial AI (Prithvi). arXiv:2310.18660
38. Cong, Y., et al. (2022). SatMAE. NeurIPS. arXiv:2207.08051
39. Reed, C. J., et al. (2023). Scale-MAE. ICCV. arXiv:2212.14532
40. Guo, X., et al. (2024). SkySense. CVPR. arXiv:2312.10115
41. Marsocci, V., et al. (2024). PANGAEA. arXiv:2412.04204
42. Lacoste, A., et al. (2023). GEO-Bench. arXiv:2306.03831
43. Simumba, N., et al. (2025). GEO-Bench-2. arXiv:2511.15658
44. Faulkenberry, R., Prasad, S. (2026). DINO Soars. CVPR MORSE Workshop. arXiv:2605.03175
45. Iqbal, J., Ali, M. (2020). Weakly Supervised Domain Adaptation for Built-up Region Segmentation in Aerial and Satellite Imagery. arXiv:2007.02277
