# BTP Reading List — Top-Venue References

**Project:** Domain adaptation for rooftop & solar-panel segmentation, AIRS (Christchurch) → Jaipur
**Owner:** Sirjan Singh · LNMIIT

> Read at least the abstract, the method figure, and the main results table of every
> paper before listing it. The one-line takeaway column is what you should be able to
> say out loud if asked.

---

## 1. Cross-domain semantic segmentation — the core method line

| Paper | Venue | Takeaway |
|---|---|---|
| DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation | CVPR 2022 | The Transformer backbone contributes more than the adaptation algorithm itself — justifies the SegFormer ablation |
| HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation | ECCV 2022 | Small high-res crops for detail + large low-res crops for context, fused by learned scale attention |
| MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation | CVPR 2023 | Masked-region prediction on the target domain; MIC(HRDA) is the strongest published configuration |
| FDA: Fourier Domain Adaptation for Semantic Segmentation | CVPR 2020 | Swap low-frequency Fourier amplitude source↔target, keep phase. No network, no adversarial training |

*Relevance:* Tier 1 (FDA) and the Phase E stretch goal (MIC/HRDA). DAFormer's
architecture finding is why Phase C runs both ResNet-34 and SegFormer.

---

## 2. Building extraction datasets and benchmarks

| Paper | Venue | Takeaway |
|---|---|---|
| Chen et al., *Aerial Imagery for Roof Segmentation* (AIRS) | ISPRS J. Photogrammetry & Remote Sensing, 2019 | The source dataset; 7.5 cm/px, roof-outline labels; PSPNet baseline IoU 0.899 |
| Maggiori et al., *Can Semantic Labeling Methods Generalize to Any City?* (Inria Aerial Image Labeling) | IGARSS 2017 | 30 cm, 810 km², 5 cities — nearly GSD-matched to Jaipur; the highest-value Phase C addition |
| Ji et al., *Fully Convolutional Networks for Multi-Source Building Extraction* (WHU Building Dataset) | IEEE TGRS, 2019 | 30 cm aerial benchmark, Christchurch-derived; second GSD-matched source |
| Van Etten et al., *SpaceNet: A Remote Sensing Dataset and Challenge Series* | CoRR / CVPR-W | Khartoum is semi-arid, dense, low-rise — the closest public morphological analogue to Jaipur |

*Relevance:* the multi-source robust-training recipe in Phase C.

---

## 3. Foundation models and weak supervision

| Paper | Venue | Takeaway |
|---|---|---|
| Kirillov et al., *Segment Anything* | ICCV 2023 | Promptable segmentation; zero-shot SAM segments *something*, it does not know what a building is |
| Ravi et al., *SAM 2: Segment Anything in Images and Videos* | ICLR 2025 | The version actually used as the label engine |
| CrossEarth: Geospatial Vision Foundation Model for Domain-Generalizable Remote Sensing Semantic Segmentation | arXiv 2410.22629 (2024) | First RS-specific domain-*generalisation* foundation model; 32-setting cross-domain benchmark |

*Relevance:* Phase D — point-prompting footprint centroids to re-anchor noisy
open-data labels to visible roof edges.

---

## 4. Architectures under comparison

| Paper | Venue | Takeaway |
|---|---|---|
| SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers | NeurIPS 2021 | The MiT encoder in the Phase C backbone ablation; LayerNorm, so AdaBN does not apply |
| U-Net: Convolutional Networks for Biomedical Image Segmentation | MICCAI 2015 | The current architecture |
| Deep Residual Learning for Image Recognition (ResNet) | CVPR 2016 | The current encoder (ResNet-34, ImageNet-pretrained) |

---

## 5. Dense-area instance quality

| Paper | Venue | Takeaway |
|---|---|---|
| Girard et al., *Polygonal Building Extraction by Frame Field Learning* | CVPR 2021 | Crisp, polygonised, per-instance footprints — the answer to shared-wall blocks merging into one blob |

*Relevance:* the fix for risk R11 if per-building capacity (not just total area)
is required. Otherwise cite as future work.

---

## 6. Domain-venue papers — keep in related work, not on the headline slide

These are remote-sensing and geoscience venues rather than top CS conferences,
so they are less citable as "good publications" — but nothing in sections 1–5
says anything about Indian urban fabric, and these are what actually support the
0.70–0.85 expected-IoU band.

| Paper | Venue |
|---|---|
| Decoding urban complexity: terrain-specific building segmentation for Indian cities | Elsevier, 2025 |
| Instance-segmentation-based building extraction in a dense urban area (Mumbai, ~15,000 objects) | Multimedia Tools & Applications, 2023 |
| A deep learning approach for automated building footprint extraction from Cartosat imagery | J. Indian Soc. Remote Sensing |
| Building footprint extraction in dense areas using super-resolution and frame field learning | arXiv 2309.01656 |

---

## 7. Data sources — cite as datasets, not as papers

| Resource | Licence |
|---|---|
| Google Open Buildings v3 (1.8 B detections, per-building confidence, Plus Codes) | CC BY 4.0 **or** ODbL v1.0 |
| Microsoft GlobalMLBuildingFootprints (110 M India polygons, 2024 update) | ODbL v1.0 (share-alike) |
| VIDA combined Google–Microsoft–OSM Open Buildings | ODbL |
| OpenStreetMap, Jaipur | ODbL |

---

## 8. Gap to close

There is currently **no solar-PV detection literature** in this list — no BDAPPV
paper, nothing on rooftop PV detection, nothing on the solar-water-heater
confusion that Stage 2 will run into. If Phase F is meant to be a contribution
rather than a demo, this section needs its own reading before the report.

---

## Practical note on citation management

Sections 1–5 have clean DBLP entries and give you usable BibTeX directly.
Section 6 is largely outside DBLP — pull BibTeX from the publisher and verify
the DOI by hand.
