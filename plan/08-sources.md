# 08 — Sources

All external claims in this plan folder trace to one of the following.
Retrieved 2026-08-04. Anything not listed here is either measured directly from
this repository's own files or is explicitly labelled as an estimate.

---

## Domain adaptation — core methods

| Ref | Source |
|-----|--------|
| DAFormer, HRDA, MIC (author page, all three papers + code) | https://lhoyer.github.io/ |
| HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation (ECCV 2022) — project page | https://vas.mpi-inf.mpg.de/hrda/ |
| HRDA — official code | https://github.com/lhoyer/HRDA |
| MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation (CVPR 2023) — paper | https://arxiv.org/pdf/2212.01322 |
| MIC — official code | https://github.com/lhoyer/MIC |
| FDA: Fourier Domain Adaptation for Semantic Segmentation (CVPR 2020) — code | https://github.com/YanchaoYang/FDA |
| FDA — overview | https://www.researchgate.net/publication/343456374_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation |
| Transformer-Based Visual Segmentation: A Survey | https://arxiv.org/pdf/2304.09854 |

Key claims drawn from these: DAFormer/HRDA improve SOTA UDA by >10 mIoU across
five benchmarks; MIC(HRDA) is the strongest published configuration; DAFormer's
gains come substantially from swapping a ResNet backbone for a Transformer plus
rare-class sampling, ImageNet feature-distance loss, and LR warm-up; HRDA
combines small high-resolution crops with large low-resolution crops under
learned scale attention; FDA requires no network and no adversarial training.

---

## Domain adaptation — remote sensing specific

| Ref | Source |
|-----|--------|
| CrossEarth: Geospatial Vision Foundation Model for Domain Generalizable RS Semantic Segmentation | https://arxiv.org/pdf/2410.22629 |
| CrossEarth — code | https://github.com/Cuzyoung/CrossEarth |
| Self-Training Guided Disentangled Adaptation for Cross-Domain RS Semantic Segmentation (ST-DASegNet) | https://arxiv.org/pdf/2301.05526 |
| Decomposition-based UDA for Remote Sensing Image Semantic Segmentation | https://arxiv.org/pdf/2404.04531 |
| UDA for RS semantic segmentation with the 2D discrete wavelet transform (Sci Rep) | https://www.nature.com/articles/s41598-024-74781-y |
| Depth-Aware Adversarial Domain Adaptation for Cross-Domain RS Segmentation | https://doi.org/10.3390/rs18071099 |
| Elevation-Aware Domain Adaptation for Semantic Segmentation of Aerial Images | https://doi.org/10.3390/rs17142529 |
| UDA for Remote Sensing Data Classification Model Transfer (ISPRS Annals 2025) | https://isprs-annals.copernicus.org/articles/X-4-W6-2025/17/2025/isprs-annals-X-4-W6-2025-17-2025.html |
| Integrating UDA and SAM for building extraction from high-resolution RS images (2025) | https://www.tandfonline.com/doi/full/10.1080/17538947.2025.2491108 |
| Enabling Country-Scale Land Cover Mapping with Meter-Resolution Satellite Imagery | https://arxiv.org/pdf/2209.00727 |

CrossEarth is described by its authors as the first vision foundation model for
remote-sensing domain *generalisation*, using an Earth-Style Injection data
pipeline plus multi-task training, evaluated on a purpose-built benchmark of 32
cross-domain settings. The depth- and elevation-aware methods require a DSM,
which this project does not have for Jaipur — cited, not implemented.

---

## SAM / weak supervision

| Ref | Source |
|-----|--------|
| The Segment Anything Model (SAM) for Remote Sensing Applications: From Zero to One Shot | https://arxiv.org/html/2306.16623v1 |
| Sparse point annotations for remote sensing image segmentation (PENet, Sci Rep 2025) | https://www.nature.com/articles/s41598-025-12969-6 |
| Weakly Supervised Semantic Segmentation of RS Images Using Siamese Affinity Network (2025) | https://www.mdpi.com/2072-4292/17/5/808 |
| Damaged-building extraction with YOLO-E + SAM2 (2025) | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12299807/ |

Basis for the claim that SAM-assisted and point-prompt weak supervision is now
a standard way to bootstrap remote-sensing labels without a manual annotation
budget, and that SAM branches are used specifically to recover object
boundaries that point labels lack.

---

## Building footprint datasets

| Ref | Source | Licence |
|-----|--------|---------|
| Microsoft GlobalMLBuildingFootprints | https://github.com/microsoft/GlobalMLBuildingFootprints | ODbL v1.0 |
| — India/Nepal coverage note (110 M India, 7 M Nepal, 2024 update) | same repo README | — |
| Google Open Buildings | https://sites.research.google/gr/open-buildings/ | CC BY 4.0 **or** ODbL v1.0 (choose one) |
| Google Open Buildings V3 Polygons (Earth Engine catalog) | https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings_v3_polygons | CC BY 4.0 |
| VIDA combined Google–Microsoft–OSM Open Buildings | https://source.coop/vida/google-microsoft-osm-open-buildings | ODbL |
| Global Google–Microsoft Open Buildings (GEE community catalog) | https://gee-community-catalog.org/projects/global_buildings/ | — |
| Microsoft Building Footprint Data (OSM wiki) | https://wiki.openstreetmap.org/wiki/Microsoft_Building_Footprint_Data | ODbL |
| From Footprints to Functions: global + semantic building footprint dataset (Sci Data 2025) | https://www.nature.com/articles/s41597-025-06132-z | — |

Open Buildings covers Africa, South Asia (India, Bangladesh, Pakistan, Nepal),
Southeast Asia, and parts of Latin America, with 1.8 B detections; each record
carries a polygon, a confidence score, and a Plus Code; V3 was inferred May 2023.

---

## Building segmentation in Indian / dense urban contexts

| Ref | Source |
|-----|--------|
| Decoding urban complexity: deep learning-based terrain-specific building segmentation for Indian cities | https://www.sciencedirect.com/science/article/abs/pii/S2352938525002265 |
| A Deep Learning Approach for Automated Building Footprint Extraction from Cartosat Imagery (J. Indian Soc. Remote Sens.) | https://link.springer.com/article/10.1007/s12524-026-02427-9 |
| Instance segmentation based building extraction in a dense urban area (Mumbai slums, ~15,000 objects) | https://link.springer.com/article/10.1007/s11042-023-15905-w |
| Revolutionizing urban mapping: deep learning and data fusion for building footprint segmentation (Sci Rep) | https://www.nature.com/articles/s41598-024-64231-0 |
| Building Footprint Extraction in Dense Areas using Super Resolution and Frame Field Learning | https://arxiv.org/pdf/2309.01656 |
| Deep Learning-Based Building Footprint Extraction from UAV Fused Spectral + Elevation | https://link.springer.com/chapter/10.1007/978-3-031-88217-3_27 |
| satellite-image-deep-learning/techniques (curated index) | https://github.com/satellite-image-deep-learning/techniques |

Basis for the claims about unplanned settlements and informal housing clusters
having irregular, overlapping, densely packed structures without the clear
edges of formal construction, and for the expectation that published dense
Indian urban results land in the 0.70–0.85 band rather than the 0.90 band.

---

## Licensing

| Ref | Source |
|-----|--------|
| Google Maps Platform Service Specific Terms | https://cloud.google.com/maps-platform/terms/maps-service-terms |
| Google Maps Platform Terms of Service | https://cloud.google.com/maps-platform/terms |
| Google Maps Additional Terms of Service | https://www.google.com/help/terms_maps/ |
| Policies for Maps Datasets API | https://developers.google.com/maps/documentation/datasets/policies |

Basis for the claim that Maps content may not be used to train, test, validate
or fine-tune machine-learning and AI models, that derived content (e.g. 3D
building models from imagery, indexes built from Street View) is prohibited,
and that caching/storing/creating derivative map datasets is restricted.

> ⚠️ Terms of service change. **Re-read the current Service Specific Terms
> yourself before relying on this summary in a submitted document.**

---

## Compute

| Ref | Source |
|-----|--------|
| Colab compute-unit consumption comparison (2024) | https://varlog.info/colab-credit-consumption-comparison-2024/ |
| Colab alternatives, pricing and limits (Aug 2026) | https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025 |
| Colab Enterprise pricing (L4 hourly rate) | https://cloud.google.com/colab/pricing |
| Google Colab Pro / Pro+ (NCSU summary) | https://software.ncsu.edu/google-colab-pro-colab-pro |

⚠️ **These sources disagree** (T4 at 1.19 vs 1.76 CU/hr; A100-40 at 5.40 vs
15 CU/hr) and Google changes the rates without notice. Treat every compute-unit
figure in [`05-compute-and-schedule.md`](05-compute-and-schedule.md) as
unverified until you have measured the drawdown yourself in a live runtime.

---

## Project-internal (measured, not cited)

These came from reading this repository directly and need no external source:

- Jaipur tile dimensions, `ModelPixelScale`, `ModelTiepoint`, and derived GSD —
  read from `daraset/map67_1-1.tif` and `map67_1-2.tif` TIFF tags.
- AIRS results (IoU 0.9016 / 0.8664), architecture, loss, and CLI flags —
  `README.md`, `rooftop/train.py`.
- AIRS paper baselines (FPN 0.882, FPN+MSFF 0.888, PSPNet 0.899; Chen et al.,
  2019, ISPRS) — `.planning/PROJECT.md`, `airs.pdf`.
- BDAPPV setup and the black-mask negative handling — `README.md`,
  `solar_panel/train_solar.py`.
- DGX constraints (glibc 2.17, Docker requirement, non-persistent `/scratch`,
  NFS slowness) — `README.md`, `.planning/PROJECT.md`.
