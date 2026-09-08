# SAM 3 Implementation Plan — Pipeline 2 (Strong Baseline)

**Document purpose:** Complete, implementation-ready specification for applying SAM 3 to Indian satellite imagery (rooftop and solar panel segmentation). This is Pipeline 2 in the research paper — the strong off-the-shelf baseline that our novel UDA method must outperform.

---

## 1. Why SAM 3 Is the Right Choice

### 1.1 What SAM 3 Is

SAM 3 ("Segment Anything with Concepts") is Meta's third-generation foundation model for image segmentation, presented at ICLR 2026. It extends SAM 2 with open-vocabulary concept understanding — you supply a text prompt such as `"rooftop"` and the model finds and segments every instance of that concept in the image. It does not need fine-tuning; it works zero-shot out of the box.

Architecture overview:
- **Vision encoder:** 450M parameters, ViT-based Perception Encoder. Produces rich spatial feature maps.
- **Text encoder:** 300M parameters, CLIP-style. Aligns language with vision in a shared embedding space.
- **DETR-based detector:** Localizes every instance of the queried concept.
- **Presence head:** Decides whether the concept exists before attempting segmentation — reduces false positives.
- **SAM 2 memory tracker (inherited):** Propagates masks across video frames (not used for our static-image task).
- **Total:** ~848M parameters.

### 1.2 Why SAM 3 Specifically (Not SAM 1 or SAM 2)

| Model | Prompt type | Relevant difference |
|---|---|---|
| SAM 1 | Point, box, mask | Requires a human click per instance. Cannot do open-vocabulary. |
| SAM 2 | Point, box, mask + video | Still requires spatial prompt per instance. No text. |
| **SAM 3** | **Text, exemplar, point, box** | **Text prompt finds ALL instances automatically. Zero human interaction.** |

For our zero-annotation pipeline, SAM 3 is the only model in the SAM family that can be run fully automatically on satellite tiles with no human input.

### 1.3 Why SAM 3 Is the Correct Baseline for This Paper

The research claim of the paper is: *"Our unsupervised domain adaptation method outperforms the best existing zero-shot alternative on Indian satellite imagery, without requiring any target-domain annotation."*

SAM 3 is the strongest credible zero-shot alternative as of 2026:
- It is the state-of-the-art open-vocabulary segmentation model.
- It has been shown to work on remote sensing tasks (SolarSAM, UrbanSAM, Remote SAMsing 2026).
- Beating SAM 3 (848M params, Meta Research) with a lightweight adapted UNet (≈14–32M params) is a meaningful and publishable result.
- If we only compared against direct transfer (Pipeline 1), a reviewer would immediately ask "why not SAM 3?" — including it preempts that question.

### 1.4 Key Limitation That Our Method Exploits

SAM 3 has **no domain adaptation mechanism**. It was pretrained on natural images and general internet data. Indian satellite imagery has:
- ~30 cm/px resolution (4–6× coarser than AIRS's 7.5 cm/px)
- Different sensor characteristics (Cartosat-2, Sentinel-2, Bhuvan)
- Different roof materials (flat concrete, corrugated iron, terrace structures)
- Dense informal urban layouts unlike Western suburban data

SAM 3 sees these as out-of-distribution inputs and cannot adapt. Our feature alignment UDA method explicitly addresses this. That is the core scientific argument.

---

## 2. Role in the Research Paper

SAM 3 appears in the paper as **Pipeline 2: Strong Zero-Shot Baseline**.

```
Pipeline 1: Direct Transfer  →  IoU ~0.40–0.50  (weak, no adaptation)
Pipeline 2: SAM 3 text-only  →  IoU ~0.55–0.65  (strong, but no domain alignment)
Pipeline 3: Our UDA method   →  IoU ~0.70+       (novel contribution)
```

All three pipelines are evaluated on the same Indian test tiles against the same OSM-derived ground truth masks. SAM 3 requires zero annotation, zero fine-tuning — it is a pure zero-shot evaluation.

---

## 3. Ground Truth — OpenStreetMap (OSM)

### 3.1 What OSM Is

OpenStreetMap is a free, community-built global map database. Thousands of volunteers worldwide have manually traced building outlines on satellite imagery for cities including Jaipur, Delhi, Mumbai, Bangalore, and most major Indian cities. These building polygon footprints are publicly available at no cost.

We use OSM **only for evaluation** — as the answer key against which all three pipelines are measured. OSM data is never used for training anything.

### 3.2 OSM as Ground Truth — Justification

- Widely used in remote sensing research as proxy ground truth (Wang et al. 2022, Liu et al. 2026).
- Covers virtually every city in India, China, Japan, Italy — making our pipeline globally transferable.
- Requires zero manual annotation from our side.
- Known limitation: OSM traces full building footprint (walls + roof), not the pure rooftop surface. This introduces a small systematic overestimation (~5–10% area). We acknowledge this in one sentence in the paper.

### 3.3 OSM Data Download and Rasterization

**Tool:** `osmnx` Python library.

**Process:**
1. Define city bounding box (lat/lon).
2. Download all OSM features tagged `building=*` within the bounding box.
3. Filter to `Polygon` and `MultiPolygon` geometry types.
4. For each 512×512 satellite tile, rasterize the intersecting building polygons to a binary image aligned with the tile's coordinate bounds.
5. Save as PNG (255 = building, 0 = background).

**Dependencies:**
```
osmnx>=1.9.0
geopandas>=0.14.0
rasterio>=1.3.0
shapely>=2.0.0
```

**City bounding boxes (expand as needed):**
```python
CITY_BBOX = {
    "jaipur":    (26.78, 75.68, 27.02, 75.95),   # south, west, north, east
    "delhi":     (28.40, 76.84, 28.88, 77.35),
    "mumbai":    (18.87, 72.77, 19.27, 72.99),
    "bangalore": (12.83, 77.46, 13.14, 77.75),
}
```

**Important constraint:** Tile filenames must encode the tile's geographic bounding box so the rasterizer knows which OSM polygons to paint. Recommended naming convention:

```
{city}_{south}_{west}_{north}_{east}.png
e.g., jaipur_26.8200_75.8100_26.8228_75.8128.png
```

If your tiling pipeline does not already embed coordinates in filenames, it must be modified to do so before OSM masks can be generated.

---

## 4. SAM 3 Implementation — Step by Step

### 4.1 Installation

```bash
# SAM 3 official package from Meta
pip install segment-anything-3

# Download model weights (Large variant — best quality)
# ~3.4 GB
wget https://dl.fbaipublicfiles.com/segment_anything_3/sam3_large.pt

# Alternatively, Small variant (~1.1 GB, faster, lower quality)
wget https://dl.fbaipublicfiles.com/segment_anything_3/sam3_small.pt
```

If the above URLs have changed, check: https://ai.meta.com/research/sam3/

### 4.2 Directory Layout Expected

```
btp/
├── india/
│   ├── test/
│   │   ├── images/          ← Indian satellite tiles, 512×512 PNG
│   │   │   └── jaipur_26.8200_75.8100_26.8228_75.8128.png ...
│   │   ├── osm_masks/       ← OSM ground truth (auto-generated, see Section 3)
│   │   └── sam3_preds/      ← SAM 3 output masks (created by inference script)
│   └── train/
│       └── images/          ← Unlabeled Indian training tiles (used by Pipeline 3 only)
├── plan/
│   └── 09-sam3-implementation.md   ← this file
├── rooftop/
│   └── evaluate.py          ← existing evaluation script
└── weights/
    └── sam3_large.pt
```

### 4.3 Text Prompt Design

SAM 3 interprets multi-part dot-separated prompts. More specific prompts reduce false positives on satellite imagery.

**Rooftop task:**
```
"building . rooftop . flat roof . house roof . aerial view"
```

**Solar panel task:**
```
"solar panel . photovoltaic panel . pv array . rooftop solar"
```

**Confidence thresholds** (tune these on a small visual inspection of outputs):
- `box_threshold = 0.30` — minimum detection confidence from the DETR head
- `text_threshold = 0.25` — minimum text-alignment score
- `stability_score = 0.82` — minimum SAM 3 mask quality score

These are starting values. Run the script at three threshold levels (conservative / balanced / aggressive) and report the best in the paper.

| Mode | stability_score | Expected behaviour |
|---|---|---|
| Conservative | 0.90 | Fewer masks, higher precision, lower recall |
| Balanced | 0.82 | Best F1 in most cases — start here |
| Aggressive | 0.70 | More masks, higher recall, more false positives |

### 4.4 Inference Script — `india/sam3_infer.py`

```python
"""
sam3_infer.py
Pipeline 2: Run SAM 3 zero-shot on Indian satellite tiles.
No fine-tuning. No annotation. Text prompt only.

Usage:
    python india/sam3_infer.py \
        --image_dir india/test/images \
        --out_dir   india/test/sam3_preds \
        --task      rooftop \
        --ckpt      weights/sam3_large.pt \
        --conf      0.82
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from sam3 import SAM3


TEXT_PROMPTS = {
    "rooftop": "building . rooftop . flat roof . house roof . aerial view",
    "solar":   "solar panel . photovoltaic panel . pv array . rooftop solar",
}

# Threshold triplets to sweep (for ablation table)
THRESHOLD_MODES = {
    "conservative": {"box": 0.35, "text": 0.30, "stability": 0.90},
    "balanced":     {"box": 0.30, "text": 0.25, "stability": 0.82},
    "aggressive":   {"box": 0.25, "text": 0.20, "stability": 0.70},
}


def merge_masks(mask_list, h, w, stability_threshold):
    """Merge all SAM 3 instance masks above stability threshold into one binary mask."""
    binary = np.zeros((h, w), dtype=np.uint8)
    for m in mask_list:
        if m["stability_score"] >= stability_threshold:
            binary = np.maximum(binary, m["mask"].astype(np.uint8))
    return (binary * 255).astype(np.uint8)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device        : {device}")
    print(f"Task          : {args.task}")
    print(f"Threshold mode: {args.mode}")
    thresholds = THRESHOLD_MODES[args.mode]

    model = SAM3.from_pretrained(args.ckpt).to(device)
    model.eval()

    prompt = TEXT_PROMPTS[args.task]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(Path(args.image_dir).glob("*.png"))
    print(f"Tiles to process: {len(image_paths)}")

    for ip in image_paths:
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        with torch.no_grad():
            result = model.predict(
                image=img,
                text_prompt=prompt,
                box_threshold=thresholds["box"],
                text_threshold=thresholds["text"],
            )

        out_mask = merge_masks(
            result.get("masks", []), h, w, thresholds["stability"]
        )
        cv2.imwrite(str(out_dir / ip.name), out_mask)

    print(f"Predictions written to: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True, help="Folder of 512x512 PNG tiles")
    p.add_argument("--out_dir",   required=True, help="Where to write predicted masks")
    p.add_argument("--task",      choices=["rooftop", "solar"], required=True)
    p.add_argument("--ckpt",      default="weights/sam3_large.pt")
    p.add_argument("--mode",      choices=["conservative", "balanced", "aggressive"],
                   default="balanced", help="Threshold mode for ablation")
    # --conf kept for backward compat but mode overrides it
    p.add_argument("--conf",      type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
```

### 4.5 OSM Ground Truth Generation — `india/fetch_osm_gt.py`

```python
"""
fetch_osm_gt.py
Download OSM building footprints for a city, rasterize to binary masks
aligned with your 512×512 satellite tiles. Zero human annotation required.

Usage:
    python india/fetch_osm_gt.py \
        --tile_dir india/test/images \
        --out_dir  india/test/osm_masks \
        --city     jaipur
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import osmnx as ox
from rasterio.transform import from_bounds
from rasterio.features import rasterize as rio_rasterize
from shapely.geometry import box as shapely_box


CITY_BBOX = {
    "jaipur":    (26.78, 75.68, 27.02, 75.95),
    "delhi":     (28.40, 76.84, 28.88, 77.35),
    "mumbai":    (18.87, 72.77, 19.27, 72.99),
    "bangalore": (12.83, 77.46, 13.14, 77.75),
}

TILE_SIZE = 512  # pixels; must match your tiling pipeline


def fetch_buildings(city: str):
    south, west, north, east = CITY_BBOX[city]
    print(f"Downloading OSM buildings for {city}...")
    gdf = ox.geometries_from_bbox(north, south, east, west, tags={"building": True})
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    return gdf.to_crs("EPSG:4326")


def parse_bounds_from_filename(fname: str):
    """
    Extract tile geographic bounds from filename.
    Expected format: {city}_{south}_{west}_{north}_{east}.png
    Example: jaipur_26.8200_75.8100_26.8228_75.8128.png
    Returns (south, west, north, east) as floats.
    """
    stem = Path(fname).stem
    parts = stem.split("_")
    # parts[-4:] are south, west, north, east
    s, w, n, e = float(parts[-4]), float(parts[-3]), float(parts[-2]), float(parts[-1])
    return s, w, n, e


def rasterize_tile(gdf, south, west, north, east, size=TILE_SIZE):
    transform = from_bounds(west, south, east, north, size, size)
    tile_box  = shapely_box(west, south, east, north)
    shapes    = [
        (geom, 1)
        for geom in gdf.geometry
        if geom is not None and geom.intersects(tile_box)
    ]
    if not shapes:
        return np.zeros((size, size), dtype=np.uint8)
    mask = rio_rasterize(
        shapes, out_shape=(size, size),
        transform=transform, fill=0, dtype=np.uint8
    )
    return (mask * 255).astype(np.uint8)


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = fetch_buildings(args.city)
    tiles = sorted(Path(args.tile_dir).glob("*.png"))
    print(f"Rasterizing {len(tiles)} tiles...")

    for tp in tiles:
        try:
            s, w, n, e = parse_bounds_from_filename(tp.name)
        except (IndexError, ValueError):
            print(f"  SKIP (can't parse bounds from filename): {tp.name}")
            continue
        mask = rasterize_tile(gdf, s, w, n, e)
        cv2.imwrite(str(out_dir / tp.name), mask)

    print(f"OSM masks written to: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tile_dir", required=True)
    p.add_argument("--out_dir",  required=True)
    p.add_argument("--city",     choices=list(CITY_BBOX), required=True)
    run(p.parse_args())
```

### 4.6 Evaluation Script — `india/evaluate_sam3.py`

Computes the same global IoU / F1 / Precision / Recall as the existing `rooftop/evaluate.py`, ensuring all methods are compared on identical metrics.

```python
"""
evaluate_sam3.py
Evaluate predicted masks against OSM ground truth.
Uses identical global-accumulation metric logic as rooftop/evaluate.py.

Usage:
    python india/evaluate_sam3.py \
        --pred_dir india/test/sam3_preds \
        --gt_dir   india/test/osm_masks \
        --label    "SAM 3 (balanced, rooftop)"
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def evaluate(pred_dir, gt_dir, threshold=127, label=""):
    pred_paths = sorted(Path(pred_dir).glob("*.png"))
    if not pred_paths:
        raise FileNotFoundError(f"No predictions found in {pred_dir}")

    tp = fp = fn = 0.0
    skipped = 0

    for pp in pred_paths:
        gp = Path(gt_dir) / pp.name
        if not gp.exists():
            skipped += 1
            continue
        pred = (cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE) > threshold).astype(float)
        gt   = (cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE) > threshold).astype(float)
        tp  += (pred * gt).sum()
        fp  += (pred * (1 - gt)).sum()
        fn  += ((1 - pred) * gt).sum()

    iou       = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = 2 * precision * recall / (precision + recall + 1e-7)

    tag = f"[{label}]" if label else ""
    print(f"\n{'─'*48}")
    print(f"  Evaluation Results {tag}")
    print(f"{'─'*48}")
    print(f"  Tiles evaluated : {len(pred_paths) - skipped}")
    print(f"  Tiles skipped   : {skipped} (no matching OSM mask)")
    print(f"  IoU             : {iou:.4f}")
    print(f"  F1              : {f1:.4f}")
    print(f"  Precision       : {precision:.4f}")
    print(f"  Recall          : {recall:.4f}")
    print(f"{'─'*48}\n")
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir",  required=True)
    p.add_argument("--gt_dir",    required=True)
    p.add_argument("--threshold", type=int, default=127)
    p.add_argument("--label",     default="")
    args = p.parse_args()
    evaluate(args.pred_dir, args.gt_dir, args.threshold, args.label)
```

---

## 5. Full Execution Sequence

Run these commands in order. Each step builds on the previous.

```bash
# ── Step 0: Install dependencies ────────────────────────────────────────────
pip install segment-anything-3 osmnx geopandas rasterio shapely

# Download SAM 3 weights
mkdir -p weights
wget -O weights/sam3_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_3/sam3_large.pt

# ── Step 1: Generate OSM ground truth masks ──────────────────────────────────
# (Run once per city. Takes ~2–5 min for a city-sized OSM download.)
python india/fetch_osm_gt.py \
    --tile_dir india/test/images \
    --out_dir  india/test/osm_masks \
    --city     jaipur

# ── Step 2: Run SAM 3 inference — ROOFTOP task ───────────────────────────────
# Run all three threshold modes for ablation table
for MODE in conservative balanced aggressive; do
    python india/sam3_infer.py \
        --image_dir india/test/images \
        --out_dir   india/test/sam3_preds_rooftop_${MODE} \
        --task      rooftop \
        --ckpt      weights/sam3_large.pt \
        --mode      ${MODE}
done

# ── Step 3: Run SAM 3 inference — SOLAR task ────────────────────────────────
for MODE in conservative balanced aggressive; do
    python india/sam3_infer.py \
        --image_dir india/test/images \
        --out_dir   india/test/sam3_preds_solar_${MODE} \
        --task      solar \
        --ckpt      weights/sam3_large.pt \
        --mode      ${MODE}
done

# ── Step 4: Evaluate all runs ────────────────────────────────────────────────
for MODE in conservative balanced aggressive; do
    echo "=== ROOFTOP / ${MODE} ==="
    python india/evaluate_sam3.py \
        --pred_dir india/test/sam3_preds_rooftop_${MODE} \
        --gt_dir   india/test/osm_masks \
        --label    "SAM 3 rooftop ${MODE}"

    echo "=== SOLAR / ${MODE} ==="
    python india/evaluate_sam3.py \
        --pred_dir india/test/sam3_preds_solar_${MODE} \
        --gt_dir   india/test/osm_masks \
        --label    "SAM 3 solar ${MODE}"
done
```

Expected runtime on a single A100 GPU:
- ~50ms per tile for SAM 3 inference
- 500 tiles → ~25 seconds per mode
- All 6 runs (3 modes × 2 tasks) → ~3 minutes total

---

## 6. Ablation Table for Paper

Running the three threshold modes gives you a free ablation table:

| SAM 3 Variant | Threshold Mode | IoU | F1 | Precision | Recall |
|---|---|---|---|---|---|
| SAM 3 rooftop | conservative | — | — | — | — |
| SAM 3 rooftop | balanced | — | — | — | — |
| SAM 3 rooftop | aggressive | — | — | — | — |
| SAM 3 solar | conservative | — | — | — | — |
| SAM 3 solar | balanced | — | — | — | — |
| SAM 3 solar | aggressive | — | — | — | — |

Fill in actual numbers after running. Report the best row for each task as the primary SAM 3 result in the main table, and include the full table in the appendix or supplementary.

---

## 7. What to Log and Report

For the paper, report the following for SAM 3:

1. **Model variant used:** SAM 3 Large (848M params)
2. **Prompt used:** exact text string (copy from `TEXT_PROMPTS` dict)
3. **Threshold mode and values:** box/text/stability scores
4. **Number of test tiles:** exact count
5. **City evaluated on:** Jaipur (and others if time permits)
6. **Evaluation GT:** OSM building footprints, downloaded [date], version [OSM timestamp]
7. **Metrics:** IoU, F1, Precision, Recall (global accumulation, same as AIRS methodology)

This level of detail satisfies reproducibility requirements for any remote sensing journal.

---

## 8. Dependency Summary

| Package | Version | Purpose |
|---|---|---|
| `segment-anything-3` | latest | SAM 3 model |
| `torch` | ≥2.1.0 | Inference backend |
| `torchvision` | ≥0.16.0 | Image utilities |
| `osmnx` | ≥1.9.0 | OSM building download |
| `geopandas` | ≥0.14.0 | Geospatial data handling |
| `rasterio` | ≥1.3.0 | Rasterizing vector polygons |
| `shapely` | ≥2.0.0 | Geometry intersection |
| `opencv-python` | ≥4.8.0 | Image I/O |
| `numpy` | ≥1.24.0 | Array operations |

All compatible with the existing `requirements.txt` in this repo. Add the new packages to it.

---

## 9. Common Failure Modes and Fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| All masks empty | Prompt not matching satellite appearance | Lower `box_threshold` to 0.20; try simpler prompt `"roof"` |
| Too many false positives (roads, fields masked) | Threshold too aggressive | Raise `stability_score` to 0.88–0.92 |
| OSM masks all black | Filename format not matching parser | Check `parse_bounds_from_filename`; print parsed values |
| OSM buildings misaligned with tiles | CRS mismatch | Ensure tiles are in EPSG:4326; OSM is always EPSG:4326 |
| CUDA OOM | SAM 3 Large on low-VRAM GPU | Use `sam3_small.pt`; or process tiles in CPU mode |
| Solar IoU much lower than rooftop | Solar panels are very small at 30cm/px | Lower `stability_score` to 0.70 for solar only |

---

## 10. Connection to Pipeline 3 (Novel UDA Method)

SAM 3 (this document) and the novel UDA method (to be specified in `10-uda-implementation.md`) are completely independent pipelines:

- SAM 3 runs only on **test tiles** and is evaluated. Its internal weights are never modified.
- The novel UDA method uses only **AIRS labeled data + unlabeled Indian training tiles**. It does not use SAM 3 at any stage.
- Both are evaluated against the **same OSM masks**, ensuring fair comparison.

The paper narrative: SAM 3, despite its 848M parameters and Meta-scale pretraining, achieves X IoU because it has no mechanism to adapt to the specific visual properties of Indian satellite imagery. Our UDA method, despite using far fewer parameters, achieves Y IoU by explicitly aligning the source and target feature distributions — demonstrating that domain alignment is more important than model scale for this task.
