# 04 — The Full Pipeline

*End to end, from 16 GeoTIFFs on disk to a rooftop-solar capacity estimate for
Jaipur.*

---

## 1. System overview

```mermaid
flowchart TB
    subgraph D["① DATA"]
        D1["daraset/map67_c-r.tif<br/>16 × (11168 × 13856)<br/>EPSG:3857, 26.6 cm, 175 km²"]
        D2["Open Buildings<br/>+ Microsoft GlobalML<br/>GeoJSON"]
        D3["AIRS 7.5 cm<br/>Inria 30 cm<br/>SpaceNet-Khartoum 30 cm"]
    end

    subgraph L["② LABEL ENGINE"]
        L1["Clip + confidence filter"]
        L2["Rasterise to pixel grid"]
        L3["Shift-correct<br/>(phase correlation)"]
        L4["SAM2 point-prompt refine"]
        L5["Boundary relax (4 px ignore)"]
        L6["Hand-correct 400 tiles<br/>25 per mosaic tile<br/>→ sealed eval split"]
    end

    subgraph T["③ TRAINING"]
        T1["Source-robust pretrain<br/>multi-source + heavy aug + FDA"]
        T2["Weak finetune<br/>on ~11,140 noisy Jaipur tiles"]
        T3["Clean finetune<br/>on hand-labelled train split"]
        T4["Self-training rounds<br/>(optional: MIC/HRDA)"]
    end

    subgraph I["④ INFERENCE"]
        I1["Sliding window 512²<br/>Hann blend, 10% overlap"]
        I2["Multi-scale + flip TTA"]
        I3["Threshold from target val"]
        I4["Morphological cleanup<br/>+ min-area filter"]
    end

    subgraph O["⑤ OUTPUT"]
        O1["Rooftop mask<br/>georeferenced GeoTIFF"]
        O2["Stage 2: solar panel mask"]
        O3["Area m² · usable m² · kW"]
        O4["Polygonise → GeoJSON"]
    end

    D1 --> L1
    D2 --> L1
    L1 --> L2 --> L3 --> L4 --> L5
    L5 --> L6
    D3 --> T1
    L5 --> T2
    T1 --> T2 --> T3 --> T4
    L6 -.sealed, eval only.-> I3
    T4 --> I1 --> I2 --> I3 --> I4
    I4 --> O1 --> O2 --> O3
    O1 --> O4

    style L fill:#fce8e6,stroke:#ea4335,color:#000
    style T fill:#fef7e0,stroke:#fbbc04,color:#000
    style O fill:#e6f4ea,stroke:#34a853,color:#000
```

---

## 2. Stage 0 — Tiling the Jaipur GeoTIFFs

Your existing `rooftop/tile_airs.py` tiles 10,000² AIRS images. Jaipur needs a
**georeferencing-aware** variant, because every tile must carry its affine
transform so predictions can be written back to map coordinates.

```
map67_<col>-<row>.tif   col,row ∈ 1..4     (16 files, ~8.2 GB)
each 11168 × 13856 px
        ↓  512 × 512, 10 % overlap (stride 461)
   24 cols × 29 rows = 696 tiles per file
        ────────────────────────────────
   696 × 16 ≈ 11,140 tiles   over ≈ 175 km²
```

Stage the source rasters on **G:** or the DGX — they do not fit on D:. See
[`01-situation-and-assets.md`](01-situation-and-assets.md) §7.1.

Per tile, persist a small sidecar JSON:

```json
{
  "tile_id": "map67_1-1_r012_c008",
  "src": "map67_1-1.tif",
  "row_off": 5532, "col_off": 3688,
  "width": 512, "height": 512,
  "transform": [0.2985821, 0.0, 8430350.4, 0.0, -0.2985821, 3116522.1],
  "crs": "EPSG:3857",
  "gsd_ground_m": 0.26613
}
```

**Quality filtering.** Reject tiles that are >30 % blank/black (mosaic edges) or
have near-zero variance (cloud, water, blur). Expect to lose 3–8 %.

**Split policy — read this twice.** Split tiles **spatially, by contiguous
block**, never randomly. Randomly-split overlapping tiles leak: two 10 %-
overlapping tiles from the same building put the same pixels in train and test,
and your reported IoU becomes fiction. Carve the mosaic into a checkerboard of
1 km blocks and assign whole blocks to train/val/test.

With 16 mosaic tiles you can do better than block splitting: **hold out 2–3
entire mosaic tiles** as the test region. Geographically disjoint beats
block-disjoint, and it is a materially stronger claim in the report.

```mermaid
flowchart LR
    A["Mosaic 175 km²<br/>16 tiles, ~11,140 crops"] --> B["Reserve 2-3 whole<br/>mosaic tiles for TEST"]
    B --> C["Remaining 13-14 tiles<br/>→ 1 km blocks"]
    C --> D["TRAIN blocks (~82%)"]
    C --> E["VAL blocks (~18%)<br/>threshold tuning"]
    B --> F["TEST region<br/><b>SEALED</b>"]
    F -.->|"touch once, at the end"| G["Reported numbers"]
    style F fill:#fce8e6,stroke:#ea4335,color:#000
```

Pick the test tiles to span fabric types — one dense walled-city tile, one
planned-colony tile, one peri-urban tile. A test set drawn only from the easy
outskirts would flatter the model.

---

## 3. Stage 1 — The label engine

This is the part that does not exist yet and is worth the most.

```mermaid
sequenceDiagram
    participant GJ as Open Buildings GeoJSON
    participant R as rasterio/geopandas
    participant P as Phase correlation
    participant S as SAM 2
    participant H as Human (you)
    participant OUT as Label set

    GJ->>R: clip to tile bounds (EPSG:3857)
    R->>R: drop confidence < 0.75
    R->>R: rasterise polygons → uint8 mask
    R->>P: mask + current model prob-map
    P->>P: estimate (dx, dy) per 1024px block
    P->>S: shifted polygons → centroids
    S->>S: point-prompt each centroid on the RGB tile
    S->>S: keep SAM mask if IoU(poly, sam) > 0.5<br/>else fall back to shifted polygon
    S->>OUT: weak labels (~11,140 tiles)
    OUT->>H: sample 400 tiles, 25 per mosaic tile
    H->>H: correct in QGIS / labelme (~8-12 h)
    H->>OUT: clean labels → 200 train / 60 val / 140 SEALED test
```

### Why each step is there

| Step | Removes which failure |
|------|----------------------|
| Confidence filter | Hallucinated detections in low-contrast areas |
| Shift correction | The 2–8 m georeferencing offset between the footprint source and Google's mosaic |
| SAM2 refine | The roof-vs-footprint semantic mismatch, and coarse polygon edges |
| IoU-0.5 fallback | Stops SAM from replacing a building with "the whole courtyard" when the prompt lands badly |
| Boundary relaxation | Residual sub-metre misalignment dominating the loss |
| Hand correction | Gives you *one* trustworthy set to report against |

### Hand-labelling budget

**400 tiles**, stratified at 25 per mosaic tile so all 16 areas and all urban
fabric types are represented. At 512² each takes ~2–4 minutes with a pre-filled
SAM2 mask to correct (vs 15+ minutes from blank). **Budget 13–17 hours of your
own time.** This is the single largest non-compute cost in the project and it
is unavoidable.
Do it early, in one or two sittings, before you are under deadline pressure.

Tools: QGIS with the raster + a GeoJSON edit layer is the most reliable. Label
Studio or `labelme` also work if you prefer per-tile PNGs.

---

## 4. Stage 2 — Training schedule

```mermaid
flowchart TD
    S0["ImageNet-pretrained encoder"] --> S1

    S1["<b>Run 1 — Robust source</b><br/>AIRS↓(0.15–0.40m) + Inria + Khartoum<br/>heavy aug + FDA(Jaipur)<br/>60 epochs · DGX 2×GPU"]

    S1 --> S2["<b>Run 2 — Weak target</b><br/>+ ~11,140 noisy Jaipur tiles<br/>boundary-relaxed loss<br/>LR ÷ 5 · 30 epochs"]

    S2 --> S3["<b>Run 3 — Clean finetune</b><br/>200 hand-labelled train tiles<br/>LR ÷ 10 · 20 epochs · early stop on target-val"]

    S3 --> S4["<b>Run 4 — Self-training</b><br/>pseudo-label all tiles at p>0.8 / p<0.2<br/>repeat ×2–3"]

    S4 --> S5["<b>Run 5 (stretch)</b><br/>MIC(HRDA) with mit_b5<br/>only if DGX time is free"]

    S3 -.->|"checkpoint the<br/>deliverable here"| DELIV["Shippable model"]

    style DELIV fill:#e6f4ea,stroke:#34a853,color:#000
    style S5 fill:#f1f3f4,stroke:#9aa0a6,color:#000
```

**Checkpoint the deliverable after Run 3.** Runs 4 and 5 are improvements; Run 3
is the point at which you have a working, defensible, reportable system. Never
be in a position where the deadline arrives mid-experiment with nothing saved.

### Loss

Keep `0.5 × BCE + 0.5 × Dice`, with two modifications for the target stage:

- `ignore_index` support so boundary-relaxed and pseudo-label `ignore` pixels
  contribute zero gradient.
- Optional per-sample weight: weak-label tiles at 0.5, clean tiles at 1.0, when
  training on a mixed batch.

### Metrics to log every epoch

Global IoU, F1, precision, recall (you already accumulate these correctly —
globally, not per-batch-averaged; keep that). **Add:**

- Boundary IoU (IoU restricted to a 5 px band around GT edges) — this is what
  catches the "merged blocks" failure that plain IoU hides in dense areas.
- Per-density-bin IoU: split the eval tiles into low / medium / high building
  cover and report all three. Your walled-city performance will be much worse
  than your outskirts performance, and averaging conceals it.

---

## 5. Stage 3 — Inference and geospatial output

```mermaid
flowchart LR
    A["Jaipur GeoTIFF"] --> B["Sliding window<br/>512², stride 384"]
    B --> C["TTA: scales 0.75/1.0/1.25<br/>× flips h/v"]
    C --> D["Average logits"]
    D --> E["Hann-window blend<br/>→ seamless prob map"]
    E --> F["Threshold τ*<br/>from target VAL"]
    F --> G["Remove blobs < 8 m²<br/>(≈113 px)"]
    G --> H["Morphological close 3×3"]
    H --> I["Write GeoTIFF<br/>same CRS + transform"]
    I --> J["rasterio.features.shapes<br/>→ GeoJSON polygons"]
    J --> K["Simplify (Douglas-Peucker, 0.5 m)<br/>+ orthogonalise"]
```

Your `infer.py` already does the sliding window + Hann blending. What it needs
added: TTA, geotransform propagation, and the polygonisation tail.

---

## 6. Stage 4 — Solar and capacity estimation

```mermaid
flowchart TB
    RM["Rooftop mask<br/>(Stage 1 output)"] --> CR["Crop each roof<br/>+ 10% context margin"]
    CR --> UP["Upsample to the GSD<br/>the solar model expects<br/>26.6 cm → BDAPPV scale"]
    UP --> SP["Solar panel model"]
    SP --> FP["<b>Solar water heater filter</b><br/>shape/aspect/texture rules<br/>or a 3-class head"]
    FP --> PM["Panel mask"]

    RM --> AR["Roof area<br/>= n_px × 0.0708 m²"]
    PM --> PA["Panel area<br/>= n_px × 0.0708 m²"]

    AR --> US["Usable area<br/>= roof × k_usable"]
    US --> CAP["Capacity kW<br/>= usable_m² × η × 1.0 kW/m²"]
    PA --> EX["Existing installed kW"]
    CAP --> NET["Remaining potential"]
    EX --> NET

    style FP fill:#fce8e6,stroke:#ea4335,color:#000
```

### Constants — update these, the old ones are wrong for Jaipur

| Constant | AIRS value (current code) | **Jaipur value** |
|----------|---------------------------|------------------|
| GSD | 0.075 m/px | **0.26613 m/px** |
| Area per pixel | 56.25 cm² | **708.3 cm² = 0.07083 m²** |
| `k_usable` (fraction of roof usable after setbacks, tanks, stairwells, shading) | — | **0.55–0.70** for Indian flat roofs; use 0.60 and state it |
| Module efficiency η | — | 0.20 (modern mono-PERC) |
| Peak irradiance | — | 1.0 kW/m² STC |

So: `kW ≈ roof_pixels × 0.07083 × 0.60 × 0.20`.

**Make the GSD a parameter read from the GeoTIFF, never a hard-coded literal.**
Hard-coding it is exactly how the 10 cm assumption would have propagated a
2.8× error into every capacity number in your report.

### The solar water heater problem

Flat-plate and evacuated-tube solar water heaters are far more common on
Jaipur roofs than PV, and from directly overhead they present as dark
rectangular arrays — visually near-identical to a PV module to a model trained
only on French PV. Options, cheapest first:

1. **Report precision/recall separately** and characterise the confusion
   honestly. This is a legitimate finding, not a failure.
2. **Geometric rules** — evacuated-tube heaters show a strong periodic 1-D
   ripple and are usually accompanied by a cylindrical tank casting a round
   shadow. At 26.6 cm this is marginal but not hopeless.
3. **Add a third class** and hand-label ~200 water-heater instances from your
   Jaipur tiles. Best solution, ~4 hours of labelling, and it turns a weakness
   into a contribution.

---

## 7. Evaluation protocol (write this down and stick to it)

| Rule | Reason |
|------|--------|
| Spatial block splits, never random tile splits | Overlapping tiles leak |
| Threshold tuned on target **val**, reported on target **test** | Otherwise you tuned on your test set |
| Sealed test split touched once, at the end | Every peek is an implicit fit |
| Report AIRS in-domain IoU alongside Jaipur IoU | The gap *is* the result |
| Report per-density-bin and boundary IoU | Aggregate IoU hides dense-area failure |
| Report zero-shot baseline in every table | Without it, no one can see what adaptation bought |

The headline table in your report should look like:

| Model | AIRS test IoU | Jaipur test IoU | Δ |
|-------|---------------|-----------------|---|
| Current (AIRS-only, zero-shot) | 0.8664 | *(measure in Phase A)* | — |
| + Tier-1 alignment | 0.8664 | | |
| + robust multi-source | | | |
| + weak supervision | | | |
| + self-training | | | |
