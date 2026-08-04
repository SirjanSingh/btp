# 06 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the AIRS-trained rooftop segmentation model to Jaipur imagery at
26.6 cm/px, with a labelled target-domain evaluation set and a documented,
reproducible ablation from zero-shot baseline to final model.

**Architecture:** A new `adapt/` package holds everything target-domain
specific: georeferencing-aware tiling, a weak-label engine built on open
building footprints plus SAM2, domain-adaptation augmentations (FDA, GSD
simulation), and an evaluation harness. Existing `rooftop/` training code is
extended, not replaced — `train.py` gains a multi-dataset loader and an
`ignore_index`-aware loss so the same script serves source, weak, and clean
finetuning stages.

**Tech Stack:** PyTorch 2.1.2 + CUDA 11.8, `segmentation-models-pytorch`,
`albumentations`, `rasterio`, `geopandas`, `shapely`, `pyproj`,
`transformers` (SAM2), `opencv-python`, `pytest`.

## Global Constraints

- Target ground GSD is **0.26613 m/px**. Never hard-code `0.075` or `0.10`
  anywhere; read `ModelPixelScale` from the GeoTIFF and correct by
  `cos(latitude)`.
- Area per Jaipur pixel is **0.070825 m²**.
- All Jaipur splits are **spatial blocks of 1 km**, never random tiles.
- The hand-labelled **test** split is sealed: it is read exactly once, in Task 12.
- `daraset/`, `*.pdf`, and `rooftop/*.tif` must be gitignored before any commit.
- No commit may contain a `Co-Authored-By:` or AI-attribution trailer.
- Every new module gets a pytest file under `tests/`.

---

### Task 0: Repository hygiene

**Files:**
- Modify: `.gitignore`
- Create: `tests/__init__.py`, `adapt/__init__.py`

- [ ] **Step 1: Add the large untracked artefacts to `.gitignore`**

Append to `.gitignore`:

```gitignore
# ── Jaipur target-domain data (1+ GB) ────────────────
daraset/
jaipur/
jaipur_crops/
adapt/labels/
adapt/cache/

# ── Papers & reference rasters ───────────────────────
*.pdf
rooftop/*.tif
rooftop/eval_results/*.png
```

- [ ] **Step 2: Verify nothing large is staged**

Run: `git status --porcelain | head -30`
Expected: no `daraset/`, no `airs.pdf`, no `christchurch_370.tif`.

- [ ] **Step 3: Create the package skeleton**

```bash
mkdir -p adapt tests
printf '' > adapt/__init__.py
printf '' > tests/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore adapt/__init__.py tests/__init__.py
git commit -m "chore: gitignore target-domain data, scaffold adapt package"
```

---

### Task 1: Geospatial helper — read true GSD from a GeoTIFF

**Files:**
- Create: `adapt/geo.py`
- Test: `tests/test_geo.py`

**Interfaces:**
- Produces: `true_gsd(path: str) -> float`,
  `tile_bounds(transform, row_off, col_off, h, w) -> tuple[float,float,float,float]`,
  `TileMeta` dataclass with fields
  `tile_id: str, src: str, row_off: int, col_off: int, height: int, width: int,
  transform: tuple[float,...], crs: str, gsd_ground_m: float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geo.py
import math
import pytest
from adapt.geo import mercator_to_ground_gsd, true_gsd, TileMeta


def test_mercator_to_ground_gsd_at_jaipur():
    # Web Mercator z19 pixel scale, at Jaipur's latitude
    got = mercator_to_ground_gsd(0.2985821417389427, lat_deg=26.95759)
    assert got == pytest.approx(0.26613, abs=1e-4)


def test_mercator_to_ground_gsd_at_equator_is_identity():
    assert mercator_to_ground_gsd(0.5, lat_deg=0.0) == pytest.approx(0.5)


def test_tile_meta_area_per_pixel():
    m = TileMeta(
        tile_id="t", src="s.tif", row_off=0, col_off=0, height=512, width=512,
        transform=(0.2985821, 0.0, 0.0, 0.0, -0.2985821, 0.0),
        crs="EPSG:3857", gsd_ground_m=0.26613,
    )
    assert m.area_per_pixel_m2 == pytest.approx(0.070825, abs=1e-6)


def test_true_gsd_reads_real_tile():
    got = true_gsd("daraset/map67_1-1.tif")
    assert got == pytest.approx(0.26613, abs=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_geo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adapt.geo'`

- [ ] **Step 3: Implement**

```python
# adapt/geo.py
"""Geospatial helpers for Web Mercator target-domain imagery.

Web Mercator (EPSG:3857) reports distances in "Mercator metres", which are
stretched relative to true ground metres by 1/cos(latitude). Every GSD in this
project must be corrected before use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import rasterio

EARTH_RADIUS_M = 6378137.0


def mercator_y_to_lat(y: float) -> float:
    """Inverse Web Mercator northing -> latitude in degrees."""
    return math.degrees(math.atan(math.sinh(y / EARTH_RADIUS_M)))


def mercator_to_ground_gsd(pixel_scale_m: float, lat_deg: float) -> float:
    """Convert a Web Mercator pixel scale to true ground metres per pixel."""
    return pixel_scale_m * math.cos(math.radians(lat_deg))


def true_gsd(path: str) -> float:
    """Read a GeoTIFF and return its true ground sample distance in m/px.

    For EPSG:3857 rasters the stored pixel size is corrected by cos(lat) at the
    raster centre. For projected CRSs already in ground metres the stored pixel
    size is returned unchanged.
    """
    with rasterio.open(path) as src:
        px = abs(src.transform.a)
        epsg = src.crs.to_epsg() if src.crs else None
        if epsg != 3857:
            return px
        centre_y = (src.bounds.top + src.bounds.bottom) / 2.0
        return mercator_to_ground_gsd(px, mercator_y_to_lat(centre_y))


@dataclass(frozen=True)
class TileMeta:
    tile_id: str
    src: str
    row_off: int
    col_off: int
    height: int
    width: int
    transform: tuple[float, ...]
    crs: str
    gsd_ground_m: float

    @property
    def area_per_pixel_m2(self) -> float:
        return self.gsd_ground_m ** 2

    def to_dict(self) -> dict:
        return asdict(self)


def tile_bounds(transform, row_off: int, col_off: int, h: int, w: int):
    """Return (left, bottom, right, top) in the raster's CRS for a tile."""
    a, _, c, _, e, f = transform[0], transform[1], transform[2], transform[3], transform[4], transform[5]
    left = c + col_off * a
    top = f + row_off * e
    right = left + w * a
    bottom = top + h * e
    return (left, min(bottom, top), right, max(bottom, top))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_geo.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add adapt/geo.py tests/test_geo.py
git commit -m "feat(adapt): geospatial helpers with Mercator GSD correction"
```

---

### Task 2: Georeferencing-aware tiler for the Jaipur mosaics

**Files:**
- Create: `adapt/tile_jaipur.py`
- Test: `tests/test_tile_jaipur.py`

**Interfaces:**
- Consumes: `adapt.geo.TileMeta`, `adapt.geo.true_gsd`
- Produces: `tile_raster(src_path, out_dir, crop=512, overlap=0.1,
  min_valid_frac=0.7) -> list[TileMeta]`, and
  `assign_blocks(metas, block_m=1000.0, fracs=(0.7,0.15,0.15), seed=0) -> dict[str,str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tile_jaipur.py
import numpy as np
import rasterio
from rasterio.transform import Affine
import pytest

from adapt.tile_jaipur import tile_raster, assign_blocks


@pytest.fixture
def fake_raster(tmp_path):
    path = tmp_path / "fake.tif"
    h = w = 1200
    data = np.random.randint(20, 240, (3, h, w), dtype=np.uint8)
    data[:, :, :200] = 0  # a blank strip that must be rejected
    transform = Affine(0.2985821, 0.0, 8429249.6, 0.0, -0.2985821, 3118173.77)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=3,
        dtype="uint8", crs="EPSG:3857", transform=transform,
    ) as dst:
        dst.write(data)
    return str(path)


def test_tiling_produces_expected_count_and_meta(fake_raster, tmp_path):
    out = tmp_path / "tiles"
    metas = tile_raster(fake_raster, str(out), crop=512, overlap=0.1)
    assert len(metas) > 0
    m = metas[0]
    assert m.width == 512 and m.height == 512
    assert m.crs == "EPSG:3857"
    assert m.gsd_ground_m == pytest.approx(0.26613, abs=1e-3)
    assert (out / f"{m.tile_id}.png").exists()
    assert (out / f"{m.tile_id}.json").exists()


def test_blank_tiles_are_rejected(fake_raster, tmp_path):
    metas = tile_raster(fake_raster, str(tmp_path / "t"), crop=512, overlap=0.0)
    # the left 200px strip is black; no surviving tile may start at col 0
    assert all(m.col_off != 0 for m in metas)


def test_assign_blocks_is_spatial_not_random(fake_raster, tmp_path):
    metas = tile_raster(fake_raster, str(tmp_path / "t"), crop=512, overlap=0.0)
    split = assign_blocks(metas, block_m=200.0, fracs=(0.7, 0.15, 0.15), seed=0)
    assert set(split.values()) <= {"train", "val", "test"}
    # every tile is assigned
    assert len(split) == len(metas)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tile_jaipur.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adapt.tile_jaipur'`

- [ ] **Step 3: Implement**

```python
# adapt/tile_jaipur.py
"""Tile the Jaipur Web Mercator mosaics into 512x512 crops that carry their
own geotransform, and assign them to spatial train/val/test blocks.

Spatial block splitting is mandatory: overlapping tiles that are split randomly
put the same pixels in train and test and invalidate every reported number.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

from adapt.geo import TileMeta, true_gsd


def tile_raster(
    src_path: str,
    out_dir: str,
    crop: int = 512,
    overlap: float = 0.1,
    min_valid_frac: float = 0.7,
    min_std: float = 4.0,
) -> list[TileMeta]:
    os.makedirs(out_dir, exist_ok=True)
    gsd = true_gsd(src_path)
    stem = os.path.splitext(os.path.basename(src_path))[0]
    stride = max(1, int(round(crop * (1.0 - overlap))))
    metas: list[TileMeta] = []

    with rasterio.open(src_path) as src:
        for r_i, row in enumerate(range(0, src.height - crop + 1, stride)):
            for c_i, col in enumerate(range(0, src.width - crop + 1, stride)):
                win = Window(col, row, crop, crop)
                arr = src.read((1, 2, 3), window=win)  # (3, H, W)

                valid = (arr.sum(axis=0) > 0).mean()
                if valid < min_valid_frac or arr.std() < min_std:
                    continue

                tile_id = f"{stem}_r{r_i:03d}_c{c_i:03d}"
                wt = src.window_transform(win)
                meta = TileMeta(
                    tile_id=tile_id, src=os.path.basename(src_path),
                    row_off=row, col_off=col, height=crop, width=crop,
                    transform=(wt.a, wt.b, wt.c, wt.d, wt.e, wt.f),
                    crs=str(src.crs), gsd_ground_m=gsd,
                )
                Image.fromarray(np.transpose(arr, (1, 2, 0))).save(
                    os.path.join(out_dir, f"{tile_id}.png")
                )
                with open(os.path.join(out_dir, f"{tile_id}.json"), "w") as fh:
                    json.dump(meta.to_dict(), fh)
                metas.append(meta)
    return metas


def assign_blocks(
    metas: list[TileMeta],
    block_m: float = 1000.0,
    fracs: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, str]:
    """Assign each tile to train/val/test by the spatial block it falls in."""
    def block_key(m: TileMeta) -> tuple[int, int]:
        # Block in the raster's own projected units. block_m is given in ground
        # metres, so convert to projected (Mercator) metres first.
        step = block_m * abs(m.transform[0]) / m.gsd_ground_m
        return (int(m.transform[2] // step), int(m.transform[5] // step))

    keys = sorted({block_key(m) for m in metas})
    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_tr = max(1, int(round(n * fracs[0])))
    n_va = max(1, int(round(n * fracs[1]))) if n - n_tr > 1 else 0
    lookup = {}
    for i, k in enumerate(keys):
        lookup[k] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return {m.tile_id: lookup[block_key(m)] for m in metas}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tile_jaipur.py -v`
Expected: 3 passed.

- [ ] **Step 5: Tile the real data and record the counts**

Stage the data on **G:** or the DGX first — it does not fit on D: (see
`plan/01-situation-and-assets.md` §7.1).

```bash
python -c "
from adapt.tile_jaipur import tile_raster, assign_blocks
from collections import Counter
import glob, json, os

SRC = os.environ.get('JAIPUR_SRC', 'G:/btp_data/daraset')
OUT = os.environ.get('JAIPUR_OUT', 'G:/btp_data/jaipur_crops')

metas = []
for f in sorted(glob.glob(f'{SRC}/map67_*-*.tif')):
    n = tile_raster(f, f'{OUT}/tiles')
    print(f'{os.path.basename(f)}: {len(n)} tiles')
    metas += n

# hold out whole mosaic tiles for test, then block-split the rest
s = assign_blocks(metas)
os.makedirs(OUT, exist_ok=True)
json.dump(s, open(f'{OUT}/split.json','w'), indent=1)
print(len(metas), 'tiles total'); print(Counter(s.values()))
"
```

Expected: 696 tiles per source file, ~11,140 total for all 16.

Record the counts in `plan/01-situation-and-assets.md`.

- [ ] **Step 6: Commit**

```bash
git add adapt/tile_jaipur.py tests/test_tile_jaipur.py
git commit -m "feat(adapt): georeferenced tiler with spatial block splits"
```

---

### Task 3: Zero-shot baseline — measure the gap before changing anything

**Files:**
- Create: `adapt/eval_target.py`
- Test: `tests/test_eval_target.py`

**Interfaces:**
- Produces: `evaluate(pred_dir, gt_dir, thresholds) -> dict` returning
  `{"iou": float, "f1": float, "precision": float, "recall": float,
    "boundary_iou": float, "best_threshold": float, "per_density": dict}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_target.py
import numpy as np
import pytest
from adapt.eval_target import global_iou, boundary_iou, sweep_threshold


def test_global_iou_perfect():
    gt = np.zeros((32, 32), np.uint8); gt[8:24, 8:24] = 1
    assert global_iou(gt.astype(bool), gt.astype(bool)) == pytest.approx(1.0)


def test_global_iou_half():
    gt = np.zeros((10, 10), bool); gt[:, :5] = True
    pr = np.zeros((10, 10), bool); pr[:, 2:7] = True
    # intersection 30, union 70
    assert global_iou(gt, pr) == pytest.approx(30 / 70)


def test_global_iou_ignores_ignore_index():
    gt = np.zeros((4, 4), np.int8); gt[0] = 1; gt[1] = -1
    pr = np.zeros((4, 4), bool); pr[0] = True
    assert global_iou(gt, pr, ignore_value=-1) == pytest.approx(1.0)


def test_boundary_iou_penalises_dilated_prediction():
    gt = np.zeros((64, 64), bool); gt[16:48, 16:48] = True
    fat = np.zeros((64, 64), bool); fat[12:52, 12:52] = True
    assert boundary_iou(gt, fat, band_px=5) < global_iou(gt, fat)


def test_sweep_threshold_picks_best():
    gt = np.zeros((8, 8), bool); gt[:4] = True
    prob = np.where(gt, 0.6, 0.1)
    best, table = sweep_threshold(gt, prob, np.arange(0.05, 0.95, 0.05))
    assert 0.1 < best < 0.6
    assert len(table) == len(np.arange(0.05, 0.95, 0.05))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_target.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adapt.eval_target'`

- [ ] **Step 3: Implement**

```python
# adapt/eval_target.py
"""Target-domain evaluation with ignore-index support, boundary IoU, and a
threshold sweep. Metrics are accumulated globally, never averaged per-tile.
"""
from __future__ import annotations

import numpy as np
import cv2


def _valid(gt: np.ndarray, ignore_value):
    if ignore_value is None:
        return np.ones(gt.shape, bool)
    return gt != ignore_value


def global_iou(gt, pred, ignore_value=None) -> float:
    v = _valid(np.asarray(gt), ignore_value)
    g = np.asarray(gt).astype(np.int16)[v] > 0
    p = np.asarray(pred).astype(bool)[v]
    inter = np.logical_and(g, p).sum()
    union = np.logical_or(g, p).sum()
    return float(inter / union) if union else 1.0


def prf(gt, pred, ignore_value=None) -> tuple[float, float, float]:
    v = _valid(np.asarray(gt), ignore_value)
    g = np.asarray(gt).astype(np.int16)[v] > 0
    p = np.asarray(pred).astype(bool)[v]
    tp = np.logical_and(g, p).sum()
    fp = np.logical_and(~g, p).sum()
    fn = np.logical_and(g, ~p).sum()
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return float(prec), float(rec), float(f1)


def _edge_band(mask: np.ndarray, band_px: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(m, k, iterations=band_px)
    ero = cv2.erode(m, k, iterations=band_px)
    return (dil - ero).astype(bool)


def boundary_iou(gt, pred, band_px: int = 5) -> float:
    """IoU restricted to a band around the ground-truth boundary.

    Catches the dense-urban failure mode where whole blocks are merged into one
    blob: plain IoU stays high while boundaries are entirely wrong.
    """
    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)
    band = _edge_band(gt, band_px)
    if not band.any():
        return 1.0
    return global_iou(gt[band], pred[band])


def sweep_threshold(gt, prob, thresholds):
    """Return (best_threshold, [(t, iou), ...]) maximising global IoU."""
    table = [(float(t), global_iou(gt, np.asarray(prob) >= t)) for t in thresholds]
    best = max(table, key=lambda x: x[1])[0]
    return best, table


def density_bin(gt: np.ndarray) -> str:
    frac = float(np.asarray(gt).astype(bool).mean())
    if frac < 0.20:
        return "low"
    if frac < 0.45:
        return "medium"
    return "high"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_target.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run the zero-shot baseline and record it**

Once Task 6 has produced labels, run the existing AIRS checkpoint over the
Jaipur val tiles and fill in the headline table in
`plan/04-pipeline.md` §7. Until then, run it qualitatively on
`test_indian/` and save overlays.

- [ ] **Step 6: Commit**

```bash
git add adapt/eval_target.py tests/test_eval_target.py
git commit -m "feat(adapt): target-domain metrics with boundary IoU and threshold sweep"
```

---

### Task 4: Domain-adaptation augmentations — GSD simulation and FDA

**Files:**
- Create: `adapt/augment.py`
- Test: `tests/test_augment.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `SimulateGSD(src_gsd, tgt_gsd_range, jpeg_range, p)` (Albumentations
  `DualTransform`), `FDA(target_images, beta, p)` (Albumentations
  `ImageOnlyTransform`), `robust_train_transform(src_gsd, target_tiles) -> A.Compose`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_augment.py
import numpy as np
import pytest
from adapt.augment import SimulateGSD, FDA, fda_blend


def test_simulate_gsd_preserves_shape_and_mask_alignment():
    img = np.random.randint(0, 255, (512, 512, 3), np.uint8)
    mask = np.zeros((512, 512), np.uint8); mask[100:300, 100:300] = 1
    t = SimulateGSD(src_gsd=0.075, tgt_gsd_range=(0.26, 0.27), p=1.0)
    out = t(image=img, mask=mask)
    assert out["image"].shape == img.shape
    assert out["mask"].shape == mask.shape
    # the square is still roughly where it was, and still ~the same fraction
    assert out["mask"].mean() == pytest.approx(mask.mean(), abs=0.05)


def test_simulate_gsd_actually_removes_high_frequency():
    img = np.random.randint(0, 255, (256, 256, 3), np.uint8)
    t = SimulateGSD(src_gsd=0.075, tgt_gsd_range=(0.26, 0.27), jpeg_range=None, p=1.0)
    out = t(image=img, mask=np.zeros((256, 256), np.uint8))["image"]
    lap = lambda a: np.abs(np.diff(a.astype(np.int16), axis=0)).mean()
    assert lap(out) < lap(img)


def test_fda_blend_keeps_source_phase():
    src = np.random.randint(0, 255, (64, 64, 3), np.uint8)
    tgt = np.random.randint(0, 255, (64, 64, 3), np.uint8)
    out = fda_blend(src, tgt, beta=0.01)
    assert out.shape == src.shape and out.dtype == np.uint8
    # beta=0 must be a no-op
    assert np.allclose(fda_blend(src, tgt, beta=0.0), src, atol=2)


def test_fda_transform_runs_in_pipeline():
    tgt = [np.random.randint(0, 255, (128, 128, 3), np.uint8) for _ in range(3)]
    t = FDA(target_images=tgt, beta=0.01, p=1.0)
    img = np.random.randint(0, 255, (128, 128, 3), np.uint8)
    assert t(image=img)["image"].shape == img.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_augment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adapt.augment'`

- [ ] **Step 3: Implement**

```python
# adapt/augment.py
"""Domain-adaptation augmentations.

SimulateGSD models the *degradation chain* of web mosaic imagery -- anti-
aliased downsample, JPEG compression, mild resharpen -- not just a naive
resize. A naive resize produces unrealistically clean low-resolution images and
the model will not transfer.

FDA (Yang & Soatto, CVPR 2020) swaps the low-frequency Fourier amplitude of a
source image for a target image's, keeping source phase. No network, no
adversarial training.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform


class SimulateGSD(DualTransform):
    """Degrade a high-resolution source crop to a sampled target GSD."""

    def __init__(self, src_gsd=0.075, tgt_gsd_range=(0.15, 0.40),
                 jpeg_range=(55, 95), sharpen_p=0.5, p=1.0):
        super().__init__(p=p)
        self.src_gsd = src_gsd
        self.tgt_gsd_range = tgt_gsd_range
        self.jpeg_range = jpeg_range
        self.sharpen_p = sharpen_p

    def get_params(self):
        return {"tgt_gsd": random.uniform(*self.tgt_gsd_range),
                "jpeg_q": (random.randint(*self.jpeg_range)
                           if self.jpeg_range else None),
                "do_sharpen": random.random() < self.sharpen_p}

    def apply(self, img, tgt_gsd=0.266, jpeg_q=None, do_sharpen=False, **kw):
        h, w = img.shape[:2]
        f = self.src_gsd / tgt_gsd                      # < 1 -> shrink
        sh, sw = max(8, int(round(h * f))), max(8, int(round(w * f)))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        if jpeg_q is not None:
            ok, buf = cv2.imencode(".jpg", small,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
            if ok:
                small = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if do_sharpen:
            blur = cv2.GaussianBlur(small, (0, 0), 1.0)
            small = cv2.addWeighted(small, 1.4, blur, -0.4, 0)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    def apply_to_mask(self, mask, tgt_gsd=0.266, **kw):
        h, w = mask.shape[:2]
        f = self.src_gsd / tgt_gsd
        sh, sw = max(8, int(round(h * f))), max(8, int(round(w * f)))
        small = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def get_transform_init_args_names(self):
        return ("src_gsd", "tgt_gsd_range", "jpeg_range", "sharpen_p")


def fda_blend(src: np.ndarray, tgt: np.ndarray, beta: float = 0.01) -> np.ndarray:
    """Replace the central beta-fraction of src's amplitude spectrum with tgt's."""
    if beta <= 0:
        return src.copy()
    h, w = src.shape[:2]
    tgt = cv2.resize(tgt, (w, h), interpolation=cv2.INTER_AREA)
    out = np.empty_like(src, dtype=np.float32)
    b = int(np.floor(min(h, w) * beta))
    cy, cx = h // 2, w // 2
    for c in range(3):
        fs = np.fft.fftshift(np.fft.fft2(src[..., c].astype(np.float32)))
        ft = np.fft.fftshift(np.fft.fft2(tgt[..., c].astype(np.float32)))
        amp_s, pha_s = np.abs(fs), np.angle(fs)
        amp_t = np.abs(ft)
        if b > 0:
            amp_s[cy - b:cy + b + 1, cx - b:cx + b + 1] = \
                amp_t[cy - b:cy + b + 1, cx - b:cx + b + 1]
        rec = np.fft.ifft2(np.fft.ifftshift(amp_s * np.exp(1j * pha_s)))
        out[..., c] = np.real(rec)
    return np.clip(out, 0, 255).astype(np.uint8)


class FDA(ImageOnlyTransform):
    """Fourier Domain Adaptation against a pool of unlabelled target images."""

    def __init__(self, target_images, beta=0.01, p=0.3):
        super().__init__(p=p)
        if not target_images:
            raise ValueError("FDA needs at least one target image")
        self.target_images = target_images
        self.beta = beta

    def apply(self, img, **kw):
        return fda_blend(img, random.choice(self.target_images), self.beta)

    def get_transform_init_args_names(self):
        return ("beta",)


def robust_train_transform(src_gsd: float = 0.075, target_tiles=None,
                           crop: int = 512) -> A.Compose:
    """The Phase-C augmentation policy: span the target domain's variation."""
    tf = [
        SimulateGSD(src_gsd=src_gsd, tgt_gsd_range=(0.15, 0.40), p=0.8),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(0.4, 0.4, 0.4, 0.15, p=0.8),
        A.RandomGamma(gamma_limit=(60, 160), p=0.5),
        A.CLAHE(p=0.3),
        A.GaussNoise(p=0.3),
        A.OneOf([A.MotionBlur(), A.GaussianBlur(), A.Sharpen()], p=0.4),
        A.RandomShadow(p=0.2),
    ]
    if target_tiles:
        tf.append(FDA(target_images=target_tiles, beta=0.01, p=0.3))
    tf.append(A.PadIfNeeded(crop, crop))
    return A.Compose(tf)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_augment.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add adapt/augment.py tests/test_augment.py
git commit -m "feat(adapt): GSD degradation and Fourier domain adaptation augments"
```

---

### Task 5: Weak labels from open building footprints

**Files:**
- Create: `adapt/openbuildings.py`
- Test: `tests/test_openbuildings.py`

**Interfaces:**
- Consumes: `adapt.geo.TileMeta`
- Produces: `download_hint() -> str`,
  `rasterize_for_tile(gdf, meta) -> np.ndarray (uint8, 0/1)`,
  `estimate_shift(mask, prob, max_shift_px=40) -> tuple[int,int]`,
  `apply_shift(mask, dy, dx) -> np.ndarray`,
  `relax_boundary(mask, band_px=4, ignore_value=255) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_openbuildings.py
import numpy as np
import pytest
from shapely.geometry import box
import geopandas as gpd

from adapt.geo import TileMeta
from adapt.openbuildings import (
    rasterize_for_tile, estimate_shift, apply_shift, relax_boundary,
)


@pytest.fixture
def meta():
    return TileMeta(
        tile_id="t", src="s.tif", row_off=0, col_off=0, height=128, width=128,
        transform=(0.2985821, 0.0, 1000.0, 0.0, -0.2985821, 2000.0),
        crs="EPSG:3857", gsd_ground_m=0.26613,
    )


def test_rasterize_places_polygon_in_correct_pixels(meta):
    # a 10x10 px square starting 20px in from the top-left
    a = meta.transform[0]
    left = 1000.0 + 20 * a
    top = 2000.0 - 20 * a
    gdf = gpd.GeoDataFrame(
        {"confidence": [0.9]},
        geometry=[box(left, top - 10 * a, left + 10 * a, top)],
        crs="EPSG:3857",
    )
    m = rasterize_for_tile(gdf, meta)
    assert m.shape == (128, 128)
    assert m[25, 25] == 1
    assert m[5, 5] == 0


def test_rasterize_drops_low_confidence(meta):
    a = meta.transform[0]
    gdf = gpd.GeoDataFrame(
        {"confidence": [0.4]},
        geometry=[box(1000.0, 2000.0 - 10 * a, 1000.0 + 10 * a, 2000.0)],
        crs="EPSG:3857",
    )
    assert rasterize_for_tile(gdf, meta, min_confidence=0.75).sum() == 0


def test_estimate_shift_recovers_a_known_offset():
    prob = np.zeros((128, 128), np.float32); prob[40:80, 40:80] = 1.0
    mask = np.zeros((128, 128), np.uint8); mask[47:87, 51:91] = 1  # +7, +11
    dy, dx = estimate_shift(mask, prob, max_shift_px=20)
    assert (dy, dx) == pytest.approx((-7, -11), abs=2)


def test_apply_shift_moves_the_mask():
    m = np.zeros((16, 16), np.uint8); m[4:8, 4:8] = 1
    s = apply_shift(m, 2, 3)
    assert s[6:10, 7:11].sum() == 16


def test_relax_boundary_marks_an_ignore_band():
    m = np.zeros((64, 64), np.uint8); m[16:48, 16:48] = 1
    r = relax_boundary(m, band_px=3, ignore_value=255)
    assert (r == 255).sum() > 0
    assert r[32, 32] == 1            # interior untouched
    assert r[0, 0] == 0              # far background untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_openbuildings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adapt.openbuildings'`

- [ ] **Step 3: Implement**

```python
# adapt/openbuildings.py
"""Turn openly-licensed building footprints into weak labels for the Jaipur
tiles.

These footprints come from a different sensor at a coarser GSD and are commonly
offset by 2-8 m, and they describe ground footprints rather than roof outlines.
They are noisy labels, not ground truth -- hence shift correction and boundary
relaxation before they are ever fed to a loss.

Sources (both openly licensed -- see plan/08-sources.md):
  Microsoft GlobalMLBuildingFootprints  (ODbL v1.0)   110M India footprints
  Google Open Buildings v3              (CC BY 4.0 / ODbL)
  VIDA combined Google+Microsoft+OSM    (ODbL)        easiest single download
"""
from __future__ import annotations

import cv2
import numpy as np
from rasterio.transform import Affine
from rasterio.features import rasterize


DOWNLOAD_HINT = """\
Microsoft: https://github.com/microsoft/GlobalMLBuildingFootprints
    dataset-links.csv -> pick the quadkeys covering Jaipur (26.96N, 75.72E)
Google:    https://sites.research.google/gr/open-buildings/#open-buildings-download
    S2 cell for Rajasthan
VIDA:      https://source.coop/vida/google-microsoft-osm-open-buildings
Load with: gpd.read_file(path).to_crs('EPSG:3857')
"""


def download_hint() -> str:
    return DOWNLOAD_HINT


def rasterize_for_tile(gdf, meta, min_confidence: float = 0.75) -> np.ndarray:
    """Rasterise footprint polygons onto a tile's pixel grid."""
    if "confidence" in gdf.columns:
        gdf = gdf[gdf["confidence"] >= min_confidence]
    if len(gdf) == 0:
        return np.zeros((meta.height, meta.width), np.uint8)
    a, b, c, d, e, f = meta.transform
    tf = Affine(a, b, c, d, e, f)
    return rasterize(
        ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty),
        out_shape=(meta.height, meta.width),
        transform=tf, fill=0, dtype=np.uint8, all_touched=False,
    )


def estimate_shift(mask: np.ndarray, prob: np.ndarray,
                   max_shift_px: int = 40) -> tuple[int, int]:
    """Estimate the (dy, dx) that best aligns `mask` onto `prob`.

    Uses normalised cross-correlation over a bounded search window, which is
    more robust here than FFT phase correlation because both inputs are
    near-binary and sparse.
    """
    m = mask.astype(np.float32)
    p = prob.astype(np.float32)
    m = m - m.mean()
    p = p - p.mean()
    pad = max_shift_px
    mp = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    res = cv2.matchTemplate(mp, p, cv2.TM_CCORR_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    dx = max_loc[0] - pad
    dy = max_loc[1] - pad
    return int(dy), int(dx)


def apply_shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate a mask by (dy, dx), zero-filling the vacated edge."""
    h, w = mask.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def relax_boundary(mask: np.ndarray, band_px: int = 4,
                   ignore_value: int = 255) -> np.ndarray:
    """Mark a band around every label edge as `ignore` so residual
    misalignment contributes no gradient."""
    m = (mask > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    band = cv2.dilate(m, k, iterations=band_px) - cv2.erode(m, k, iterations=band_px)
    out = m.copy()
    out[band.astype(bool)] = ignore_value
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_openbuildings.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add adapt/openbuildings.py tests/test_openbuildings.py
git commit -m "feat(adapt): weak labels from open building footprints with shift correction"
```

---

### Task 6: SAM2 label refinement

**Files:**
- Create: `adapt/sam_refine.py`
- Test: `tests/test_sam_refine.py`

**Interfaces:**
- Consumes: `adapt.openbuildings.rasterize_for_tile`
- Produces: `polygon_centroids_px(gdf, meta) -> np.ndarray (N, 2)`,
  `refine_with_sam(image, centroids, poly_mask, predictor, min_iou=0.5) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam_refine.py
import numpy as np
import pytest
from adapt.sam_refine import refine_with_sam, _iou


class FakePredictor:
    """Stands in for SAM2; returns a fixed square around each prompt point."""
    def __init__(self, half=10, offset=0):
        self.half, self.offset = half, offset
    def set_image(self, img):
        self.shape = img.shape[:2]
    def predict(self, point_coords, point_labels, multimask_output=False):
        m = np.zeros(self.shape, bool)
        x, y = int(point_coords[0][0]) + self.offset, int(point_coords[0][1])
        m[max(0, y - self.half):y + self.half, max(0, x - self.half):x + self.half] = True
        return m[None], np.array([0.9]), None


def test_refine_accepts_well_matched_sam_mask():
    img = np.zeros((128, 128, 3), np.uint8)
    poly = np.zeros((128, 128), np.uint8); poly[54:74, 54:74] = 1
    out = refine_with_sam(img, np.array([[64, 64]]), poly,
                          FakePredictor(half=10), min_iou=0.5)
    assert out.sum() > 0
    assert _iou(out > 0, poly > 0) > 0.8


def test_refine_falls_back_when_sam_disagrees():
    img = np.zeros((128, 128, 3), np.uint8)
    poly = np.zeros((128, 128), np.uint8); poly[54:74, 54:74] = 1
    # SAM returns a mask displaced far away -> must fall back to the polygon
    out = refine_with_sam(img, np.array([[64, 64]]), poly,
                          FakePredictor(half=10, offset=50), min_iou=0.5)
    assert _iou(out > 0, poly > 0) == pytest.approx(1.0)


def test_refine_with_no_centroids_returns_polygon_mask():
    img = np.zeros((32, 32, 3), np.uint8)
    poly = np.zeros((32, 32), np.uint8); poly[8:16, 8:16] = 1
    out = refine_with_sam(img, np.empty((0, 2)), poly, FakePredictor())
    assert np.array_equal(out, poly)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sam_refine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adapt.sam_refine'`

- [ ] **Step 3: Implement**

```python
# adapt/sam_refine.py
"""Snap coarse building footprints onto the roof edges actually visible in the
target imagery, using SAM 2 with each polygon's centroid as a point prompt.

SAM knows nothing about "buildings" -- prompted zero-shot it segments *some*
region. Prompted at a known building centroid and accepted only when it agrees
with the prior polygon, it is an excellent label refiner. The IoU gate is what
stops SAM replacing a house with the whole courtyard it sits in.
"""
from __future__ import annotations

import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def polygon_centroids_px(gdf, meta) -> np.ndarray:
    """Centroid of every polygon, in (x, y) pixel coordinates of the tile."""
    a, _, c, _, e, f = meta.transform
    pts = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        cx, cy = geom.centroid.x, geom.centroid.y
        px = (cx - c) / a
        py = (cy - f) / e
        if 0 <= px < meta.width and 0 <= py < meta.height:
            pts.append((px, py))
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def refine_with_sam(image: np.ndarray, centroids: np.ndarray,
                    poly_mask: np.ndarray, predictor,
                    min_iou: float = 0.5) -> np.ndarray:
    """Return a refined 0/1 mask.

    For each centroid: prompt SAM, isolate the connected component of
    `poly_mask` containing that centroid, and accept SAM's mask only if it
    overlaps that component by at least `min_iou`. Otherwise keep the polygon.
    """
    import cv2

    if centroids is None or len(centroids) == 0:
        return poly_mask.copy()

    predictor.set_image(image)
    n_lbl, comps = cv2.connectedComponents((poly_mask > 0).astype(np.uint8))
    out = np.zeros(poly_mask.shape, np.uint8)

    for (x, y) in centroids:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= yi < comps.shape[0] and 0 <= xi < comps.shape[1]):
            continue
        lbl = comps[yi, xi]
        prior = (comps == lbl) if lbl > 0 else (poly_mask > 0)

        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        cand = np.asarray(masks[0]).astype(bool)
        out[(cand if _iou(cand, prior) >= min_iou else prior)] = 1

    # anything the prompts never reached keeps its polygon label
    out[(poly_mask > 0) & (out == 0)] = 1
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sam_refine.py -v`
Expected: 3 passed.

- [ ] **Step 5: Wire in the real SAM2 predictor (Colab, L4/A100)**

```python
from transformers import Sam2Processor, Sam2Model   # or the official sam2 repo
# build a predictor exposing .set_image(np.ndarray) and
# .predict(point_coords, point_labels, multimask_output) -> (masks, scores, _)
```

Run over all Jaipur tiles, writing `adapt/labels/weak/<tile_id>.png`.
Budget ~140 compute units (see `plan/05-compute-and-schedule.md`).

- [ ] **Step 6: Commit**

```bash
git add adapt/sam_refine.py tests/test_sam_refine.py
git commit -m "feat(adapt): SAM2 point-prompt label refinement with IoU gating"
```

---

### Task 7: Ignore-index-aware loss and mixed-source dataset

**Files:**
- Create: `adapt/losses.py`, `adapt/datasets.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Consumes: nothing
- Produces: `BceDiceIgnore(ignore_index=255, bce_w=0.5, dice_w=0.5)` (an
  `nn.Module` taking `(logits: (B,1,H,W), target: (B,1,H,W))`),
  `MixedSegDataset(specs: list[DatasetSpec])` where
  `DatasetSpec = (img_dir, mask_dir, sample_weight, transform)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_losses.py
import torch
import pytest
from adapt.losses import BceDiceIgnore


def test_ignored_pixels_contribute_no_gradient():
    loss = BceDiceIgnore(ignore_index=255)
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    target = torch.full((1, 1, 4, 4), 255.0)
    target[0, 0, 0, :] = 1.0
    out = loss(logits, target)
    out.backward()
    g = logits.grad[0, 0]
    assert torch.allclose(g[1:], torch.zeros(3, 4), atol=1e-8)
    assert g[0].abs().sum() > 0


def test_perfect_prediction_gives_near_zero_loss():
    loss = BceDiceIgnore(ignore_index=255)
    target = torch.zeros(1, 1, 8, 8); target[:, :, :4] = 1.0
    logits = torch.where(target > 0.5, 12.0, -12.0)
    assert loss(logits, target).item() < 0.02


def test_all_ignored_returns_zero_not_nan():
    loss = BceDiceIgnore(ignore_index=255)
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    target = torch.full((1, 1, 4, 4), 255.0)
    out = loss(logits, target)
    assert torch.isfinite(out) and out.item() == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adapt.losses'`

- [ ] **Step 3: Implement**

```python
# adapt/losses.py
"""BCE + Dice with an ignore index.

Needed for two things this project depends on: boundary-relaxed weak labels
(Task 5) and confidence-thresholded pseudo-labels (Task 9). Both mark pixels
that must contribute exactly zero gradient.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BceDiceIgnore(nn.Module):
    def __init__(self, ignore_index: int = 255, bce_w: float = 0.5,
                 dice_w: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce_w, self.dice_w, self.smooth = bce_w, dice_w, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        valid = (target != self.ignore_index).float()
        n_valid = valid.sum()
        if n_valid == 0:
            return logits.sum() * 0.0

        tgt = torch.where(valid.bool(), target, torch.zeros_like(target))

        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
        bce = (bce * valid).sum() / n_valid

        prob = torch.sigmoid(logits) * valid
        tgt_v = tgt * valid
        inter = (prob * tgt_v).sum()
        denom = prob.sum() + tgt_v.sum()
        dice = 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)

        return self.bce_w * bce + self.dice_w * dice
```

```python
# adapt/datasets.py
"""Mixed-source segmentation dataset: concatenates several (images, masks)
directories with per-source sample weights and per-source transforms, so one
training script serves the source, weak-target, and clean-target stages.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


@dataclass
class DatasetSpec:
    img_dir: str
    mask_dir: Optional[str]
    sample_weight: float = 1.0
    transform: object = None
    name: str = ""


class MixedSegDataset(Dataset):
    """Missing mask file -> all-zeros mask (a genuine negative example)."""

    def __init__(self, specs: list[DatasetSpec]):
        self.items = []
        for si, spec in enumerate(specs):
            names = sorted(f for f in os.listdir(spec.img_dir)
                           if f.lower().endswith(IMG_EXT))
            for n in names:
                self.items.append((si, n))
        self.specs = specs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        si, name = self.items[i]
        spec = self.specs[si]
        img = cv2.cvtColor(
            cv2.imread(os.path.join(spec.img_dir, name), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        mask = np.zeros(img.shape[:2], np.uint8)
        if spec.mask_dir:
            mp = os.path.join(spec.mask_dir, os.path.splitext(name)[0] + ".png")
            if os.path.exists(mp):
                mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                # 0/1 datasets (AIRS) and 0/255 datasets both normalise to 0/1,
                # while 255 as an *ignore* marker is preserved by the caller
                # writing 255 only when relax_boundary was applied.
                if mask.max() == 255 and (mask == 255).mean() > 0.5:
                    mask = (mask > 127).astype(np.uint8)

        if spec.transform is not None:
            out = spec.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        img = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
        return img, mask[None].astype(np.float32), spec.sample_weight
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_losses.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add adapt/losses.py adapt/datasets.py tests/test_losses.py
git commit -m "feat(adapt): ignore-index BCE+Dice loss and mixed-source dataset"
```

---

### Task 8: Tier-1 free adaptation — AdaBN, histogram matching, TTA

**Files:**
- Create: `adapt/tier1.py`
- Test: `tests/test_tier1.py`

**Interfaces:**
- Consumes: `adapt.eval_target.global_iou`
- Produces: `adabn(model, loader, n_batches=50) -> nn.Module`,
  `match_histogram(src, ref) -> np.ndarray`,
  `tta_predict(model, img_tensor, scales=(0.75,1.0,1.25), flips=True) -> torch.Tensor`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tier1.py
import numpy as np
import torch
import torch.nn as nn
import pytest
from adapt.tier1 import adabn, match_histogram, tta_predict


def test_adabn_updates_bn_stats_but_not_weights():
    m = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))
    w0 = m[0].weight.detach().clone()
    rm0 = m[1].running_mean.detach().clone()
    batches = [(torch.randn(2, 3, 16, 16) * 5 + 3,) for _ in range(4)]
    adabn(m, batches, n_batches=4)
    assert torch.allclose(m[0].weight, w0)               # weights frozen
    assert not torch.allclose(m[1].running_mean, rm0)    # stats moved
    assert not m.training                                 # left in eval mode


def test_match_histogram_moves_mean_towards_reference():
    src = (np.random.rand(64, 64, 3) * 60 + 20).astype(np.uint8)
    ref = (np.random.rand(64, 64, 3) * 60 + 160).astype(np.uint8)
    out = match_histogram(src, ref)
    assert out.shape == src.shape and out.dtype == np.uint8
    assert abs(int(out.mean()) - int(ref.mean())) < abs(int(src.mean()) - int(ref.mean()))


def test_tta_predict_is_invariant_to_a_flip_equivariant_model():
    class Identity1(nn.Module):
        def forward(self, x):
            return x[:, :1]
    x = torch.randn(1, 3, 32, 32)
    out = tta_predict(Identity1(), x, scales=(1.0,), flips=True)
    assert out.shape == (1, 1, 32, 32)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tier1.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adapt.tier1'`

- [ ] **Step 3: Implement**

```python
# adapt/tier1.py
"""Tier-1 adaptation: everything that needs no training and no target labels.

Measure each of these separately and report them in the ablation table -- they
are close to free, and reviewers will ask whether you tried them.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def adabn(model: nn.Module, batches, n_batches: int = 50) -> nn.Module:
    """Re-estimate BatchNorm running statistics on unlabelled target data.

    Weights are never updated: only the BN buffers move. No-op for models
    without BatchNorm (e.g. SegFormer, which uses LayerNorm).
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        model.eval()
        return model

    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None      # cumulative moving average

    model.train()
    for i, batch in enumerate(batches):
        if i >= n_batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        model(x)
    model.eval()
    return model


def match_histogram(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-channel CDF matching of `src` onto `ref`."""
    out = np.empty_like(src)
    for c in range(src.shape[2]):
        s = src[..., c].ravel()
        r = ref[..., c].ravel()
        s_vals, s_idx, s_cnt = np.unique(s, return_inverse=True, return_counts=True)
        r_vals, r_cnt = np.unique(r, return_counts=True)
        s_q = np.cumsum(s_cnt).astype(np.float64) / s.size
        r_q = np.cumsum(r_cnt).astype(np.float64) / r.size
        interp = np.interp(s_q, r_q, r_vals)
        out[..., c] = interp[s_idx].reshape(src.shape[:2]).astype(src.dtype)
    return out


@torch.no_grad()
def tta_predict(model, x: torch.Tensor, scales=(0.75, 1.0, 1.25),
                flips: bool = True) -> torch.Tensor:
    """Average logits over scales and flips. Returns (B, 1, H, W)."""
    h, w = x.shape[-2:]
    acc, n = None, 0
    for s in scales:
        xs = x if s == 1.0 else F.interpolate(
            x, size=(int(h * s), int(w * s)), mode="bilinear", align_corners=False)
        variants = [(xs, None)]
        if flips:
            variants += [(torch.flip(xs, [-1]), [-1]),
                         (torch.flip(xs, [-2]), [-2]),
                         (torch.flip(xs, [-1, -2]), [-1, -2])]
        for xv, dims in variants:
            y = model(xv)
            if dims is not None:
                y = torch.flip(y, dims)
            if y.shape[-2:] != (h, w):
                y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            acc = y if acc is None else acc + y
            n += 1
    return acc / n
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tier1.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the Phase-B ablation and fill in the table**

Evaluate the existing AIRS checkpoint on the Jaipur val split under each
setting, cumulatively: `zero-shot` → `+ upsample to 7.5 cm` → `+ AdaBN` →
`+ histogram match` → `+ TTA` → `+ tuned threshold`. Write the results into
`plan/04-pipeline.md` §7.

- [ ] **Step 6: Commit**

```bash
git add adapt/tier1.py tests/test_tier1.py
git commit -m "feat(adapt): tier-1 training-free adaptation (AdaBN, hist match, TTA)"
```

---

### Task 9: Pseudo-label generation for self-training

**Files:**
- Create: `adapt/pseudo.py`
- Test: `tests/test_pseudo.py`

**Interfaces:**
- Consumes: `adapt.tier1.tta_predict`
- Produces: `make_pseudo_label(prob, hi=0.8, lo=0.2, ignore_value=255) -> np.ndarray`,
  `class_balanced_pseudo(prob, keep_frac=0.5, ignore_value=255) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pseudo.py
import numpy as np
import pytest
from adapt.pseudo import make_pseudo_label, class_balanced_pseudo


def test_confident_pixels_get_hard_labels_and_the_rest_are_ignored():
    prob = np.array([[0.95, 0.05], [0.5, 0.6]], np.float32)
    out = make_pseudo_label(prob, hi=0.8, lo=0.2, ignore_value=255)
    assert out[0, 0] == 1 and out[0, 1] == 0
    assert out[1, 0] == 255 and out[1, 1] == 255


def test_all_uncertain_yields_all_ignore():
    prob = np.full((4, 4), 0.5, np.float32)
    assert (make_pseudo_label(prob) == 255).all()


def test_class_balanced_keeps_requested_fraction():
    prob = np.linspace(0, 1, 100).astype(np.float32).reshape(10, 10)
    out = class_balanced_pseudo(prob, keep_frac=0.5, ignore_value=255)
    kept = (out != 255).mean()
    assert kept == pytest.approx(0.5, abs=0.06)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pseudo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adapt.pseudo'`

- [ ] **Step 3: Implement**

```python
# adapt/pseudo.py
"""Pseudo-label generation for confidence self-training.

Two selection policies. The fixed-threshold one is simpler; the class-balanced
one is safer, because a global threshold on a domain-shifted model tends to
keep almost only background and the model then collapses towards predicting
nothing.
"""
from __future__ import annotations

import numpy as np


def make_pseudo_label(prob: np.ndarray, hi: float = 0.8, lo: float = 0.2,
                      ignore_value: int = 255) -> np.ndarray:
    prob = np.asarray(prob, np.float32)
    out = np.full(prob.shape, ignore_value, np.uint8)
    out[prob >= hi] = 1
    out[prob <= lo] = 0
    return out


def class_balanced_pseudo(prob: np.ndarray, keep_frac: float = 0.5,
                          ignore_value: int = 255) -> np.ndarray:
    """Keep the `keep_frac` most confident pixels, split evenly between the
    two classes by distance from 0.5."""
    prob = np.asarray(prob, np.float32)
    conf = np.abs(prob - 0.5)
    n_keep = int(round(prob.size * keep_frac))
    out = np.full(prob.shape, ignore_value, np.uint8)
    if n_keep <= 0:
        return out
    idx = np.argpartition(conf.ravel(), -n_keep)[-n_keep:]
    flat = out.ravel()
    flat[idx] = (prob.ravel()[idx] >= 0.5).astype(np.uint8)
    return flat.reshape(prob.shape)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pseudo.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add adapt/pseudo.py tests/test_pseudo.py
git commit -m "feat(adapt): confidence and class-balanced pseudo-labelling"
```

---

### Task 10: Wire the adaptation stages into `rooftop/train.py`

**Files:**
- Modify: `rooftop/train.py`
- Test: `tests/test_train_cli.py`

**Interfaces:**
- Consumes: `adapt.datasets.MixedSegDataset`, `adapt.datasets.DatasetSpec`,
  `adapt.losses.BceDiceIgnore`, `adapt.augment.robust_train_transform`
- Produces: new CLI flags on `train.py` —
  `--extra_train_dir DIR:WEIGHT` (repeatable), `--ignore_index INT`,
  `--robust_aug`, `--fda_dir DIR`, `--sim_gsd_range LO HI`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_cli.py
import subprocess, sys, json, os
import numpy as np, cv2, pytest


@pytest.fixture
def tiny_dataset(tmp_path):
    for split in ("train", "val"):
        for sub in ("images", "masks"):
            (tmp_path / split / sub).mkdir(parents=True)
        for i in range(4):
            img = np.random.randint(0, 255, (64, 64, 3), np.uint8)
            m = np.zeros((64, 64), np.uint8); m[16:48, 16:48] = 1
            cv2.imwrite(str(tmp_path / split / "images" / f"{i}.png"), img)
            cv2.imwrite(str(tmp_path / split / "masks" / f"{i}.png"), m)
    return tmp_path


def test_train_accepts_new_adaptation_flags(tiny_dataset, tmp_path):
    r = subprocess.run([
        sys.executable, "rooftop/train.py",
        "--train_dir", str(tiny_dataset / "train"),
        "--val_dir", str(tiny_dataset / "val"),
        "--epochs", "1", "--batch_size", "2", "--workers", "0",
        "--img_size", "64",
        "--robust_aug", "--sim_gsd_range", "0.15", "0.40",
        "--ignore_index", "255",
        "--ckpt_dir", str(tmp_path / "ck"), "--log_dir", str(tmp_path / "lg"),
    ], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    assert list((tmp_path / "ck").glob("*.pth"))


def test_extra_train_dir_is_parsed_with_weight(tiny_dataset, tmp_path):
    r = subprocess.run([
        sys.executable, "rooftop/train.py",
        "--train_dir", str(tiny_dataset / "train"),
        "--val_dir", str(tiny_dataset / "val"),
        "--extra_train_dir", f"{tiny_dataset / 'train'}:0.5",
        "--epochs", "1", "--batch_size", "2", "--workers", "0",
        "--img_size", "64",
        "--ckpt_dir", str(tmp_path / "ck2"), "--log_dir", str(tmp_path / "lg2"),
    ], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "8 images" in r.stdout or "8 samples" in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_cli.py -v`
Expected: FAIL — `train.py: error: unrecognized arguments: --robust_aug`

- [ ] **Step 3: Add the flags and wire them up**

In `rooftop/train.py`'s argument parser, add:

```python
p.add_argument("--extra_train_dir", action="append", default=[],
               metavar="DIR[:WEIGHT]",
               help="Additional (images/,masks/) root, optional :weight. Repeatable.")
p.add_argument("--ignore_index", type=int, default=None,
               help="Mask value treated as ignore (255 for weak/pseudo labels)")
p.add_argument("--robust_aug", action="store_true",
               help="Use the Phase-C domain-generalisation augmentation policy")
p.add_argument("--fda_dir", type=str, default=None,
               help="Directory of unlabelled target tiles for FDA augmentation")
p.add_argument("--sim_gsd_range", type=float, nargs=2, default=None,
               metavar=("LO", "HI"),
               help="Simulate this ground-GSD range in metres, e.g. 0.15 0.40")
p.add_argument("--img_size", type=int, default=512)
```

Then, where the dataset is constructed:

```python
from adapt.datasets import MixedSegDataset, DatasetSpec
from adapt.augment import robust_train_transform
from adapt.losses import BceDiceIgnore

fda_tiles = None
if args.fda_dir:
    import glob, cv2
    fda_tiles = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
                 for f in sorted(glob.glob(f"{args.fda_dir}/*.png"))[:200]]

train_tf = (robust_train_transform(src_gsd=0.075, target_tiles=fda_tiles,
                                   crop=args.img_size)
            if args.robust_aug else default_train_transform(args.img_size))

specs = [DatasetSpec(f"{args.train_dir}/images", f"{args.train_dir}/masks",
                     1.0, train_tf, name="primary")]
for entry in args.extra_train_dir:
    path, _, w = entry.partition(":")
    specs.append(DatasetSpec(f"{path}/images", f"{path}/masks",
                             float(w) if w else 1.0, train_tf, name=path))

train_ds = MixedSegDataset(specs)
print(f"[MixedSegDataset] {len(train_ds)} images across {len(specs)} sources")

criterion = (BceDiceIgnore(ignore_index=args.ignore_index)
             if args.ignore_index is not None else criterion)
```

Keep the existing single-directory path working unchanged when no new flags are
passed — the source-domain reproduction must stay byte-compatible.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train_cli.py -v`
Expected: 2 passed.

- [ ] **Step 5: Verify the AIRS baseline is unchanged**

Run a 3-epoch, 200-sample AIRS run with no new flags and confirm the loss curve
matches a pre-change run to within noise. If it does not, the refactor broke
something.

- [ ] **Step 6: Commit**

```bash
git add rooftop/train.py tests/test_train_cli.py
git commit -m "feat(rooftop): multi-source training, ignore-index loss, robust aug flags"
```

---

### Task 11: Georeferenced inference and capacity estimation

**Files:**
- Create: `adapt/infer_geo.py`
- Test: `tests/test_infer_geo.py`

**Interfaces:**
- Consumes: `adapt.geo.true_gsd`, `adapt.tier1.tta_predict`
- Produces: `capacity_kw(mask, gsd_m, k_usable=0.60, efficiency=0.20) -> dict`,
  `mask_to_geojson(mask, transform, crs, min_area_m2=8.0) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_infer_geo.py
import numpy as np
import pytest
from adapt.infer_geo import capacity_kw, mask_to_geojson


def test_capacity_uses_the_geotiff_gsd_not_a_hardcoded_one():
    mask = np.ones((100, 100), np.uint8)     # 10,000 px
    r = capacity_kw(mask, gsd_m=0.26613)
    assert r["area_m2"] == pytest.approx(10000 * 0.070825, rel=1e-4)
    assert r["usable_m2"] == pytest.approx(r["area_m2"] * 0.60, rel=1e-6)
    assert r["capacity_kw"] == pytest.approx(r["usable_m2"] * 0.20, rel=1e-6)


def test_capacity_scales_with_gsd_squared():
    mask = np.ones((10, 10), np.uint8)
    a = capacity_kw(mask, gsd_m=0.075)["area_m2"]
    b = capacity_kw(mask, gsd_m=0.150)["area_m2"]
    assert b == pytest.approx(4 * a, rel=1e-6)


def test_small_blobs_are_dropped_from_geojson():
    mask = np.zeros((64, 64), np.uint8)
    mask[2:4, 2:4] = 1            # 4 px -> 0.28 m2, below the 8 m2 floor
    mask[20:50, 20:50] = 1        # 900 px -> 63.7 m2, kept
    gj = mask_to_geojson(mask, (0.26613, 0, 0, 0, -0.26613, 0),
                         "EPSG:3857", min_area_m2=8.0)
    assert len(gj["features"]) == 1
    assert gj["features"][0]["properties"]["area_m2"] > 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_infer_geo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adapt.infer_geo'`

- [ ] **Step 3: Implement**

```python
# adapt/infer_geo.py
"""Georeferenced inference outputs: capacity estimation and polygonisation.

The GSD is always a parameter read from the raster. Hard-coding 0.075 (AIRS) or
0.10 (the mistaken Jaipur assumption) propagates a 12.6x or 2.8x error into
every area and capacity figure in the report.
"""
from __future__ import annotations

import numpy as np
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape, mapping

# Defaults for Indian flat-roof residential PV. Stated, not hidden.
K_USABLE_DEFAULT = 0.60      # after parapet setbacks, tanks, stairwells, shading
EFFICIENCY_DEFAULT = 0.20    # modern mono-PERC module efficiency
STC_IRRADIANCE_KW_M2 = 1.0


def capacity_kw(mask: np.ndarray, gsd_m: float,
                k_usable: float = K_USABLE_DEFAULT,
                efficiency: float = EFFICIENCY_DEFAULT) -> dict:
    px = int(np.asarray(mask).astype(bool).sum())
    area = px * gsd_m ** 2
    usable = area * k_usable
    return {
        "pixels": px,
        "gsd_m": gsd_m,
        "area_per_pixel_m2": gsd_m ** 2,
        "area_m2": area,
        "usable_m2": usable,
        "capacity_kw": usable * efficiency * STC_IRRADIANCE_KW_M2,
        "k_usable": k_usable,
        "efficiency": efficiency,
    }


def mask_to_geojson(mask: np.ndarray, transform, crs: str,
                    min_area_m2: float = 8.0,
                    simplify_m: float = 0.5) -> dict:
    a, b, c, d, e, f = transform
    tf = Affine(a, b, c, d, e, f)
    gsd = abs(a)
    feats = []
    m = (np.asarray(mask) > 0).astype(np.uint8)
    for geom, val in shapes(m, mask=m.astype(bool), transform=tf):
        if val != 1:
            continue
        poly = shape(geom)
        area = poly.area                    # projected units squared
        if area < min_area_m2:
            continue
        if simplify_m:
            poly = poly.simplify(simplify_m, preserve_topology=True)
        feats.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {"area_m2": float(area),
                           "gsd_m": float(gsd)},
        })
    return {"type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": crs}},
            "features": feats}
```

> Note: `mask_to_geojson` computes area in the transform's own units. For an
> EPSG:3857 transform those are Mercator metres — pass a transform already
> scaled to ground metres, or correct the reported `area_m2` by `cos²(lat)`.
> The test above uses a ground-metre transform, which is the intended usage.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_infer_geo.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add adapt/infer_geo.py tests/test_infer_geo.py
git commit -m "feat(adapt): georeferenced capacity estimation and polygonisation"
```

---

### Task 12: The sealed evaluation, run once

**Files:**
- Create: `adapt/run_final_eval.py`
- Modify: `plan/04-pipeline.md` (§7 headline table)
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above

- [ ] **Step 1: Confirm the test split has never been read**

```bash
git log --all --oneline -S "split == 'test'" -- adapt/ rooftop/ | cat
```
Expected: only this task's commit. If anything earlier touched the test split,
the number is compromised — say so in the report rather than hiding it.

- [ ] **Step 2: Run every checkpoint over the sealed test split, once**

```bash
python adapt/run_final_eval.py \
    --tiles jaipur_crops/tiles \
    --split jaipur_crops/split.json --which test \
    --labels adapt/labels/clean \
    --ckpts rooftop/checkpoints/zeroshot.pth \
            rooftop/checkpoints/robust_source.pth \
            rooftop/checkpoints/weak_finetune.pth \
            rooftop/checkpoints/clean_finetune.pth \
            rooftop/checkpoints/selftrain_r3.pth \
    --threshold_from jaipur_crops/val_threshold.json \
    --out plan/results/final_eval.json
```

- [ ] **Step 3: Fill in the headline table**

Write the real numbers into `plan/04-pipeline.md` §7 and the expected-outcome
table in `plan/README.md`. **If a phase did not help, report that it did not
help.** A negative ablation result is a result.

- [ ] **Step 4: Update the top-level README**

Add a "Domain Adaptation — Jaipur" section pointing at `plan/`, the final
numbers, and the corrected 26.6 cm GSD.

- [ ] **Step 5: Commit**

```bash
git add plan/ README.md adapt/run_final_eval.py
git commit -m "docs: final Jaipur domain-adaptation results and ablation table"
```

---

## Self-review notes

- **Spec coverage.** Tasks 1–2 cover the tiling and GSD correction from `01`;
  Task 4 covers Gaps 1, 2, 5 from `02`; Tasks 5–6 cover Gaps 3, 4; Task 8
  covers Tier 1 of `03`; Task 10 covers Tier 2; Task 9 covers Tier 3; Task 11
  covers the capacity constants from `04` §6.
- **Known gap — Stage 2 (solar) has no task here.** It is deliberately deferred:
  it depends on Stage 1's output and on a labelling decision (whether to add a
  solar-water-heater class) that should be made after seeing Phase-D results.
  Write `plan/09-stage2-solar.md` when Task 12 is done.
- **Known gap — MIC/HRDA has no task here.** It is a stretch goal in `03` §3.2
  and lives in a separate `mmsegmentation` environment; it does not belong in
  this repo's task graph until the DGX time is confirmed.
