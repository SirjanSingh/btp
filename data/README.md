# data/ — canonical dataset location for this server

`/scratch` is unusable on this node and `/` is full, so all datasets live here under
`/home` (~1.1 TB free). Gitignored.

```
data/
├── airs/            SOURCE domain, raw AIRS (7.5 cm)
│                    → <split>/image/*.tif + <split>/label/*.png  (0/1 masks)
│                    splits per rooftop/dataset/{train,val}.txt  (857 / 94 christchurch_*)
├── airs_crops/      tiled 512×512 → train|val|test / images/*.png + masks/*.png
│                    produced by:  python rooftop/tile_airs.py --src_dir data/airs/<split> ...
├── jaipur/          TARGET domain, 16 GeoTIFF mosaic tiles map67_<r>-<c>.tif (8.0 GB)
│                    EPSG:3857, GSD 0.26618 m/px (measured by D6), ZERO labels
└── open_buildings/  jaipur_open_buildings.csv — Google Open Buildings v3 (CC BY 4.0),
                     523,283 polygons over the AOI, WKT lon/lat (EPSG:4326) +
                     confidence + area_in_meters. Weak labels / target prior only:
                     these are GROUND FOOTPRINTS, not roof outlines (MASTER_CONTEXT Gap 4).
```

**Restore status as of 2026-09-09:** `jaipur/` and `open_buildings/` are complete.
**`airs/` is not** — 50 of 857 images, and only 3 image/label pairs actually match
(`christchurch_15`, `_48`, `_77`); `airs_crops/` is still empty. The Drive pull was
interrupted; re-run `scripts/fetch_drive_folder.py <folder> data/airs` (it skips
files that already completed), then tile.

**Do not try to finish the AIRS pull from Drive — it is not there.** The master Drive
folder holds 75 unique labelled AIRS pairs against `train.txt`'s 857, and each
`label/` folder is half `_vis` previews. Verified 2026-09-09 with gdown's 50-file
cap raised; full enumeration and folder IDs in `plan/01-situation-and-assets.md` §2.
Real AIRS comes from <https://www.airs-dataset.com/> (~28 GB) — but read
`.planning/STATE.md` todo 1 first, because switching source domain is on the table.

Stage-2 (BDAPPV solar) data, when needed, goes in `data/bdappv/` and `data/bdappv_crops/`.
