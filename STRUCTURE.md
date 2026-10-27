# BTP — Repository Structure

Project root: `~/btp/`  (moved here 2026-09-08 from `~/airs/btp/`)

```
btp/
├── rooftop/                  # Stage 1 — rooftop segmentation (AIRS)
│   ├── train.py              # training (DataParallel, AMP, cosine LR, global-IoU)
│   ├── evaluate.py           # test-set eval + threshold sweep + overlays
│   ├── infer.py              # sliding-window inference, any image size
│   ├── tile_airs.py          # 10k×10k AIRS tiles → 512×512 crops
│   ├── notebooks/            # demo + exploration notebooks
│   ├── dataset/              # official split lists: train.txt (857), val.txt (94)
│   ├── checkpoints/          # *.pth weights (gitignored) + tb_logs/
│   ├── logs/                 # per-run JSON/TXT epoch logs
│   ├── eval_results/         # eval overlays + per-sample IoU CSV (gitignored)
│   └── infer_results/        # inference outputs (gitignored)
│
├── solar_panel/              # Stage 2 — solar-panel segmentation (BDAPPV)
│   ├── train_solar.py        # training (black-mask fallback for negatives)
│   ├── evaluate_solar.py
│   ├── infer_solar.py        # + panel area (m²) & peak power (kW)
│   ├── prep_bdappv.py        # BDAPPV → resized/split crops
│   ├── checkpoints/          # timestamped run dirs (gitignored)
│   ├── logs/  eval_results/
│
├── data/                     # ← canonical data location on THIS server (gitignored)
│   ├── airs/                 #   raw AIRS: image/ + label/ per split
│   └── airs_crops/           #   tiled 512×512: train|val|test / images|masks
│
├── .planning/                # GSD planning docs — PROJECT / ROADMAP / STATE / research
├── Dockerfile                # pytorch 2.1.2 + cu118 + smp/albumentations
├── requirements.txt
└── README.md

archived elsewhere:
  ~/archive/rooftop-prototype/U-net_btp/   # the from-scratch U-Net prototype (superseded)
```

## Server notes (lnmdgx, node checked 2026-09-08)

- **No `/scratch`** usable here (root-owned, empty) and **`/` is 100 % full (~2 GB free)** —
  the README's `/scratch → /tmp` staging path does **not** apply. Put everything under
  `~/btp/data/` (`/home` has ~1.1 TB free).
- Host Python has **no** `segmentation-models-pytorch` — all training runs **inside Docker**.
- 8× Tesla V100-32GB, shared node (usually 7/8 busy — check `nvidia-smi` before launching).

## Known cleanup still pending (needs `chown`, see PHASE2_RUNBOOK.md §0)

Files/dirs created by earlier Docker-as-root runs are still `root:root` and can't be
moved/deleted without the ownership fix:
- `logs/` at repo root — exact duplicate of `rooftop/logs/*.txt|json` (pre-refactor leftover)
- `rooftop/checkpoints/solar_panel/`, `rooftop/logs/solar_panel/` — stray Stage-2 tb-logs
- `solar_panel/solar_panel/` — nested duplicate (one stale infer PNG)
- `solar_panel/checkpoints/*_180709`, `*_050947`, `*_051053` — empty failed-launch run dirs
