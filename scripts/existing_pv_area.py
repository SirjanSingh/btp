#!/usr/bin/env python
"""
existing_pv_area.py - how much Jaipur roof already has solar panels, over the whole study area.

For every 512x512 crop in data/jaipur_weak (train + val, the same 9,072 crops d16 uses):
  - rooftop model -> roof mask (threshold chosen on the hand-label calib half)
  - solar model, crop upsampled --scale times (BDAPPV pixel size) -> panel mask
Sums roof area, panel area and panel-on-roof area per tile and in total.

Energy use: existing panels already sit on usable roof, so
  remaining usable area = k_usable * roof - panel_on_roof
and the existing capacity is panel_on_roof * eta (1 kW/m2 at STC).

Usage (DGX, in run_docker.sh):
  python scripts/existing_pv_area.py --roof_ckpt <r34 c0_shift> --roof_thr 0.4 \
      --solar_ckpt <solar best.pth> --scale 3.4 --out diagnostics/existing_pv.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import segmentation_models_pytorch as smp

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
GSD = 0.266


def load(path, enc, dev):
    ck = torch.load(path, map_location="cpu")
    m = smp.Unet(enc, encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(ck.get("model_state", ck))
    return m.to(dev).eval()


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}", flush=True)
    roof_m = load(a.roof_ckpt, "resnet34", dev)
    sol_m = load(a.solar_ckpt, "resnet34", dev)
    files = []
    for s in ("train", "val"):
        d = os.path.join(a.crops, s, "images")
        files += [os.path.join(d, f) for f in sorted(os.listdir(d))]
    tiles = defaultdict(lambda: np.zeros(3, np.int64))   # roof, panel, panel_on_roof (px)
    S = int(round(512 * a.scale / 32)) * 32
    with torch.no_grad():
        for i in range(0, len(files), a.batch):
            fb = files[i:i + a.batch]
            x = torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGB")) for f in fb]))
            x = ((x.permute(0, 3, 1, 2).float() / 255. - MEAN) / STD).to(dev)
            roof = torch.sigmoid(roof_m(x))[:, 0] > a.roof_thr
            xs = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)
            ps = []
            for j in range(0, len(xs), a.solar_batch):
                ps.append(torch.sigmoid(sol_m(xs[j:j + a.solar_batch])))
            p = F.interpolate(torch.cat(ps), size=(512, 512), mode="bilinear", align_corners=False)[:, 0] > a.solar_thr
            for f, r, q in zip(fb, roof, p):
                t = os.path.basename(f).split("_")[0] + "_" + os.path.basename(f).split("_")[1]
                tiles[t] += np.array([int(r.sum()), int(q.sum()), int((r & q).sum())])
            if i % (a.batch * 50) == 0:
                print(f"{i}/{len(files)}", flush=True)
    px = GSD ** 2
    tot = sum(tiles.values())
    out = dict(
        n_crops=len(files), roof_ckpt=a.roof_ckpt, roof_thr=a.roof_thr, solar_ckpt=a.solar_ckpt,
        solar_scale=a.scale, solar_thr=a.solar_thr,
        roof_km2=tot[0] * px / 1e6, panel_km2=tot[1] * px / 1e6, panel_on_roof_km2=tot[2] * px / 1e6,
        panel_share_of_roof=tot[2] / max(tot[0], 1),
        existing_capacity_MWp_at_eta020=tot[2] * px * 0.20 / 1e3,
        per_tile={t: dict(roof_km2=v[0] * px / 1e6, panel_on_roof_m2=v[2] * px,
                          share=v[2] / max(v[0], 1)) for t, v in sorted(tiles.items())},
        note="panel area is a lower bound: the solar model misses part of some arrays (gallery check)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_tile"}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--crops", default="data/jaipur_weak")
    p.add_argument("--roof_ckpt", required=True)
    p.add_argument("--roof_thr", type=float, default=0.4)
    p.add_argument("--solar_ckpt", required=True)
    p.add_argument("--solar_thr", type=float, default=0.5)
    p.add_argument("--scale", type=float, default=3.4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--solar_batch", type=int, default=2)
    p.add_argument("--out", default="diagnostics/existing_pv.json")
    main(p.parse_args())
