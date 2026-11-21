#!/usr/bin/env python
"""
solar_on_hand_crops.py - run a BDAPPV-trained solar-panel model on the 152 hand-labelled Jaipur
crops and show what it finds, to see whether it can be used to subtract roof area that already
has panels.

There are no Jaipur panel labels, so this gives no accuracy number. It reports how much of the
hand-drawn roof area the model calls "panel" and writes a gallery of the crops with the most
panel pixels so the detections can be checked by eye.

BDAPPV Google imagery is about 0.1 m/px and Jaipur is 0.266 m/px, so panels look about 2.6x
smaller here than in training. --scale upsamples each crop before inference (then the mask is
resized back) to test whether matching the pixel size helps.

Usage (DGX, inside run_docker.sh):
  python scripts/solar_on_hand_crops.py --index data/hand_r12b2/index.csv --ckpt <best.pth> \
      --scales 1 2 2.6 --out diagnostics/solar_on_hand
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import segmentation_models_pytorch as smp

MEAN = np.array((0.485, 0.456, 0.406), np.float32)
STD = np.array((0.229, 0.224, 0.225), np.float32)


def load(path, dev):
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("model_state", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(sd)
    return m.to(dev).eval()


@torch.no_grad()
def predict(m, img, scale, dev):
    x = torch.from_numpy(((img.astype(np.float32) / 255. - MEAN) / STD).transpose(2, 0, 1)[None]).to(dev)
    H, W = x.shape[-2:]
    if scale != 1:
        x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
        # pad to a multiple of 32 for the U-Net
        ph, pw = (-x.shape[-2]) % 32, (-x.shape[-1]) % 32
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    p = torch.sigmoid(m(x))
    if scale != 1:
        p = p[..., :int(round(H * scale)), :int(round(W * scale))]
        p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
    return p[0, 0].cpu().numpy()


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}", flush=True)
    rows = list(csv.DictReader(open(a.index)))
    m = load(a.ckpt, dev)
    os.makedirs(a.out, exist_ok=True)
    summary = {"ckpt": a.ckpt, "threshold": a.thr, "n_crops": len(rows), "scales": {}}
    for s in a.scales:
        roof_px = solar_px = solar_on_roof = crops_hit = 0
        per = []
        for r in rows:
            img = np.asarray(Image.open(r["img"]).convert("RGB"))
            hand = np.asarray(Image.open(r["mask"]).convert("L")) > 127
            p = predict(m, img, s, dev) > a.thr
            roof_px += int(hand.sum()); solar_px += int(p.sum()); solar_on_roof += int((p & hand).sum())
            crops_hit += int(p.sum() >= a.min_px)
            per.append((int(p.sum()), r, p))
        summary["scales"][str(s)] = dict(
            solar_px=solar_px, solar_on_roof_px=solar_on_roof, roof_px=roof_px,
            solar_frac_of_roof=solar_on_roof / max(roof_px, 1),
            solar_off_roof_frac=1 - solar_on_roof / max(solar_px, 1),
            crops_with_detection=crops_hit)
        print(s, json.dumps(summary["scales"][str(s)]), flush=True)
        # gallery: top crops by panel pixels
        per.sort(key=lambda t: -t[0])
        d = os.path.join(a.out, f"scale_{s}")
        os.makedirs(d, exist_ok=True)
        html = [f"<html><head><meta charset='utf-8'><title>Solar model on Jaipur, scale {s}</title>"
                "<style>body{font-family:sans-serif;background:#111;color:#eee}img{max-width:100%}</style></head><body>"
                f"<h2>BDAPPV solar model on Jaipur crops (scale {s}, threshold {a.thr})</h2>"
                "<p>Left: image. Right: yellow = predicted panel, thin cyan = hand-drawn roof.</p>"]
        for n, r, p in per[:a.n]:
            if n == 0:
                break
            img = np.asarray(Image.open(r["img"]).convert("RGB"))
            hand = np.asarray(Image.open(r["mask"]).convert("L")) > 127
            ov = img.astype(np.float32) * 0.6
            edge = hand ^ np.roll(hand, 1, 0) | hand ^ np.roll(hand, 1, 1)
            ov[edge] = (0, 220, 255)
            ov[p] = ov[p] * 0.3 + np.array((255, 220, 0)) * 0.7
            W = Image.new("RGB", (1032, 512), (17, 17, 17))
            W.paste(Image.fromarray(img), (0, 0)); W.paste(Image.fromarray(ov.astype(np.uint8)), (520, 0))
            ImageDraw.Draw(W).text((526, 6), f"{n} panel px ({n * 0.266 ** 2:.0f} m2)", fill=(255, 255, 255))
            fn = os.path.splitext(r["name"])[0] + ".jpg"
            W.save(os.path.join(d, fn), quality=88)
            html.append(f"<h3>{r['name']} - {r['bin']} - {n} px</h3><img src='{fn}'>")
        open(os.path.join(d, "index.html"), "w").write("\n".join(html + ["</body></html>"]))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0, 2.6])
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--min_px", type=int, default=50)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", default="diagnostics/solar_on_hand")
    main(p.parse_args())
