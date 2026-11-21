#!/usr/bin/env python
"""
hand_label_gallery.py - side-by-side pictures of model predictions on hand-labelled crops.

Per crop, one row of panels:
  image | hand label | Open Buildings | model A | model B ...
Every mask panel is drawn over the image and coloured against the HAND label:
  green = roof found (true positive), red = predicted roof that is not roof (false positive),
  blue = roof missed (false negative). The IoU for that crop is printed on the panel.

Output: one PNG per crop plus index.html in --out. The imagery licence is unresolved,
so keep the output local and off the public repo.

Usage (DGX, inside run_docker.sh):
  python scripts/hand_label_gallery.py --index data/hand_r12b2/index.csv --half test --n 10 \
     --model "Old default (MiT-B2 eroded, t=0.5)|<ckpt>|0.5" \
     --model "Picked with hand labels (ResNet34 all-conf, t=0.3)|<ckpt>|0.3" --out gallery
"""
import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
import segmentation_models_pytorch as smp

MEAN = np.array((0.485, 0.456, 0.406), np.float32)
STD = np.array((0.229, 0.224, 0.225), np.float32)


def load_model(path, dev):
    sd = torch.load(path, map_location="cpu")["model_state"]
    for enc in ("resnet34", "mit_b2", "mit_b5"):
        m = smp.Unet(enc, encoder_weights=None, in_channels=3, classes=1)
        try:
            m.load_state_dict(sd)
            return m.to(dev).eval()
        except RuntimeError:
            pass
    raise RuntimeError(path)


def overlay(img, pred, ref, title):
    out = img.astype(np.float32) * 0.55
    col = np.zeros_like(out)
    col[pred & ref] = (40, 220, 60)
    col[pred & ~ref] = (240, 50, 40)
    col[~pred & ref] = (40, 120, 255)
    m = (pred | ref)[..., None]
    out = np.where(m, out * 0.35 + col * 0.65, out)
    u = (pred | ref).sum()
    iou = (pred & ref).sum() / u if u else float("nan")
    return panel(out.astype(np.uint8), f"{title}  IoU {iou:.2f}" if u else f"{title}  (empty)")


def panel(arr, title):
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 512, 22], fill=(0, 0, 0))
    d.text((6, 5), title, fill=(255, 255, 255))
    return im


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}", flush=True)
    rows = [r for r in csv.DictReader(open(a.index)) if not a.half or r.get("half") == a.half]
    # Spread over density bins: up to n crops, round-robin by bin, skipping empty crops.
    by = {}
    for r in rows:
        if r["bin"] != "empty":
            by.setdefault(r["bin"], []).append(r)
    order = ["very_dense", "dense", "moderate", "sparse"]
    pick = []
    while len(pick) < a.n and any(by.get(b) for b in order):
        for b in order:
            if by.get(b) and len(pick) < a.n:
                pick.append(by[b].pop(len(by[b]) // 2))
    models = []
    for spec in a.model:
        name, ckpt, thr = spec.split("|")
        models.append((name, load_model(ckpt, dev), float(thr)))
    os.makedirs(a.out, exist_ok=True)
    html = ["<html><head><meta charset='utf-8'><title>Hand-label gallery</title><style>body{font-family:sans-serif;background:#111;color:#eee}"
            "img{max-width:100%}</style></head><body><h2>Predictions vs hand-drawn roofs</h2>"
            "<p>Colours are against the hand label: <b style='color:#28dc3c'>green</b> roof found, "
            "<b style='color:#f03228'>red</b> predicted but not roof, <b style='color:#4a8cff'>blue</b> roof missed.</p>"]
    for r in pick:
        img = np.asarray(Image.open(r["img"]).convert("RGB"))
        hand = np.asarray(Image.open(r["mask"]).convert("L")) > 127
        ob = np.asarray(Image.open(r["ob"]).convert("L")) > 127
        x = torch.from_numpy(((img.astype(np.float32) / 255. - MEAN) / STD).transpose(2, 0, 1)[None]).to(dev)
        panels = [panel(img, "Image"), overlay(img, hand, hand, "Hand label"), overlay(img, ob, hand, "Open Buildings")]
        with torch.no_grad():
            for name, m, thr in models:
                p = torch.sigmoid(m(x))[0, 0].cpu().numpy() > thr
                panels.append(overlay(img, p, hand, name))
        W = Image.new("RGB", (512 * len(panels) + 8 * (len(panels) - 1), 512), (17, 17, 17))
        for i, p in enumerate(panels):
            W.paste(p, (i * 520, 0))
        fn = os.path.splitext(r["name"])[0] + ".jpg"
        W.save(os.path.join(a.out, fn), quality=88)
        html.append(f"<h3>{r['name']} - {r['bin']} - labelled by {r['annotator']}</h3><img src='{fn}'>")
        print("wrote", fn, flush=True)
    open(os.path.join(a.out, "index.html"), "w").write("\n".join(html + ["</body></html>"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--half", default="test")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--model", action="append", required=True, help="name|ckpt|threshold")
    p.add_argument("--out", default="diagnostics/hand_labels/gallery")
    main(p.parse_args())
