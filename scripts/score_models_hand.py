#!/usr/bin/env python
"""
score_models_hand.py - score trained rooftop checkpoints against HAND-DRAWN roof
labels (labelling/r12_batch*), next to their agreement with Open Buildings on the
same crops.

WHY. Every Jaipur number before Nov 2026 was agreement with Open Buildings (OB)
ground footprints, which are also the training labels. The hand labels measure
roofs directly. If the OB-based ranking of models and the hand-label ranking
disagree, the September conclusions need revisiting.

Inputs
  --index   CSV with columns name,img,mask,ob,bin,annotator (paths relative to
            the repo root or absolute). One row per labelled crop.
  --models  CSV with columns tag,group,ckpt. Encoder is detected from the
            state dict (resnet34 / mit_b2 / mit_b5).
Outputs
  --out     JSON with, per model: pooled pixel IoU/P/R vs hand and vs OB at
            --threshold, predicted/hand roof-area ratio, building counts,
            per-density-bin and per-annotator IoU, and a threshold sweep
            (the sweep is tuned on the eval set, so it is an oracle, not a result).
  A markdown table is printed and written next to the JSON.

Preprocessing matches rooftop/train.py and scripts/merge_split_rate.py:
RGB / 255, ImageNet mean/std, full 512x512 crop, sigmoid > threshold.

Usage (DGX):
  python scripts/score_models_hand.py --index data/hand_r12b2/index.csv \
      --models data/hand_r12b2/models.csv --out diagnostics/hand_labels/models.json --gpu 4
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp

MEAN = np.array((0.485, 0.456, 0.406), np.float32)
STD = np.array((0.229, 0.224, 0.225), np.float32)
SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7)


def load_bin(p):
    return np.asarray(Image.open(p).convert("L")) > 127


def load_model(path, dev):
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("model_state", ck) if isinstance(ck, dict) else ck
    errs = []
    # Checkpoints from rooftop/train.py record their encoder; try that first so
    # newer encoders (e.g. tu-convnext_tiny.dinov3_lvd1689m) load without a guess.
    first = (ck.get("args") or {}).get("encoder") if isinstance(ck, dict) else None
    for enc in dict.fromkeys([e for e in (first, "resnet34", "mit_b2", "mit_b5") if e]):
        m = smp.Unet(enc, encoder_weights=None, in_channels=3, classes=1)
        try:
            m.load_state_dict(sd, strict=True)
            return m.to(dev).eval(), enc
        except RuntimeError as e:
            errs.append(f"{enc}: {str(e)[:120]}")
    raise RuntimeError(f"no encoder matched {path}:\n" + "\n".join(errs))


def count(mask, min_area_px):
    # 4-connectivity, same as scipy.ndimage.label's default used in eval_hand_labels.py
    n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    return int((st[1:, cv2.CC_STAT_AREA] >= min_area_px).sum())


def prf(tp, fp, fn):
    return dict(iou=tp / max(tp + fp + fn, 1), precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1))


class Acc:
    """Pooled pixel counts against one reference."""
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, pred, ref):
        self.tp += int((pred & ref).sum())
        self.fp += int((pred & ~ref).sum())
        self.fn += int((~pred & ref).sum())

    def out(self):
        return prf(self.tp, self.fp, self.fn)


def score_preds(preds, rows, refs_hand, refs_ob, min_area_px):
    """preds: dict thr -> list of bool masks aligned with rows."""
    res = {}
    for thr, ps in preds.items():
        h, o = Acc(), Acc()
        for p, rh, ro in zip(ps, refs_hand, refs_ob):
            h.add(p, rh)
            o.add(p, ro)
        res[thr] = dict(hand=h.out(), ob=o.out(),
                        area_ratio_pred_over_hand=sum(int(p.sum()) for p in ps)
                        / max(sum(int(r.sum()) for r in refs_hand), 1))
    main = res[min(res, key=lambda t: abs(t - 0.5))] if 0.5 not in res else res[0.5]
    ps = preds[0.5] if 0.5 in preds else next(iter(preds.values()))
    by = {}
    for key in [k for k in ("bin", "annotator", "split", "half") if k in rows[0]]:
        groups = {}
        for i, r in enumerate(rows):
            groups.setdefault(r[key], []).append(i)
        by[key] = {}
        for g, idx in sorted(groups.items()):
            a = Acc()
            for i in idx:
                a.add(ps[i], refs_hand[i])
            by[key][g] = dict(crops=len(idx), **a.out())
    main = dict(main)
    main["buildings_pred"] = sum(count(p, min_area_px) for p in ps)
    main["buildings_hand"] = sum(count(r, min_area_px) for r in refs_hand)
    for key, v in by.items():
        main[f"by_{key}"] = v
    # Crop-level bootstrap of pooled hand IoU (same resample indices for every
    # model, so differences between models can be read off paired).
    tp = np.array([int((p & r).sum()) for p, r in zip(ps, refs_hand)])
    un = np.array([int((p | r).sum()) for p, r in zip(ps, refs_hand)])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ps), size=(2000, len(ps)))
    boots = tp[idx].sum(1) / np.maximum(un[idx].sum(1), 1)
    main["hand_iou_boot95"] = [round(float(np.percentile(boots, 2.5)), 4),
                               round(float(np.percentile(boots, 97.5)), 4)]
    main["_crop_tp"], main["_crop_union"] = tp.tolist(), un.tolist()
    # Per crop and threshold: [tp, fp, fn, pred_px, hand_px], so a threshold can be
    # picked on one half of the crops and reported on the other.
    main["_crop_counts"] = {str(t): [[int((p & r).sum()), int((p & ~r).sum()), int((~p & r).sum()),
                                      int(p.sum()), int(r.sum())] for p, r in zip(pl, refs_hand)]
                            for t, pl in preds.items()}
    main["_names"] = [r["name"] for r in rows]
    main["sweep_hand_iou_ORACLE"] = {str(t): round(v["hand"]["iou"], 4) for t, v in res.items()}
    return main


def main(a):
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}", flush=True)
    rows = list(csv.DictReader(open(a.index)))
    missing = [r["img"] for r in rows if not os.path.isfile(r["img"])]
    missing += [r["ob"] for r in rows if not os.path.isfile(r["ob"])]
    missing += [r["mask"] for r in rows if not os.path.isfile(r["mask"])]
    if missing:
        raise SystemExit(f"{len(missing)} files missing, e.g. {missing[:3]}")
    imgs = np.stack([np.asarray(Image.open(r["img"]).convert("RGB")) for r in rows])
    hand = [load_bin(r["mask"]) for r in rows]
    ob = [load_bin(r["ob"]) for r in rows]
    x_all = ((imgs.astype(np.float32) / 255. - MEAN) / STD).transpose(0, 3, 1, 2)
    print(f"{len(rows)} crops loaded", flush=True)

    results = {}
    # Open Buildings itself, as a reference row (no threshold).
    results["open_buildings"] = dict(group="reference", encoder=None, ckpt=None,
                                     **score_preds({0.5: ob}, rows, hand, ob, a.min_area_px))

    for m in csv.DictReader(open(a.models)):
        if not os.path.isfile(m["ckpt"]):
            print(f"[skip] {m['tag']}: {m['ckpt']} not found", flush=True)
            continue
        model, enc = load_model(m["ckpt"], dev)
        probs = []
        with torch.no_grad():
            for i in range(0, len(rows), a.batch):
                x = torch.from_numpy(x_all[i:i + a.batch]).to(dev)
                probs.append(torch.sigmoid(model(x))[:, 0].float().cpu().numpy())
        probs = np.concatenate(probs)
        preds = {t: [p > t for p in probs] for t in SWEEP}
        r = score_preds(preds, rows, hand, ob, a.min_area_px)
        results[m["tag"]] = dict(group=m["group"], encoder=enc, ckpt=m["ckpt"], **r)
        print(f"{m['tag']:28s} {enc:9s} hand IoU {r['hand']['iou']:.4f} "
              f"P {r['hand']['precision']:.3f} R {r['hand']['recall']:.3f} | "
              f"OB IoU {r['ob']['iou']:.4f} | area/hand {r['area_ratio_pred_over_hand']:.2f} "
              f"| bldg {r['buildings_pred']}/{r['buildings_hand']}", flush=True)
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    meta = dict(n_crops=len(rows), threshold=0.5, min_area_px=a.min_area_px,
                note="pooled pixel metrics; sweep is tuned on the eval set (oracle)")
    json.dump(dict(meta=meta, models=results), open(a.out, "w"), indent=1)

    lines = ["| model | group | enc | hand IoU | P | R | OB IoU | area pred/hand | bldg pred/hand | best-thr IoU (oracle) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(results.items(), key=lambda kv: -kv[1]["hand"]["iou"])
    for tag, r in order:
        sw = r["sweep_hand_iou_ORACLE"]
        best = max(sw.items(), key=lambda kv: kv[1])
        lines.append(f"| {tag} | {r['group']} | {r['encoder'] or '-'} | {r['hand']['iou']:.3f} | "
                     f"{r['hand']['precision']:.3f} | {r['hand']['recall']:.3f} | {r['ob']['iou']:.3f} | "
                     f"{r['area_ratio_pred_over_hand']:.2f} | {r['buildings_pred']}/{r['buildings_hand']} | "
                     f"{best[1]:.3f} @ {best[0]} |")
    md = "\n".join(lines)
    open(os.path.splitext(a.out)[0] + ".md", "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--out", default="diagnostics/hand_labels/models.json")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--min_area_px", type=int, default=50)
    main(p.parse_args())
