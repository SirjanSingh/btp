#!/usr/bin/env python
"""
eval_hand_labels.py — turn hand-labelled Jaipur crops into an eval set, and
score Open Buildings (or any model's predictions) against them.

WHY. Every Jaipur number in this repo is agreement with Open Buildings ground
footprints. The labelling batches (labelling/r12_batch*) exist to replace that
with accuracy against hand-drawn roof outlines. This script is the bridge:

  export  copy every crop that has a finished label into <out>/images + <out>/masks,
          the layout FolderDataset and self_train.py --val_target_dir read.
  score   compare a folder of binary masks (default: openbuildings_reference/)
          to the hand labels: pixel IoU / precision / recall, building counts,
          per density bin. Pass --pred_dir to score a model's saved predictions
          instead of OB.

A crop counts as labelled only when labels/<name> exists. Unfinished crops are
skipped and reported, never treated as empty.

Usage:
    python scripts/eval_hand_labels.py export --batches labelling/r12_batch1 \
        labelling/r12_batch2 --out data/jaipur_hand/val
    python scripts/eval_hand_labels.py score --batches labelling/r12_batch2
    python scripts/eval_hand_labels.py score --batches labelling/r12_batch2 \
        --pred_dir path/to/predicted_masks --tag armD
"""
import argparse
import csv
import glob
import json
import os
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage


def load_bin(path):
    return np.asarray(Image.open(path).convert("L")) > 127


def labelled_crops(batches):
    """One record per crop with a finished label, across batches and people.
    Works for both layouts: batch1 (working/, sealed/) and batch2 (<person>/)."""
    rows = []
    for b in batches:
        bins = {}
        man = os.path.join(b, "manifest.csv")
        if os.path.isfile(man):
            for r in csv.DictReader(open(man)):
                bins[r["name"]] = r["bin"]
        for img in sorted(glob.glob(os.path.join(b, "*", "images", "*.png"))):
            group_dir = os.path.dirname(os.path.dirname(img))
            name = os.path.basename(img)
            rows.append(dict(
                name=name, batch=os.path.basename(os.path.normpath(b)),
                group=os.path.basename(group_dir), bin=bins.get(name, "unknown"),
                img=img,
                label=os.path.join(group_dir, "labels", name),
                ob=os.path.join(group_dir, "openbuildings_reference", name)))
    done = [r for r in rows if os.path.isfile(r["label"])]
    return done, len(rows)


def export(a):
    done, total = labelled_crops(a.batches)
    for sub in ("images", "masks"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    for r in done:
        shutil.copy2(r["img"], os.path.join(a.out, "images", r["name"]))
        # Normalise to 0/255 so a label saved as 0/1 still reads correctly.
        m = load_bin(r["label"]).astype(np.uint8) * 255
        Image.fromarray(m).save(os.path.join(a.out, "masks", r["name"]))
    print(f"[export] {len(done)} of {total} crops labelled -> {a.out}")


def n_buildings(mask, min_area_px):
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0
    areas = np.bincount(lab.ravel())[1:]
    return int((areas >= min_area_px).sum())


def score(a):
    done, total = labelled_crops(a.batches)
    if not done:
        raise SystemExit(f"no finished labels yet (0 of {total} crops)")
    per_bin, rows = {}, []
    for r in done:
        gt = load_bin(r["label"])
        src = os.path.join(a.pred_dir, r["name"]) if a.pred_dir else r["ob"]
        if not os.path.isfile(src):
            print(f"[warn] no prediction for {r['name']}, skipped")
            continue
        pr = load_bin(src)
        tp = int((pr & gt).sum()); fp = int((pr & ~gt).sum()); fn = int((~pr & gt).sum())
        row = dict(name=r["name"], bin=r["bin"], group=r["group"], tp=tp, fp=fp, fn=fn,
                   n_hand=n_buildings(gt, a.min_area_px),
                   n_pred=n_buildings(pr, a.min_area_px))
        rows.append(row)
        acc = per_bin.setdefault(r["bin"], dict(tp=0, fp=0, fn=0, n_hand=0, n_pred=0, crops=0))
        for k in ("tp", "fp", "fn", "n_hand", "n_pred"):
            acc[k] += row[k]
        acc["crops"] += 1

    def summarise(c):
        # Pooled over pixels, not averaged per crop: empty crops have no defined
        # per-crop IoU, and they are exactly where false positives show up.
        # A ratio with nothing in it (both masks empty) is undefined, not 0.
        ratio = lambda n, d: round(n / d, 4) if d else None
        return dict(crops=c["crops"],
                    iou=ratio(c["tp"], c["tp"] + c["fp"] + c["fn"]),
                    precision=ratio(c["tp"], c["tp"] + c["fp"]),
                    recall=ratio(c["tp"], c["tp"] + c["fn"]),
                    false_positive_px=c["fp"],
                    buildings_hand=c["n_hand"], buildings_pred=c["n_pred"])

    overall = dict(tp=0, fp=0, fn=0, n_hand=0, n_pred=0, crops=0)
    for c in per_bin.values():
        for k in overall:
            overall[k] += c[k]
    out = {
        "compared": a.pred_dir or "openbuildings_reference",
        "tag": a.tag,
        "labelled": len(done), "total_crops": total,
        "min_area_px": a.min_area_px,
        "overall": summarise(overall),
        "by_bin": {b: summarise(c) for b, c in sorted(per_bin.items())},
        "note": "hand labels trace ROOF outlines; OB traces ground footprints",
    }
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(dict(out, per_crop=rows), open(a.json, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("compared", "labelled", "total_crops", "overall")},
                     indent=1))
    f = lambda v: "  n/a " if v is None else f"{v:.4f}"
    for b, s in out["by_bin"].items():
        print(f"   {b:11s} crops={s['crops']:3d}  IoU={f(s['iou'])}  P={f(s['precision'])}  "
              f"R={f(s['recall'])}  FP px={s['false_positive_px']}  "
              f"bldg hand/pred={s['buildings_hand']}/{s['buildings_pred']}")
    print(f"[ok] {a.json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("export", "score"):
        s = sub.add_parser(name)
        s.add_argument("--batches", nargs="+",
                       default=["labelling/r12_batch1", "labelling/r12_batch2"])
    sub.choices["export"].add_argument("--out", default="data/jaipur_hand/val")
    sc = sub.choices["score"]
    sc.add_argument("--pred_dir", default=None,
                    help="folder of binary masks named like the crops; default is OB")
    sc.add_argument("--tag", default="openbuildings")
    sc.add_argument("--min_area_px", type=int, default=50,
                    help="ignore specks below this size when counting buildings")
    sc.add_argument("--json", default="diagnostics/hand_labels/score.json")
    a = p.parse_args()
    export(a) if a.cmd == "export" else score(a)
