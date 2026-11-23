#!/usr/bin/env python
"""
score_crossearth_hand.py - score a CrossEarth (mmsegmentation) checkpoint, zero-shot, on the
152 hand-labelled Jaipur crops, with the same protocol as hand_ensemble_budget.py:
threshold picked on the calib half, IoU reported on the test half, crop bootstrap CI.

CrossEarth (Gong et al., TPAMI 2026, github.com/VisionXLab/CrossEarth) is a multi-class
domain-generalisation segmenter. We never train it here. Its class-probability map is
collapsed to one "roof" probability:
    roof_prob = sum of class probabilities over --roof_class_ids
so the SAME binary metric used for our U-Nets applies. Which class id means "building"
depends on the checkpoint's benchmark config (e.g. its LoveDA / Potsdam / building label
map): READ THE CONFIG'S `classes` / `metainfo` AND PASS THE RIGHT IDS. This script prints
the class names it finds so you can check, and refuses to run without --roof_class_ids.

Nothing here has been run yet. It was written without GPU or data access.

Environment: CrossEarth's own (mmseg>=1.0, mmcv>=2.0, torch 2.0.1, xformers 0.0.20), NOT the
repo's rooftop image. See experiments/2026-11-22-crossearth-hand-scoring/README.md.

Usage (DGX, inside the CrossEarth image):
  python scripts/score_crossearth_hand.py \
      --index data/hand_r12b2/index.csv \
      --config CrossEarth/configs/CrossEarth_dinov2/<pick>.py \
      --checkpoint CrossEarth/checkpoints/<pick>.pth \
      --roof_class_ids 1 --tag crossearth_<pick> --gpu 4 \
      --out diagnostics/hand_labels/crossearth_<pick>.json

index.csv columns are the ones score_models_hand.py uses: name,img,mask,ob,bin,annotator,half.
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
import torch

# Importable without mmseg so the metric helpers can be unit-checked off the DGX.
FINE_SWEEP = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def load_bin(p):
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127


def counts(prob, hand, thr):
    out = []
    for p, h in zip(prob, hand):
        q = p > thr
        out.append((int((q & h).sum()), int((q & ~h).sum()), int((~q & h).sum()),
                    int(q.sum()), int(h.sum())))
    return np.array(out, np.int64)


def agg(c):
    tp, fp, fn, pp, hp = c.sum(0)
    return float(tp / max(tp + fp + fn, 1)), float(pp / max(hp, 1))


def dihedral(fn, img):
    """Mean of fn(img) over the 8 dihedral transforms; fn: HxWx3 -> HxW float probs."""
    acc = 0
    for k in range(4):
        for flip in (False, True):
            t = np.rot90(img, k)
            if flip:
                t = t[:, ::-1]
            p = fn(np.ascontiguousarray(t))
            if flip:
                p = p[:, ::-1]
            acc = acc + np.rot90(p, -k)
    return acc / 8


def make_predictor(model, roof_ids, logit_mode):
    from mmseg.apis import inference_model

    def predict(img_bgr):
        r = inference_model(model, img_bgr)
        logits = r.seg_logits.data.float()          # C,H,W
        if logit_mode == "softmax":
            p = torch.softmax(logits, 0)
        else:  # "raw": already score-like (e.g. Mask2Former); renormalise over classes
            p = logits.clamp(min=0)
            p = p / p.sum(0, keepdim=True).clamp(min=1e-6)
        return p[roof_ids].sum(0).cpu().numpy()
    return predict


def main(a):
    import sys
    sys.path.insert(0, os.path.abspath(a.repo))          # CrossEarth's custom backbone/head/dataset
    import CrossEarth  # noqa: F401  (registers modules with mmseg)
    import rs_dataset  # noqa: F401
    from mmseg.apis import init_model

    rows = list(csv.DictReader(open(a.index)))
    miss = [r[k] for r in rows for k in ("img", "mask") if not os.path.isfile(r[k])]
    if miss:
        raise SystemExit(f"{len(miss)} files missing, e.g. {miss[:3]}")
    hand = [load_bin(r["mask"]) for r in rows]
    cal = [i for i, r in enumerate(rows) if r["half"] == "calib"]
    tst = [i for i, r in enumerate(rows) if r["half"] == "test"]
    assert cal and tst, "index.csv needs a 'half' column with calib/test"

    model = init_model(a.config, a.checkpoint, device=f"cuda:{a.gpu}")
    names = getattr(model, "dataset_meta", {}).get("classes")
    print("classes found in checkpoint meta:", names, flush=True)
    n_cls = len(names) if names else None
    if any(i < 0 or (n_cls and i >= n_cls) for i in a.roof_class_ids):
        raise SystemExit(f"--roof_class_ids {a.roof_class_ids} out of range for classes {names}")
    print("collapsing to roof =", [names[i] for i in a.roof_class_ids] if names else a.roof_class_ids,
          flush=True)

    predict = make_predictor(model, a.roof_class_ids, a.logit_mode)
    plain, tta = [], []
    for k, r in enumerate(rows):
        img = cv2.imread(r["img"], cv2.IMREAD_COLOR)        # BGR, as mmseg expects for ndarrays
        plain.append(predict(img).astype(np.float16))
        if a.tta:
            tta.append(dihedral(predict, img).astype(np.float16))
        if k % 20 == 0:
            print(f"{k}/{len(rows)}", flush=True)

    table = {}
    variants = {a.tag: plain}
    if a.tta:
        variants[a.tag + "+tta"] = tta
    for tag, prob in variants.items():
        cs = {t: counts(prob, hand, t) for t in FINE_SWEEP}
        thr = max(FINE_SWEEP, key=lambda t: agg(cs[t][cal])[0])     # chosen on calib only
        ci, _ = agg(cs[thr][cal])
        ti, ta = agg(cs[thr][tst])
        # crop bootstrap of test IoU
        c = cs[thr][tst]
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(c), size=(2000, len(c)))
        tp = c[:, 0][idx].sum(1); un = (c[:, 0] + c[:, 1] + c[:, 2])[idx].sum(1)
        b = tp / np.maximum(un, 1)
        table[tag] = dict(thr=thr, calib_iou=ci, test_iou=ti, test_area_ratio=ta,
                          test_iou_boot95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                          iou_at_0_5_all=agg(counts(prob, hand, 0.5))[0],
                          oracle_best_thr_all=max(FINE_SWEEP, key=lambda t: agg(cs[t])[0]))
        print(f"{tag:40s} thr {thr:.2f} calib {ci:.3f} TEST {ti:.3f} area {ta:.2f} "
              f"95% {table[tag]['test_iou_boot95'][0]:.3f}-{table[tag]['test_iou_boot95'][1]:.3f}",
              flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(meta=dict(config=a.config, checkpoint=a.checkpoint, roof_class_ids=a.roof_class_ids,
                             classes=names, logit_mode=a.logit_mode, n_crops=len(rows),
                             n_calib=len(cal), n_test=len(tst),
                             note="zero-shot, no Jaipur data used; thr from calib, IoU on test"),
                   results=table), open(a.out, "w"), indent=1)
    # Same layout as hand_ensemble_budget's cache: tag -> N,H,W float16.
    np.savez_compressed(os.path.splitext(a.out)[0] + "_probs.npz",
                        **{t: np.stack(p) for t, p in variants.items()})
    print("wrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--repo", default=os.path.expanduser("~/crossearth_work/CrossEarth"),
                   help="CrossEarth checkout (added to sys.path so its modules register)")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--roof_class_ids", type=int, nargs="+", required=True)
    p.add_argument("--logit_mode", choices=("softmax", "raw"), default="softmax",
                   help="'raw' for heads (e.g. Mask2Former) whose seg_logits are already scores")
    p.add_argument("--tag", default="crossearth")
    p.add_argument("--tta", action="store_true", help="also report 8-way dihedral TTA")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default="diagnostics/hand_labels/crossearth.json")
    main(p.parse_args())
