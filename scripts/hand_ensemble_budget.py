#!/usr/bin/env python
"""
hand_ensemble_budget.py - steps 2 and 3 of the "toward higher IoU" plan, on the frozen
152 hand-labelled crops.

Step 3 (ensemble + TTA): every checkpoint in --models is run on every crop, plain and with
8-way dihedral TTA (4 rotations x flip, predictions mapped back and averaged). Ensembles are
the mean probability of the listed members. Threshold is picked on the calib half from a fine
sweep and IoU is reported on the test half, with a paired crop bootstrap against --baseline
(plain, no TTA).

Step 2 (error budget) for the best test configuration and the baseline:
  - IoU when pixels within k px of the hand roof edge are ignored (k = 1, 2, 3): how much of
    the error sits on boundaries no model can place exactly at 26.6 cm/px.
  - error split: missed buildings (hand components with no predicted pixel), false buildings
    (predicted components touching no hand pixel), and the rest (boundary / partial).

Usage (DGX, in run_docker.sh):
  python scripts/hand_ensemble_budget.py --index data/hand_r12b2/index.csv \
      --models data/hand_r12b2/models_ens.csv --baseline r34_c0_shift \
      --out diagnostics/hand_labels/ensemble_budget.json
"""
import argparse
import csv
import itertools
import json
import os

import cv2
import numpy as np
import torch

from score_models_hand import MEAN, STD, load_bin, load_model
from PIL import Image

SWEEP = [round(t, 2) for t in np.arange(0.15, 0.71, 0.05)]


def dihedral_probs(model, x):
    """x: B,3,H,W tensor. Mean sigmoid over the 8 dihedral transforms."""
    acc = 0
    for k in range(4):
        for flip in (False, True):
            t = torch.rot90(x, k, (2, 3))
            if flip:
                t = torch.flip(t, (3,))
            p = torch.sigmoid(model(t))
            if flip:
                p = torch.flip(p, (3,))
            acc = acc + torch.rot90(p, -k, (2, 3))
    return acc / 8


def run_models(rows, models_csv, dev, batch):
    imgs = np.stack([np.asarray(Image.open(r["img"]).convert("RGB")) for r in rows])
    x_all = ((imgs.astype(np.float32) / 255. - MEAN) / STD).transpose(0, 3, 1, 2)
    probs = {}
    for m in csv.DictReader(open(models_csv)):
        model, enc = load_model(m["ckpt"], dev)
        plain, tta = [], []
        with torch.no_grad():
            for i in range(0, len(rows), batch):
                x = torch.from_numpy(x_all[i:i + batch]).to(dev)
                plain.append(torch.sigmoid(model(x))[:, 0].cpu().numpy())
                tta.append(dihedral_probs(model, x)[:, 0].cpu().numpy())
        probs[m["tag"]] = np.concatenate(plain).astype(np.float16)
        probs[m["tag"] + "+tta"] = np.concatenate(tta).astype(np.float16)
        print(f"ran {m['tag']} ({enc})", flush=True)
        del model
        torch.cuda.empty_cache()
    return probs


def counts(prob, hand, thr):
    out = []
    for p, h in zip(prob, hand):
        q = p > thr
        out.append((int((q & h).sum()), int((q & ~h).sum()), int((~q & h).sum()), int(q.sum()), int(h.sum())))
    return np.array(out, np.int64)


def agg(c):
    tp, fp, fn, pp, hp = c.sum(0)
    return tp / max(tp + fp + fn, 1), pp / max(hp, 1)


def boundary_band(h, k):
    if k == 0:
        return np.zeros_like(h)
    ker = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    hu = h.astype(np.uint8)
    return (cv2.dilate(hu, ker) - cv2.erode(hu, ker)).astype(bool)


def budget(prob, hand, thr, idx):
    """Error budget on crops idx at threshold thr."""
    res = {}
    for k in (0, 1, 2, 3):
        tp = fp = fn = 0
        for i in idx:
            q, h = prob[i] > thr, hand[i]
            keep = ~boundary_band(h, k)
            tp += int((q & h & keep).sum()); fp += int((q & ~h & keep).sum()); fn += int((~q & h & keep).sum())
        res[f"iou_ignore_{k}px_band"] = tp / max(tp + fp + fn, 1)
    miss = false = fp_all = fn_all = 0
    for i in idx:
        q, h = prob[i] > thr, hand[i]
        fp_all += int((q & ~h).sum()); fn_all += int((~q & h).sum())
        n, lab = cv2.connectedComponents(h.astype(np.uint8), connectivity=4)
        for c in range(1, n):
            comp = lab == c
            if not (comp & q).any():
                miss += int(comp.sum())
        n, lab = cv2.connectedComponents(q.astype(np.uint8), connectivity=4)
        for c in range(1, n):
            comp = lab == c
            if not (comp & h).any():
                false += int(comp.sum())
    err = fp_all + fn_all
    res.update(error_px=err,
               share_missed_buildings=miss / max(err, 1),
               share_false_buildings=false / max(err, 1),
               share_boundary_partial=(err - miss - false) / max(err, 1))
    return res


def main(a):
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}", flush=True)
    rows = list(csv.DictReader(open(a.index)))
    hand = [load_bin(r["mask"]) for r in rows]
    cal = [i for i, r in enumerate(rows) if r["half"] == "calib"]
    tst = [i for i, r in enumerate(rows) if r["half"] == "test"]
    cache = os.path.splitext(a.out)[0] + "_probs.npz"
    if os.path.isfile(cache) and not a.rerun:
        probs = dict(np.load(cache))
    else:
        probs = run_models(rows, a.models, dev, a.batch)
        np.savez_compressed(cache, **probs)

    members = [t for t in probs if not t.endswith("+tta")]
    configs = dict(probs)
    for r in range(2, len(members) + 1):
        for combo in itertools.combinations(members, r):
            for suf in ("", "+tta"):
                configs["ens(" + ",".join(combo) + ")" + suf] = np.mean(
                    [probs[c + suf].astype(np.float32) for c in combo], 0).astype(np.float16)

    rng = np.random.default_rng(0)
    bidx = rng.integers(0, len(tst), (2000, len(tst)))
    def boot(c):
        s = c[tst][bidx].sum(1)
        return s[:, 0] / (s[:, 0] + s[:, 1] + s[:, 2])
    base_thr = None
    table = {}
    for tag, pr in configs.items():
        cs = {t: counts(pr, hand, t) for t in SWEEP}
        thr = max(SWEEP, key=lambda t: agg(cs[t][cal])[0])
        it, at = agg(cs[thr][tst])
        table[tag] = dict(thr=thr, calib_iou=agg(cs[thr][cal])[0], test_iou=it, test_area=at,
                          _c=cs[thr])
    b = boot(table[a.baseline]["_c"])
    for tag, v in table.items():
        d = boot(v["_c"]) - b
        v["diff_vs_baseline"] = [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    best = max(table, key=lambda t: table[t]["calib_iou"])   # chosen on calib, not test
    best_test = max(table, key=lambda t: table[t]["test_iou"])
    out = dict(
        baseline=a.baseline, chosen_on_calib=best, best_on_test_ORACLE=best_test,
        configs={t: {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k != "_c"}
                 for t, v in sorted(table.items(), key=lambda kv: -kv[1]["test_iou"])},
        budget={t: budget(configs[t], hand, table[t]["thr"], tst) for t in {a.baseline, best, best_test}})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{'config':60s} thr  calib  TEST   area  diff vs {a.baseline} (95%)")
    for t, v in list(out["configs"].items())[:25]:
        d = v["diff_vs_baseline"]
        print(f"{t:60s} {v['thr']:.2f} {v['calib_iou']:.3f}  {v['test_iou']:.3f}  {v['test_area']:.2f}  "
              f"{d[0]:+.3f} ({d[1]:+.3f} to {d[2]:+.3f})")
    print("chosen on calib:", best, "| best on test (oracle):", best_test)
    print(json.dumps(out["budget"], indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--baseline", default="r34_c0_shift")
    p.add_argument("--out", default="diagnostics/hand_labels/ensemble_budget.json")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--rerun", action="store_true")
    main(p.parse_args())
