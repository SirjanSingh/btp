#!/usr/bin/env python
"""
Merge rate and split rate — the failure pixel IoU cannot see.

D4 measured **78 % of Jaipur buildings touching a neighbour**, so a model that
fuses adjacent buildings will be wrong about building COUNTS while its IoU looks
fine. `MASTER_CONTEXT` §6.2: pixel IoU cannot detect this, and boundary IoU
largely cannot either — a Hong Kong study cut under-segmentation 35.7 % → 5.0 %
while Boundary F1 sat at 79.78 %. The two move independently.

That matters because the pipeline ends in a **per-building kW estimate**. Merging
two houses changes the count, the per-roof area distribution and therefore
`k_usable` — with no effect on IoU at all.

Definitions used here (stated explicitly, because the literature varies):

  A predicted component P and a label component L are **associated** when their
  intersection covers at least `--min_overlap` of L's area.

  MERGE  — one P associated with >= 2 label components. Reported as the fraction
           of LABEL components caught up in such a merge (under-segmentation).
  SPLIT  — one L associated with >= 2 predicted components. Reported as the
           fraction of label components so divided (over-segmentation).
  MISSED — an L with no associated P at all.

Counting over label components rather than events means the numbers read as
"what fraction of real buildings are mis-instanced", which is the quantity the
kW estimate depends on.

Usage:
    python scripts/merge_split_rate.py --ckpt <path> --encoder resnet34
"""
import argparse
import json
import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def instance_stats(pred, gt, min_overlap, min_area_px, split_overlap=0.10):
    """-> (n_gt, n_merged, n_split_strict, n_split, n_missed, n_pred, frag_sum).

    TWO split definitions, because the strict one was silently blind.

    `split_strict` uses the same >=50% association as merge: a label counts as
    split only when two predictions EACH cover half of it. Nothing smaller than
    half a building qualifies, so a model shattering a building into thirds
    scores zero. That is exactly what happened at 0.8 m erosion -- split read
    0.0 while pred/label hit 1.47 -- and I reported the silence as a surprising
    result for three consecutive runs before noticing.

    `split` uses a low bar (>=10% of the label) and `frag_sum` counts every
    predicted component touching a label at all, so fragmentation is visible
    rather than rounded away. A metric with no failing case in your data has not
    been validated.
    """
    n_p, p_lbl = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    n_g, g_lbl = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    n_p -= 1
    n_g -= 1
    if n_g <= 0:
        return 0, 0, 0, 0, max(n_p, 0)

    g_area = np.bincount(g_lbl.ravel(), minlength=n_g + 1)
    keep_g = {i for i in range(1, n_g + 1) if g_area[i] >= min_area_px}
    if not keep_g:
        return 0, 0, 0, 0, n_p

    # Joint histogram of (pred_label, gt_label) over pixels where both fire.
    # One bincount beats an O(n_p * n_g) loop of boolean ANDs.
    both = (p_lbl > 0) & (g_lbl > 0)
    pairs = p_lbl[both].astype(np.int64) * (n_g + 1) + g_lbl[both].astype(np.int64)
    counts = np.bincount(pairs)
    nz = np.nonzero(counts)[0]

    p_to_g, g_to_p, g_to_p_loose = {}, {}, {}
    for code in nz:
        pi, gi = divmod(int(code), n_g + 1)
        if pi == 0 or gi == 0 or gi not in keep_g:
            continue
        if counts[code] >= min_overlap * g_area[gi]:
            p_to_g.setdefault(pi, set()).add(gi)
            g_to_p.setdefault(gi, set()).add(pi)
        if counts[code] >= split_overlap * g_area[gi]:
            g_to_p_loose.setdefault(gi, set()).add(pi)

    merged = set()
    for pi, gs in p_to_g.items():
        if len(gs) >= 2:
            merged |= gs
    split_strict = {gi for gi, ps in g_to_p.items() if len(ps) >= 2}
    split_loose = {gi for gi, ps in g_to_p_loose.items() if len(ps) >= 2}
    missed = {gi for gi in keep_g if gi not in g_to_p}
    frag_sum = sum(len(ps) for ps in g_to_p_loose.values())
    return (len(keep_g), len(merged), len(split_strict), len(split_loose),
            len(missed), n_p, frag_sum)


@torch.no_grad()
def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = smp.Unet(a.encoder, encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(a.ckpt, map_location=dev)["model_state"])
    m = m.to(dev).eval()

    img_dir = os.path.join(a.val_dir, "images")
    msk_dir = os.path.join(a.val_dir, "masks")
    names = sorted(os.listdir(img_dir))
    if a.limit:
        names = names[:a.limit]

    tot = dict(gt=0, merged=0, split_strict=0, split=0, missed=0, pred=0, frag=0)
    for i in range(0, len(names), a.batch):
        chunk = names[i:i + a.batch]
        xs = []
        for n in chunk:
            im = cv2.cvtColor(cv2.imread(os.path.join(img_dir, n)), cv2.COLOR_BGR2RGB)
            xs.append(((im.astype(np.float32) / 255. - MEAN) / STD).transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(xs)).to(dev)
        pr = (torch.sigmoid(m(x))[:, 0] > a.threshold).cpu().numpy()
        if a.dilate_px:
            # Training on eroded labels shrinks predictions, so scoring against
            # un-eroded truth charges the method for a deliberate offset.
            # Dilating back was meant to restore scale without re-merging.
            #
            # MEASURED 2026-09-10: it does NOT hold. On the 0.4 m eroded MiT-B2,
            # dilate_px=2 cut missed 0.3223 -> 0.2496 but pushed merge 0.3256 ->
            # 0.4163 and pred/label 0.9745 -> 0.8071. Two components 3 px apart
            # are closed by a 2 px dilation on each side, and after 0.4 m (~1.5
            # px) erosion most neighbour gaps are exactly that narrow. Dilation
            # trades misses back for merges at a bad rate. Keep it off unless
            # the erosion is large enough to leave a gap wider than 2*dilate_px
            # -- which is what the 0.8 m sweep is for.
            k = np.ones((2 * a.dilate_px + 1,) * 2, np.uint8)
            pr = np.stack([cv2.dilate(p.astype(np.uint8), k).astype(bool)
                           for p in pr])
        for n, p in zip(chunk, pr):
            g = cv2.imread(os.path.join(msk_dir, n), cv2.IMREAD_GRAYSCALE) > 127
            ng, nm, nss, ns, nmiss, npred, fs = instance_stats(
                p, g, a.min_overlap, a.min_area_px, a.split_overlap)
            tot["gt"] += ng; tot["merged"] += nm
            tot["split_strict"] += nss; tot["split"] += ns
            tot["missed"] += nmiss; tot["pred"] += npred; tot["frag"] += fs
        if (i // a.batch) % 20 == 0:
            print(f"   {i+len(chunk)}/{len(names)}", flush=True)

    g = max(tot["gt"], 1)
    out = {
        "checkpoint": a.ckpt, "encoder": a.encoder, "val_dir": a.val_dir,
        "threshold": a.threshold, "min_overlap": a.min_overlap,
        "min_area_px": a.min_area_px,
        "n_label_components": tot["gt"], "n_pred_components": tot["pred"],
        "merge_rate": round(tot["merged"] / g, 4),
        "split_rate": round(tot["split"] / g, 4),
        "split_rate_strict": round(tot["split_strict"] / g, 4),
        "fragments_per_label": round(tot["frag"] / g, 4),
        "split_overlap": a.split_overlap,
        "missed_rate": round(tot["missed"] / g, 4),
        "pred_per_label": round(tot["pred"] / g, 4),
    }
    print(json.dumps(out, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"[ok] {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--val_dir", default="data/jaipur_weak/val")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min_overlap", type=float, default=0.5,
                   help="fraction of a label component that must be covered")
    p.add_argument("--min_area_px", type=int, default=50,
                   help="ignore label blobs under this size; at 26.6 cm, 50 px "
                        "is ~3.5 m2 and mostly labelling specks")
    p.add_argument("--split_overlap", type=float, default=0.10,
                   help="a label counts as split if >=2 predictions each cover "
                        "this fraction of it; the strict 0.5 bar misses "
                        "fragmentation entirely")
    p.add_argument("--dilate_px", type=int, default=0,
                   help="dilate predictions by N px before scoring, to undo "
                        "label erosion; 0.4 m ~ 1.5 px, 0.8 m ~ 3 px at 26.6 cm")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="")
    main(p.parse_args())
