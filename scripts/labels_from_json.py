#!/usr/bin/env python
"""
labels_from_json.py — turn polygon annotations into the binary masks the
pipeline expects, and check them before they are submitted.

WHY. Every browser and desktop labelling tool exports *polygons* (labelme JSON,
makesense.ai VOC/JSON), but `merge_split_rate.py` scores **binary PNG masks**
matching each image pixel-for-pixel. Without this step a teammate finishes the
tedious part and then discovers their output is the wrong format.

THE CHECKS MATTER AS MUCH AS THE CONVERSION. A silently malformed label set is
worse than a missing one: it produces a number nobody questions. So this
verifies size, binarity, and -- the one that actually bites -- that adjacent
buildings were drawn as SEPARATE polygons rather than one merged blob, by
comparing the component count against Open Buildings' count for the same crop.
A hand label with far fewer components than OB usually means buildings were
traced as a single region, which defeats the whole point of the batch.

Usage:
    # convert a folder of labelme .json into masks
    python scripts/labels_from_json.py --json_dir path/to/json \
        --out labelling/r12_batch1/working/labels

    # verify an existing folder of masks (no conversion)
    python scripts/labels_from_json.py --verify_only \
        --out labelling/r12_batch1/working/labels \
        --images labelling/r12_batch1/working/images
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def polys_from_labelme(d):
    """labelme stores [[x,y], ...] per shape; ignore any non-polygon shape."""
    out = []
    for sh in d.get("shapes", []):
        pts = sh.get("points") or []
        if sh.get("shape_type", "polygon") == "polygon" and len(pts) >= 3:
            out.append(np.asarray(pts, dtype=np.int32))
    return out


def convert(a):
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.json_dir, "*.json")))
    if not files:
        raise SystemExit(f"no .json files in {a.json_dir}")
    n = 0
    for jf in files:
        d = json.load(open(jf))
        h = d.get("imageHeight") or a.size
        w = d.get("imageWidth") or a.size
        polys = polys_from_labelme(d)
        m = np.zeros((h, w), np.uint8)
        # fillPoly per polygon, not all at once: drawing them together can merge
        # touching shapes into one region, which is exactly what must not happen.
        for p in polys:
            cv2.fillPoly(m, [p], 255)
        name = os.path.splitext(os.path.basename(jf))[0] + ".png"
        cv2.imwrite(os.path.join(a.out, name), m)
        n += 1
        print(f"  {name}: {len(polys)} polygons")
    print(f"[convert] wrote {n} masks to {a.out}")


def verify(a):
    """Report problems loudly; return count of masks with issues."""
    imgs = sorted(os.listdir(a.images))
    bad = 0
    rows = []
    for name in imgs:
        mp = os.path.join(a.out, name)
        if not os.path.isfile(mp):
            rows.append((name, "MISSING", "", "", ""))
            bad += 1
            continue
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        im = cv2.imread(os.path.join(a.images, name))
        issues = []
        if m is None:
            rows.append((name, "UNREADABLE", "", "", ""))
            bad += 1
            continue
        if im is not None and m.shape[:2] != im.shape[:2]:
            issues.append(f"size {m.shape[:2]} != image {im.shape[:2]}")
        vals = set(np.unique(m).tolist())
        if not vals <= {0, 255}:
            issues.append(f"not binary (values {sorted(vals)[:5]}...)")
        b = (m > 127).astype(np.uint8)
        ncomp, _ = cv2.connectedComponents(b)
        ncomp -= 1
        fg = float(b.mean())
        ref = os.path.join(os.path.dirname(a.out.rstrip("/")),
                           "openbuildings_reference", name)
        ob = ""
        if os.path.isfile(ref):
            r = cv2.imread(ref, cv2.IMREAD_GRAYSCALE)
            if r is not None:
                nob, _ = cv2.connectedComponents((r > 127).astype(np.uint8))
                ob = nob - 1
                # Far fewer components than OB is the classic "traced a block as
                # one shape" error. Far more is fine -- OB merges buildings too.
                if ob >= 5 and ncomp < 0.5 * ob:
                    issues.append(f"only {ncomp} regions vs OB's {ob} — "
                                  "were touching buildings merged?")
        if issues:
            bad += 1
        rows.append((name, "; ".join(issues) or "ok", ncomp, ob, f"{fg:.3f}"))

    w = max(len(r[0]) for r in rows) + 2
    print(f"\n{'file':<{w}}{'regions':>8}{'OB':>6}{'fg':>7}  status")
    for name, status, ncomp, ob, fg in rows:
        print(f"{name:<{w}}{str(ncomp):>8}{str(ob):>6}{fg:>7}  {status}")
    done = sum(1 for r in rows if r[1] != "MISSING")
    print(f"\n[verify] {done}/{len(imgs)} labelled, {bad} with issues")
    if bad == 0 and done == len(imgs):
        print("[verify] ALL GOOD — ready to submit")
    return bad


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir", help="folder of labelme .json files")
    p.add_argument("--out", required=True, help="where masks go / already are")
    p.add_argument("--images", help="matching images folder, for verification")
    p.add_argument("--size", type=int, default=512,
                   help="fallback mask size if the JSON omits it")
    p.add_argument("--verify_only", action="store_true")
    a = p.parse_args()
    if not a.verify_only:
        if not a.json_dir:
            raise SystemExit("--json_dir required unless --verify_only")
        convert(a)
    if a.images:
        verify(a)
    elif a.verify_only:
        raise SystemExit("--images required with --verify_only")
