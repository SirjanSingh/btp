#!/usr/bin/env python
"""
build_selftrain_set.py — assemble a self-training set: real source + pseudo-labelled target.

WHY A SCRIPT. Every self-training arm so far (S2, S4, S6a, S6b) built this
directory with an ad-hoc shell loop that was never recorded, so the exact
composition of each arm's training set is not reconstructable from the repo.
That is the one thing an arm cannot be re-run without.

RELATIVE SYMLINKS, ALWAYS. The training container mounts the repo at
/workspace, so a symlink stored as an absolute host path (/home/23ucs715/...)
dangles inside the container and the loader sees an empty directory rather than
an error -- PITFALLS 3.15. Links are therefore written relative to the link's
own directory.

Usage:
    python scripts/build_selftrain_set.py \
        --source_dir data/bdappv_split/google_train \
        --target_images data/bdappv_split/ign_train/images \
        --target_masks data/bdappv_pseudo_r2/ign_train/masks \
        --out data/bdappv_st_r2/train
"""
import argparse
import json
import os


def link_all(src_dir, dst_dir, names=None):
    """Symlink every file in src_dir into dst_dir, relative. Returns count."""
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for name in sorted(names if names is not None else os.listdir(src_dir)):
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(dst_dir, name)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.relpath(src, dst_dir), dst)
        n += 1
    return n


def main(a):
    img_out = os.path.join(a.out, "images")
    msk_out = os.path.join(a.out, "masks")

    n_src_i = link_all(os.path.join(a.source_dir, "images"), img_out)
    n_src_m = link_all(os.path.join(a.source_dir, "masks"), msk_out)
    if n_src_i != n_src_m:
        raise SystemExit(f"source images/masks mismatch: {n_src_i} vs {n_src_m}")

    # Only target images that actually received a pseudo-mask. An image whose
    # mask is missing would otherwise be silently paired with the wrong file or
    # dropped by the loader, and either way the arm's real size would be unknown.
    mask_names = set(os.listdir(a.target_masks))
    n_tgt_m = link_all(a.target_masks, msk_out, names=mask_names)
    n_tgt_i = link_all(a.target_images, img_out, names=mask_names)
    if n_tgt_i != n_tgt_m:
        raise SystemExit(f"target images/masks mismatch: {n_tgt_i} vs {n_tgt_m}")

    # Verify every link resolves -- a dangling link is the failure this script
    # exists to prevent, and it must not be reported as success.
    dangling = [f for d in (img_out, msk_out) for f in os.listdir(d)
                if not os.path.exists(os.path.join(d, f))]
    if dangling:
        raise SystemExit(f"{len(dangling)} dangling links, e.g. {dangling[:3]}")

    summary = {
        "out": a.out,
        "source_dir": a.source_dir, "n_source_pairs": n_src_i,
        "target_images": a.target_images, "target_masks": a.target_masks,
        "n_target_pairs": n_tgt_i,
        "n_total_pairs": n_src_i + n_tgt_i,
        "note": "source masks are real labels; target masks are pseudo-labels",
    }
    json.dump(summary, open(os.path.join(a.out, "composition.json"), "w"), indent=2)
    print(f"[st-set] source {n_src_i} + target {n_tgt_i} = "
          f"{n_src_i + n_tgt_i} pairs, all links resolve")
    print(f"[ok] {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", default="data/bdappv_split/google_train")
    p.add_argument("--target_images", default="data/bdappv_split/ign_train/images")
    p.add_argument("--target_masks", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
