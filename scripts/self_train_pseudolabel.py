#!/usr/bin/env python
"""
CBST-style pseudo-labelling for the google -> ign self-training arm.

WHY HERE AND NOT JAIPUR. Jaipur has zero labels, so a self-training run there
can never be scored — only argued about. BDAPPV splits by sensor into `google_`
and `ign_`, both fully labelled, so the target IoU is measurable and every
hyperparameter (threshold policy, class ratio, how many rounds) can be *tuned*
against a real number. `MASTER_CONTEXT`'s ordering principle: settle it here,
freeze, then transfer to Jaipur blind. S1 established the source-only floor at
**0.5611** against an in-domain ceiling of **0.8723**; this tries to close some
of those 31 points.

The ign labels are deliberately NOT used to make pseudo-labels — they are only
read at evaluation. Treating a labelled set as unlabelled is what makes this a
valid UDA rehearsal rather than a supervised run in disguise.

WHY CLASS-RATIO THRESHOLDING RATHER THAN A FIXED 0.95. A fixed high threshold
keeps only what the model is already confident about, which on a shifted domain
means keeping its existing bias and amplifying it: sparse predictions produce
sparse pseudo-labels produce a sparser teacher. CBST instead picks the threshold
so the *selected foreground fraction* matches a target ratio, which holds the
class balance steady across rounds. This is the mechanism `MASTER_CONTEXT`
flags as the fix for foreground collapse.

Usage:
    python scripts/self_train_pseudolabel.py --ckpt <source_only.pth> \
        --src_dir data/bdappv_split/ign_train --out_dir data/bdappv_pseudo/ign_train
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


@torch.no_grad()
def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = smp.Unet(a.encoder, encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(a.ckpt, map_location=dev)["model_state"])
    m = m.to(dev).eval()

    img_dir = os.path.join(a.src_dir, "images")
    names = sorted(os.listdir(img_dir))
    if a.limit:
        names = names[:a.limit]
    os.makedirs(os.path.join(a.out_dir, "masks"), exist_ok=True)
    print(f"[st] {len(names)} target images from {img_dir}")

    # Pass 1 — collect the probability distribution so the threshold can be set
    # from data rather than guessed. Sampling every Nth pixel keeps this cheap;
    # the quantile of a 1-in-16 sample is indistinguishable at these sizes.
    probs = []
    for i in range(0, len(names), a.batch):
        xs = []
        for n in names[i:i + a.batch]:
            im = cv2.cvtColor(cv2.imread(os.path.join(img_dir, n)), cv2.COLOR_BGR2RGB)
            if im.shape[:2] != (512, 512):
                im = cv2.resize(im, (512, 512))
            xs.append(((im.astype(np.float32) / 255. - MEAN) / STD).transpose(2, 0, 1))
        p = torch.sigmoid(m(torch.from_numpy(np.stack(xs)).to(dev)))[:, 0]
        probs.append(p[:, ::4, ::4].reshape(-1).cpu().numpy())
    allp = np.concatenate(probs)

    # CBST: the threshold is the (1 - ratio) quantile, so exactly `ratio` of
    # pixels are labelled foreground. That pins the pseudo-label class balance
    # to the source prior instead of letting it drift down each round.
    thr = float(np.quantile(allp, 1.0 - a.class_ratio))
    print(f"[st] class-ratio {a.class_ratio}: threshold = {thr:.4f}  "
          f"(a fixed 0.5 would select {float((allp > 0.5).mean()):.4f})")

    # Pass 2 — write pseudo-masks. Pixels between the two bands are ambiguous;
    # with a binary mask they must fall somewhere, so they go to background,
    # which is the conservative choice for a precision-critical task.
    kept_fg = 0
    total = 0
    for i in range(0, len(names), a.batch):
        chunk = names[i:i + a.batch]
        xs, sizes = [], []
        for n in chunk:
            im = cv2.cvtColor(cv2.imread(os.path.join(img_dir, n)), cv2.COLOR_BGR2RGB)
            sizes.append(im.shape[:2])
            if im.shape[:2] != (512, 512):
                im = cv2.resize(im, (512, 512))
            xs.append(((im.astype(np.float32) / 255. - MEAN) / STD).transpose(2, 0, 1))
        p = torch.sigmoid(m(torch.from_numpy(np.stack(xs)).to(dev)))[:, 0].cpu().numpy()
        for n, pm, hw in zip(chunk, p, sizes):
            b = (pm > thr).astype(np.uint8)
            # Inference runs at 512 but BDAPPV crops are 400x400. A mask written
            # at model resolution mismatches its image and albumentations rejects
            # the pair outright. Resize back to the ORIGINAL size, nearest so the
            # mask stays binary.
            if b.shape != hw:
                b = cv2.resize(b, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
            kept_fg += int(b.sum())
            total += b.size
            cv2.imwrite(os.path.join(a.out_dir, "masks", n), b * 255)

    summary = {
        "checkpoint": a.ckpt, "source_dir": a.src_dir, "n_images": len(names),
        "class_ratio_target": a.class_ratio, "threshold_chosen": round(thr, 5),
        "pseudo_fg_fraction": round(kept_fg / max(total, 1), 5),
        "note": "target labels never read; used at evaluation only",
    }
    json.dump(summary, open(os.path.join(a.out_dir, "pseudo_summary.json"), "w"),
              indent=2)
    print(f"[st] pseudo-label fg fraction {summary['pseudo_fg_fraction']:.4f}")
    print(f"[ok] {a.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--src_dir", default="data/bdappv_split/ign_train")
    p.add_argument("--out_dir", default="data/bdappv_pseudo/ign_train")
    p.add_argument("--class_ratio", type=float, default=0.06,
                   help="fraction of pixels to label foreground; set from the "
                        "SOURCE prior, not guessed")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
