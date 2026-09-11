#!/usr/bin/env python
"""
d14_energy_budget.py — which term actually controls the headline energy figure?

WHY. `MASTER_CONTEXT` §7 specifies the chain but leaves three constants open:
GTI/PVOUT and PR are [TODO], `k_usable` is [ASSUMED] at 0.60 with a noted +/-25%
swing. Meanwhile a full day has gone into measuring segmentation error. Nobody
has asked the question that decides where effort belongs:

    if every term is uncertain, which one controls the answer?

If segmentation contributes a few percent and `k_usable` contributes twenty, then
further segmentation work cannot move the headline figure and the cheapest real
improvement is the superstructure labelling pass -- which is exactly what §7.4
proposes.

AREA, NOT COUNT. Energy depends on total usable roof **area**. This project's
headline instance metric, `pred/label`, is a *count* ratio and is the wrong input
here: a model can count buildings perfectly and still mis-estimate area. So the
segmentation term is measured directly as predicted foreground area over labelled
foreground area, across the three seeds already trained, giving a mean and spread.

NO FABRICATED IRRADIANCE. PVOUT for Jaipur is [TODO] in the plan and is not
invented here. The run uses the plan's own stated India range (GHI 3.8-6.5
kWh/m2/day, western Rajasthan at the high end) converted to a wide PVOUT interval,
and reports how much of the output variance that ignorance costs. If it turns out
to dominate, looking the number up is the priority; if not, it can wait.

Usage:
    python scripts/d14_energy_budget.py --ckpts <a.pth> <b.pth> <c.pth>
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
def area_ratio(ckpt, encoder, val_dir, threshold, batch):
    """Predicted foreground area / labelled foreground area over the val set."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = smp.Unet(encoder, encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ckpt, map_location=dev)["model_state"])
    m = m.to(dev).eval()
    img_dir, msk_dir = os.path.join(val_dir, "images"), os.path.join(val_dir, "masks")
    names = sorted(os.listdir(img_dir))
    pred_px = lab_px = 0
    for i in range(0, len(names), batch):
        chunk = names[i:i + batch]
        xs = []
        for n in chunk:
            im = cv2.cvtColor(cv2.imread(os.path.join(img_dir, n)), cv2.COLOR_BGR2RGB)
            xs.append(((im.astype(np.float32) / 255. - MEAN) / STD).transpose(2, 0, 1))
        p = torch.sigmoid(m(torch.from_numpy(np.stack(xs)).to(dev)))[:, 0].cpu().numpy()
        for n, pm in zip(chunk, p):
            gt = cv2.imread(os.path.join(msk_dir, n), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            pred_px += int((pm > threshold).sum())
            lab_px += int((gt > 127).sum())
    return pred_px / max(lab_px, 1), pred_px, lab_px


def main(a):
    ratios = []
    for c in a.ckpts:
        r, pp, lp = area_ratio(c, a.encoder, a.val_dir, a.threshold, a.batch)
        ratios.append(r)
        print(f"[area] {os.path.basename(os.path.dirname(c)):<44} ratio {r:.4f}")
    ratios = np.array(ratios)
    seg_mean, seg_sd = float(ratios.mean()), float(ratios.std(ddof=1)) if len(ratios) > 1 else 0.0
    print(f"\n[area] segmentation area ratio = {seg_mean:.4f} +/- {seg_sd:.4f} (1 sd, n={len(ratios)})")

    # Term: (name, mean, sd, provenance). sd is a 1-sigma stand-in for the stated
    # range, not a measured dispersion except where marked MEASURED.
    terms = [
        ("segmentation area ratio", seg_mean, seg_sd, f"MEASURED n={len(ratios)} seeds"),
        ("k_usable", 0.60, 0.075, "ASSUMED — plan notes +/-25% swing"),
        ("eta (module efficiency)", 0.20, 0.01, "PLANNED — mono-PERC"),
        ("PVOUT kWh/kWp/yr", 1650.0, 150.0, "TODO — plan's India range, NOT Jaipur-specific"),
        ("PR", 0.775, 0.025, "TODO — plan states 0.75-0.80"),
    ]
    print(f"\n{'term':<26}{'mean':>10}{'sd':>9}{'rel sd':>9}  provenance")
    for nm, mu, sd, prov in terms:
        print(f"{nm:<26}{mu:>10.4f}{sd:>9.4f}{sd/mu:>8.1%}  {prov}")

    # The chain is a product, so relative variances add. Each term's share of
    # total output variance is its squared relative sd over the sum.
    rel = np.array([sd / mu for _, mu, sd, _ in terms])
    share = rel ** 2 / (rel ** 2).sum()
    tot_rel = float(np.sqrt((rel ** 2).sum()))
    print(f"\n{'term':<26}{'share of output variance':>26}")
    order = np.argsort(-share)
    for i in order:
        print(f"{terms[i][0]:<26}{100*share[i]:>25.1f}%")
    print(f"\ntotal relative uncertainty on the energy figure: +/-{100*tot_rel:.1f}% (1 sd)")
    print(f"                                                  +/-{200*tot_rel:.1f}% (2 sd)")

    out = {
        "val_dir": a.val_dir, "threshold": a.threshold,
        "segmentation_area_ratio": {"mean": round(seg_mean, 4),
                                    "sd": round(seg_sd, 4),
                                    "per_seed": [round(float(r), 4) for r in ratios]},
        "terms": [{"name": nm, "mean": mu, "sd": sd, "rel_sd": round(sd / mu, 4),
                   "variance_share": round(float(share[i]), 4), "provenance": prov}
                  for i, (nm, mu, sd, prov) in enumerate(terms)],
        "total_rel_uncertainty_1sd": round(tot_rel, 4),
        "note": "product chain -> relative variances add. PVOUT is the plan's "
                "India-wide range, not a Jaipur lookup; it is included to size "
                "the cost of that ignorance, not to produce a figure.",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[ok] {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--encoder", default="mit_b2")
    p.add_argument("--val_dir", default="data/jaipur_weak/val")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", default="diagnostics/d14_energy_budget.json")
    main(p.parse_args())
