#!/usr/bin/env python
"""
d16_capacity_estimate.py — the project's first end-to-end capacity and energy figure.

WHY. `MASTER_CONTEXT` repeatedly refers to "the headline GW figure" and compares
against a published ~1.89 GW for an Indian metropolis, but **nothing in this repo
computes one**. The pipeline has always stopped before the arithmetic. D14 built
the uncertainty budget and D15 corrected a 22.5% double-count in the formula, so
the chain is finally specifiable.

TWO CORRECTIONS THIS APPLIES THAT A NAIVE RUN WOULD MISS.

1. **Do NOT blindly un-erode.** The obvious move -- the model trains on 0.4 m
   eroded labels, so divide its area by the 0.8748 eroded/un-eroded ratio to
   recover real roofs -- is WRONG here, and I shipped it on the first run for a
   14.3% inflation. The teacher is trained on eroded labels but **validated and
   best-epoch-selected on the UN-eroded val set**, so its output already sits at
   un-eroded extent: measured pred/label = 0.9847. The script now computes that
   ratio and refuses the correction when it is near 1.0, printing what it
   ignored. A correction derived from how a model was TRAINED must be checked
   against what it actually EMITS.

2. **PVOUT already contains system losses** (D15). `PVOUT x PR` double-counts.
   This uses PVOUT x a small rooftop-specific derate instead.

The train crops were seen during training, so predicting over the full AOI is
partly in-sample. That is unavoidable for a city-wide total and is reported
alongside a val-only extrapolation so the two can be compared.

Usage:
    python scripts/d16_capacity_estimate.py --ckpt <best.pth> --encoder mit_b2
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
M2_PER_PX = 0.266 ** 2          # 0.070756 m^2 -- MASTER_CONTEXT says 0.07083


@torch.no_grad()
def sum_area(m, dev, split_dir, threshold, batch, label):
    img_dir = os.path.join(split_dir, "images")
    msk_dir = os.path.join(split_dir, "masks")
    names = sorted(os.listdir(img_dir))
    pred_px = lab_px = tot_px = 0
    for i in range(0, len(names), batch):
        chunk = names[i:i + batch]
        xs = []
        for n in chunk:
            im = cv2.cvtColor(cv2.imread(os.path.join(img_dir, n)), cv2.COLOR_BGR2RGB)
            xs.append(((im.astype(np.float32) / 255. - MEAN) / STD).transpose(2, 0, 1))
        p = torch.sigmoid(m(torch.from_numpy(np.stack(xs)).to(dev)))[:, 0].cpu().numpy()
        for n, pm in zip(chunk, p):
            pred_px += int((pm > threshold).sum())
            tot_px += int(pm.size)
            gt = cv2.imread(os.path.join(msk_dir, n), cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                lab_px += int((gt > 127).sum())
        if i % (batch * 100) == 0:
            print(f"  [{label}] {i}/{len(names)}")
    return pred_px, lab_px, tot_px, len(names)


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = smp.Unet(a.encoder, encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(a.ckpt, map_location=dev)["model_state"])
    m = m.to(dev).eval()

    res = {}
    for split in ("train", "val"):
        d = os.path.join(a.crops, split)
        if os.path.isdir(d):
            res[split] = sum_area(m, dev, d, a.threshold, a.batch, split)
            pp, lp, tp, nc = res[split]
            print(f"[{split}] {nc} crops · pred {pp/1e6:.1f} Mpx · "
                  f"label {lp/1e6:.1f} Mpx · fg {pp/max(tp,1):.4f}")

    pred_px = sum(v[0] for v in res.values())
    tot_px = sum(v[2] for v in res.values())
    n_crops = sum(v[3] for v in res.values())

    lab_px = sum(v[1] for v in res.values())
    aoi_m2 = tot_px * M2_PER_PX
    pred_m2 = pred_px * M2_PER_PX

    # SELF-CHECK, because I got this wrong on the first run. The un-erosion
    # correction assumes the model's output matches the ERODED labels it trained
    # on. It does not: the teacher is validated and best-epoch-selected on the
    # UN-eroded val set, so its predictions calibrate to un-eroded extent. The
    # measured ratio against the un-eroded labels summed above settles it --
    # near 1.0 means no correction is due, and applying one inflates the headline
    # figure by 14%. Assumed corrections must be checked against what the model
    # actually emits, not against how it was trained.
    pred_vs_uneroded = pred_px / max(lab_px, 1)
    if abs(pred_vs_uneroded - 1.0) < 0.10 and a.erosion_area_ratio < 0.98:
        print(f"[check] pred/label vs UN-ERODED labels = {pred_vs_uneroded:.4f} "
              f"-- already un-eroded extent; IGNORING erosion_area_ratio "
              f"{a.erosion_area_ratio} (would inflate by "
              f"{100*(1/a.erosion_area_ratio - 1):.1f}%)")
        eff_ratio = 1.0
    else:
        eff_ratio = a.erosion_area_ratio
    roof_m2 = pred_m2 / eff_ratio
    usable_m2 = roof_m2 * a.k_usable
    kwp = usable_m2 * a.eta * 1.0                        # 1 kW/m^2 at STC
    kwh = kwp * a.pvout * a.rooftop_derate               # correction 2 (D15)

    # Product chain -> relative variances add.
    terms = [
        ("segmentation area", 1.0156, 0.0219, "MEASURED n=3 seeds (D14)"),
        ("erosion un-do", eff_ratio, 0.010, "SELF-CHECKED — 1.0 when output is already un-eroded"),
        ("k_usable", a.k_usable, 0.075, "ASSUMED — 68% of variance (D14/D15)"),
        ("eta", a.eta, 0.010, "PLANNED — mono-PERC"),
        ("PVOUT", a.pvout, 100.0, "SECONDARY — not citable, needs GSA map (D15)"),
        ("rooftop derate", a.rooftop_derate, 0.030, "ASSUMED (D15)"),
    ]
    rel = np.array([sd / mu for _, mu, sd, _ in terms])
    tot_rel = float(np.sqrt((rel ** 2).sum()))
    share = rel ** 2 / (rel ** 2).sum()

    print(f"\n{'':<26}{'value':>16}")
    print(f"{'AOI covered':<26}{aoi_m2/1e6:>13.1f} km2   ({n_crops} crops)")
    print(f"{'predicted roof area':<26}{pred_m2/1e6:>13.2f} km2"
          f"   (pred/label vs un-eroded {pred_vs_uneroded:.4f})")
    print(f"{'roof area used':<26}{roof_m2/1e6:>13.2f} km2"
          f"   ({100*roof_m2/aoi_m2:.1f}% of AOI)")
    print(f"{'usable area':<26}{usable_m2/1e6:>13.2f} km2   (k_usable {a.k_usable})")
    print(f"{'INSTALLED CAPACITY':<26}{kwp/1e6:>13.2f} GWp")
    print(f"{'ANNUAL ENERGY':<26}{kwh/1e9:>13.2f} TWh/yr")
    print(f"\nuncertainty +/-{100*tot_rel:.1f}% (1 sd)  ->  "
          f"{kwp/1e6*(1-tot_rel):.2f}-{kwp/1e6*(1+tot_rel):.2f} GWp, "
          f"{kwh/1e9*(1-tot_rel):.2f}-{kwh/1e9*(1+tot_rel):.2f} TWh/yr")
    print(f"\n{'term':<22}{'rel sd':>9}{'share':>9}  provenance")
    for i in np.argsort(-share):
        nm, mu, sd, prov = terms[i]
        print(f"{nm:<22}{rel[i]:>8.1%}{share[i]:>9.1%}  {prov}")

    json.dump({
        "checkpoint": a.ckpt, "threshold": a.threshold, "n_crops": n_crops,
        "aoi_km2": round(aoi_m2 / 1e6, 3),
        "predicted_roof_km2": round(pred_m2 / 1e6, 3),
        "pred_vs_uneroded_label": round(pred_vs_uneroded, 4),
        "erosion_ratio_applied": eff_ratio,
        "roof_area_km2": round(roof_m2 / 1e6, 3),
        "roof_fraction_of_aoi": round(roof_m2 / aoi_m2, 4),
        "usable_area_km2": round(usable_m2 / 1e6, 3),
        "installed_capacity_GWp": round(kwp / 1e6, 3),
        "annual_energy_TWh": round(kwh / 1e9, 3),
        "rel_uncertainty_1sd": round(tot_rel, 4),
        "terms": [{"name": nm, "mean": mu, "sd": sd, "rel_sd": round(float(rel[i]), 4),
                   "variance_share": round(float(share[i]), 4), "provenance": prov}
                  for i, (nm, mu, sd, prov) in enumerate(terms)],
        "caveats": [
            "PVOUT is secondary-source and not citable (D15)",
            "train crops were seen during training; full-AOI total is partly in-sample",
            "areas are agreement with Open Buildings, which over-covers (D10)",
            "PVOUT x rooftop_derate, never PVOUT x PR (D15)",
        ],
    }, open(a.out, "w"), indent=2)
    print(f"[ok] {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--encoder", default="mit_b2")
    p.add_argument("--crops", default="data/jaipur_weak")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--erosion_area_ratio", type=float, default=1.0,
                   help="eroded label area / un-eroded label area (0.20172/0.2306)")
    p.add_argument("--k_usable", type=float, default=0.60)
    p.add_argument("--eta", type=float, default=0.20)
    p.add_argument("--pvout", type=float, default=1750.0)
    p.add_argument("--rooftop_derate", type=float, default=0.95)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", default="diagnostics/d16_capacity.json")
    main(p.parse_args())
