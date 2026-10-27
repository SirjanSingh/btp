#!/usr/bin/env python
"""
build_run_ledger.py — one consolidated record of every training and eval run.

WHY: checkpoints get deleted when the 40 GB quota runs low, and a `.pth` is the
one artifact nobody can regenerate. The *numbers* inside it, though, are what a
paper actually needs — per-epoch curves, the best score and where it happened,
the config that produced it, the hardware it ran on. This walks every log in the
repo and consolidates them into one JSON plus a human-readable table, so the
stats survive independently of the weights.

Run it BEFORE deleting any checkpoint, and after every experiment.

Reads:
    logs/*.json                     rooftop training (rich, per-epoch)
    rooftop/logs/*.json             rooftop eval + training
    solar_panel/logs/**/*.json      solar training + eval
    experiments/*/outputs/*.json    everything from 2026-09 onward
    *.txt siblings                  fallback when no JSON exists

Writes:
    experiments/RUN_LEDGER.json     machine-readable, full per-epoch history
    experiments/RUN_LEDGER.md       the table you paste into a paper appendix

Usage:
    python scripts/build_run_ledger.py
"""
import glob
import json
import os
import re
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_GLOBS = [
    "logs/*.json",
    "rooftop/logs/*.json",
    "solar_panel/logs/*.json",
    "solar_panel/logs/**/*.json",
    # outputs*, not outputs: the eroded-labels experiment used outputs_mit/ and
    # outputs_resnet34/, and the narrower glob silently skipped both runs -- the
    # same directory-naming assumption that let checkpoints_mit/ past .gitignore.
    "experiments/*/outputs*/*.json",
    "diagnostics/*.json",
    "diagnostics/*/*.json",
]
# " 12    0.2788    0.2622   0.6350   0.7767   0.7181   0.8458   8.26e-05  <- best"
TXT_ROW = re.compile(
    r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")


def stage_of(path):
    if "solar" in path:
        return "solar"
    if "diagnostics" in path:
        return "diagnostic"
    return "rooftop"


def parse_txt(path):
    """Recover per-epoch rows from a .txt log when no JSON was written."""
    rows = []
    try:
        with open(path, errors="ignore") as fh:
            for line in fh:
                m = TXT_ROW.match(line)
                if m:
                    rows.append({
                        "epoch": int(m.group(1)),
                        "tr_loss": float(m.group(2)),
                        "va_loss": float(m.group(3)),
                        "iou": float(m.group(4)),
                        "f1": float(m.group(5)),
                        "precision": float(m.group(6)),
                        "recall": float(m.group(7)),
                    })
    except OSError:
        pass
    return rows


def collect():
    seen, runs = set(), []
    for g in JSON_GLOBS:
        for path in glob.glob(os.path.join(ROOT, g), recursive=True):
            # Dedup on the resolved path: rooftop/logs contains a symlink that
            # makes several logs reachable under two globs, which otherwise
            # double-counts them in the ledger.
            key = os.path.realpath(path)
            if key in seen:
                continue
            seen.add(key)
            try:
                d = json.load(open(path))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(d, dict):
                continue
            rel = os.path.relpath(path, ROOT)
            epochs = d.get("epochs") or []
            best = max(epochs, key=lambda e: e.get("iou", -1)) if epochs else {}
            # Eval runs have no epoch list; their scores sit in a threshold sweep.
            # Without this they render as blank rows and look like failures.
            if not best and isinstance(d.get("threshold_sweep"), list) \
                    and d["threshold_sweep"]:
                best = max(d["threshold_sweep"], key=lambda e: e.get("iou", -1))
                best = dict(best, epoch=f"thr {best.get('threshold')}")
            runs.append({
                "run_name": d.get("run_name", os.path.basename(path)[:-5]),
                "log": rel,
                "stage": stage_of(rel),
                "kind": "eval" if "eval" in rel or "threshold_sweep" in d
                        else ("diagnostic" if "diagnostics" in rel else "train"),
                "timestamp": d.get("timestamp", ""),
                "config": d.get("config", {}),
                "hardware": d.get("hardware", {}),
                "n_epochs": len(epochs),
                "best_epoch": best.get("epoch"),
                "best_iou": best.get("iou"),
                "best_f1": best.get("f1"),
                "best_precision": best.get("precision"),
                "best_recall": best.get("recall"),
                "summary": d.get("summary", {}),
                "threshold_sweep": d.get("threshold_sweep"),
                "results": d.get("results"),
                "epochs": epochs,
            })

    # .txt logs with no JSON sibling — crashed runs live here, and a run that
    # died at epoch 3 is still evidence about what not to repeat.
    for g in ("logs/*.txt", "rooftop/logs/*.txt", "solar_panel/logs/*.txt"):
        for path in glob.glob(os.path.join(ROOT, g)):
            if os.path.exists(path[:-4] + ".json"):
                continue
            rel = os.path.relpath(path, ROOT)
            rows = parse_txt(path)
            best = max(rows, key=lambda e: e["iou"]) if rows else {}
            runs.append({
                "run_name": os.path.basename(path)[:-4],
                "log": rel,
                "stage": stage_of(rel),
                "kind": "train (txt only)",
                "timestamp": "",
                "config": {}, "hardware": {},
                "n_epochs": len(rows),
                "best_epoch": best.get("epoch"),
                "best_iou": best.get("iou"),
                "best_f1": best.get("f1"),
                "best_precision": best.get("precision"),
                "best_recall": best.get("recall"),
                "summary": {}, "threshold_sweep": None, "results": None,
                "epochs": rows,
                "note": "no JSON — recovered from text log"
                        + ("; ZERO epochs completed (crashed run)" if not rows else ""),
            })

    # Several logs exist as genuine copies under both logs/ and <stage>/logs/,
    # so realpath dedup does not catch them. Collapse on run identity and keep
    # the richest record -- most epochs, then a real score over a blank one.
    best_of = {}
    for r in runs:
        k = (r["run_name"], r["kind"].replace(" (txt only)", ""))
        cur = best_of.get(k)
        if cur is None or (r["n_epochs"], r["best_iou"] is not None) > \
                (cur["n_epochs"], cur["best_iou"] is not None):
            best_of[k] = r
    runs = list(best_of.values())

    runs.sort(key=lambda r: (r["timestamp"] or r["run_name"]))
    return runs


def checkpoints_present():
    return {os.path.relpath(p, ROOT)
            for p in glob.glob(os.path.join(ROOT, "**/*.pth"), recursive=True)}


def main():
    runs = collect()
    ckpts = checkpoints_present()

    out_json = os.path.join(ROOT, "experiments/RUN_LEDGER.json")
    with open(out_json, "w") as fh:
        json.dump({
            "generated": datetime.now().isoformat(timespec="seconds"),
            "n_runs": len(runs),
            "checkpoints_on_disk": sorted(ckpts),
            "runs": runs,
        }, fh, indent=1)

    lines = [
        "# Run ledger — every training and evaluation run, consolidated",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/build_run_ledger.py`. "
        "Rebuild it after every run, and **always before deleting a checkpoint**.",
        "",
        "Weights get deleted when the 40 GB quota runs low; these numbers do not. "
        "Full per-epoch curves for every run are in `RUN_LEDGER.json` — this table is "
        "just the summary.",
        "",
        f"**{len(runs)} runs · {len(ckpts)} checkpoints currently on disk**",
        "",
        "| Run | Stage | Kind | Epochs | Best IoU | @ep | F1 | Prec | Rec |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        iou = f"**{r['best_iou']:.4f}**" if r["best_iou"] else "—"
        lines.append(
            f"| `{r['run_name']}` | {r['stage']} | {r['kind']} | {r['n_epochs']} | "
            f"{iou} | {r['best_epoch'] or '—'} | "
            f"{r['best_f1']:.4f} |" .replace("None", "—") if r["best_f1"] else
            f"| `{r['run_name']}` | {r['stage']} | {r['kind']} | {r['n_epochs']} | "
            f"{iou} | {r['best_epoch'] or '—'} | — | — | — |")
        if r["best_f1"]:
            lines[-1] += f" {r['best_precision']:.4f} | {r['best_recall']:.4f} |"

    crashed = [r for r in runs if r["n_epochs"] == 0]
    if crashed:
        lines += ["", "## Runs that completed zero epochs", "",
                  "Kept deliberately — a crashed configuration is evidence too.", ""]
        lines += [f"- `{r['run_name']}` ({r['log']})" for r in crashed]

    with open(os.path.join(ROOT, "experiments/RUN_LEDGER.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[ledger] {len(runs)} runs, {len(ckpts)} checkpoints on disk")
    print("[ledger] wrote experiments/RUN_LEDGER.{json,md}")


if __name__ == "__main__":
    main()
