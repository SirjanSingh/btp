"""Generate training curve plots for rooftop and solar panel models."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────
def load(path):
    with open(path) as f:
        return json.load(f)["epochs"]

def extract(epochs, keys=("epoch", "tr_loss", "va_loss", "iou")):
    rows = {k: [] for k in keys}
    for e in epochs:
        for k in keys:
            rows[k].append(e[k])
    return rows

# ── rooftop: single run, 100 epochs ──────────────────────────────────────────
ROOT = Path(__file__).parent
rt = extract(load(ROOT / "rooftop/logs/unet_resnet34_2000samples_100ep_20260330_125206.json"))

# ── solar panel: stitched from 4 sequential runs ─────────────────────────────
sp_raw = (
    load(ROOT / "solar_panel/logs/unet_resnet34_100ep_20260403_062049.json")   # ep 1-33
    + load(ROOT / "solar_panel/logs/unet_resnet34_100ep_20260403_163032.json") # ep 34-38
    + load(ROOT / "solar_panel/logs/unet_resnet34_100ep_20260403_185446.json") # ep 39-55
    + [e for e in load(ROOT / "solar_panel/logs/unet_resnet34_100ep_20260404_061348.json")
       if e["epoch"] >= 56]                                                    # ep 56-93
)
# deduplicate on epoch number, keep first occurrence
seen, sp_deduped = set(), []
for e in sp_raw:
    if e["epoch"] not in seen:
        seen.add(e["epoch"])
        sp_deduped.append(e)
sp_deduped.sort(key=lambda e: e["epoch"])
sp = extract(sp_deduped)

# ── stitch boundaries (for vertical markers on solar panel plot) ──────────────
boundaries = [33, 38, 55]   # epoch numbers where runs were resumed

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("U-Net + ResNet-34 Training Curves", fontsize=15, fontweight="bold", y=0.98)

COLORS = {"tr": "#2196F3", "va": "#F44336", "iou": "#4CAF50"}

def plot_loss(ax, data, title, boundaries=None):
    ax.plot(data["epoch"], data["tr_loss"], color=COLORS["tr"], lw=1.8, label="Train loss")
    ax.plot(data["epoch"], data["va_loss"], color=COLORS["va"], lw=1.8, label="Val loss")
    if boundaries:
        for b in boundaries:
            ax.axvline(b, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (BCE + Dice)")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)

def plot_iou(ax, data, title, boundaries=None):
    ax.plot(data["epoch"], data["iou"], color=COLORS["iou"], lw=1.8, label="Val IoU")
    best_epoch = data["epoch"][data["iou"].index(max(data["iou"]))]
    best_iou   = max(data["iou"])
    ax.scatter([best_epoch], [best_iou], color="gold", s=80, zorder=5,
               edgecolors="black", lw=0.8, label=f"Best  IoU={best_iou:.4f} @ ep{best_epoch}")
    if boundaries:
        for b in boundaries:
            ax.axvline(b, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0.6)

# row 0 – rooftop
plot_loss(axes[0, 0], rt, "Rooftop – Loss")
plot_iou (axes[0, 1], rt, "Rooftop – Val IoU")

# row 1 – solar panel
plot_loss(axes[1, 0], sp, "Solar Panel – Loss", boundaries)
plot_iou (axes[1, 1], sp, "Solar Panel – Val IoU", boundaries)

# dashed-line legend for solar panel plots
resume_patch = mpatches.Patch(color="gray", alpha=0.6, label="Resume boundary")
for ax in axes[1]:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [resume_patch], labels + ["Resume boundary"], framealpha=0.9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = ROOT / "training_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
