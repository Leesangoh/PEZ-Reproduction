"""Phase 2 figure: feature convention x probe strength for V-JEPA 2-L.

Produces a 2x3 grid:
    rows    = {post-block capture, pre-block capture}
    columns = {trainable (strong), ridge_weak, adamw100}
Each panel overlays speed / direction / acceleration R^2 vs. layer.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))

VARIANTS = [
    # (csv_name, feature_label, probe_label)
    ("results_vjepa2_L.csv",                     "post-block", "trainable (strong)"),
    ("results_vjepa2_L_postblock_ridge.csv",     "post-block", "ridge_weak (alpha in {1,10,100})"),
    ("results_vjepa2_L_postblock_adamw100.csv",  "post-block", "adamw100 (single-HP)"),
    ("results_vjepa2_L_preblock.csv",            "pre-block",  "trainable (strong)"),
    ("results_vjepa2_L_preblock_ridge.csv",      "pre-block",  "ridge_weak (alpha in {1,10,100})"),
    ("results_vjepa2_L_preblock_adamw100.csv",   "pre-block",  "adamw100 (single-HP)"),
]

COLORS = {"speed": "#1f77b4", "direction": "#d62728", "acceleration": "#2ca02c"}

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)

for ax, (csv, feat_label, probe_label) in zip(axes.flat, VARIANTS):
    df = pd.read_csv(os.path.join(ROOT, csv))
    for probe in ["speed", "direction", "acceleration"]:
        sub = df[df.probe == probe].sort_values("layer")
        ax.plot(sub.layer, sub.mean_r2, marker="o", markersize=3.5,
                color=COLORS[probe], label=probe.capitalize(), linewidth=1.6)
        ax.fill_between(
            sub.layer,
            sub.mean_r2 - sub.std_r2,
            sub.mean_r2 + sub.std_r2,
            color=COLORS[probe], alpha=0.15,
        )

    # PEZ zone — paper places PEZ around ~1/3 depth (~L8 for 24-layer ViT-L).
    ax.axvspan(7, 10, color="orange", alpha=0.1, label="Paper PEZ zone" if ax is axes[0, 0] else None)

    ax.set_title(f"{feat_label}  |  {probe_label}", fontsize=10)
    ax.set_xlabel("Layer index")
    ax.set_ylim(-0.2, 1.05)
    ax.grid(True, alpha=0.3)

axes[0, 0].set_ylabel("R² (mean over 5-fold CV, position-grouped)")
axes[1, 0].set_ylabel("R² (mean over 5-fold CV, position-grouped)")
axes[0, 0].legend(loc="lower right", fontsize=8)

fig.suptitle(
    "V-JEPA 2-L · feature convention × probe strength (Phase 2)",
    fontsize=13,
)
fig.tight_layout()

out = os.path.join(ROOT, "pez_reproduction_vitl_phase2_2x3.png")
fig.savefig(out, dpi=150)
print("Saved", out)
