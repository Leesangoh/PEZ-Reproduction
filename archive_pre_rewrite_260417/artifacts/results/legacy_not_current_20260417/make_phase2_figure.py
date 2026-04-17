"""Phase 2 figure: feature convention x weak probe for V-JEPA 2-L.

Produces a 2x2 grid:
    rows    = {post-block feature, pre-block feature}
    columns = {ridge_weak (alpha in {1,10,100}),
               adamw100 (single-HP lr=1e-3 wd=0.1, 100 epochs)}

Each panel overlays speed / direction / acceleration R^2 vs. layer, with
the paper's approximate PEZ zone (roughly 1/3 depth, i.e. L7-L10 for a
24-layer ViT-L) shaded for reference.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))

VARIANTS = [
    ("results_vjepa2_L_postblock_ridge.csv",     "post-block", "ridge_weak"),
    ("results_vjepa2_L_postblock_adamw100.csv",  "post-block", "adamw100"),
    ("results_vjepa2_L_preblock_ridge.csv",      "pre-block",  "ridge_weak"),
    ("results_vjepa2_L_preblock_adamw100.csv",   "pre-block",  "adamw100"),
]

COLORS = {"speed": "#1f77b4", "direction": "#d62728", "acceleration": "#2ca02c"}

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

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

    ax.axvspan(7, 10, color="orange", alpha=0.1,
               label="Paper PEZ zone" if ax is axes[0, 0] else None)

    ax.set_title(f"{feat_label}  |  {probe_label}", fontsize=11)
    ax.set_xlabel("Layer index")
    ax.set_ylim(-0.2, 1.05)
    ax.grid(True, alpha=0.3)

for ax in axes[:, 0]:
    ax.set_ylabel("R² (mean over 5-fold CV, position-grouped)")
axes[0, 0].legend(loc="lower right", fontsize=8)

fig.suptitle(
    "V-JEPA 2-L · feature convention × weak probe (Phase 2)",
    fontsize=13,
)
fig.tight_layout()

out = os.path.join(ROOT, "pez_reproduction_vitl_phase2_2x2.png")
fig.savefig(out, dpi=150)
print("Saved", out)
