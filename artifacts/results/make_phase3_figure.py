"""Phase 3 figure: position grouping vs LOO-direction grouping for V-JEPA 2-L.

A 2x2 grid:
    rows    = {pre-block x trainable, pre-block x adamw100}
    columns = {position grouping (Phase 2), direction-LOO grouping (Phase 3)}

Each panel overlays speed / direction / acceleration R^2 vs. layer. The
LOO-direction runs are 8-fold (one direction angle entirely held out each
fold), so they measure generalization to unseen angles and should collapse
direction R^2 if the probe was memorizing angles under position grouping.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(ROOT, "archive")

# 8-fold LOO direction is ill-posed for the direction probe (val fold contains
# a single angle -> constant sin/cos target -> ss_tot = 0 -> R^2 undefined).
# We use 4-fold LOO instead: each val fold contains 2 of 8 direction angles,
# which gives non-zero target variance while still holding those angles out
# of the training split entirely.

# (csv_path, row_label, col_label)
VARIANTS = [
    (os.path.join(ARCHIVE, "results_vjepa2_L_preblock.csv"),
     "pre-block · trainable",     "position grouping (5-fold)"),
    (os.path.join(ROOT, "results_vjepa2_L_preblock_trainable_loo_dir4.csv"),
     "pre-block · trainable",     "direction grouping (4-fold)"),
    (os.path.join(ROOT, "results_vjepa2_L_preblock_adamw100.csv"),
     "pre-block · adamw100",      "position grouping (5-fold)"),
    (os.path.join(ROOT, "results_vjepa2_L_preblock_adamw100_loo_dir4.csv"),
     "pre-block · adamw100",      "direction grouping (4-fold)"),
]

COLORS = {"speed": "#1f77b4", "direction": "#d62728", "acceleration": "#2ca02c"}

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

for ax, (csv, row_label, col_label) in zip(axes.flat, VARIANTS):
    if not os.path.exists(csv):
        ax.text(0.5, 0.5, f"missing: {os.path.basename(csv)}",
                transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.set_title(f"{row_label}  |  {col_label}", fontsize=10)
        continue
    df = pd.read_csv(csv)
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

    ax.set_title(f"{row_label}  |  {col_label}", fontsize=11)
    ax.set_xlabel("Layer index")
    ax.set_ylim(-0.2, 1.05)
    ax.grid(True, alpha=0.3)

for ax in axes[:, 0]:
    ax.set_ylabel("R² (mean across folds)")
axes[0, 0].legend(loc="lower right", fontsize=8)

fig.suptitle(
    "V-JEPA 2-L · position grouping vs direction grouping (Phase 3)",
    fontsize=13,
)
fig.tight_layout()

out = os.path.join(ROOT, "pez_reproduction_vitl_phase3_loo.png")
fig.savefig(out, dpi=150)
print("Saved", out)
