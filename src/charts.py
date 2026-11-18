"""Static charts for the README and deck."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from features import DATA

OUT = DATA.parent / "outputs" / "charts"; OUT.mkdir(parents=True, exist_ok=True)
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def title(ax, t, sub=None):
    ax.set_title(t, loc="left", fontsize=15, color=INK, fontweight="bold", pad=24 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=11, color=INK2)


# 1. Benchmarks (QWK)
b = pd.read_csv(DATA.parent / "outputs" / "benchmark_results.csv")
b = b.iloc[::-1]
fig, ax = plt.subplots(figsize=(11, 5.6))
colors = [S1 if m.startswith("Deployed") else "#9ec5f4" for m in b.model]
ax.barh(b.model, b.qwk, color=colors, height=0.6)
for y, v in enumerate(b.qwk):
    ax.text(v + 0.005, y, f"{v:.3f}", va="center", color=INK, fontsize=11)
ax.set_xlabel("Quadratic weighted kappa (higher is better), 5-fold CV grouped by rescuer")
ax.grid(axis="y", visible=False); ax.set_xlim(0, 0.5)
title(ax, "AI photo understanding lifts prediction quality", "Same folds for every model; no rescuer appears in both train and test")
fig.tight_layout(); fig.savefig(OUT / "benchmarks.png", dpi=180); plt.close(fig)

# 2. Action effects
e = pd.read_csv(DATA.parent / "outputs" / "action_effects.csv").sort_values("mean_days_saved")
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.barh(e.label, e.mean_days_saved, color=S1, height=0.6)
for y, (v, n) in enumerate(zip(e.mean_days_saved, e.share_eligible)):
    ax.text(v + 0.08, y, f"{v:.1f} days  ({n:.0%} of listings eligible)", va="center", color=INK, fontsize=10.5)
ax.set_xlabel("Average expected days saved per eligible listing")
ax.grid(axis="y", visible=False); ax.set_xlim(0, 8)
title(ax, "Listing fixes move adoption speed; vet status does not", "Out-of-fold model estimates; vaccination and sterilization show no measurable speed-up in this data")
fig.tight_layout(); fig.savefig(OUT / "action_effects.png", dpi=180); plt.close(fig)

# 3. Portfolio budget curve
s = pd.read_csv(DATA.parent / "outputs" / "impact_summary.csv")
fig, ax = plt.subplots(figsize=(11, 5.6))
for col, lab, c in [("opt_saved", "AI optimizer", S1), ("cheapest_first_saved", "Rule: cheapest actions first", S2),
                    ("vet_first_saved", "Rule: vet care first", S3)]:
    ax.plot(s.budget_per_listing, s[col] / 1000, color=c, lw=2, marker="o", ms=6, label=lab)
    ax.text(s.budget_per_listing.iloc[-1] + 4, s[col].iloc[-1] / 1000 + (0.4 if "vet" in lab else 0), lab, color=INK, va="center", fontsize=10.5)
ax.axhline(s.free_wins_saved.iloc[0] / 1000, color=INK2, lw=1, ls="--")
ax.text(60, s.free_wins_saved.iloc[0] / 1000 - 0.6, "Free quick wins alone (RM0)", color=INK2, fontsize=10.5)
ax.set_xlabel("Budget per listing (RM)"); ax.set_ylabel("Animal-days saved (thousands)")
ax.set_xlim(0, 270); ax.set_ylim(10, 23)
ax.legend(loc="upper left", bbox_to_anchor=(0.62, 0.55), frameon=False, fontsize=10)
title(ax, "Same budget, far more days saved", "50 real Malaysian rescuers, 4,730 animals, scored by models that never saw them")
fig.tight_layout(); fig.savefig(OUT / "budget_curve.png", dpi=180); plt.close(fig)

# 4. Photo scores vs days
d = pd.read_csv(DATA / "pets_model_table.csv")
fig, ax = plt.subplots(figsize=(11, 5.6))
for col, lab, c in [("PhotoAppeal", "CLIP photo appeal (AI vision)", S1), ("TechPhotoScore", "Pixel quality score (sharpness, contrast, color)", S2)]:
    g = d.groupby(pd.qcut(d[col], 5, labels=False)).Days.mean()
    ax.plot(np.arange(1, 6), g.values, color=c, lw=2, marker="o", ms=7, label=lab)
ax.set_xticks(range(1, 6), ["1 (worst)", "2", "3", "4", "5 (best)"])
ax.set_xlabel("Main photo score quintile"); ax.set_ylabel("Average days to adoption")
ax.legend(frameon=False, loc="upper center")
title(ax, "Pixel metrics point the wrong way; AI vision gets photos right", "Average days by quintile of each main-photo score, all 14,993 listings")
fig.tight_layout(); fig.savefig(OUT / "photo_scores.png", dpi=180); plt.close(fig)

# 5. Calibration
c = pd.read_csv(DATA.parent / "outputs" / "calibration_deciles.csv")
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([15, 100], [15, 100], color=GRID, lw=1.5)
ax.plot(c.pred, c.actual, color=S1, lw=2, marker="o", ms=7)
ax.set_xlabel("Predicted days (decile average)"); ax.set_ylabel("Actual days (decile average)")
title(ax, "Predicted days are well calibrated", "Out-of-fold, deciles of 1,500 listings")
fig.tight_layout(); fig.savefig(OUT / "calibration.png", dpi=180); plt.close(fig)
print(sorted(p.name for p in OUT.iterdir()))
