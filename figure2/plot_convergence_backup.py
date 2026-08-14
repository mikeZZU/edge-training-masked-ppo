"""
plot_convergence.py -- Figure 2(b): Masked-PPO validation convergence.

Three curves (seed 1, 2, 3) of validation_reward vs episode.
Raw curves plus light rolling-mean auxiliary lines.
Best-checkpoint markers + annotations verified against ppo_summary_paper.csv.

Output: outputs/figure2_convergence.png + outputs/figure2_convergence.pdf
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -- Paths ----------------------------------------------------------
DATA_DIR = Path("D:/pythonWork/formal_test/figure2")
OUT_DIR  = DATA_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Paper style ----------------------------------------------------
def apply_paper_style():
    """Apply consistent paper-ready style."""
    plt.rcParams.update({
        "font.family":          "serif",
        "font.serif":           ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":            9,
        "axes.titlesize":       9,
        "axes.labelsize":       8.5,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.fontsize":      7.5,
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "axes.edgecolor":       "#444444",
        "axes.linewidth":       0.8,
        "grid.alpha":           0.3,
        "grid.linestyle":       "--",
        "grid.color":           "#888888",
        "xtick.color":          "#444444",
        "ytick.color":          "#444444",
        "legend.framealpha":    0.85,
        "legend.edgecolor":     "#999999",
        "legend.fancybox":      False,
        "savefig.dpi":          600,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.03,
    })


def rolling_mean(series: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple centred rolling mean; edges use available data only."""
    smoothed = np.full_like(series, np.nan)
    half = window // 2
    n = len(series)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed[i] = np.mean(series[lo:hi])
    return smoothed


# -- Load data ------------------------------------------------------
df_train = pd.read_csv(DATA_DIR / "ppo_training_paper.csv")
df_summ  = pd.read_csv(DATA_DIR / "ppo_summary_paper.csv")

# Clean numerics
for col in df_train.columns:
    if col not in ("seed", "episode"):
        df_train[col] = pd.to_numeric(df_train[col], errors="coerce")

print("[plot_convergence] ppo_training_paper.csv shape:", df_train.shape)
print("[plot_convergence] ppo_summary_paper.csv shape:", df_summ.shape)
print("[plot_convergence] Seeds:", sorted(df_train["seed"].unique()))
print("[plot_convergence] Episodes per seed:", df_train.groupby("seed")["episode"].count().to_dict())

# -- Find best episode per seed from training data ------------------
print("\n[plot_convergence] Best-checkpoint analysis:")
best_info = {}  # seed -> {episode, reward, ...}
for seed in sorted(df_train["seed"].unique()):
    mask = df_train["seed"] == seed
    df_s = df_train[mask].sort_values("episode")

    # Find episode with max validation_reward
    idx_best = df_s["validation_reward"].idxmax()
    ep_best  = df_s.loc[idx_best, "episode"]
    rw_best  = df_s.loc[idx_best, "validation_reward"]

    # Cross-check with summary CSV
    summ_row = df_summ[df_summ["seed"] == seed]
    if len(summ_row) == 1:
        summ_rw = summ_row["best_validation_reward"].values[0]
        match = "OK" if abs(rw_best - summ_rw) < 1e-6 else "MISMATCH"
    else:
        summ_rw = float("nan")
        match = "NO_SUMMARY"

    print(f"  Seed {seed}: Best Ep = {int(ep_best)}, "
          f"Reward = {rw_best:.6f}, "
          f"Summary = {summ_rw:.6f}, "
          f"Match: {match}")

    best_info[seed] = {
        "episode": int(ep_best),
        "reward":  rw_best,
        "match":   match,
    }

# -- Build figure ---------------------------------------------------
apply_paper_style()

# Colour palette for 3 seeds
COLORS = {
    1: "#2166AC",  # blue
    2: "#B2182B",  # red
    3: "#4DAF4A",  # green
}
ROLL_WINDOW = 5

fig, ax = plt.subplots(figsize=(7.2, 3.8))
fig.subplots_adjust(left=0.09, right=0.95, top=0.93, bottom=0.13)

for seed in sorted(df_train["seed"].unique()):
    mask = df_train["seed"] == seed
    df_s = df_train[mask].sort_values("episode")
    eps  = df_s["episode"].values
    rw   = df_s["validation_reward"].values
    c    = COLORS[seed]

    # Raw validation curve
    ax.plot(eps, rw, linewidth=1.5, color=c, alpha=0.65,
            label=f"Seed {seed} (raw)")

    # Rolling mean (auxiliary, lighter + dashed)
    rw_smooth = rolling_mean(rw, window=ROLL_WINDOW)
    ax.plot(eps, rw_smooth, linewidth=1.2, color=c, linestyle="--", alpha=0.85,
            label=f"Seed {seed} (MA-{ROLL_WINDOW})")

    # Best checkpoint marker
    bi  = best_info[seed]
    ax.plot(bi["episode"], bi["reward"], marker="*", color=c,
            markersize=10, markeredgewidth=0.5, markeredgecolor="#333333",
            zorder=10)

    # Vertical dashed line + annotation
    ylim = ax.get_ylim()
    # Place label slightly above the best point
    y_text = bi["reward"] + (ylim[1] - ylim[0]) * 0.06
    ax.axvline(x=bi["episode"], color=c, linestyle=":", linewidth=0.8,
               alpha=0.6)
    ax.annotate(f"Best: Ep. {bi['episode']}",
                xy=(bi["episode"], bi["reward"]),
                xytext=(bi["episode"] + 2, y_text),
                fontsize=7.5,
                color=c,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=c, lw=0.8),
                ha="left", va="bottom")

# Labels and grid
ax.set_xlabel("Episode")
ax.set_ylabel("Validation reward")
ax.grid(True, linestyle="--", alpha=0.3, color="#888888")
ax.legend(loc="lower right", frameon=True, ncol=2)

# Sub-label
ax.text(-0.04, 1.01, "(b) PPO validation convergence",
        transform=ax.transAxes, fontsize=9, fontweight="bold",
        va="bottom", ha="left")

# -- Save -----------------------------------------------------------
for fmt in ["png", "pdf"]:
    out = OUT_DIR / f"figure2_convergence.{fmt}"
    fig.savefig(out, dpi=600 if fmt == "png" else None, format=fmt)
    print(f"[plot_convergence] Saved: {out}")

plt.close(fig)
print("[plot_convergence] Done.")
