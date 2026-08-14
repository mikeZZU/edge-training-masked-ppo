"""
plot_figure2.py -- Combined Figure 2 for paper.

Single figure with two sub-figures:
  (a) Split-point profile (two stacked panels: latency + feature/memory)
  (b) PPO validation convergence (3 seeds, raw + rolling mean, best markers)

Output: outputs/figure2_combined.png + outputs/figure2_combined.pdf
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
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


def bytes_to_mb(b: float) -> float:
    """Convert bytes to megabytes using 1024^2."""
    return b / (1024 * 1024)


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


# -- Colour palette --------------------------------------------------
C_DEVICE  = "#2166AC"   # blue
C_SERVER  = "#B2182B"   # red
C_FEATURE = "#4DAF4A"   # green
C_MEMORY  = "#FF7F00"   # orange

SEED_COLORS = {
    1: "#2166AC",  # blue
    2: "#B2182B",  # red
    3: "#4DAF4A",  # green
}
ROLL_WINDOW = 5

# -- Load data -------------------------------------------------------
df_prof  = pd.read_csv(DATA_DIR / "profile_paper.csv")
df_train = pd.read_csv(DATA_DIR / "ppo_training_paper.csv")
df_summ  = pd.read_csv(DATA_DIR / "ppo_summary_paper.csv")

# Sort & clean
df_prof = df_prof.sort_values("action").reset_index(drop=True)
for col in df_prof.columns:
    if col != "action":
        df_prof[col] = pd.to_numeric(df_prof[col], errors="coerce")
df_prof["feature_mb"] = df_prof["feature_bytes"].apply(bytes_to_mb)

for col in df_train.columns:
    if col not in ("seed", "episode"):
        df_train[col] = pd.to_numeric(df_train[col], errors="coerce")

# -- Compute best episodes per seed ----------------------------------
print("[plot_figure2] Best-checkpoint analysis (from training data):")
best_info = {}
for seed in sorted(df_train["seed"].unique()):
    df_s = df_train[df_train["seed"] == seed]
    idx_best = df_s["validation_reward"].idxmax()
    ep_best  = int(df_s.loc[idx_best, "episode"])
    rw_best  = df_s.loc[idx_best, "validation_reward"]

    # Verify against summary
    summ_row = df_summ[df_summ["seed"] == seed]
    summ_rw = summ_row["best_validation_reward"].values[0]
    match = "OK" if abs(rw_best - summ_rw) < 1e-6 else "MISMATCH"
    final_rw = df_s[df_s["episode"] == df_s["episode"].max()]["validation_reward"].values[0]
    print(f"  Seed {seed}: Best Ep = {ep_best}, Reward = {rw_best:.6f}, "
          f"Summary = {summ_rw:.6f}, Match: {match}, "
          f"Final(ep{df_s['episode'].max()}) = {final_rw:.2f}")

    best_info[seed] = {"episode": ep_best, "reward": rw_best, "match": match}

# Print profile summary
total_lat = df_prof["device_ms"] + df_prof["server_ms"]
print(f"[plot_figure2] Profile total latency: {total_lat.mean():.2f} ms (constant across actions)")

# -- Build combined figure -------------------------------------------
apply_paper_style()

fig = plt.figure(figsize=(7.2, 8.8))

# Outer GridSpec: 2 rows (a) profile, (b) convergence
outer_gs = GridSpec(
    nrows=2, ncols=1,
    height_ratios=[1.0, 0.78],
    hspace=0.38,
    left=0.10, right=0.92, top=0.965, bottom=0.055,
)

# ====================================================================
# (a) Split-point profile  (nested GridSpec for two panels)
# ====================================================================
gs_a = outer_gs[0].subgridspec(
    nrows=2, ncols=1,
    height_ratios=[1.0, 0.65],
    hspace=0.08,
)

ax_a_top = fig.add_subplot(gs_a[0])
ax_a_bot = fig.add_subplot(gs_a[1], sharex=ax_a_top)

actions = df_prof["action"].values

# -- (a) Top: Latency --
ax_a_top.plot(actions, df_prof["device_ms"], "o-",
              color=C_DEVICE, linewidth=1.5, markersize=4,
              markerfacecolor="white", markeredgewidth=0.8,
              label="Device latency")
ax_a_top.plot(actions, df_prof["server_ms"], "s-",
              color=C_SERVER, linewidth=1.5, markersize=4,
              markerfacecolor="white", markeredgewidth=0.8,
              label="Server latency")
ax_a_top.set_ylabel("Latency (ms)")
ax_a_top.grid(True, linestyle="--", alpha=0.3, color="#888888")
ax_a_top.legend(loc="upper right", frameon=True)
ax_a_top.tick_params(labelbottom=False)

# -- (a) Bottom: Feature size + Peak memory --
ax_a_bot.plot(actions, df_prof["feature_mb"], "D-",
              color=C_FEATURE, linewidth=1.5, markersize=4,
              markerfacecolor="white", markeredgewidth=0.8,
              label="Feature size")
ax_a_bot.set_ylabel("Feature size (MB)")

ax_a_bot_right = ax_a_bot.twinx()
ax_a_bot_right.plot(actions, df_prof["memory_mb"], "^--",
                    color=C_MEMORY, linewidth=1.5, markersize=4,
                    markerfacecolor="white", markeredgewidth=0.8,
                    label="Peak memory")
ax_a_bot_right.set_ylabel("Peak memory (MB)")

# Combine bottom legends
lines1, labels1 = ax_a_bot.get_legend_handles_labels()
lines2, labels2 = ax_a_bot_right.get_legend_handles_labels()
ax_a_bot.legend(lines1 + lines2, labels1 + labels2,
                loc="upper left", frameon=True)

ax_a_bot.set_xlabel("Action / Split point")
ax_a_bot.grid(True, linestyle="--", alpha=0.3, color="#888888")
ax_a_bot.set_xticks(actions)
ax_a_bot.set_xticklabels([str(a) for a in actions])

# (a) sub-label at top-left
ax_a_top.text(-0.083, 0.97, "(a) Split-point profile",
              transform=ax_a_top.transAxes, fontsize=9,
              fontweight="bold", va="top", ha="left")

# ====================================================================
# (b) PPO validation convergence
# ====================================================================
ax_b = fig.add_subplot(outer_gs[1])

for seed in sorted(df_train["seed"].unique()):
    mask = df_train["seed"] == seed
    df_s = df_train[mask].sort_values("episode")
    eps  = df_s["episode"].values
    rw   = df_s["validation_reward"].values
    c    = SEED_COLORS[seed]

    # Raw validation curve (primary)
    ax_b.plot(eps, rw, linewidth=1.5, color=c, alpha=0.60,
              label=f"Seed {seed} (raw)")

    # Rolling mean (auxiliary, dashed, lighter)
    rw_smooth = rolling_mean(rw, window=ROLL_WINDOW)
    ax_b.plot(eps, rw_smooth, linewidth=1.2, color=c,
              linestyle="--", alpha=0.85,
              label=f"Seed {seed} (MA-{ROLL_WINDOW})")

    # Best checkpoint marker
    bi = best_info[seed]
    ax_b.plot(bi["episode"], bi["reward"], marker="*", color=c,
              markersize=11, markeredgewidth=0.5,
              markeredgecolor="#333333", zorder=10)

    # Vertical dashed line through best point
    ax_b.axvline(x=bi["episode"], color=c, linestyle=":",
                 linewidth=0.8, alpha=0.55)

    # Annotation -- handle early-episode best (seed 3 at ep 2)
    ylim = ax_b.get_ylim()
    y_span = ylim[1] - ylim[0]

    if bi["episode"] <= 5:
        # Best very early: place label below and to the right
        txt_x = bi["episode"] + 6
        txt_y = bi["reward"] - y_span * 0.18
        ha = "left"
        va = "top"
    else:
        # Normal case: label above and slightly right
        txt_x = bi["episode"] + 2
        txt_y = bi["reward"] + y_span * 0.07
        ha = "left"
        va = "bottom"

    ax_b.annotate(
        f"Best: Ep. {bi['episode']}",
        xy=(bi["episode"], bi["reward"]),
        xytext=(txt_x, txt_y),
        fontsize=7.5, color=c, fontweight="bold",
        arrowprops=dict(
            arrowstyle="->", color=c, lw=0.8,
            connectionstyle="arc3,rad=0.2",
        ),
        ha=ha, va=va,
    )

ax_b.set_xlabel("Episode")
ax_b.set_ylabel("Validation reward")
ax_b.grid(True, linestyle="--", alpha=0.3, color="#888888")

# Legend: two columns, positioned to minimize overlap with data
ax_b.legend(loc="upper left", frameon=True, ncol=2,
            columnspacing=0.6, handlelength=1.5, borderaxespad=0.4)

# (b) sub-label at top-left
ax_b.text(-0.06, 1.01, "(b) PPO validation convergence",
          transform=ax_b.transAxes, fontsize=9,
          fontweight="bold", va="bottom", ha="left")

# -- Final layout ---------------------------------------------------
# NOTE: Manual GridSpec left/right/top/bottom/hspace provides layout.
# tight_layout skipped here because twin axes (ax_a_bot_right) raise
# a warning and the manual params already produce correct output.

# -- Save -----------------------------------------------------------
for fmt in ["png", "pdf"]:
    out = OUT_DIR / f"figure2_combined.{fmt}"
    if fmt == "png":
        fig.savefig(out, dpi=600, format=fmt)
    else:
        fig.savefig(out, format=fmt)
    print(f"[plot_figure2] Saved: {out}")

plt.close(fig)
print("[plot_figure2] Done.")
