"""
plot_convergence_mean_std.py
Figure 2(b) -- experimental version: mean + std across 3 seeds.

Layer order:
  1. Three individual seed curves (faded, no legend)
  2. Standard-deviation shade (fill_between)
  3. Mean reward curve (bold, legend only)

Output: outputs/figure2_convergence_mean_std.png + .pdf

Does NOT overwrite the raw-seed version (figure2_convergence.*).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_DIR = Path(r"D:\pythonWork\formal_test\figure2")
OUT_DIR = DATA_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "ppo_training_paper.csv"


# ---------------------------------------------------------------------
# Paper style
# ---------------------------------------------------------------------
def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
                "serif",
            ],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "grid.alpha": 0.2,
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
            "grid.color": "#999999",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle="--",
        linewidth=0.65,
        alpha=0.2,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#444444")
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.5,
        width=0.7,
        pad=3,
    )


# ---------------------------------------------------------------------
# Load and validate data
# ---------------------------------------------------------------------
if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"CSV file not found: {TRAIN_CSV}")

df = pd.read_csv(TRAIN_CSV)

required_columns = {"seed", "episode", "validation_reward"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(
        "ppo_training_paper.csv is missing required columns: "
        f"{sorted(missing_columns)}"
    )

for col in ["seed", "episode", "validation_reward"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

invalid_rows = df[["seed", "episode", "validation_reward"]].isna().any(axis=1)
if invalid_rows.any():
    print(
        "[plot_conv_ms] Dropping rows with invalid numeric values:",
        int(invalid_rows.sum()),
    )
    df = df.loc[~invalid_rows].copy()

df["seed"] = df["seed"].astype(int)
df["episode"] = df["episode"].astype(int)

df = (
    df.sort_values(["seed", "episode"])
    .drop_duplicates(subset=["seed", "episode"], keep="last")
    .reset_index(drop=True)
)

seeds = sorted(df["seed"].unique())

print("[plot_conv_ms] Loaded:", TRAIN_CSV)
print("[plot_conv_ms] Shape:", df.shape)
print("[plot_conv_ms] Seeds:", seeds)
print("[plot_conv_ms] Episodes per seed:", df.groupby("seed")["episode"].count().to_dict())

# Check for missing episodes
episode_max = int(df["episode"].max())
missing_report = {}
for seed in seeds:
    full_set = set(range(1, episode_max + 1))
    actual   = set(df.loc[df["seed"] == seed, "episode"])
    missing  = sorted(full_set - actual)
    missing_report[seed] = len(missing)
    if missing:
        print(f"[plot_conv_ms] Seed {seed}: missing episodes = {missing}")

total_missing = sum(missing_report.values())
print(f"[plot_conv_ms] Total missing episodes: {total_missing}")

# ---------------------------------------------------------------------
# Build mean / std per episode
# ---------------------------------------------------------------------
pivot = df.pivot_table(
    index="episode",
    columns="seed",
    values="validation_reward",
    aggfunc="mean",
)

mean_reward = pivot.mean(axis=1)
std_reward  = pivot.std(axis=1)
episodes_arr = pivot.index.values

print(f"[plot_conv_ms] Episodes in pivot: {len(episodes_arr)} ({episodes_arr[0]}-{episodes_arr[-1]})")
print(f"[plot_conv_ms] Mean reward range: {mean_reward.min():.2f} - {mean_reward.max():.2f}")
print(f"[plot_conv_ms] Std  reward range: {std_reward.min():.2f} - {std_reward.max():.2f}")

# ---------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------
apply_paper_style()

# Seed colors (muted, for faded background curves)
SEED_COLORS = {
    1: "#4C78A8",
    2: "#E15759",
    3: "#59A14F",
}
MEAN_COLOR = "#1F4E79"

fig, ax = plt.subplots(figsize=(7.2, 4.15))
fig.subplots_adjust(
    left=0.105,
    right=0.975,
    bottom=0.145,
    top=0.84,
)
style_axis(ax)

# ---------------------------------------------------------------------
# Layer 1: Individual seed curves (faded, no legend)
# ---------------------------------------------------------------------
for seed in seeds:
    seed_df = df.loc[df["seed"] == seed].sort_values("episode")
    eps = seed_df["episode"].to_numpy()
    rw  = seed_df["validation_reward"].to_numpy()
    c   = SEED_COLORS.get(seed)

    ax.plot(
        eps, rw,
        color=c,
        linewidth=0.8,
        linestyle="-",
        marker="o",
        markersize=1.5,
        markerfacecolor=c,
        markeredgecolor=c,
        markevery=3,
        alpha=0.25,
        zorder=1,
        label="_nolegend_",
    )

# ---------------------------------------------------------------------
# Layer 2: Standard-deviation shade
# ---------------------------------------------------------------------
ax.fill_between(
    episodes_arr,
    mean_reward - std_reward,
    mean_reward + std_reward,
    color=MEAN_COLOR,
    alpha=0.18,
    linewidth=0,
    zorder=2,
    label="_nolegend_",
)

# ---------------------------------------------------------------------
# Layer 3: Mean reward curve
# ---------------------------------------------------------------------
ax.plot(
    episodes_arr,
    mean_reward.values,
    color=MEAN_COLOR,
    linewidth=2.2,
    linestyle="-",
    marker="",
    zorder=3,
    label="Mean reward (3 seeds)",
)

# ---------------------------------------------------------------------
# Axis labels and limits
# ---------------------------------------------------------------------
ax.set_title(
    "(b) Evolution of validation reward during PPO training",
    loc="left",
    fontsize=10,
    fontweight="bold",
    pad=20,
)

ax.set_xlabel("Episode")
ax.set_ylabel("Validation reward")

# Y-axis: auto with 10% margin around mean +/- std envelope
y_lo = (mean_reward - std_reward).min()
y_hi = (mean_reward + std_reward).max()
y_span = y_hi - y_lo
y_pad = max(5, 0.10 * y_span)

ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

# X-axis
x_padding = max(2, 0.015 * (episode_max - 1))
ax.set_xlim(1 - x_padding, episode_max + x_padding)
ax.set_xticks(list(range(0, episode_max + 1, 20)))

# ---------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.72, 1.08),
    ncol=1,
    frameon=False,
    fontsize=7.5,
    handlelength=1.6,
)

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
for fmt in ["png", "pdf"]:
    out = OUT_DIR / f"figure2_convergence_mean_std.{fmt}"
    fig.savefig(
        out,
        dpi=600 if fmt == "png" else None,
        format=fmt,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    print(f"[plot_conv_ms] Saved: {out}")

plt.close(fig)
print("[plot_conv_ms] Done.")
