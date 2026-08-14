"""
plot_convergence.py
Figure 2(b): Masked-PPO validation reward evolution.

Raw per-episode validation rewards under dynamic resource conditions.
No smoothing, no best-checkpoint markers.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


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
            "grid.alpha": 0.32,
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
        alpha=0.32,
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

required_columns = {
    "seed",
    "episode",
    "validation_reward",
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        "ppo_training_paper.csv is missing required columns: "
        f"{sorted(missing_columns)}"
    )

for column in ["seed", "episode", "validation_reward"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

invalid_rows = df[
    ["seed", "episode", "validation_reward"]
].isna().any(axis=1)

if invalid_rows.any():
    print(
        "[plot_convergence] Dropping rows with invalid numeric values:",
        int(invalid_rows.sum()),
    )
    df = df.loc[~invalid_rows].copy()

df["seed"] = df["seed"].astype(int)
df["episode"] = df["episode"].astype(int)

df = (
    df.sort_values(["seed", "episode"])
    .drop_duplicates(
        subset=["seed", "episode"],
        keep="last",
    )
    .reset_index(drop=True)
)

print("[plot_convergence] Loaded:", TRAIN_CSV)
print("[plot_convergence] Shape:", df.shape)
print("[plot_convergence] Columns:", list(df.columns))
print(
    "[plot_convergence] Episodes per seed:",
    df.groupby("seed")["episode"].count().to_dict(),
)

for seed, seed_df in df.groupby("seed"):
    print(
        f"[plot_convergence] Seed {seed}: "
        f"episode {seed_df['episode'].min()}–"
        f"{seed_df['episode'].max()}, "
        f"reward {seed_df['validation_reward'].min():.3f}–"
        f"{seed_df['validation_reward'].max():.3f}"
    )


# ---------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------
apply_paper_style()

SEED_COLORS = {
    1: "#4C78A8",
    2: "#C44E52",
    3: "#55A868",
}

fig, ax = plt.subplots(figsize=(7.2, 4.15))

fig.subplots_adjust(
    left=0.105,
    right=0.975,
    bottom=0.145,
    top=0.84,
)

style_axis(ax)


# ---------------------------------------------------------------------
# Draw each seed
# ---------------------------------------------------------------------
for seed in sorted(df["seed"].unique()):
    seed_df = (
        df.loc[df["seed"] == seed]
        .sort_values("episode")
        .copy()
    )

    episodes = seed_df["episode"].to_numpy()
    rewards = seed_df["validation_reward"].to_numpy()
    color = SEED_COLORS.get(seed)

    ax.plot(
        episodes,
        rewards,
        color=color,
        linewidth=1.05,
        linestyle="-",
        marker="o",
        markersize=2.0,
        markerfacecolor=color,
        markeredgecolor=color,
        markevery=3,
        alpha=0.8,
        zorder=2,
        label=f"Seed {seed}",
    )


# ---------------------------------------------------------------------
# Axis labels and limits
# ---------------------------------------------------------------------
ax.set_title(
    "(b) PPO validation reward evolution",
    loc="left",
    fontsize=10,
    fontweight="bold",
    pad=20,
)

ax.set_xlabel("Episode")
ax.set_ylabel("Validation reward")

episode_min = df["episode"].min()
episode_max = df["episode"].max()

x_padding = max(2, 0.015 * (episode_max - episode_min))

ax.set_xlim(
    episode_min - x_padding,
    episode_max + x_padding,
)

ax.set_ylim(-120, 540)

# Use readable x ticks.
ax.set_xticks(
    list(range(0, int(episode_max) + 1, 20))
)


# ---------------------------------------------------------------------
# Legend (outside axes, top-right)
# ---------------------------------------------------------------------
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.72, 1.08),
    ncol=3,
    frameon=False,
    fontsize=7.5,
    handlelength=1.6,
    columnspacing=1.0,
    borderaxespad=0.0,
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
for fmt in ["png", "pdf"]:
    output_path = OUT_DIR / f"figure2_convergence.{fmt}"

    fig.savefig(
        output_path,
        dpi=600 if fmt == "png" else None,
        format=fmt,
        bbox_inches="tight",
        pad_inches=0.08,
    )

    print(f"[plot_convergence] Saved: {output_path}")

plt.close(fig)

print("[plot_convergence] Done.")
