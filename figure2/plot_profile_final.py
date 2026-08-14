"""
plot_profile.py -- Figure 2(a): ResNet50 split-point profile.

Two vertically stacked panels sharing the x-axis (action / split point):
  Top:    Device latency (ms) + Server latency (ms)
  Bottom: Feature size (MB) + Peak device memory (MB)

Output: outputs/figure2_profile.png + outputs/figure2_profile.pdf
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
        "axes.titlesize":       10,
        "axes.labelsize":       9,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.fontsize":      7.5,
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "axes.edgecolor":       "#444444",
        "axes.linewidth":       0.8,
        "grid.alpha":           0.32,
        "grid.linestyle":       "--",
        "grid.linewidth":       0.65,
        "grid.color":           "#999999",
        "xtick.color":          "#333333",
        "ytick.color":          "#333333",
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "savefig.dpi":          600,
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
        "axes.unicode_minus":   False,
    })


def bytes_to_mb(b: float) -> float:
    """Convert bytes to megabytes using 1024^2."""
    return b / (1024 * 1024)


# -- Load data ------------------------------------------------------
df = pd.read_csv(DATA_DIR / "profile_paper.csv")
df = df.sort_values("action").reset_index(drop=True)
print("[plot_profile] profile_paper.csv loaded, shape:", df.shape)
print("[plot_profile] Columns:", list(df.columns))

# Convert columns to numeric (defensive)
for col in df.columns:
    if col != "action":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Feature bytes to MB (1024-based)
df["feature_mb"] = df["feature_bytes"].apply(bytes_to_mb)
print(f"[plot_profile] feature_bytes range: {df['feature_bytes'].min()} -- {df['feature_bytes'].max()} B")
print(f"[plot_profile] feature_mb    range: {df['feature_mb'].min():.3f} -- {df['feature_mb'].max():.3f} MB")
print(f"[plot_profile] memory_mb     range: {df['memory_mb'].min():.2f} -- {df['memory_mb'].max():.2f} MB")
print(f"[plot_profile] device_ms     range: {df['device_ms'].min():.2f} -- {df['device_ms'].max():.2f} ms")
print(f"[plot_profile] server_ms     range: {df['server_ms'].min():.2f} -- {df['server_ms'].max():.2f} ms")
print("[plot_profile] Unit judgment: device_ms=milliseconds, server_ms=milliseconds, "
      "feature_bytes=bytes->MB(/1024^2), memory_mb=megabytes")

# -- Palette --------------------------------------------------------
C_DEVICE  = "#3B6EA8"   # muted blue
C_SERVER  = "#B44A4A"   # muted red
C_FEATURE = "#55A868"   # muted green
C_MEMORY  = "#E17C05"   # muted orange

# -- Build figure ---------------------------------------------------
apply_paper_style()

fig = plt.figure(figsize=(7.2, 5.3))

# GridSpec with extra top/bottom margin for title clearance
gs = fig.add_gridspec(
    nrows=2, ncols=1,
    height_ratios=[1.0, 0.68],
    hspace=0.18,
    left=0.105, right=0.885, top=0.86, bottom=0.105,
)

ax_top = fig.add_subplot(gs[0, 0])
ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

actions = df["action"].values

# -- Top panel: Latency ---------------------------------------------
ax_top.plot(actions, df["device_ms"],
            color=C_DEVICE, linewidth=1.5, linestyle="-",
            marker="o", markersize=3.2,
            markerfacecolor=C_DEVICE, markeredgecolor=C_DEVICE,
            label="Device latency")
ax_top.plot(actions, df["server_ms"],
            color=C_SERVER, linewidth=1.5, linestyle="-",
            marker="o", markersize=3.2,
            markerfacecolor=C_SERVER, markeredgecolor=C_SERVER,
            label="Server latency")

ax_top.set_ylabel("Latency (ms)")
ax_top.grid(True, linestyle="--", alpha=0.32, linewidth=0.65, color="#999999")
ax_top.tick_params(labelbottom=False)

# Y-axis breathing room for latency panel
latency_values = pd.concat(
    [df["device_ms"], df["server_ms"]],
    ignore_index=True,
)
latency_min = latency_values.min()
latency_max = latency_values.max()
latency_pad = max(2.0, 0.06 * (latency_max - latency_min))
ax_top.set_ylim(
    max(0, latency_min - latency_pad),
    latency_max + latency_pad,
)

# X-axis padding (set once; shared)
ax_top.set_xlim(actions.min() - 0.6, actions.max() + 0.6)

# Title replaces old internal text
ax_top.set_title(
    "(a) Split-point profile",
    loc="left",
    fontsize=10,
    fontweight="bold",
    pad=18,
)

# Legend moved to figure-level (outside axes) -- see below

# Diagnostic
total_lat = df["device_ms"] + df["server_ms"]
print(f"[plot_profile] Total latency (device+server) ~ {total_lat.mean():.2f} ms "
      f"(min={total_lat.min():.2f}, max={total_lat.max():.2f})")

# -- Bottom panel: Feature size + Peak memory -----------------------
ax_bot.plot(actions, df["feature_mb"],
            color=C_FEATURE, linewidth=1.5, linestyle="-",
            marker="o", markersize=3.2,
            markerfacecolor=C_FEATURE, markeredgecolor=C_FEATURE,
            label="Feature size")
ax_bot.set_ylabel("Feature size (MB)")

# Memory on right y-axis (dashed to distinguish right-axis metric)
ax_bot_right = ax_bot.twinx()
ax_bot_right.plot(actions, df["memory_mb"],
                  color=C_MEMORY, linewidth=1.5, linestyle="--",
                  marker="o", markersize=3.2,
                  markerfacecolor=C_MEMORY, markeredgecolor=C_MEMORY,
                  label="Peak memory")
ax_bot_right.set_ylabel("Peak memory (MB)")

# -- Unified figure-level legend (manual handles, all circles) ------
legend_handles = [
    Line2D([0], [0], color=C_DEVICE,  marker="o", markersize=4,
           markerfacecolor=C_DEVICE,  markeredgecolor=C_DEVICE,
           linewidth=1.5, linestyle="-",  label="Device latency"),
    Line2D([0], [0], color=C_SERVER,  marker="o", markersize=4,
           markerfacecolor=C_SERVER,  markeredgecolor=C_SERVER,
           linewidth=1.5, linestyle="-",  label="Server latency"),
    Line2D([0], [0], color=C_FEATURE, marker="o", markersize=4,
           markerfacecolor=C_FEATURE, markeredgecolor=C_FEATURE,
           linewidth=1.5, linestyle="-",  label="Feature size"),
    Line2D([0], [0], color=C_MEMORY,  marker="o", markersize=4,
           markerfacecolor=C_MEMORY,  markeredgecolor=C_MEMORY,
           linewidth=1.5, linestyle="--", label="Peak memory"),
]

fig.legend(
    handles=legend_handles,
    loc="upper right",
    bbox_to_anchor=(0.96, 0.97),
    ncol=2,
    frameon=False,
    fontsize=7.0,
    handlelength=1.8,
    columnspacing=1.0,
)

ax_bot.set_xlabel("Action / Split point")
ax_bot.grid(True, linestyle="--", alpha=0.32, linewidth=0.65, color="#999999")

# x-axis ticks
ax_bot.set_xticks(actions)
ax_bot.set_xticklabels([str(a) for a in actions])

# -- Save -----------------------------------------------------------
for fmt in ["png", "pdf"]:
    out = OUT_DIR / f"figure2_profile.{fmt}"
    fig.savefig(
        out,
        dpi=600 if fmt == "png" else None,
        format=fmt,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    print(f"[plot_profile] Saved: {out}")

plt.close(fig)
print("[plot_profile] Done.")
