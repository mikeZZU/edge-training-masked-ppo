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
import matplotlib.ticker as mticker
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
# Subdued, distinguishable colors for paper
C_DEVICE  = "#2166AC"   # blue
C_SERVER  = "#B2182B"   # red
C_FEATURE = "#4DAF4A"   # green
C_MEMORY  = "#FF7F00"   # orange

# -- Build figure ---------------------------------------------------
apply_paper_style()

fig = plt.figure(figsize=(7.2, 5.0))

# Use GridSpec: bottom panel ~40% height of top
gs = fig.add_gridspec(
    nrows=2, ncols=1,
    height_ratios=[1.0, 0.65],
    hspace=0.08,
    left=0.10, right=0.89, top=0.93, bottom=0.11,
)

ax_top = fig.add_subplot(gs[0, 0])
ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

actions = df["action"].values

# -- Top panel: Latency ---------------------------------------------
ax_top.plot(actions, df["device_ms"], "o-",
            color=C_DEVICE, linewidth=1.5, markersize=4,
            markerfacecolor="white", markeredgewidth=0.8,
            label="Device latency")
ax_top.plot(actions, df["server_ms"], "s-",
            color=C_SERVER, linewidth=1.5, markersize=4,
            markerfacecolor="white", markeredgewidth=0.8,
            label="Server latency")

ax_top.set_ylabel("Latency (ms)")
ax_top.grid(True, linestyle="--", alpha=0.3, color="#888888")
ax_top.legend(loc="upper right", frameon=True)
ax_top.tick_params(labelbottom=False)  # hide x-tick labels on top

# Annotate total
total_lat = df["device_ms"] + df["server_ms"]
print(f"[plot_profile] Total latency (device+server) ~ {total_lat.mean():.2f} ms "
      f"(min={total_lat.min():.2f}, max={total_lat.max():.2f})")

# -- Bottom panel: Feature size + Peak memory -----------------------
ax_bot.plot(actions, df["feature_mb"], "D-",
            color=C_FEATURE, linewidth=1.5, markersize=4,
            markerfacecolor="white", markeredgewidth=0.8,
            label="Feature size")
ax_bot.set_ylabel("Feature size (MB)")

# Memory on right y-axis
ax_bot_right = ax_bot.twinx()
ax_bot_right.plot(actions, df["memory_mb"], "^--",
                  color=C_MEMORY, linewidth=1.5, markersize=4,
                  markerfacecolor="white", markeredgewidth=0.8,
                  label="Peak memory")
ax_bot_right.set_ylabel("Peak memory (MB)")

# Combine legends from both axes in the bottom panel
lines1, labels1 = ax_bot.get_legend_handles_labels()
lines2, labels2 = ax_bot_right.get_legend_handles_labels()
ax_bot.legend(lines1 + lines2, labels1 + labels2,
              loc="upper left", frameon=True)

ax_bot.set_xlabel("Action / Split point")
ax_bot.grid(True, linestyle="--", alpha=0.3, color="#888888")

# x-axis ticks
ax_bot.set_xticks(actions)
ax_bot.set_xticklabels([str(a) for a in actions])

# Sub-label
ax_top.text(-0.06, 0.96, "(a) Split-point profile", transform=ax_top.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")

# -- Save -----------------------------------------------------------
for fmt in ["png", "pdf"]:
    out = OUT_DIR / f"figure2_profile.{fmt}"
    fig.savefig(out, dpi=600 if fmt == "png" else None, format=fmt)
    print(f"[plot_profile] Saved: {out}")

plt.close(fig)
print("[plot_profile] Done.")
