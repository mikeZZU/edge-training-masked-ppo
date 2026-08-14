from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch,
)
from matplotlib.path import Path as MplPath


# ============================================================
# 1. Workspace and output files
# ============================================================

WORKDIR = Path(r"D:\pythonWork\formal_test\figure1")
WORKDIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = WORKDIR / "masked_ppo_framework.png"
PDF_PATH = WORKDIR / "masked_ppo_framework.pdf"

# Delete only old outputs.
# Do not delete any other files in the workspace.
for output_file in (
    PNG_PATH,
    PDF_PATH,
):
    if output_file.exists():
        output_file.unlink()


# ============================================================
# 2. Matplotlib configuration
# ============================================================

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "DejaVu Sans",
    ],
    "font.size": 8.5,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,

    # Keep vector text editable in PDF.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 3. Color palette
# ============================================================

COLORS = {
    "ink": "#1F2937",
    "muted": "#5B6572",
    "line": "#2F3B4A",

    "state_border": "#4E79A7",
    "state_fill": "#EEF5FB",

    "mask_border": "#D99A17",
    "mask_fill": "#FFF7DF",

    "ppo_border": "#8E6BBE",
    "ppo_fill": "#F5F0FA",

    "decision_border": "#4C9A73",
    "decision_fill": "#EDF8F2",

    "training_border": "#64748B",
    "training_fill": "#F5F7FA",

    "feedback_border": "#7C8796",
    "feedback_fill": "#F2F4F6",

    "communication": "#2F6FB3",
    "white": "#FFFFFF",
}


# ============================================================
# 4. Canvas
# ============================================================

fig, ax = plt.subplots(
    figsize=(15.2, 4.9),
    facecolor="white",
)

ax.set_xlim(0, 19.2)
ax.set_ylim(0, 6.0)
ax.axis("off")


# ============================================================
# 5. Helper functions
# ============================================================

def rounded_box(
    x,
    y,
    width,
    height,
    face_color,
    edge_color,
    linewidth=1.15,
    linestyle="-",
    radius=0.06,
    zorder=2,
):
    """
    Draw one rounded rectangular box.
    """

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=(
            f"round,pad=0.02,"
            f"rounding_size={radius}"
        ),
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )

    ax.add_patch(patch)
    return patch


def draw_card(
    x,
    y,
    width,
    height,
    title,
    face_color,
    edge_color,
    title_fontsize=9.3,
    header_height=0.46,
):
    """
    Draw a main module with a title header.
    """

    rounded_box(
        x=x,
        y=y,
        width=width,
        height=height,
        face_color=face_color,
        edge_color=edge_color,
        linewidth=1.15,
        radius=0.06,
        zorder=2,
    )

    separator_y = (
        y
        + height
        - header_height
    )

    ax.plot(
        [x, x + width],
        [separator_y, separator_y],
        color=edge_color,
        linewidth=0.90,
        zorder=3,
    )

    ax.text(
        x + width / 2,
        y + height - 0.15,
        title,
        ha="center",
        va="top",
        fontsize=title_fontsize,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=4,
    )

    return separator_y


def straight_arrow(
    x_start,
    y_start,
    x_end,
    y_end,
    color=None,
    linewidth=1.15,
    mutation_scale=10,
):
    """
    Draw a straight arrow.
    """

    if color is None:
        color = COLORS["line"]

    arrow_patch = FancyArrowPatch(
        (x_start, y_start),
        (x_end, y_end),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=7,
    )

    ax.add_patch(arrow_patch)


def orthogonal_arrow(
    points,
    color=None,
    linewidth=1.15,
    mutation_scale=10,
):
    """
    Draw a right-angle polyline arrow.
    No curves are used.
    """

    if color is None:
        color = COLORS["line"]

    vertices = [points[0]]
    path_codes = [MplPath.MOVETO]

    for point in points[1:]:
        vertices.append(point)
        path_codes.append(MplPath.LINETO)

    path = MplPath(
        vertices,
        path_codes,
    )

    arrow_patch = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        fill=False,
        zorder=6,
    )

    ax.add_patch(arrow_patch)


# ============================================================
# 6. Fixed module geometry
# ============================================================

TOP_Y = 3.25
TOP_HEIGHT = 2.10

STATE_BOX = (
    0.35,
    TOP_Y,
    2.35,
    TOP_HEIGHT,
)

STATE_CONSTRUCTION_BOX = (
    2.95,
    TOP_Y,
    2.35,
    TOP_HEIGHT,
)

MASK_BOX = (
    5.55,
    TOP_Y,
    2.35,
    TOP_HEIGHT,
)

PPO_BOX = (
    8.15,
    TOP_Y,
    2.35,
    TOP_HEIGHT,
)

SPLIT_BOX = (
    10.75,
    TOP_Y,
    2.15,
    TOP_HEIGHT,
)

TRAINING_BOX = (
    13.15,
    3.05,
    5.65,
    2.50,
)

FEEDBACK_BOX = (
    4.40,
    0.55,
    10.15,
    1.18,
)


# ============================================================
# 7. Dynamic Resource State
# ============================================================

x, y, width, height = STATE_BOX

state_separator_y = draw_card(
    x=x,
    y=y,
    width=width,
    height=height,
    title="Dynamic Resource State",
    face_color=COLORS["state_fill"],
    edge_color=COLORS["state_border"],
)

state_rows = [
    (
        "Bandwidth",
        r"$B_t$",
    ),
    (
        "Available memory",
        r"$M_{\mathrm{avail},t}$",
    ),
    (
        "Compute factor",
        r"$\alpha_t$",
    ),
    (
        "Previous action",
        r"$a_{t-1}$",
    ),
]

state_row_y = (
    state_separator_y
    - 0.29
)

for label, symbol in state_rows:
    ax.text(
        x + 0.16,
        state_row_y,
        label,
        ha="left",
        va="center",
        fontsize=7.7,
        color=COLORS["ink"],
        zorder=4,
    )

    ax.text(
        x + width - 0.15,
        state_row_y,
        symbol,
        ha="right",
        va="center",
        fontsize=8.9,
        color=COLORS["ink"],
        zorder=4,
    )

    state_row_y -= 0.38


# ============================================================
# 8. State Construction
# ============================================================

x, y, width, height = STATE_CONSTRUCTION_BOX

state_construction_separator_y = draw_card(
    x=x,
    y=y,
    width=width,
    height=height,
    title="State Construction",
    face_color=COLORS["state_fill"],
    edge_color=COLORS["state_border"],
)

ax.text(
    x + width / 2,
    state_construction_separator_y - 0.38,
    "Normalize resource features",
    ha="center",
    va="center",
    fontsize=7.7,
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    x + width / 2,
    state_construction_separator_y - 0.75,
    (
        r"$\bar{B}_t,\ "
        r"\bar{M}_{\mathrm{avail},t},\ "
        r"\bar{\alpha}_t$"
    ),
    ha="center",
    va="center",
    fontsize=9.1,
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    x + width / 2,
    state_construction_separator_y - 1.10,
    r"One-hot encode $a_{t-1}$",
    ha="center",
    va="center",
    fontsize=7.6,
    color=COLORS["ink"],
    zorder=4,
)

rounded_box(
    x=x + 0.28,
    y=y + 0.12,
    width=width - 0.56,
    height=0.40,
    face_color=COLORS["white"],
    edge_color=COLORS["state_border"],
    linewidth=0.85,
    radius=0.04,
    zorder=3,
)

ax.text(
    x + width / 2,
    y + 0.32,
    r"$s_t\in\mathbb{R}^{21}$",
    ha="center",
    va="center",
    fontsize=9.0,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)


# ============================================================
# 9. Memory Feasibility Mask
# ============================================================

x, y, width, height = MASK_BOX

mask_separator_y = draw_card(
    x=x,
    y=y,
    width=width,
    height=height,
    title="Memory Feasibility Mask",
    face_color=COLORS["mask_fill"],
    edge_color=COLORS["mask_border"],
    title_fontsize=8.9,
)

ax.text(
    x + width / 2,
    mask_separator_y - 0.42,
    r"$M_{\mathrm{peak}}(a)"
    r"\leq M_{\mathrm{avail},t}$",
    ha="center",
    va="center",
    fontsize=9.2,
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    x + width / 2,
    mask_separator_y - 0.85,
    "Remove infeasible split points",
    ha="center",
    va="center",
    fontsize=7.7,
    color=COLORS["ink"],
    zorder=4,
)

rounded_box(
    x=x + 0.32,
    y=y + 0.28,
    width=width - 0.64,
    height=0.45,
    face_color=COLORS["white"],
    edge_color=COLORS["mask_border"],
    linewidth=0.85,
    radius=0.04,
    zorder=3,
)

ax.text(
    x + width / 2,
    y + 0.505,
    r"Feasible set $\mathcal{A}_t^f$",
    ha="center",
    va="center",
    fontsize=8.3,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    x + width / 2,
    y + 0.13,
    "18 candidate actions",
    ha="center",
    va="center",
    fontsize=7.0,
    color=COLORS["muted"],
    zorder=4,
)


# ============================================================
# 10. Masked-PPO Agent
# ============================================================

x, y, width, height = PPO_BOX

ppo_separator_y = draw_card(
    x=x,
    y=y,
    width=width,
    height=height,
    title="Masked-PPO Agent",
    face_color=COLORS["ppo_fill"],
    edge_color=COLORS["ppo_border"],
)

ax.text(
    x + width / 2,
    ppo_separator_y - 0.38,
    "Lightweight actor\u2013critic MLP",
    ha="center",
    va="center",
    fontsize=7.6,
    color=COLORS["ink"],
    zorder=4,
)

rounded_box(
    x=x + 0.27,
    y=ppo_separator_y - 1.05,
    width=width - 0.54,
    height=0.56,
    face_color=COLORS["white"],
    edge_color=COLORS["ppo_border"],
    linewidth=0.85,
    radius=0.04,
    zorder=3,
)

ax.text(
    x + width / 2,
    ppo_separator_y - 0.77,
    "21  \u2192  32  \u2192  32  \u2192  18",
    ha="center",
    va="center",
    fontsize=8.9,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    x + width / 2,
    y + 0.34,
    r"Masked policy on $\mathcal{A}_t^f$",
    ha="center",
    va="center",
    fontsize=7.5,
    color=COLORS["muted"],
    zorder=4,
)


# ============================================================
# 11. Split-Point Selection
# ============================================================

x, y, width, height = SPLIT_BOX

split_separator_y = draw_card(
    x=x,
    y=y,
    width=width,
    height=height,
    title="Split-Point Selection",
    face_color=COLORS["decision_fill"],
    edge_color=COLORS["decision_border"],
    title_fontsize=8.8,
)

ax.text(
    x + width / 2,
    split_separator_y - 0.40,
    "Choose a feasible action",
    ha="center",
    va="center",
    fontsize=7.6,
    color=COLORS["ink"],
    zorder=4,
)

rounded_box(
    x=x + 0.24,
    y=split_separator_y - 1.17,
    width=width - 0.48,
    height=0.66,
    face_color=COLORS["white"],
    edge_color=COLORS["decision_border"],
    linewidth=0.85,
    radius=0.04,
    zorder=3,
)

ax.text(
    x + width / 2,
    split_separator_y - 0.84,
    r"$a_t\in\{0,\ldots,17\}$",
    ha="center",
    va="center",
    fontsize=9.0,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    x + width / 2,
    y + 0.31,
    "Selected split point",
    ha="center",
    va="center",
    fontsize=7.1,
    color=COLORS["muted"],
    zorder=4,
)


# ============================================================
# 12. Edge\u2013Cloud Split Training
# ============================================================

x, y, width, height = TRAINING_BOX

rounded_box(
    x=x,
    y=y,
    width=width,
    height=height,
    face_color=COLORS["training_fill"],
    edge_color=COLORS["training_border"],
    linewidth=1.20,
    linestyle=(0, (4, 2)),
    radius=0.06,
    zorder=2,
)

ax.text(
    x + width / 2,
    y + height - 0.16,
    "Edge\u2013Cloud Split Training",
    ha="center",
    va="top",
    fontsize=9.6,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)

ax.plot(
    [x, x + width],
    [
        y + height - 0.50,
        y + height - 0.50,
    ],
    color=COLORS["training_border"],
    linewidth=0.90,
    zorder=3,
)

sub_box_y = y + 0.42
sub_box_height = 1.50
sub_box_width = 1.55

edge_box_x = x + 0.34

cloud_box_x = (
    x
    + width
    - 0.34
    - sub_box_width
)

rounded_box(
    x=edge_box_x,
    y=sub_box_y,
    width=sub_box_width,
    height=sub_box_height,
    face_color=COLORS["white"],
    edge_color=COLORS["training_border"],
    linewidth=0.90,
    radius=0.04,
    zorder=3,
)

rounded_box(
    x=cloud_box_x,
    y=sub_box_y,
    width=sub_box_width,
    height=sub_box_height,
    face_color=COLORS["white"],
    edge_color=COLORS["training_border"],
    linewidth=0.90,
    radius=0.04,
    zorder=3,
)

# Edge device
ax.text(
    edge_box_x + sub_box_width / 2,
    sub_box_y + sub_box_height - 0.24,
    "Edge Device",
    ha="center",
    va="center",
    fontsize=8.0,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    edge_box_x + sub_box_width / 2,
    sub_box_y + sub_box_height - 0.61,
    "ResNet-50 prefix",
    ha="center",
    va="center",
    fontsize=7.4,
    fontstyle="italic",
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    edge_box_x + sub_box_width / 2,
    sub_box_y + 0.39,
    "forward + backward",
    ha="center",
    va="center",
    fontsize=7.0,
    color=COLORS["muted"],
    zorder=4,
)

# Cloud server
ax.text(
    cloud_box_x + sub_box_width / 2,
    sub_box_y + sub_box_height - 0.24,
    "Cloud Server",
    ha="center",
    va="center",
    fontsize=8.0,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    cloud_box_x + sub_box_width / 2,
    sub_box_y + sub_box_height - 0.61,
    "ResNet-50 suffix",
    ha="center",
    va="center",
    fontsize=7.4,
    fontstyle="italic",
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    cloud_box_x + sub_box_width / 2,
    sub_box_y + 0.39,
    "forward + backward",
    ha="center",
    va="center",
    fontsize=7.0,
    color=COLORS["muted"],
    zorder=4,
)

# Two neat and parallel communication arrows
communication_left_x = (
    edge_box_x
    + sub_box_width
    + 0.12
)

communication_right_x = (
    cloud_box_x
    - 0.12
)

feature_arrow_y = (
    sub_box_y
    + 1.00
)

gradient_arrow_y = (
    sub_box_y
    + 0.65
)

# Edge -> Cloud
straight_arrow(
    x_start=communication_left_x,
    y_start=feature_arrow_y,
    x_end=communication_right_x,
    y_end=feature_arrow_y,
    color=COLORS["communication"],
    linewidth=1.40,
    mutation_scale=10,
)

# Cloud -> Edge
straight_arrow(
    x_start=communication_right_x,
    y_start=gradient_arrow_y,
    x_end=communication_left_x,
    y_end=gradient_arrow_y,
    color=COLORS["communication"],
    linewidth=1.40,
    mutation_scale=10,
)

ax.text(
    (
        communication_left_x
        + communication_right_x
    )
    / 2,
    feature_arrow_y + 0.12,
    "Intermediate feature",
    ha="center",
    va="bottom",
    fontsize=6.8,
    color=COLORS["communication"],
    zorder=4,
)

ax.text(
    (
        communication_left_x
        + communication_right_x
    )
    / 2,
    gradient_arrow_y - 0.12,
    "Gradient return",
    ha="center",
    va="top",
    fontsize=6.8,
    color=COLORS["communication"],
    zorder=4,
)

ax.text(
    x + width / 2,
    y + 0.15,
    r"split at $a_t$",
    ha="center",
    va="center",
    fontsize=7.4,
    fontweight="bold",
    color=COLORS["training_border"],
    zorder=4,
)


# ============================================================
# 13. Main horizontal process arrows
# ============================================================

main_boxes = [
    STATE_BOX,
    STATE_CONSTRUCTION_BOX,
    MASK_BOX,
    PPO_BOX,
    SPLIT_BOX,
    TRAINING_BOX,
]

for left_box, right_box in zip(
    main_boxes[:-1],
    main_boxes[1:],
):
    left_x, left_y, left_width, left_height = left_box
    right_x, right_y, right_width, right_height = right_box

    straight_arrow(
        x_start=left_x + left_width + 0.05,
        y_start=left_y + left_height / 2,
        x_end=right_x - 0.06,
        y_end=right_y + right_height / 2,
        color=COLORS["line"],
        linewidth=1.15,
        mutation_scale=10,
    )


# ============================================================
# 14. Performance Feedback
# ============================================================

x, y, width, height = FEEDBACK_BOX

rounded_box(
    x=x,
    y=y,
    width=width,
    height=height,
    face_color=COLORS["feedback_fill"],
    edge_color=COLORS["feedback_border"],
    linewidth=1.15,
    radius=0.05,
    zorder=2,
)

ax.text(
    x + width / 2,
    y + height - 0.15,
    "Performance Feedback",
    ha="center",
    va="top",
    fontsize=9.3,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)

feedback_separator_y = (
    y
    + height
    - 0.40
)

ax.plot(
    [x, x + width],
    [
        feedback_separator_y,
        feedback_separator_y,
    ],
    color=COLORS["feedback_border"],
    linewidth=0.85,
    zorder=3,
)

ax.text(
    x + width / 2,
    y + 0.33,
    (
        r"Latency $T_t(a_t)$"
        r"   \u2022   Switching cost"
        r"   \u2022   Reward $r_t$"
        r"   \u2022   Next state $s_{t+1}$"
    ),
    ha="center",
    va="center",
    fontsize=7.9,
    color=COLORS["ink"],
    zorder=4,
)


# ============================================================
# 15. Training-to-feedback arrow
# ============================================================

training_x, training_y, training_width, training_height = (
    TRAINING_BOX
)

feedback_x, feedback_y, feedback_width, feedback_height = (
    FEEDBACK_BOX
)

orthogonal_arrow(
    points=[
        (
            training_x + training_width / 2,
            training_y,
        ),
        (
            training_x + training_width / 2,
            1.98,
        ),
        (
            feedback_x + feedback_width - 0.55,
            1.98,
        ),
        (
            feedback_x + feedback_width - 0.55,
            feedback_y + feedback_height,
        ),
    ],
    color=COLORS["line"],
    linewidth=1.15,
    mutation_scale=10,
)


# ============================================================
# 16. Feedback-to-next-state arrow
# ============================================================

state_x, state_y, state_width, state_height = STATE_BOX

orthogonal_arrow(
    points=[
        (
            feedback_x,
            feedback_y + feedback_height / 2,
        ),
        (
            0.18,
            feedback_y + feedback_height / 2,
        ),
        (
            0.18,
            2.87,
        ),
        (
            state_x + state_width / 2,
            2.87,
        ),
        (
            state_x + state_width / 2,
            state_y,
        ),
    ],
    color=COLORS["line"],
    linewidth=1.15,
    mutation_scale=10,
)

ax.text(
    0.34,
    2.05,
    "next time step",
    ha="left",
    va="center",
    fontsize=6.8,
    color=COLORS["muted"],
    rotation=90,
    zorder=4,
)


# ============================================================
# 17. Save PNG and PDF
# ============================================================

fig.subplots_adjust(
    left=0.012,
    right=0.992,
    top=0.985,
    bottom=0.04,
)

# 600 dpi PNG for preview and submission backup
fig.savefig(
    PNG_PATH,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.04,
    facecolor="white",
)

# Vector PDF for insertion into the paper
fig.savefig(
    PDF_PATH,
    bbox_inches="tight",
    pad_inches=0.04,
    facecolor="white",
)

plt.close(fig)


# ============================================================
# 18. Verify outputs
# ============================================================

for output_file in (
    PNG_PATH,
    PDF_PATH,
):
    if not output_file.exists():
        raise FileNotFoundError(
            f"Output was not generated: "
            f"{output_file}"
        )

    if output_file.stat().st_size == 0:
        raise RuntimeError(
            f"Output file is empty: "
            f"{output_file}"
        )

    file_size_kb = (
        output_file.stat().st_size
        / 1024
    )

    print(
        f"{output_file} | "
        f"{file_size_kb:.1f} KB"
    )
