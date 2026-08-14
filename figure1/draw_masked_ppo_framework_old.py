from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch,
    Rectangle,
)
from matplotlib.path import Path as MplPath


# ============================================================
# 1. Fixed workspace and output files
# ============================================================

WORKDIR = Path(r"D:\pythonWork\formal_test\figure1")
WORKDIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = WORKDIR / "masked_ppo_framework.png"
PDF_PATH = WORKDIR / "masked_ppo_framework.pdf"
SVG_PATH = WORKDIR / "masked_ppo_framework.svg"

# Delete only previous outputs.
# Do not delete any other files in the workspace.
for output_file in (
    PNG_PATH,
    PDF_PATH,
    SVG_PATH,
):
    if output_file.exists():
        output_file.unlink()


# ============================================================
# 2. Matplotlib global configuration
# ============================================================

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "DejaVu Sans",
    ],
    "font.size": 8.2,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,

    # Keep fonts editable in vector outputs.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ============================================================
# 3. EI-conference color palette
# ============================================================

COLORS = {
    "ink": "#1F2937",
    "muted": "#5B6572",
    "line": "#2F3B4A",

    "state_border": "#4E79A7",
    "state_fill": "#EEF5FB",

    "policy_border": "#8E6BBE",
    "policy_fill": "#F5F0FA",

    "mask_border": "#D99A17",
    "mask_fill": "#FFF7DF",

    "decision_border": "#4C9A73",
    "decision_fill": "#EDF8F2",

    "training_border": "#64748B",
    "training_fill": "#F5F7FA",

    "feedback_border": "#7C8796",
    "feedback_fill": "#F2F4F6",

    "profile_border": "#A56C2C",
    "profile_fill": "#FBF4EB",

    "communication": "#2F6FB3",

    "white": "#FFFFFF",
    "masked_cell": "#D7DCE2",
}


# ============================================================
# 4. Canvas
# ============================================================

fig, ax = plt.subplots(
    figsize=(15.4, 6.15),
    facecolor="white",
)

ax.set_xlim(0, 19.2)
ax.set_ylim(0, 7.25)
ax.axis("off")


# ============================================================
# 5. Drawing helper functions
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
    radius=0.055,
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
    header_height=0.52,
    title_fontsize=9.7,
):
    """
    Draw a main framework card with
    a separated title header.
    """

    rounded_box(
        x=x,
        y=y,
        width=width,
        height=height,
        face_color=face_color,
        edge_color=edge_color,
        linewidth=1.18,
        radius=0.055,
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
        y + height - 0.17,
        title,
        ha="center",
        va="top",
        fontsize=title_fontsize,
        fontweight="bold",
        linespacing=1.02,
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
    Draw one straight arrow.
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
    return arrow_patch


def orthogonal_arrow(
    points,
    color=None,
    linewidth=1.15,
    mutation_scale=10,
):
    """
    Draw one orthogonal polyline arrow.

    The arrow follows the supplied points.
    No curved line is used.
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
    return arrow_patch


def center_right(box):
    """
    Return the center point of a box's right edge.
    """

    x, y, width, height = box

    return (
        x + width,
        y + height / 2,
    )


def center_left(box):
    """
    Return the center point of a box's left edge.
    """

    x, y, width, height = box

    return (
        x,
        y + height / 2,
    )


# ============================================================
# 6. Fixed geometry
# ============================================================

TOP_Y = 4.55
TOP_HEIGHT = 2.05

STATE_BOX = (
    0.45,
    TOP_Y,
    2.30,
    TOP_HEIGHT,
)

STATE_CONSTRUCTION_BOX = (
    3.08,
    TOP_Y,
    2.30,
    TOP_HEIGHT,
)

POLICY_BOX = (
    5.72,
    TOP_Y,
    2.50,
    TOP_HEIGHT,
)

MASK_SELECTION_BOX = (
    8.56,
    TOP_Y,
    2.55,
    TOP_HEIGHT,
)

DECISION_BOX = (
    11.45,
    TOP_Y,
    2.15,
    TOP_HEIGHT,
)

TRAINING_BOX = (
    13.94,
    4.35,
    4.75,
    2.55,
)

PROFILE_BOX = (
    6.35,
    2.55,
    4.00,
    1.25,
)

FEEDBACK_BOX = (
    2.65,
    0.55,
    11.65,
    1.30,
)


# ============================================================
# 7. Module 1: Dynamic Resource State
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
    title_fontsize=9.1,
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
        "Compute coefficient",
        r"$\alpha_t$",
    ),
    (
        "Previous action",
        r"$a_{t-1}$",
    ),
]

state_row_y = (
    state_separator_y
    - 0.31
)

for label, symbol in state_rows:
    ax.text(
        x + 0.16,
        state_row_y,
        label,
        ha="left",
        va="center",
        fontsize=7.75,
        color=COLORS["ink"],
        zorder=4,
    )

    ax.text(
        x + width - 0.15,
        state_row_y,
        symbol,
        ha="right",
        va="center",
        fontsize=9.10,
        color=COLORS["ink"],
        zorder=4,
    )

    state_row_y -= 0.38


# ============================================================
# 8. Module 2: State Construction
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
    state_construction_separator_y - 0.31,
    "Normalize continuous features",
    ha="center",
    va="center",
    fontsize=7.60,
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    x + width / 2,
    state_construction_separator_y - 0.62,
    (
        r"$\bar{B}_t,\ "
        r"\bar{M}_{\mathrm{avail},t},\ "
        r"\bar{\alpha}_t$"
    ),
    ha="center",
    va="center",
    fontsize=9.30,
    color=COLORS["ink"],
    zorder=4,
)

ax.plot(
    [
        x + 0.27,
        x + width - 0.27,
    ],
    [
        state_construction_separator_y - 0.84,
        state_construction_separator_y - 0.84,
    ],
    color=COLORS["state_border"],
    linewidth=0.75,
    zorder=3,
)

ax.text(
    x + width / 2,
    state_construction_separator_y - 0.96,
    (
        r"Concatenate "
        r"$\mathrm{onehot}(a_{t-1})$"
    ),
    ha="center",
    va="center",
    fontsize=7.60,
    color=COLORS["ink"],
    zorder=4,
)

rounded_box(
    x=x + 0.24,
    y=y + 0.08,
    width=width - 0.48,
    height=0.40,
    face_color=COLORS["white"],
    edge_color=COLORS["state_border"],
    linewidth=0.85,
    radius=0.04,
    zorder=3,
)

ax.text(
    x + width / 2,
    y + 0.28,
    r"$s_t\in\mathbb{R}^{21}$",
    ha="center",
    va="center",
    fontsize=9.20,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)


# ============================================================
# 9. Module 3: Policy Network
# ============================================================

x, y, width, height = POLICY_BOX

policy_separator_y = draw_card(
    x=x,
    y=y,
    width=width,
    height=height,
    title="Policy Network",
    face_color=COLORS["policy_fill"],
    edge_color=COLORS["policy_border"],
)

policy_center_y = (
    policy_separator_y
    - 0.74
)

policy_node_width = 0.42
policy_node_height = 0.70
policy_node_gap = 0.14
policy_start_x = x + 0.20

policy_nodes = [
    (
        "Input",
        "21",
    ),
    (
        "FC1",
        "32",
    ),
    (
        "FC2",
        "32",
    ),
]

for index, (
    node_name,
    node_units,
) in enumerate(policy_nodes):

    node_x = (
        policy_start_x
        + index
        * (
            policy_node_width
            + policy_node_gap
        )
    )

    rounded_box(
        x=node_x,
        y=(
            policy_center_y
            - policy_node_height / 2
        ),
        width=policy_node_width,
        height=policy_node_height,
        face_color=COLORS["white"],
        edge_color=COLORS["policy_border"],
        linewidth=0.85,
        radius=0.035,
        zorder=3,
    )

    ax.text(
        node_x + policy_node_width / 2,
        policy_center_y + 0.10,
        node_name,
        ha="center",
        va="center",
        fontsize=6.30,
        color=COLORS["muted"],
        zorder=4,
    )

    ax.text(
        node_x + policy_node_width / 2,
        policy_center_y - 0.14,
        node_units,
        ha="center",
        va="center",
        fontsize=8.60,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=4,
    )

    if index < 2:
        straight_arrow(
            x_start=(
                node_x
                + policy_node_width
            ),
            y_start=policy_center_y,
            x_end=(
                node_x
                + policy_node_width
                + policy_node_gap
                - 0.03
            ),
            y_end=policy_center_y,
            color=COLORS["policy_border"],
            linewidth=0.85,
            mutation_scale=8,
        )


last_shared_layer_right = (
    policy_start_x
    + 2
    * (
        policy_node_width
        + policy_node_gap
    )
    + policy_node_width
)

branch_x = (
    x
    + width
    - 0.58
)

straight_arrow(
    x_start=last_shared_layer_right,
    y_start=policy_center_y,
    x_end=branch_x - 0.08,
    y_end=policy_center_y,
    color=COLORS["policy_border"],
    linewidth=0.85,
    mutation_scale=8,
)

rounded_box(
    x=branch_x,
    y=policy_center_y + 0.10,
    width=0.42,
    height=0.34,
    face_color=COLORS["white"],
    edge_color=COLORS["policy_border"],
    linewidth=0.80,
    radius=0.03,
    zorder=3,
)

rounded_box(
    x=branch_x,
    y=policy_center_y - 0.44,
    width=0.42,
    height=0.34,
    face_color=COLORS["white"],
    edge_color=COLORS["policy_border"],
    linewidth=0.80,
    radius=0.03,
    zorder=3,
)

ax.text(
    branch_x + 0.21,
    policy_center_y + 0.27,
    "Actor",
    ha="center",
    va="center",
    fontsize=6.10,
    color=COLORS["muted"],
    zorder=4,
)

ax.text(
    branch_x + 0.21,
    policy_center_y + 0.16,
    "18",
    ha="center",
    va="center",
    fontsize=8.00,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    branch_x + 0.21,
    policy_center_y - 0.27,
    "Critic",
    ha="center",
    va="center",
    fontsize=6.10,
    color=COLORS["muted"],
    zorder=4,
)

ax.text(
    branch_x + 0.21,
    policy_center_y - 0.38,
    "1",
    ha="center",
    va="center",
    fontsize=8.00,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    x + width / 2,
    y + 0.19,
    "shared actor\u2013critic MLP",
    ha="center",
    va="center",
    fontsize=7.20,
    color=COLORS["muted"],
    zorder=4,
)


# ============================================================
# 10. Module 4: Masked Action Selection
# ============================================================

x, y, width, height = MASK_SELECTION_BOX

mask_separator_y = draw_card(
    x=x,
    y=y,
    width=width,
    height=height,
    title="Masked Action Selection",
    face_color=COLORS["mask_fill"],
    edge_color=COLORS["mask_border"],
)

ax.text(
    x + width / 2,
    mask_separator_y - 0.28,
    "Actor logits + feasible mask",
    ha="center",
    va="center",
    fontsize=7.65,
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    x + width / 2,
    mask_separator_y - 0.63,
    r"infeasible logits $\rightarrow -\infty$",
    ha="center",
    va="center",
    fontsize=8.30,
    color=COLORS["ink"],
    zorder=4,
)

mask_values = [
    1, 1, 0, 1, 0, 1, 1, 0, 1,
    1, 0, 1, 0, 1, 1, 0, 1, 0,
]

mask_columns = 9
mask_cell_size = 0.18
mask_cell_gap = 0.035

mask_grid_width = (
    mask_columns * mask_cell_size
    + (
        mask_columns - 1
    )
    * mask_cell_gap
)

mask_grid_start_x = (
    x
    + (
        width
        - mask_grid_width
    )
    / 2
)

mask_grid_start_y = (
    mask_separator_y
    - 1.10
)

for index, value in enumerate(mask_values):
    row = index // mask_columns
    column = index % mask_columns

    cell_x = (
        mask_grid_start_x
        + column
        * (
            mask_cell_size
            + mask_cell_gap
        )
    )

    cell_y = (
        mask_grid_start_y
        - row
        * (
            mask_cell_size
            + mask_cell_gap
        )
    )

    if value == 1:
        cell_face_color = COLORS["white"]
        cell_edge_color = COLORS["mask_border"]
    else:
        cell_face_color = COLORS["masked_cell"]
        cell_edge_color = "#9AA0A6"

    cell_patch = Rectangle(
        (cell_x, cell_y),
        mask_cell_size,
        mask_cell_size,
        facecolor=cell_face_color,
        edgecolor=cell_edge_color,
        linewidth=0.65,
        zorder=4,
    )

    ax.add_patch(cell_patch)

    ax.text(
        cell_x + mask_cell_size / 2,
        cell_y + mask_cell_size / 2,
        str(value),
        ha="center",
        va="center",
        fontsize=6.20,
        color=COLORS["ink"],
        zorder=5,
    )

ax.text(
    x + width / 2,
    y + 0.24,
    "masked softmax  \u2022  argmax",
    ha="center",
    va="center",
    fontsize=7.55,
    color=COLORS["muted"],
    zorder=4,
)


# ============================================================
# 11. Module 5: Split Decision
# ============================================================

x, y, width, height = DECISION_BOX

decision_separator_y = draw_card(
    x=x,
    y=y,
    width=width,
    height=height,
    title="Split Decision",
    face_color=COLORS["decision_fill"],
    edge_color=COLORS["decision_border"],
)

ax.text(
    x + width / 2,
    decision_separator_y - 0.31,
    "Select feasible split point",
    ha="center",
    va="center",
    fontsize=7.55,
    color=COLORS["ink"],
    zorder=4,
)

rounded_box(
    x=x + 0.18,
    y=decision_separator_y - 1.18,
    width=width - 0.36,
    height=0.74,
    face_color=COLORS["white"],
    edge_color=COLORS["decision_border"],
    linewidth=0.85,
    radius=0.04,
    zorder=3,
)

ax.text(
    x + width / 2,
    decision_separator_y - 0.70,
    (
        r"$a_t=\arg\max_{a\in\mathcal{A}_t^f}"
        r"\pi_\theta(a\mid s_t)$"
    ),
    ha="center",
    va="center",
    fontsize=7.55,
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    x + width / 2,
    decision_separator_y - 1.00,
    r"$a_t\in\{0,\ldots,17\}$",
    ha="center",
    va="center",
    fontsize=8.30,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    x + width / 2,
    y + 0.20,
    "18 ResNet-50 split points",
    ha="center",
    va="center",
    fontsize=7.15,
    color=COLORS["muted"],
    zorder=4,
)


# ============================================================
# 12. Module 6: Edge\u2013Cloud Split Training
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
    radius=0.055,
    zorder=2,
)

ax.text(
    x + width / 2,
    y + height - 0.18,
    "Edge\u2013Cloud Split Training",
    ha="center",
    va="top",
    fontsize=9.90,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)

ax.plot(
    [x, x + width],
    [
        y + height - 0.55,
        y + height - 0.55,
    ],
    color=COLORS["training_border"],
    linewidth=0.90,
    zorder=3,
)

training_subbox_y = y + 0.40
training_subbox_height = 1.48
training_subbox_width = 1.43

edge_box_x = x + 0.34

cloud_box_x = (
    x
    + width
    - 0.34
    - training_subbox_width
)

rounded_box(
    x=edge_box_x,
    y=training_subbox_y,
    width=training_subbox_width,
    height=training_subbox_height,
    face_color=COLORS["white"],
    edge_color=COLORS["training_border"],
    linewidth=0.90,
    radius=0.04,
    zorder=3,
)

rounded_box(
    x=cloud_box_x,
    y=training_subbox_y,
    width=training_subbox_width,
    height=training_subbox_height,
    face_color=COLORS["white"],
    edge_color=COLORS["training_border"],
    linewidth=0.90,
    radius=0.04,
    zorder=3,
)

# Edge-device text
ax.text(
    edge_box_x + training_subbox_width / 2,
    training_subbox_y
    + training_subbox_height
    - 0.23,
    "Edge Device",
    ha="center",
    va="center",
    fontsize=8.20,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    edge_box_x + training_subbox_width / 2,
    training_subbox_y
    + training_subbox_height
    - 0.60,
    "ResNet-50 prefix",
    ha="center",
    va="center",
    fontsize=7.55,
    fontstyle="italic",
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    edge_box_x + training_subbox_width / 2,
    training_subbox_y + 0.48,
    "forward + backward",
    ha="center",
    va="center",
    fontsize=7.10,
    color=COLORS["muted"],
    zorder=4,
)

# Cloud-server text
ax.text(
    cloud_box_x + training_subbox_width / 2,
    training_subbox_y
    + training_subbox_height
    - 0.23,
    "Cloud Server",
    ha="center",
    va="center",
    fontsize=8.20,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    cloud_box_x + training_subbox_width / 2,
    training_subbox_y
    + training_subbox_height
    - 0.60,
    "ResNet-50 suffix",
    ha="center",
    va="center",
    fontsize=7.55,
    fontstyle="italic",
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    cloud_box_x + training_subbox_width / 2,
    training_subbox_y + 0.48,
    "forward + backward",
    ha="center",
    va="center",
    fontsize=7.10,
    color=COLORS["muted"],
    zorder=4,
)

# Perfectly aligned parallel communication arrows
communication_left_x = (
    edge_box_x
    + training_subbox_width
    + 0.12
)

communication_right_x = (
    cloud_box_x
    - 0.12
)

feature_arrow_y = (
    training_subbox_y
    + 1.00
)

gradient_arrow_y = (
    training_subbox_y
    + 0.66
)

# Upper arrow: Edge -> Cloud
straight_arrow(
    x_start=communication_left_x,
    y_start=feature_arrow_y,
    x_end=communication_right_x,
    y_end=feature_arrow_y,
    color=COLORS["communication"],
    linewidth=1.45,
    mutation_scale=10,
)

# Lower arrow: Cloud -> Edge
straight_arrow(
    x_start=communication_right_x,
    y_start=gradient_arrow_y,
    x_end=communication_left_x,
    y_end=gradient_arrow_y,
    color=COLORS["communication"],
    linewidth=1.45,
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
    fontsize=6.90,
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
    fontsize=6.90,
    color=COLORS["communication"],
    zorder=4,
)

ax.text(
    x + width / 2,
    y + 0.15,
    r"split at $a_t$",
    ha="center",
    va="center",
    fontsize=7.50,
    fontweight="bold",
    color=COLORS["training_border"],
    zorder=4,
)


# ============================================================
# 13. Main left-to-right process arrows
# ============================================================

main_boxes = [
    STATE_BOX,
    STATE_CONSTRUCTION_BOX,
    POLICY_BOX,
    MASK_SELECTION_BOX,
    DECISION_BOX,
    TRAINING_BOX,
]

for left_box, right_box in zip(
    main_boxes[:-1],
    main_boxes[1:],
):
    left_point = center_right(left_box)
    right_point = center_left(right_box)

    straight_arrow(
        x_start=left_point[0] + 0.05,
        y_start=left_point[1],
        x_end=right_point[0] - 0.06,
        y_end=right_point[1],
        color=COLORS["line"],
        linewidth=1.15,
        mutation_scale=10,
    )


# ============================================================
# 14. Offline Profile and Feasibility Mask
# ============================================================

x, y, width, height = PROFILE_BOX

profile_separator_y = draw_card(
    x=x,
    y=y,
    width=width,
    height=height,
    title="Offline Profile & Feasibility Mask",
    face_color=COLORS["profile_fill"],
    edge_color=COLORS["profile_border"],
    header_height=0.46,
    title_fontsize=8.90,
)

ax.text(
    x + width / 2,
    profile_separator_y - 0.25,
    (
        r"$M_{\mathrm{peak}}(a),\ "
        r"D_{\mathrm{dev}}(a),\ "
        r"D_{\mathrm{srv}}(a),\ "
        r"S_{\mathrm{out}}(a)$"
    ),
    ha="center",
    va="center",
    fontsize=7.65,
    color=COLORS["ink"],
    zorder=4,
)

ax.text(
    x + width / 2,
    y + 0.27,
    (
        r"$\mathcal{A}_t^f="
        r"\{a\mid "
        r"M_{\mathrm{peak}}(a)"
        r"\leq M_{\mathrm{avail},t}\}$"
    ),
    ha="center",
    va="center",
    fontsize=8.20,
    color=COLORS["ink"],
    zorder=4,
)


# ============================================================
# 15. State-to-profile and profile-to-mask arrows
# ============================================================

state_x, state_y, state_width, state_height = (
    STATE_CONSTRUCTION_BOX
)

mask_x, mask_y, mask_width, mask_height = (
    MASK_SELECTION_BOX
)

profile_x, profile_y, profile_width, profile_height = (
    PROFILE_BOX
)

orthogonal_arrow(
    points=[
        (
            state_x + state_width / 2,
            state_y,
        ),
        (
            state_x + state_width / 2,
            4.03,
        ),
        (
            profile_x + 0.72,
            4.03,
        ),
        (
            profile_x + 0.72,
            profile_y + profile_height,
        ),
    ],
    color=COLORS["profile_border"],
    linewidth=1.00,
    mutation_scale=9,
)

ax.text(
    5.05,
    3.93,
    r"$M_{\mathrm{avail},t}$",
    ha="center",
    va="top",
    fontsize=6.80,
    color=COLORS["profile_border"],
    zorder=4,
)

orthogonal_arrow(
    points=[
        (
            profile_x
            + profile_width
            - 0.72,
            profile_y
            + profile_height,
        ),
        (
            profile_x
            + profile_width
            - 0.72,
            4.03,
        ),
        (
            mask_x + mask_width / 2,
            4.03,
        ),
        (
            mask_x + mask_width / 2,
            mask_y,
        ),
    ],
    color=COLORS["profile_border"],
    linewidth=1.00,
    mutation_scale=9,
)

ax.text(
    10.10,
    3.93,
    r"$\mathcal{A}_t^f$",
    ha="center",
    va="top",
    fontsize=7.00,
    color=COLORS["profile_border"],
    zorder=4,
)


# ============================================================
# 16. Performance Feedback panel
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
    y + height - 0.14,
    "Performance Feedback and State Transition",
    ha="center",
    va="top",
    fontsize=9.50,
    fontweight="bold",
    color=COLORS["ink"],
    zorder=4,
)

feedback_separator_y = (
    y
    + height
    - 0.42
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

feedback_column_widths = [
    2.00,
    2.25,
    4.45,
    2.45,
]

feedback_titles = [
    "Latency",
    "Switching cost",
    "Reward",
    "Next state",
]

feedback_values = [
    r"$T_t(a_t)$",
    r"$\mathbb{I}(a_t\neq a_{t-1})$",
    (
        r"$r_t=1-2\delta_t"
        r"-\lambda_s"
        r"\mathbb{I}(a_t\neq a_{t-1})$"
    ),
    r"$s_{t+1}$",
]

feedback_column_x = x

for index, column_width in enumerate(
    feedback_column_widths
):
    if index > 0:
        ax.plot(
            [
                feedback_column_x,
                feedback_column_x,
            ],
            [
                y + 0.12,
                feedback_separator_y - 0.04,
            ],
            color="#B9C0C9",
            linewidth=0.75,
            zorder=3,
        )

    ax.text(
        feedback_column_x
        + column_width / 2,
        feedback_separator_y - 0.14,
        feedback_titles[index],
        ha="center",
        va="center",
        fontsize=7.70,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=4,
    )

    if index == 2:
        value_fontsize = 7.45
    else:
        value_fontsize = 7.80

    ax.text(
        feedback_column_x
        + column_width / 2,
        y + 0.29,
        feedback_values[index],
        ha="center",
        va="center",
        fontsize=value_fontsize,
        color=COLORS["ink"],
        zorder=4,
    )

    feedback_column_x += column_width


# ============================================================
# 17. Training-to-feedback orthogonal arrow
# ============================================================

feedback_x, feedback_y, feedback_width, feedback_height = (
    FEEDBACK_BOX
)

training_x, training_y, training_width, training_height = (
    TRAINING_BOX
)

orthogonal_arrow(
    points=[
        (
            training_x + training_width / 2,
            training_y,
        ),
        (
            training_x + training_width / 2,
            2.12,
        ),
        (
            feedback_x + feedback_width - 0.55,
            2.12,
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

ax.text(
    15.65,
    2.24,
    "measured outcome",
    ha="center",
    va="bottom",
    fontsize=6.80,
    color=COLORS["muted"],
    zorder=4,
)


# ============================================================
# 18. Feedback-to-next-state orthogonal arrow
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
            4.05,
        ),
        (
            state_x + state_width / 2,
            4.05,
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
    0.33,
    3.10,
    "next time step",
    ha="left",
    va="center",
    fontsize=6.90,
    color=COLORS["muted"],
    rotation=90,
    zorder=4,
)


# ============================================================
# 19. Save PNG, PDF and SVG
# ============================================================

fig.subplots_adjust(
    left=0.012,
    right=0.992,
    top=0.985,
    bottom=0.035,
)

# High-resolution PNG preview
fig.savefig(
    PNG_PATH,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.04,
    facecolor="white",
)

# Vector PDF for the EI conference paper
fig.savefig(
    PDF_PATH,
    bbox_inches="tight",
    pad_inches=0.04,
    facecolor="white",
)

# Editable SVG backup
fig.savefig(
    SVG_PATH,
    bbox_inches="tight",
    pad_inches=0.04,
    facecolor="white",
)

plt.close(fig)


# ============================================================
# 20. Verify generated outputs
# ============================================================

for output_file in (
    PNG_PATH,
    PDF_PATH,
    SVG_PATH,
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
