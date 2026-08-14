import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch

import config as cfg


# ============================================================
# 1. 文件读取
# ============================================================

def read_result(name):

    path = (
        cfg.OUTPUT_DIR
        / f"{name}_{cfg.MODE_NAME}.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Result file does not exist: {path}"
        )

    return pd.read_csv(
        path
    )


# ============================================================
# 2. Fig.1 系统框架
# ============================================================

def draw_fig1():

    fig, ax = plt.subplots(
        figsize=(11, 4.8)
    )

    ax.set_xlim(
        0,
        11,
    )

    ax.set_ylim(
        0,
        5,
    )

    ax.axis(
        "off"
    )

    def add_box(
        x,
        y,
        width,
        height,
        text,
    ):

        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03",
            linewidth=1.2,
            fill=False,
        )

        ax.add_patch(
            box
        )

        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=10,
        )

    def add_arrow(
        x1,
        y1,
        x2,
        y2,
    ):

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.2,
            },
        )

    add_box(
        0.3,
        3.3,
        1.8,
        1.0,
        "Bandwidth",
    )

    add_box(
        0.3,
        2.0,
        1.8,
        1.0,
        "Available\nMemory",
    )

    add_box(
        0.3,
        0.7,
        1.8,
        1.0,
        "Device\nCompute",
    )

    add_box(
        2.8,
        2.0,
        1.8,
        1.0,
        "State\nConstruction",
    )

    add_box(
        5.1,
        3.1,
        1.8,
        1.0,
        "Memory\nAction Mask",
    )

    add_box(
        5.1,
        1.5,
        1.8,
        1.0,
        "PPO Policy",
    )

    add_box(
        7.4,
        2.0,
        1.5,
        1.0,
        "Split\nAction 0-17",
    )

    add_box(
        9.3,
        2.0,
        1.4,
        1.0,
        "ResNet50\nSplit Training",
    )

    add_arrow(
        2.1,
        3.8,
        2.8,
        2.7,
    )

    add_arrow(
        2.1,
        2.5,
        2.8,
        2.5,
    )

    add_arrow(
        2.1,
        1.2,
        2.8,
        2.3,
    )

    add_arrow(
        4.6,
        2.5,
        5.1,
        2.0,
    )

    add_arrow(
        4.6,
        2.5,
        5.1,
        3.6,
    )

    add_arrow(
        6.9,
        3.6,
        7.4,
        2.7,
    )

    add_arrow(
        6.9,
        2.0,
        7.4,
        2.4,
    )

    add_arrow(
        8.9,
        2.5,
        9.3,
        2.5,
    )

    ax.text(
        8.1,
        0.8,
        "Latency / OOM feedback",
        ha="center",
        va="center",
        fontsize=9,
    )

    ax.annotate(
        "",
        xy=(5.9, 1.5),
        xytext=(9.9, 2.0),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.0,
            "connectionstyle":
                "arc3,rad=-0.25",
        },
    )

    fig.tight_layout()

    png_path = (
        cfg.OUTPUT_DIR
        / f"Fig1_framework_{cfg.MODE_NAME}.png"
    )

    pdf_path = (
        cfg.OUTPUT_DIR
        / f"Fig1_framework_{cfg.MODE_NAME}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return (
        png_path,
        pdf_path,
    )


# ============================================================
# 3. Fig.2(a) PPO 收敛
# ============================================================

def draw_fig2a():

    df = read_result(
        "ppo_training"
    )

    grouped = (
        df.groupby(
            "episode"
        )[
            "train_reward"
        ]
    )

    reward_mean = (
        grouped.mean()
    )

    reward_std = (
        grouped.std()
        .fillna(
            0.0
        )
    )

    fig, ax = plt.subplots(
        figsize=(6.0, 4.2)
    )

    ax.plot(
        reward_mean.index,
        reward_mean.values,
        linewidth=1.6,
    )

    ax.fill_between(
        reward_mean.index,
        reward_mean.values
        - reward_std.values,
        reward_mean.values
        + reward_std.values,
        alpha=0.20,
    )

    ax.set_xlabel(
        "Training Episode"
    )

    ax.set_ylabel(
        "Episode Reward"
    )

    ax.grid(
        alpha=0.20
    )

    fig.tight_layout()

    png_path = (
        cfg.OUTPUT_DIR
        / f"Fig2a_PPO_convergence_{cfg.MODE_NAME}.png"
    )

    pdf_path = (
        cfg.OUTPUT_DIR
        / f"Fig2a_PPO_convergence_{cfg.MODE_NAME}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return (
        png_path,
        pdf_path,
    )


# ============================================================
# 4. Fig.2(b) 系统时间-精度
# ============================================================

def draw_fig2b():

    df = read_result(
        "time_accuracy"
    )

    fig, ax = plt.subplots(
        figsize=(6.2, 4.3)
    )

    for method in [
        "Delay-only Search",
        "Memory-Constrained Search",
        "PPO",
    ]:

        method_df = (
            df[
                df["method"]
                == method
            ]
        )

        grouped = (
            method_df.groupby(
                "epoch"
            )
            .agg(
                elapsed_system_s=(
                    "elapsed_system_s",
                    "mean",
                ),
                test_accuracy=(
                    "test_accuracy",
                    "mean",
                ),
            )
        )

        ax.plot(
            grouped[
                "elapsed_system_s"
            ],
            grouped[
                "test_accuracy"
            ],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=method,
        )

    ax.axhline(
        cfg.TARGET_ACCURACY,
        linestyle="--",
        linewidth=1.0,
        label=(
            f"Target "
            f"{cfg.TARGET_ACCURACY:.0f}%"
        ),
    )

    ax.set_xlabel(
        "Accumulated System Time (s)"
    )

    ax.set_ylabel(
        "Test Accuracy (%)"
    )

    ax.grid(
        alpha=0.20
    )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    png_path = (
        cfg.OUTPUT_DIR
        / f"Fig2b_time_accuracy_{cfg.MODE_NAME}.png"
    )

    pdf_path = (
        cfg.OUTPUT_DIR
        / f"Fig2b_time_accuracy_{cfg.MODE_NAME}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return (
        png_path,
        pdf_path,
    )


# ============================================================
# 5. Table I
# ============================================================

def create_table1():

    table = pd.DataFrame(
        [
            {
                "Method":
                    "Delay-only Search",

                "Memory Constraint":
                    "No",

                "Objective":
                    "Minimum estimated latency",

                "Online Selection":
                    "Search",

                "Complexity":
                    "O(K)",
            },

            {
                "Method":
                    "Memory-Constrained Search",

                "Memory Constraint":
                    "Explicit",

                "Objective":
                    "Minimum feasible latency",

                "Online Selection":
                    "Search",

                "Complexity":
                    "O(K)",
            },

            {
                "Method":
                    "Memory-aware PPO",

                "Memory Constraint":
                    "Action Mask",

                "Objective":
                    "Feasible low-latency partition",

                "Online Selection":
                    "Policy inference",

                "Complexity":
                    "O(K + NN)",
            },
        ]
    )

    path = (
        cfg.OUTPUT_DIR
        / f"TableI_methods_{cfg.MODE_NAME}.csv"
    )

    table.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return (
        table,
        path,
    )


# ============================================================
# 6. Table II
# ============================================================

def create_table2():

    summary = read_result(
        "comparison_summary"
    )

    decision = read_result(
        "decision_time"
    )

    result_rows = []

    for method in [
        "Delay-only Search",
        "Memory-Constrained Search",
        "PPO",
    ]:

        part = (
            summary[
                summary["method"]
                == method
            ]
        )

        decision_part = (
            decision[
                decision["method"]
                == method
            ].iloc[
                0
            ]
        )

        valid_target = (
            part[
                "time_to_target_s"
            ]
            .dropna()
        )

        if len(
            valid_target
        ) > 0:

            time_mean = float(
                valid_target.mean()
            )

            if len(
                valid_target
            ) > 1:

                time_std = float(
                    valid_target.std(
                        ddof=1
                    )
                )

            else:

                time_std = 0.0

        else:

            time_mean = np.nan

            time_std = np.nan

        result_rows.append(
            {
                "Method":
                    method,

                "Time_to_Target_mean_s":
                    time_mean,

                "Time_to_Target_std_s":
                    time_std,

                "Final_System_Time_mean_s":
                    float(
                        part[
                            "final_system_time_s"
                        ].mean()
                    ),

                "OOM_Count_mean":
                    float(
                        part[
                            "oom_count"
                        ].mean()
                    ),

                "Low_Memory_Survival_mean_%":
                    float(
                        part[
                            "low_memory_survival_rate"
                        ].mean()
                    ),

                "Switch_Count_mean":
                    float(
                        part[
                            "switch_count"
                        ].mean()
                    ),

                "Decision_Median_us":
                    float(
                        decision_part[
                            "median_us"
                        ]
                    ),

                "Decision_P95_us":
                    float(
                        decision_part[
                            "p95_us"
                        ]
                    ),
            }
        )

    table = pd.DataFrame(
        result_rows
    )

    path = (
        cfg.OUTPUT_DIR
        / f"TableII_results_{cfg.MODE_NAME}.csv"
    )

    table.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return (
        table,
        path,
    )


# ============================================================
# 7. Fig.3 核心性能
# ============================================================

def draw_fig3(
    table2,
):

    methods = (
        table2[
            "Method"
        ].tolist()
    )

    short_names = [
        "Delay-only",
        "Memory-search",
        "PPO",
    ]

    x = np.arange(
        len(
            methods
        )
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.0, 4.0),
    )

    # --------------------------------------------------------
    # Fig.3(a)
    # --------------------------------------------------------

    target_values = (
        table2[
            "Time_to_Target_mean_s"
        ].to_numpy(
            dtype=np.float64
        )
    )

    target_std = (
        table2[
            "Time_to_Target_std_s"
        ].to_numpy(
            dtype=np.float64
        )
    )

    if np.all(
        np.isnan(
            target_values
        )
    ):

        values = (
            table2[
                "Final_System_Time_mean_s"
            ].to_numpy(
                dtype=np.float64
            )
        )

        axes[
            0
        ].bar(
            x,
            values,
        )

        axes[
            0
        ].set_ylabel(
            "System Time (s)"
        )

        axes[
            0
        ].set_title(
            "(a) Quick-mode System Time"
        )

    else:

        axes[
            0
        ].bar(
            x,
            target_values,
            yerr=np.nan_to_num(
                target_std,
                nan=0.0,
            ),
            capsize=3,
        )

        axes[
            0
        ].set_ylabel(
            "Time to Target Accuracy (s)"
        )

        axes[
            0
        ].set_title(
            "(a) Time to Target"
        )

    axes[
        0
    ].set_xticks(
        x
    )

    axes[
        0
    ].set_xticklabels(
        short_names,
        rotation=20,
        ha="right",
    )

    # --------------------------------------------------------
    # Fig.3(b)
    # --------------------------------------------------------

    survival = (
        table2[
            "Low_Memory_Survival_mean_%"
        ].to_numpy(
            dtype=np.float64
        )
    )

    axes[
        1
    ].bar(
        x,
        survival,
    )

    axes[
        1
    ].set_ylabel(
        "Low-memory Survival Rate (%)"
    )

    axes[
        1
    ].set_title(
        "(b) Low-memory Survival"
    )

    axes[
        1
    ].set_ylim(
        0,
        105,
    )

    axes[
        1
    ].set_xticks(
        x
    )

    axes[
        1
    ].set_xticklabels(
        short_names,
        rotation=20,
        ha="right",
    )

    # --------------------------------------------------------
    # Fig.3(c)
    # --------------------------------------------------------

    median = (
        table2[
            "Decision_Median_us"
        ].to_numpy(
            dtype=np.float64
        )
    )

    p95 = (
        table2[
            "Decision_P95_us"
        ].to_numpy(
            dtype=np.float64
        )
    )

    width = 0.35

    axes[
        2
    ].bar(
        x - width / 2,
        median,
        width=width,
        label="Median",
    )

    axes[
        2
    ].bar(
        x + width / 2,
        p95,
        width=width,
        label="P95",
    )

    axes[
        2
    ].set_ylabel(
        "Decision Time (us)"
    )

    axes[
        2
    ].set_title(
        "(c) Online Decision Time"
    )

    axes[
        2
    ].set_xticks(
        x
    )

    axes[
        2
    ].set_xticklabels(
        short_names,
        rotation=20,
        ha="right",
    )

    axes[
        2
    ].legend(
        fontsize=8
    )

    for ax in axes:

        ax.grid(
            axis="y",
            alpha=0.20,
        )

    fig.tight_layout()

    png_path = (
        cfg.OUTPUT_DIR
        / f"Fig3_core_performance_{cfg.MODE_NAME}.png"
    )

    pdf_path = (
        cfg.OUTPUT_DIR
        / f"Fig3_core_performance_{cfg.MODE_NAME}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return (
        png_path,
        pdf_path,
    )


# ============================================================
# 8. 主程序
# ============================================================

def main():

    cfg.prepare_output_dir()

    print()
    print("=" * 70)
    print("Drawing figures and tables")
    print("=" * 70)

    fig1 = draw_fig1()

    fig2a = draw_fig2a()

    fig2b = draw_fig2b()

    table1, table1_path = (
        create_table1()
    )

    table2, table2_path = (
        create_table2()
    )

    fig3 = draw_fig3(
        table2
    )

    print()
    print("Generated:")

    for path in [
        fig1[0],
        fig1[1],
        fig2a[0],
        fig2a[1],
        fig2b[0],
        fig2b[1],
        fig3[0],
        fig3[1],
        table1_path,
        table2_path,
    ]:

        print(
            path
        )

    print()
    print("Table I:")
    print(
        table1.to_string(
            index=False
        )
    )

    print()
    print("Table II:")
    print(
        table2.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("draw_figures.py finished")
    print("=" * 70)


if __name__ == "__main__":
    main()
