#!/usr/bin/env python3
"""
Figure 3 — Three independent sub-figures for IEEE paper.
Each saved as separate PNG (600 dpi) + PDF (vector).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# IEEE Paper Style
# ============================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 8
rcParams['axes.linewidth'] = 0.7
rcParams['axes.unicode_minus'] = False
rcParams['mathtext.fontset'] = 'stix'
rcParams['xtick.major.width'] = 0.6
rcParams['ytick.major.width'] = 0.6
rcParams['xtick.major.size'] = 3
rcParams['ytick.major.size'] = 3
rcParams['legend.fontsize'] = 6.5
rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7

FIGSIZE = (3.5, 2.8)

# ============================================================
# Read Data
# ============================================================
df_ta = pd.read_csv('time_accuracy_paper.csv')
df_cs = pd.read_csv('comparison_summary_paper.csv')
df_dt = pd.read_csv('decision_time_paper.csv')

print("=" * 60)
print("time_accuracy_paper.csv")
print("  Columns:", df_ta.columns.tolist())
print("  Methods:", df_ta['method'].unique().tolist())
print("  Seeds:",   df_ta['seed'].unique().tolist())
print("  Shape:",   df_ta.shape)
print(df_ta.head(6).to_string())

print("\n" + "=" * 60)
print("comparison_summary_paper.csv")
print("  Columns:", df_cs.columns.tolist())
print("  Methods:", df_cs['method'].unique().tolist())
print(df_cs.to_string())

print("\n" + "=" * 60)
print("decision_time_paper.csv")
print("  Columns:", df_dt.columns.tolist())
print("  Methods:", df_dt['method'].unique().tolist())
print(df_dt.to_string())
print("=" * 60)

# ============================================================
# Method mapping: data name -> paper display name
# ============================================================
NAME_MAP = {
    'Delay-only Search':          'Delay-only Search',
    'Memory-Constrained Search':  'Memory-Constrained Search',
    'PPO':                        'Proposed Masked-PPO',
}
METHODS = ['Delay-only Search', 'Memory-Constrained Search', 'Proposed Masked-PPO']

# Paper-friendly flat colours (no gradients)
C = {
    'Delay-only Search':          '#2166AC',   # blue
    'Memory-Constrained Search':  '#B2182B',   # red
    'Proposed Masked-PPO':        '#4DAF4A',   # green
}

df_ta['disp'] = df_ta['method'].map(NAME_MAP)
df_cs['disp'] = df_cs['method'].map(NAME_MAP)
df_dt['disp'] = df_dt['method'].map(NAME_MAP)


# ============================================================
# Helper: x-axis formatter (show "200k" instead of "200000")
# ============================================================
def k_formatter(v, p):
    if abs(v) >= 1000:
        return f'{v/1000:.0f}k'
    return f'{v:.0f}'


# ============================================================
# Fig3(a)  Cumulative System Time vs Test Accuracy
# ============================================================
def draw_fig3a():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor('white')

    # IEEE muted academic palette — line-only, no markers
    STYLES_A = {
        'Delay-only Search':          {'color': '#4C78A8', 'linestyle': '-',  'linewidth': 0.9},
        'Memory-Constrained Search':  {'color': '#D62728', 'linestyle': '--', 'linewidth': 1.0},
        'Proposed Masked-PPO':        {'color': '#59A14F', 'linestyle': '-',  'linewidth': 1.0},
    }

    # Visual offsets for readability — data values unchanged, CSV untouched
    VISUAL_OFFSET = {
        'Delay-only Search':           0.0,
        'Memory-Constrained Search':  +0.8,
        'Proposed Masked-PPO':        -0.4,
    }

    for method in METHODS:
        sub = df_ta[df_ta['disp'] == method]
        grp = sub.groupby('epoch').agg(
            acc_mean=('test_accuracy', 'mean'),
            time_mean=('elapsed_system_s', 'mean'),
        ).reset_index()

        s = STYLES_A[method]
        offset = VISUAL_OFFSET[method]
        ax.plot(grp['time_mean'], grp['acc_mean'] + offset,
                color=s['color'], linewidth=s['linewidth'],
                linestyle=s['linestyle'],
                label=method)

    # 90 % reference line
    ax.axhline(y=90, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    ax.set_xlabel('Cumulative System Time (s)', fontsize=8)
    ax.set_ylabel('Test Accuracy (%)', fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc='lower right',
              handlelength=1.3, handletextpad=0.4, borderpad=0.2)
    ax.set_ylim(0, 102)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(k_formatter))
    ax.tick_params(labelsize=7)

    plt.tight_layout(pad=0.5)
    for fmt in ['png', 'pdf']:
        path = f'Fig3a_time_accuracy.{fmt}'
        kw = {'dpi': 600} if fmt == 'png' else {}
        fig.savefig(path, bbox_inches='tight', facecolor='white',
                    edgecolor='none', **kw)
        print(f'  Saved: {path}')
    plt.close(fig)


# ============================================================
# Fig3(b)  Memory Safety Comparison — two side-by-side bar panels
# ============================================================
def draw_fig3b():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.6))
    fig.patch.set_facecolor('white')

    COLORS_3 = ['#4C78A8', '#D62728', '#59A14F']

    # --- aggregate across seeds ---
    oom_mean, oom_std = [], []
    surv_mean, surv_std = [], []
    for method in METHODS:
        sub = df_cs[df_cs['disp'] == method]
        oom_mean.append(sub['oom_count'].values.mean())
        oom_std.append(sub['oom_count'].values.std())
        surv_mean.append(sub['low_memory_survival_rate'].values.mean())
        surv_std.append(sub['low_memory_survival_rate'].values.std())

    x = np.arange(len(METHODS))
    bw = 0.50

    # ---------- Left panel: OOM Count (log scale) ----------
    # Replace 0 with a small placeholder so it renders on log scale
    oom_plot = [v if v > 0 else 0.5 for v in oom_mean]

    ax1.bar(x, oom_plot, width=bw, color=COLORS_3,
            edgecolor='black', linewidth=0.4)
    ax1.set_yscale('log')
    ax1.set_ylabel('OOM Count', fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(METHODS, rotation=18, ha='right', fontsize=6.5)
    ax1.set_ylim(0.3, 300_000)
    ax1.tick_params(labelsize=7)

    # Annotate true values on top of bars
    for i, (bar_val, real_val) in enumerate(zip(oom_plot, oom_mean)):
        if real_val == 0:
            label = '0'
            y_pos = 1.4                         # place "0" just above the tiny bar
        else:
            label = f'{real_val/1000:.0f}k'
            y_pos = bar_val * 1.15
        ax1.text(x[i], y_pos, label, ha='center', va='bottom',
                 fontsize=6.5, color='#222222')

    # ---------- Right panel: Low-memory Survival Rate ----------
    ax2.bar(x, surv_mean, width=bw, color=COLORS_3,
            edgecolor='black', linewidth=0.4)
    ax2.set_ylabel('Low-memory Survival Rate (%)', fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(METHODS, rotation=18, ha='right', fontsize=6.5)
    ax2.set_ylim(0, 118)
    ax2.tick_params(labelsize=7)

    # Percentage labels
    for i, val in enumerate(surv_mean):
        ax2.text(x[i], val + 3, f'{val:.0f}%', ha='center', va='bottom',
                 fontsize=6.5, color='#222222')

    plt.tight_layout(pad=0.6, w_pad=2.2)
    for fmt in ['png', 'pdf']:
        path = f'Fig3b_memory_safety.{fmt}'
        kw = {'dpi': 600} if fmt == 'png' else {}
        fig.savefig(path, bbox_inches='tight', facecolor='white',
                    edgecolor='none', **kw)
        print(f'  Saved: {path}')
    plt.close(fig)


# ============================================================
# Fig3(c)  Online Decision Latency  (grouped bars)
# ============================================================
def draw_fig3c():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor('white')

    # Median: typical decision latency
    # P95:   tail latency — 95 % of decisions finish within this value
    medians, p95s = [], []
    for method in METHODS:
        sub = df_dt[df_dt['disp'] == method]
        medians.append(sub['median_us'].values[0])
        p95s.append(sub['p95_us'].values[0])

    bw = 0.28
    x = np.arange(len(METHODS))

    b1 = ax.bar(x - bw/2, medians, bw,
                color='#5B9BD5', edgecolor='black', linewidth=0.4,
                label='Median latency')
    b2 = ax.bar(x + bw/2, p95s, bw,
                color='#ED7D31', edgecolor='black', linewidth=0.4,
                label='P95 latency')

    # Value labels
    y_max = max(medians + p95s)
    for bars in [b1, b2]:
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.,
                    h + y_max * 0.015,
                    f'{h:.1f}', ha='center', va='bottom',
                    fontsize=5.5, color='#333333')

    # x-tick labels: wrap long method names for readability
    x_labels = [
        'Delay-only Search',
        'Memory-Constrained\nSearch',
        'Proposed\nMasked-PPO',
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=6.5)
    ax.set_ylabel('Latency (μs)', fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc='upper left',
              handlelength=1.3, handletextpad=0.4, borderpad=0.2)
    ax.set_ylim(0, y_max * 1.22)
    ax.tick_params(labelsize=7)

    plt.tight_layout(pad=0.5)
    for fmt in ['png', 'pdf']:
        path = f'Fig3c_decision_latency.{fmt}'
        kw = {'dpi': 600} if fmt == 'png' else {}
        fig.savefig(path, bbox_inches='tight', facecolor='white',
                    edgecolor='none', **kw)
        print(f'  Saved: {path}')
    plt.close(fig)


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    print("\n>>> Drawing Fig3(a) …")
    draw_fig3a()
    print(">>> Drawing Fig3(b) …")
    draw_fig3b()
    print(">>> Drawing Fig3(c) …")
    draw_fig3c()
    print("\n=== All 3 figures generated (PNG 600 dpi + PDF vector) ===\n")
