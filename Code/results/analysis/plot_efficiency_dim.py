#!/usr/bin/env python3
"""
Combined efficiency figure (two panels, one row, no sub-captions):

  Left  -- Throughput vs Memory trade-off for the 6 canonical methods
           (log--log; shaded green = favorable region: fast + low memory).
  Right -- Efficiency vs feature dimensionality: each dataset is a point in the
           throughput--memory plane, colored/shaped by its feature-count range.

Names/colors/markers from style.py. Output: charts/efficiency_dim.pdf / .png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import style as S

# ---- left panel (methods) label placement: (dx, dy in pts, anchor) ----
OFFSET = {
    'AutoDSC':      (13, 6, 'left'),
    'ConfStream':   (0, 11, 'center'),
    'OACE':         (0, 11, 'center'),
    'MOStream':     (0, 11, 'center'),
    'RandomSearch': (0, -20, 'center'),
    'Default(Best)': (0, 11, 'center'),
}

# ---- right panel (dimensionality) ----
DIMS = {'real_insects': 33, 'real_kdd99': 38, 'real_pendigits': 16,
        'real_shuttle': 9, 'real_vehicle': 100,
        'synth_blobs_expanding': 5, 'synth_blobs_gradual': 5,
        'synth_blobs_merge_split': 5, 'synth_blobs_sudden': 5, 'synth_rbf_fast': 5}
PHASES = [(1, 25, '#0072B2', 'o', r'$\leq$25'),
          (26, 50, '#E69F00', 's', '26-50'),
          (51, 75, '#009E73', '^', '51-75'),
          (76, 100, '#D55E00', 'D', '76-100')]


def phase_of(dim):
    for lo, hi, color, marker, label in PHASES:
        if lo <= dim <= hi:
            return color, marker, label
    return '#999999', 'o', '?'


def eff_panel(ax):
    for m in S.METHODS:
        tp = S.scalar_overall(m, 'throughput_ips')
        mem = S.scalar_overall(m, 'memory_mb')
        is_auto = m == S.EMPHASIS
        ax.scatter(tp, mem, s=340 if is_auto else 150,
                   marker='*' if is_auto else S.METHOD_MARKER[m],
                   color=S.METHOD_COLOR[m], edgecolor='black',
                   linewidth=1.2 if is_auto else 0.8, zorder=5)
        dx, dy, ha = OFFSET[m]
        ax.annotate(S.METHOD_NAME[m], (tp, mem), textcoords='offset points',
                    xytext=(dx, dy), ha=ha, fontsize=8.5, fontweight='bold',
                    color=S.METHOD_COLOR[m], zorder=6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Throughput (ips) $\uparrow$', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Memory (MB) $\downarrow$', fontsize=13, fontweight='bold')
    ax.set_xlim(ax.get_xlim()[0] * 0.45, ax.get_xlim()[1] * 3.0)
    ax.set_ylim(ax.get_ylim()[0] * 0.7, ax.get_ylim()[1] * 1.6)
    ax.set_yticks([15, 25, 50, 100, 166])
    ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis='both', labelsize=10)
    S.bold_ticks(ax)
    S.clean_spines(ax)
    ax.grid(axis='both', which='major', alpha=0.18, linewidth=0.6)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.add_patch(Rectangle((350.0, yl[0]), xl[1] - 350.0, 32.0 - yl[0],
                           facecolor='#9ed99a', alpha=0.35, edgecolor='none', zorder=0))


def dim_panel(ax):
    seen = set()
    for ds, dim in DIMS.items():
        tp = S.scalar('AutoDSC', ds, 'throughput_ips')
        mem = S.scalar('AutoDSC', ds, 'memory_mb')
        if np.isnan(tp):
            continue
        color, marker, label = phase_of(dim)
        ax.scatter(tp, mem, s=130, color=color, marker=marker, edgecolor='black',
                   linewidth=0.9, zorder=3, label=label if label not in seen else None)
        seen.add(label)
    ax.set_xlabel(r'Throughput (ips) $\uparrow$', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Memory (MB) $\downarrow$', fontsize=13, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)
    S.bold_ticks(ax)
    S.clean_spines(ax)
    handles, labels = ax.get_legend_handles_labels()
    order = [l for _, _, _, _, l in PHASES if l in labels]
    hd = {l: h for h, l in zip(handles, labels)}
    ax.legend([hd[l] for l in order], order, loc='upper center',
              bbox_to_anchor=(0.5, 1.16), ncol=len(order), frameon=False,
              prop={'weight': 'bold', 'size': 8.5}, columnspacing=0.8,
              handletextpad=0.25, title=None)


def main():
    os.makedirs(S.CHARTS_DIR, exist_ok=True)
    S.set_rc()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.7))
    eff_panel(axL)
    dim_panel(axR)
    for ax in (axL, axR):
        ax.set_box_aspect(1)   # square panels, like the benchmark chart
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(S.CHARTS_DIR, f'efficiency_dim.{ext}'),
                    dpi=300, bbox_inches='tight')
    print('Saved charts/efficiency_dim.pdf / .png')


if __name__ == '__main__':
    main()
