#!/usr/bin/env python3
"""
Throughput vs Memory trade-off chart for the 6 canonical methods.

One point per method (mean over all datasets and seeds): x = throughput (log),
y = peak memory (log, to accommodate the wide range). Each point is labelled
in place with the method name and its mean ARI, so the speed/footprint/quality
trade-off is readable without a separate legend. AutoDSC is emphasized (red
star). Names/colors/markers from style.py.

Usage:
    python plot_efficiency.py
Output:
    charts/efficiency.pdf / .png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import style as S

# per-method label placement: (dx, dy) in points, and text anchor
OFFSET = {
    'AutoDSC':      (13, 6, 'left'),
    'ConfStream':   (0, 11, 'center'),
    'OACE':         (0, 11, 'center'),
    'MOStream':     (0, 11, 'center'),
    'RandomSearch': (0, -20, 'center'),
    'Default(Best)': (0, 11, 'center'),
}


def main():
    os.makedirs(S.CHARTS_DIR, exist_ok=True)
    S.set_rc()
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    for m in S.METHODS:
        tp = S.scalar_overall(m, 'throughput_ips')
        mem = S.scalar_overall(m, 'memory_mb')
        is_auto = m == S.EMPHASIS
        ax.scatter(tp, mem, s=520 if is_auto else 260,
                   marker='*' if is_auto else S.METHOD_MARKER[m],
                   color=S.METHOD_COLOR[m], edgecolor='black',
                   linewidth=1.4 if is_auto else 0.9, zorder=5)
        dx, dy, ha = OFFSET[m]
        ax.annotate(S.METHOD_NAME[m], (tp, mem),
                    textcoords='offset points', xytext=(dx, dy), ha=ha,
                    fontsize=12, fontweight='bold', color=S.METHOD_COLOR[m],
                    zorder=6)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Throughput (instances/s) $\uparrow$', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'Memory (MB) $\downarrow$', fontsize=15, fontweight='bold')
    # pad limits so labels are not clipped
    ax.set_xlim(ax.get_xlim()[0] * 0.5, ax.get_xlim()[1] * 2.2)
    ax.set_ylim(ax.get_ylim()[0] * 0.78, ax.get_ylim()[1] * 1.4)
    ax.set_yticks([15, 25, 50, 100, 166])
    ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis='both', labelsize=12)
    S.bold_ticks(ax)
    S.clean_spines(ax)
    ax.grid(axis='both', which='major', alpha=0.18, linewidth=0.6)

    # shade the favorable region: fast (high throughput) + low memory
    xl, yl = ax.get_xlim(), ax.get_ylim()
    rx0, ry1 = 350.0, 32.0   # throughput >= rx0 and memory <= ry1
    ax.add_patch(Rectangle((rx0, yl[0]), xl[1] - rx0, ry1 - yl[0],
                           facecolor='#9ed99a', alpha=0.35, edgecolor='none',
                           zorder=0))

    plt.tight_layout()
    S.save_fig(fig, 'efficiency', 'efficiency')
    print('Saved charts/efficiency/efficiency.pdf / .png')


if __name__ == '__main__':
    main()
