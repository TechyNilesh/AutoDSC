#!/usr/bin/env python3
"""
Dimensionality scalability of AutoDSC (MOStream Fig.-4 style).

A controlled 5-cluster blob stream (sudden drift) is generated at increasing
feature dimensionality. We plot AutoDSC's clustering quality (ARI, left axis)
and throughput (instances/s, right axis) as dimensionality grows. Quality stays
stable while throughput degrades only gracefully with d (distance computations
scale with dimension) --- i.e.\ no curse-of-dimensionality collapse.

Reads:  extra/dim_sweep.json   (from run_extra_analysis.py)
Output: charts/dimscaling/dim_sweep.pdf/.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'extra', 'dim_sweep.json')

ARI_C = S.METHOD_COLOR['AutoDSC']     # vermilion (quality, ours)
THR_C = '#0072B2'                     # blue (throughput)


def main():
    with open(DATA) as f:
        d = json.load(f)
    dims = d['dims']
    R = d['results']
    ari = np.array([R[str(x)]['ari'] for x in dims])
    ari_s = np.array([R[str(x)]['ari_std'] for x in dims])
    thr = np.array([R[str(x)]['throughput'] for x in dims])
    thr_s = np.array([R[str(x)]['throughput_std'] for x in dims])

    S.set_rc()
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # left: ARI
    ln1 = ax.plot(dims, ari, color=ARI_C, marker='o', markersize=9,
                  linewidth=S.lw('AutoDSC'), markeredgecolor='black',
                  markeredgewidth=0.6, label=r'ARI $\uparrow$', zorder=4)
    ax.fill_between(dims, ari - ari_s, ari + ari_s, color=ARI_C, alpha=0.15, zorder=1)
    ax.set_xlabel('Dimensionality (features)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'ARI $\uparrow$', fontsize=14, fontweight='bold', color=ARI_C)
    ax.set_ylim(0.4, 1.02)
    ax.tick_params(axis='y', labelcolor=ARI_C, labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    ax.set_xticks(dims)

    # right: throughput
    ax2 = ax.twinx()
    ln2 = ax2.plot(dims, thr, color=THR_C, marker='s', markersize=8,
                   linewidth=2.2, linestyle='--', markeredgecolor='black',
                   markeredgewidth=0.6, label='Throughput', zorder=3)
    ax2.fill_between(dims, thr - thr_s, thr + thr_s, color=THR_C, alpha=0.12, zorder=1)
    ax2.set_ylabel('Throughput (instances/s)', fontsize=14, fontweight='bold', color=THR_C)
    ax2.tick_params(axis='y', labelcolor=THR_C, labelsize=11)
    ax2.set_ylim(0, max(thr + thr_s) * 1.15)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.grid(axis='x', alpha=0.15, linewidth=0.6)

    # self-contained in-plot legend (both axes)
    lines = ln1 + ln2
    ax.legend(lines, [l.get_label() for l in lines], loc='lower center',
              ncol=2, frameon=False, prop={'weight': 'bold', 'size': 11},
              handletextpad=0.4, columnspacing=1.4)

    plt.tight_layout()
    S.save_fig(fig, 'dimscaling', 'dim_sweep')

    # standalone legend (separate PDF)
    handles = [
        Line2D([0], [0], color=ARI_C, marker='o', markersize=9,
               linewidth=S.lw('AutoDSC'), markeredgecolor='black',
               markeredgewidth=0.6, label=r'ARI $\uparrow$'),
        Line2D([0], [0], color=THR_C, marker='s', markersize=8, linestyle='--',
               linewidth=2.2, markeredgecolor='black', markeredgewidth=0.6,
               label='Throughput'),
    ]
    S.save_legend(handles, 'dimscaling', ncol=2, width=6.0)
    print('Saved charts/dimscaling/dim_sweep.* and legend.*')


if __name__ == '__main__':
    main()
