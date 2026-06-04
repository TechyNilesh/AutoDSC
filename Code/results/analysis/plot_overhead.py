#!/usr/bin/env python3
"""
Per-window runtime decomposition of AutoDSC (MOStream-style bars).

AutoDSC's wall-clock splits into three additive components:
  - Train + predict : incremental training + prediction of all K members
  - Multi-CVI score : computing the multi-CVI window scores for every member
  - PGSSO search    : elite update, exploration-rate adaptation, diversity
                      prune, and challenger generation (the AutoML machinery)
The three costs span ~3 orders of magnitude, so we plot the *absolute*
per-window time on a log axis: training dominates (~1 s), CVI scoring is ~100 ms,
and the AutoML search is ~1 ms --- i.e.\ negligible. This is the readable way to
show that the price of automation is dwarfed by the clustering it automates.

Reads:  extra/overhead.json   (from run_extra_analysis.py)
Output: charts/overhead/{overhead,legend}.pdf/.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import style as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'extra', 'overhead.json')

# datasets -> display name, color, hatch (consistent with the validity figure)
PANEL = {
    'real_pendigits':     (S.DATASET_NAME['real_pendigits'], '#0072B2', '//'),
    'synth_blobs_sudden': (S.DATASET_NAME['synth_blobs_sudden'], '#E69F00', 'xx'),
}
COMPONENTS = [('learn', 'Train +\npredict'),
              ('score', 'Multi-CVI\nscoring'),
              ('search', 'PGSSO\nsearch')]
BAR_KW = dict(edgecolor='black', linewidth=1.0)


def _frame(ax):
    ax.set_facecolor('white')
    ax.figure.patch.set_facecolor('white')
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which='both', linestyle='--', color='0.7',
                  linewidth=0.8, alpha=0.9)
    ax.xaxis.grid(False)
    ax.tick_params(labelsize=12)
    S.bold_ticks(ax)


def _fmt(ms):
    if ms >= 100:
        return f'{ms:.0f}'
    if ms >= 10:
        return f'{ms:.0f}'
    if ms >= 1:
        return f'{ms:.1f}'
    return f'{ms:.2f}'


def main():
    with open(DATA) as f:
        data = json.load(f)
    dsets = [d for d in PANEL if d in data]

    S.set_rc()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    x = np.arange(len(COMPONENTS))
    nb = len(dsets)
    width = 0.8 / nb

    for j, ds in enumerate(dsets):
        d = data[ds]
        w = max(1.0, d.get('windows', 1.0))
        # per-window time in milliseconds for each component
        ms = [1000.0 * d[k] / w for k, _ in COMPONENTS]
        name, color, hatch = PANEL[ds]
        xs = x + (j - (nb - 1) / 2) * width
        ax.bar(xs, ms, width, color=color, hatch=hatch, label=name, **BAR_KW)
        for xi, v in zip(xs, ms):
            ax.text(xi, v * 1.18, _fmt(v), ha='center', va='bottom',
                    fontsize=9.5, fontweight='bold', color=color)

    ax.set_yscale('log')
    ax.set_ylim(0.3, 6000)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in COMPONENTS], fontsize=12, fontweight='bold')
    ax.set_ylabel('Time per window (ms, log)', fontsize=14, fontweight='bold')
    _frame(ax)
    plt.tight_layout()
    S.save_fig(fig, 'overhead', 'overhead')

    # standalone legend (datasets), hatched patches
    handles = [Patch(facecolor=color, hatch=hatch, edgecolor='black',
                     linewidth=1.0, label=name)
               for name, color, hatch in (PANEL[d] for d in dsets)]
    S.save_legend(handles, 'overhead', ncol=len(handles), width=5.5)
    print('Saved charts/overhead/overhead.* and legend.*')


if __name__ == '__main__':
    main()
