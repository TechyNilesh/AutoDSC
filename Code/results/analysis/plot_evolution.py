#!/usr/bin/env python3
"""
Adaptive algorithm evolution heatmap for AutoDSC.

Rows = population members (learners), columns = stream windows. Each cell is
colored by the base algorithm occupying that population slot at that window,
showing how AutoDSC's population composition evolves as the stream drifts.
Within each window the population is sorted by algorithm so the composition
reads as coherent colored bands (population membership has no fixed identity
across windows). A colorbar legend maps colors to algorithms.

Reads evolution_capture.json (from capture_evolution.py on the server).

Usage:
    python plot_evolution.py --dataset real_insects
Output:
    charts/evolution.pdf / .png
"""
import json
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE = os.path.join(HERE, 'evolution_capture.json')
CHARTS_DIR = os.path.join(HERE, 'charts')

# fixed algorithm order + colors (canonical single-algorithm palette, see CLAUDE.md)
ALGOS = ['CluStream', 'DenStream', 'DBSTREAM', 'StreamKMeans']
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
CODE = {a: i for i, a in enumerate(ALGOS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='real_insects')
    ap.add_argument('--out', default='evolution')
    args = ap.parse_args()
    os.makedirs(CHARTS_DIR, exist_ok=True)

    rec = json.load(open(CAPTURE))[args.dataset]['record']
    nwin = len(rec)
    maxrows = max(len(r['population']) for r in rec)

    # build matrix [row, window] = algorithm code (sorted per column), -1 = empty
    M = np.full((maxrows, nwin), -1, dtype=float)
    for w, r in enumerate(rec):
        codes = sorted(CODE.get(a, 0) for a in r['population'])
        for row, c in enumerate(codes):
            M[row, w] = c
    M = np.ma.masked_less(M, 0)

    # x edges in #instances (thousands): the per-window 'instances' field is the
    # cumulative count at each window's right edge, so use it directly.
    cum_inst = np.array([r.get('instances', (w + 1) * 1000)
                         for w, r in enumerate(rec)], dtype=float)
    xedges = np.concatenate([[0.0], cum_inst]) / 1000.0

    cmap = ListedColormap(COLORS)
    plt.rcParams.update({'font.size': 13, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.pcolormesh(xedges, np.arange(maxrows + 1), M,
                  cmap=cmap, vmin=0, vmax=len(ALGOS) - 1,
                  edgecolors='white', linewidth=0.4)

    ax.set_xlabel('#Instances (in thousands)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Population', fontsize=15, fontweight='bold')
    xt = np.arange(0, xedges[-1] + 1, 10)
    ax.set_xticks(xt)
    ax.set_xticklabels([str(int(t)) for t in xt])
    ax.set_yticks(np.arange(maxrows) + 0.5)
    ax.set_yticklabels([str(i + 1) for i in range(maxrows)], fontsize=10)
    ax.tick_params(axis='both', labelsize=11)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')

    handles = [Patch(facecolor=COLORS[i], edgecolor='black', label=a)
               for i, a in enumerate(ALGOS)]

    plt.tight_layout()
    out_dir = os.path.join(CHARTS_DIR, 'evolution')
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(out_dir, f'{args.out}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved charts/evolution/{args.out}.pdf / .png  ({maxrows} learners x {nwin} windows)')

    # standalone shared legend (base algorithms)
    lf = plt.figure(figsize=(max(4.0, 1.7 * len(ALGOS)), 0.5))
    lf.legend(handles=handles, loc='center', ncol=len(ALGOS), frameon=False,
              columnspacing=1.4, handletextpad=0.5, prop={'weight': 'bold', 'size': 12})
    for ext in ('pdf', 'png'):
        lf.savefig(os.path.join(out_dir, f'legend.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(lf)
    print(f'Saved {out_dir}/legend.pdf / .png')


if __name__ == '__main__':
    main()
