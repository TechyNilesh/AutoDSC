#!/usr/bin/env python3
"""
Sensitivity analysis dual-axis line charts for AutoDSC, as individual panels.

One panel per swept parameter (window_size, population_size). Each panel is a
dual-axis line chart: a quality metric (default ARI) on the left y-axis and
Throughput on the right y-axis, versus the parameter value on the x-axis. Each
panel is saved as its own vector PDF under charts/sensitivity/, with a single
shared legend.pdf (ARI vs Throughput) placed above them.

Usage:
    python plot_sensitivity.py                 # ARI (left) + Throughput (right)
    python plot_sensitivity.py --quality cmm   # use CMM as the quality metric
Output:
    charts/sensitivity/window_size.pdf / .png
    charts/sensitivity/population_size.pdf / .png
    charts/sensitivity/legend.pdf / .png   (shared legend)
"""
import json
import glob
import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
import style as S

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, '..', 'sensitivity')
GROUP = 'sensitivity'

PARAMS = [('window_size', 'Window Size'), ('population_size', 'Population Size')]
Q_COLOR, T_COLOR = '#0072B2', '#E69F00'   # quality = blue, throughput = orange
QLABEL = {'ari': 'ARI', 'cmm': 'CMM', 'nmi': 'NMI', 'purity': 'Purity'}


def collect(param, quality):
    agg = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(RESULTS_DIR, f'*{param}-*.json')):
        m = re.search(rf'{param}-(\d+)', os.path.basename(f))
        if not m:
            continue
        val = int(m.group(1))
        r = json.load(open(f))
        agg[val]['q'].append(r.get(quality, 0))
        agg[val]['t'].append(r.get('throughput_ips', 0))
    vals = sorted(agg)
    q = [np.mean(agg[v]['q']) for v in vals]
    t = [np.mean(agg[v]['t']) for v in vals]
    return vals, q, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quality', default='ari', choices=['ari', 'cmm', 'nmi', 'purity'])
    args = ap.parse_args()
    S.set_rc()

    for param, pname in PARAMS:
        vals, q, t = collect(param, args.quality)
        if not vals:
            continue
        x = np.arange(len(vals))
        fig, ax = plt.subplots(figsize=(5.0, 3.6))

        ax.plot(x, q, '-', marker='^', markersize=11, linewidth=2.6,
                color=Q_COLOR, markeredgecolor='black', markeredgewidth=0.6)
        ax.set_xlabel(pname, fontsize=15, fontweight='bold')
        ax.set_ylabel(QLABEL[args.quality], fontsize=15, fontweight='bold', color=Q_COLOR)
        ax.tick_params(axis='y', labelcolor=Q_COLOR, labelsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(vals)

        ax2 = ax.twinx()
        ax2.plot(x, t, '-', marker='s', markersize=10, linewidth=2.6,
                 color=T_COLOR, markeredgecolor='black', markeredgewidth=0.6)
        ax2.set_ylabel('Throughput (ips)', fontsize=15, fontweight='bold', color=T_COLOR)
        ax2.tick_params(axis='y', labelcolor=T_COLOR, labelsize=11)

        for lbl in (ax.get_xticklabels() + ax.get_yticklabels()
                    + ax2.get_yticklabels()):
            lbl.set_fontweight('bold')
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        fig.tight_layout()
        S.save_fig(fig, GROUP, param)
        print(f'Saved charts/{GROUP}/{param}.pdf / .png')

    # shared legend identifying the two lines (same on both panels)
    handles = [
        Line2D([0], [0], color=Q_COLOR, marker='^', linestyle='-', linewidth=2.6,
               markersize=9, markeredgecolor='black', markeredgewidth=0.5,
               label=f'{QLABEL[args.quality]} (left axis)'),
        Line2D([0], [0], color=T_COLOR, marker='s', linestyle='-', linewidth=2.6,
               markersize=9, markeredgecolor='black', markeredgewidth=0.5,
               label='Throughput (right axis)'),
    ]
    S.save_legend(handles, GROUP, 'legend', ncol=2, width=6.0)
    print(f'Saved charts/{GROUP}/legend.pdf / .png')


if __name__ == '__main__':
    main()
