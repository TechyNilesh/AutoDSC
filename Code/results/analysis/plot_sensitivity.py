#!/usr/bin/env python3
"""
Sensitivity analysis dual-axis line charts for AutoDSC.

One panel per swept parameter (window_size, population_size). Each panel is a
dual-axis line chart: a quality metric (default ARI) on the left y-axis and
Throughput on the right y-axis, versus the parameter value on the x-axis.
Big bold fonts, distinct colors/markers per axis, vector PDF output.

NOTE: reads whatever is in ../sensitivity/. Re-run the sensitivity sweep with
the current code and this script regenerates automatically.

Usage:
    python plot_sensitivity.py                 # ARI (left) + Throughput (right)
    python plot_sensitivity.py --quality cmm   # use CMM as the quality metric
Output:
    charts/sensitivity.pdf / .png
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
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, '..', 'sensitivity')
CHARTS_DIR = os.path.join(HERE, 'charts')

PARAMS = [('window_size', 'Window Size'), ('population_size', 'Population Size')]
Q_COLOR, T_COLOR = '#0072B2', '#E69F00'   # quality = blue, throughput = orange
QLABEL = {'ari': 'ARI', 'cmm': 'CMM', 'nmi': 'NMI', 'purity': 'Purity'}


def collect(param, quality):
    """Return sorted values and {value:{quality, throughput}} over all data."""
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
    ap.add_argument('--out', default='sensitivity')
    args = ap.parse_args()
    os.makedirs(CHARTS_DIR, exist_ok=True)

    plt.rcParams.update({'font.size': 13, 'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(5.0 * len(PARAMS), 3.6))
    if len(PARAMS) == 1:
        axes = [axes]

    for ax, (param, pname) in zip(axes, PARAMS):
        vals, q, t = collect(param, args.quality)
        if not vals:
            continue
        x = np.arange(len(vals))

        # left axis: quality
        l1, = ax.plot(x, q, '-', marker='^', markersize=11, linewidth=2.6,
                      color=Q_COLOR, markeredgecolor='black', markeredgewidth=0.6)
        ax.set_xlabel(pname, fontsize=15, fontweight='bold')
        ax.set_ylabel(QLABEL[args.quality], fontsize=15, fontweight='bold', color=Q_COLOR)
        ax.tick_params(axis='y', labelcolor=Q_COLOR, labelsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(vals)

        # right axis: throughput
        ax2 = ax.twinx()
        l2, = ax2.plot(x, t, '-', marker='s', markersize=10, linewidth=2.6,
                       color=T_COLOR, markeredgecolor='black', markeredgewidth=0.6)
        ax2.set_ylabel('Throughput (ips)', fontsize=15, fontweight='bold', color=T_COLOR)
        ax2.tick_params(axis='y', labelcolor=T_COLOR, labelsize=11)

        for lbl in (ax.get_xticklabels() + ax.get_yticklabels()
                    + ax2.get_yticklabels()):
            lbl.set_fontweight('bold')
        # no in-panel legend (colored y-axes identify the two lines)
        # no top title (bottom x-label already names the parameter)
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

    plt.tight_layout()
    pdf = os.path.join(CHARTS_DIR, f'{args.out}.pdf')
    png = os.path.join(CHARTS_DIR, f'{args.out}.png')
    plt.savefig(pdf, bbox_inches='tight')
    plt.savefig(png, dpi=300, bbox_inches='tight')
    print(f'Saved {pdf}\nSaved {png}')


if __name__ == '__main__':
    main()
