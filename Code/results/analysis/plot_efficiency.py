#!/usr/bin/env python3
"""
Throughput vs Memory trade-off chart for AutoDSC vs baselines.

A single scatter: one point per method (mean over all datasets and seeds),
x = throughput (log scale), y = peak memory. Shows the speed/footprint
trade-off; quality (ARI/CMM) is reported separately. AutoDSC is emphasized
(black star). Colors match the benchmark-stages chart. Bold fonts, vector PDF.

Usage:
    python plot_efficiency.py
Output:
    charts/efficiency.pdf / .png
"""
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, '..', 'benchmark')
CHARTS_DIR = os.path.join(HERE, 'charts')

MODELS = ['AutoDSC', 'ConfStream', 'OnlineAutoClustEnsemble', 'RandomSearchStream',
          'MOStream', 'CluStream', 'DenStream', 'DBSTREAM', 'STREAMKMeans']
MLABELS = {
    'AutoDSC': 'AutoDSC', 'ConfStream': 'ConfStream',
    'OnlineAutoClustEnsemble': 'OAClEns', 'RandomSearchStream': 'RandSearch',
    'MOStream': 'MOStream', 'CluStream': 'CluStream', 'DenStream': 'DenStream',
    'DBSTREAM': 'DBSTREAM', 'STREAMKMeans': 'STREAMKMeans',
}
COLOR = {
    'AutoDSC': '#000000', 'ConfStream': '#0072B2', 'OnlineAutoClustEnsemble': '#E69F00',
    'RandomSearchStream': '#009E73', 'MOStream': '#D55E00', 'CluStream': '#CC79A7',
    'DenStream': '#56B4E9', 'DBSTREAM': '#F0E442', 'STREAMKMeans': '#999999',
}


def aggregate():
    agg = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(RESULTS_DIR, '*.json')):
        r = json.load(open(f))
        if r.get('model') not in MODELS:
            continue
        for k in ['throughput_ips', 'memory_mb']:
            agg[r['model']][k].append(r.get(k, 0))
    return {m: {k: np.mean(agg[m][k]) for k in ['throughput_ips', 'memory_mb']}
            for m in MODELS}


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    M = aggregate()
    plt.rcParams.update({'font.size': 13, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for m in MODELS:
        is_auto = m == 'AutoDSC'
        ax.scatter(M[m]['throughput_ips'], M[m]['memory_mb'],
                   s=380 if is_auto else 200,
                   marker='*' if is_auto else 'o',
                   color=COLOR[m], edgecolor='black',
                   linewidth=1.2 if is_auto else 0.8,
                   label=MLABELS[m], zorder=(6 if is_auto else 3))

    ax.set_xscale('log')
    ax.set_xlabel(r'Throughput (ips) $\uparrow$', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'Memory (MB) $\downarrow$', fontsize=15, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # top legend, two rows, matching the benchmark-stages chart style/font
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5,
               frameon=False, bbox_to_anchor=(0.5, 1.08),
               columnspacing=1.0, handletextpad=0.4,
               prop={'weight': 'bold', 'size': 13})

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(CHARTS_DIR, f'efficiency.{ext}'),
                    dpi=300, bbox_inches='tight')
    print('Saved charts/efficiency.pdf / .png')


if __name__ == '__main__':
    main()
