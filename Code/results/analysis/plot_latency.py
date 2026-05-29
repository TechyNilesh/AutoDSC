#!/usr/bin/env python3
"""
Latency-over-stream chart: AutoDSC per-point latency stays bounded.

Addresses the real-time-viability concern: does AutoDSC's AutoML overhead make
per-point latency grow as the stream lengthens? For each dataset we recover the
per-window runtime (difference of the cumulative window_metric.runtimes),
convert to microseconds per point, seed-average, and plot over stream stages.
Flat/non-growing lines show latency is bounded by the fixed population size.

Usage:
    python plot_latency.py
Output:
    charts/latency.pdf / .png
"""
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, '..', 'benchmark')
CHARTS_DIR = os.path.join(HERE, 'charts')

# 2 real + 2 synthetic datasets (canonical display names, see CLAUDE.md)
DATASETS = [
    ('real_kdd99', 'KDD99', '#0072B2', '-', 'o'),
    ('real_insects', 'Insects', '#D55E00', '--', 's'),
    ('synth_blobs_sudden', 'Sudden', '#CC79A7', '-.', 'D'),
    ('synth_rbf_fast', 'Fast-RBF', '#009E73', ':', 'v'),
]


def per_window_latency(dataset, stages):
    """Seed-averaged per-point latency (us) binned into `stages` stream stages."""
    series = []
    for f in glob.glob(os.path.join(RESULTS_DIR, f'AutoDSC_{dataset}_*.json')):
        wm = json.load(open(f))['window_metric']
        rt, ws = wm.get('runtimes'), wm.get('window_size')
        if not rt:
            continue
        rt = np.asarray(rt, dtype=float)
        per = np.diff(np.concatenate([[0.0], rt]))   # per-window seconds
        lat = per / ws * 1e6                          # us per point
        series.append(lat)
    if not series:
        return None, None
    n = min(len(s) for s in series)
    mean = np.vstack([s[:n] for s in series]).mean(axis=0)
    binned = np.array([c.mean() for c in np.array_split(mean, stages)])
    x = np.arange(1, stages + 1)  # stage index
    return x, binned


def main():
    stages = 12
    os.makedirs(CHARTS_DIR, exist_ok=True)
    plt.rcParams.update({'font.size': 13, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    for ds, name, color, ls, mk in DATASETS:
        x, lat = per_window_latency(ds, stages)
        if x is None:
            continue
        ax.plot(x, lat, linestyle=ls, marker=mk, markersize=7, linewidth=2.2,
                color=color, markeredgecolor='black', markeredgewidth=0.5,
                label=name)

    ax.set_xlabel('Stages of Stream', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'Latency ($\mu$s / point) $\downarrow$', fontsize=15, fontweight='bold')
    ax.set_xticks(range(1, stages + 1))
    ax.tick_params(axis='both', labelsize=12)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(DATASETS),
               frameon=False, bbox_to_anchor=(0.5, 1.05), columnspacing=1.0,
               handletextpad=0.4, prop={'weight': 'bold', 'size': 11})
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(CHARTS_DIR, f'latency.{ext}'),
                    dpi=300, bbox_inches='tight')
    print('Saved charts/latency.pdf / .png')


if __name__ == '__main__':
    main()
