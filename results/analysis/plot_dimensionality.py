#!/usr/bin/env python3
"""
Dimensionality scalability of AutoDSC (from existing benchmark results).

Each dataset is a point in the throughput--memory plane, colored by its
dimensionality phase (binned into 25-feature ranges). This shows how feature
count relates to AutoDSC's efficiency profile: low-dimensional streams sit in
the low-memory region, while higher-dimensional streams use more memory yet
keep throughput in the same band---i.e.\ efficiency scales gracefully with
dimensionality (no curse-of-dimensionality blow-up).

Usage:
    python plot_dimensionality.py
Output:
    charts/dimensionality.pdf / .png
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

DIMS = {
    'real_insects': 33, 'real_kdd99': 38, 'real_pendigits': 16,
    'real_shuttle': 9, 'real_vehicle': 100,
    'synth_blobs_expanding': 5, 'synth_blobs_gradual': 5,
    'synth_blobs_merge_split': 5, 'synth_blobs_sudden': 5, 'synth_rbf_fast': 5,
}
# dimensionality phases (25-wide bins) -> color
PHASES = [(1, 25, '#0072B2', 'o', r'$\leq$25'),
          (26, 50, '#E69F00', 's', '26-50'),
          (51, 75, '#009E73', '^', '51-75'),
          (76, 100, '#D55E00', 'D', '76-100')]


def phase_of(dim):
    for lo, hi, color, marker, label in PHASES:
        if lo <= dim <= hi:
            return color, marker, label
    return '#999999', 'o', '?'


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    agg = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(RESULTS_DIR, 'AutoDSC_*.json')):
        r = json.load(open(f))
        if r['dataset'] not in DIMS:
            continue
        for k in ['throughput_ips', 'memory_mb']:
            agg[r['dataset']][k].append(r.get(k, 0))

    plt.rcParams.update({'font.size': 13, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    seen = set()
    for ds, dim in DIMS.items():
        if not agg[ds]['throughput_ips']:
            continue
        x = np.mean(agg[ds]['throughput_ips'])
        y = np.mean(agg[ds]['memory_mb'])
        color, marker, label = phase_of(dim)
        ax.scatter(x, y, s=240, color=color, marker=marker, edgecolor='black',
                   linewidth=1.0, zorder=3, label=label if label not in seen else None)
        seen.add(label)

    ax.set_xlabel(r'Throughput (ips) $\uparrow$', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'Memory (MB) $\downarrow$', fontsize=15, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # legend ordered by phase
    handles, labels = ax.get_legend_handles_labels()
    order = [l for _, _, _, _, l in PHASES if l in labels]
    hd = {l: h for h, l in zip(handles, labels)}
    ax.legend([hd[l] for l in order], order, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(order),
              frameon=False, prop={'weight': 'bold', 'size': 11},
              columnspacing=1.2, handletextpad=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(CHARTS_DIR, f'dimensionality.{ext}'),
                    dpi=300, bbox_inches='tight')
    print('Saved charts/dimensionality.pdf / .png')


if __name__ == '__main__':
    main()
