#!/usr/bin/env python3
"""
Ablation chart generator for AutoDSC.

Reads the leave-one-out ablation results from ``results/ablation/`` and
produces a publication-grade grouped bar chart (Okabe-Ito colorblind-safe
palette, hatching for grayscale, vector PDF + PNG) into ``charts/``.

Usage:
    python plot_ablation.py                # all 10 datasets, ARI + CMM
    python plot_ablation.py --regime real  # real-world datasets only
    python plot_ablation.py --metrics ari nmi purity cmm

Output:
    charts/ablation_bar.pdf  (vector, for LaTeX)
    charts/ablation_bar.png  (300 dpi preview)
"""
import json
import glob
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, '..', 'ablation')
CHARTS_DIR = os.path.join(HERE, 'charts')

VARIANTS = ['full', 'no_recency', 'no_adaptive', 'single_cvi', 'no_diversity']
VLABELS = {
    'full': 'AutoDSC (Full)', 'no_recency': 'w/o Recency',
    'no_adaptive': 'w/o Adaptive', 'single_cvi': 'w/o Multi-CVI',
    'no_diversity': 'w/o Diversity',
}
# Okabe-Ito colorblind-safe palette; 'full' = AutoDSC vermilion-red (brand)
COLORS = {
    'full': '#D55E00', 'no_recency': '#0072B2', 'no_adaptive': '#009E73',
    'single_cvi': '#E69F00', 'no_diversity': '#CC79A7',
}
HATCHES = {'full': '', 'no_recency': '//', 'no_adaptive': '\\\\',
           'single_cvi': 'xx', 'no_diversity': '..'}
METRIC_LABEL = {'ari': r'ARI $\uparrow$', 'nmi': r'NMI $\uparrow$',
                'purity': r'Purity $\uparrow$', 'cmm': r'CMM $\uparrow$'}


def load(regime, metrics):
    agg = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(RESULTS_DIR, '*.json')):
        r = json.load(open(f))
        ds = r['dataset']
        if regime == 'real' and not ds.startswith('real_'):
            continue
        if regime == 'synth' and not ds.startswith('synth_'):
            continue
        v = r['model'].replace('AutoDSC_Ablation_', '')
        for k in metrics:
            agg[v][k].append(r.get(k, 0))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', choices=['all', 'real', 'synth'], default='all')
    ap.add_argument('--metrics', nargs='+', default=['ari', 'cmm'])
    ap.add_argument('--out', default='ablation_bar')
    args = ap.parse_args()

    os.makedirs(CHARTS_DIR, exist_ok=True)
    agg = load(args.regime, args.metrics)
    M = {v: {k: np.mean(agg[v][k]) for k in args.metrics} for v in VARIANTS}
    ratio = {v: {k: min(M[v][k] / M['full'][k], 1.0) if M['full'][k] > 0 else 1.0
                 for k in args.metrics} for v in VARIANTS}

    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'DejaVu Sans',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    x = np.arange(len(args.metrics)); w = 0.16
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    allr = [ratio[v][k] for v in VARIANTS for k in args.metrics]
    lo = min(allr); ymin = lo - (1 - lo) * 0.45 - 0.002
    for i, v in enumerate(VARIANTS):
        vals = [ratio[v][k] for k in args.metrics]
        ax.bar(x + (i - 2) * w, vals, w, label=VLABELS[v], color=COLORS[v],
               edgecolor='black', linewidth=0.7, hatch=HATCHES[v])
    ax.axhline(1.0, color='black', linewidth=1.0, linestyle='--', alpha=0.7, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABEL.get(k, k.upper()) for k in args.metrics],
                       fontsize=13, fontweight='bold')
    ax.set_ylim(ymin, 1.003)
    ax.set_ylabel('Relative Score', fontsize=13, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.grid(axis='y', alpha=0.3, linewidth=0.6)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()

    out_dir = os.path.join(CHARTS_DIR, 'ablation')
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, f'{args.out}.pdf')
    png = os.path.join(out_dir, f'{args.out}.png')
    plt.savefig(pdf, bbox_inches='tight')
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {pdf}\nSaved {png}')

    # standalone shared legend (variants)
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=COLORS[v], edgecolor='black', hatch=HATCHES[v],
                 label=VLABELS[v]) for v in VARIANTS]
    lf = plt.figure(figsize=(max(4.0, 1.7 * len(VARIANTS)), 0.5))
    lf.legend(handles=leg, loc='center', ncol=len(VARIANTS), frameon=False,
              columnspacing=1.0, handletextpad=0.4, prop={'weight': 'bold', 'size': 10})
    for ext in ('pdf', 'png'):
        lf.savefig(os.path.join(out_dir, f'legend.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(lf)
    print(f'Saved {out_dir}/legend.pdf / .png')


if __name__ == '__main__':
    main()
