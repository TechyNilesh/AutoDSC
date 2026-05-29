#!/usr/bin/env python3
"""
Staged model-comparison line charts (EDS-style).

The stream is divided into N equal stages; per stage we plot the seed-averaged
mean score for each of the 6 canonical methods (Default(Best) summarizes the
fixed algorithms). One panel per (dataset, metric) in a single row, with a
shared bold legend on top. Names/colors/markers/order come from style.py.

Usage:
    python plot_benchmark_stages.py
    python plot_benchmark_stages.py --datasets real_pendigits synth_blobs_expanding \
        --metrics ari cmm --stages 5
Output:
    charts/benchmark_stages.pdf / .png
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import style as S


def stage_means(method, dataset, metric, stages):
    c = S.curve(method, dataset, metric)
    if c is None:
        return None
    return np.array([chunk.mean() for chunk in np.array_split(c, stages)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['real_pendigits', 'synth_blobs_expanding'])
    ap.add_argument('--metrics', nargs='+', default=['ari', 'cmm'])
    ap.add_argument('--stages', type=int, default=5)
    ap.add_argument('--full', action='store_true',
                    help='use a fixed [0,1] y-axis instead of per-panel zoom')
    ap.add_argument('--out', default='benchmark_stages')
    args = ap.parse_args()
    os.makedirs(S.CHARTS_DIR, exist_ok=True)

    panels = [(ds, m) for ds in args.datasets for m in args.metrics]
    x = np.arange(1, args.stages + 1)
    S.set_rc()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 3.4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (ds, metric) in zip(axes, panels):
        panel_vals = []
        for m in S.METHODS:
            y = stage_means(m, ds, metric, args.stages)
            if y is None:
                continue
            panel_vals.append(y)
            ax.plot(x, y, '-', marker=S.METHOD_MARKER[m], markersize=S.ms(m),
                    linewidth=S.lw(m), color=S.METHOD_COLOR[m], label=S.METHOD_NAME[m],
                    markeredgecolor='black', markeredgewidth=0.6,
                    zorder=(6 if m == S.EMPHASIS else 3))
        ax.set_xlabel(f'Stages of {S.DATASET_NAME.get(ds, ds)} Stream',
                      fontsize=14, fontweight='bold')
        ax.set_ylabel(S.METRIC_LABEL[metric], fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        if not args.full and panel_vals:
            allv = np.concatenate(panel_vals)
            lo, hi = allv.min(), allv.max()
            pad = (hi - lo) * 0.10 + 1e-3
            ax.set_ylim(max(0.0, lo - pad), min(1.02, hi + pad))
        else:
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        ax.tick_params(axis='both', labelsize=12)
        S.bold_ticks(ax)
        S.clean_spines(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(S.METHODS),
               frameon=False, bbox_to_anchor=(0.5, 1.12), columnspacing=1.0,
               handletextpad=0.4, prop={'weight': 'bold', 'size': 13})
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(S.CHARTS_DIR, f'{args.out}.{ext}'),
                    dpi=300, bbox_inches='tight')
    print(f'Saved charts/{args.out}.pdf / .png')


if __name__ == '__main__':
    main()
