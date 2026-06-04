#!/usr/bin/env python3
"""
Staged model-comparison line charts (EDS-style), as individual panels.

The stream is divided into N equal stages; per stage we plot the seed-averaged
mean score for each of the 6 canonical methods (Default(Best) summarizes the
fixed algorithms). Each (dataset, metric) panel is saved as its own vector PDF
under charts/benchmark/, together with a single shared legend.pdf that is
placed above the subfigures in the paper. Names/colors/markers/order come from
style.py.

Usage:
    python plot_benchmark_stages.py
    python plot_benchmark_stages.py --datasets real_pendigits synth_blobs_expanding \
        --metrics ari cmm --stages 5
Output:
    charts/benchmark/<dataset>_<metric>.pdf / .png  (one per panel)
    charts/benchmark/legend.pdf / .png              (shared legend)
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import style as S

GROUP = 'benchmark'


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
    args = ap.parse_args()
    S.set_rc()

    panels = [(ds, m) for ds in args.datasets for m in args.metrics]

    for ds, metric in panels:
        fig, ax = plt.subplots(figsize=(3.8, 3.4))
        x = S.stage_thousands(ds, args.stages)   # cumulative #instances (thousands)
        panel_vals = []
        for m in S.METHODS:
            y = stage_means(m, ds, metric, args.stages)
            if y is None:
                continue
            panel_vals.append(y)
            ax.plot(x, y, '-', marker=S.METHOD_MARKER[m], markersize=S.ms(m),
                    linewidth=S.lw(m), color=S.METHOD_COLOR[m],
                    label=S.METHOD_NAME[m], markeredgecolor='black',
                    markeredgewidth=0.6, zorder=(6 if m == S.EMPHASIS else 3))
        ax.set_xlabel(S.STREAM_XLABEL, fontsize=14, fontweight='bold')
        ax.set_ylabel(S.METRIC_LABEL[metric], fontsize=15, fontweight='bold')
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
        fig.tight_layout()
        name = f'{S.DATASET_NAME.get(ds, ds).lower().replace("-", "")}_{metric}'
        S.save_fig(fig, GROUP, name)
        print(f'Saved charts/{GROUP}/{name}.pdf / .png')

    # shared legend for all panels (all six canonical methods)
    S.save_legend(S.method_handles(S.METHODS), GROUP, 'legend',
                  ncol=len(S.METHODS))
    print(f'Saved charts/{GROUP}/legend.pdf / .png')


if __name__ == '__main__':
    main()
