#!/usr/bin/env python3
"""
Staged model-comparison line charts (EDS-style).

The stream is divided into N equal stages; per stage we plot the seed-averaged
mean score for each method. Big bold markers, large axis fonts, y in [0,1] --
matching the reference EDS_CMM style. One panel per (dataset, metric), laid out
in a single row, with a shared bold legend on top.

Usage:
    python plot_benchmark_stages.py
    python plot_benchmark_stages.py --datasets real_pendigits synth_blobs_expanding \
        --metrics ari cmm --stages 5

Output:
    charts/benchmark_stages.pdf / .png
"""
import json
import glob
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, '..', 'benchmark')
CHARTS_DIR = os.path.join(HERE, 'charts')

MODELS = ['AutoDSC', 'ConfStream', 'OnlineAutoClustEnsemble', 'RandomSearchStream',
          'MOStream', 'CluStream', 'DenStream', 'DBSTREAM', 'STREAMKMeans']
# adaptive/AutoML methods (always shown) vs single fixed algorithms (collapsible)
AUTOML = ['AutoDSC', 'ConfStream', 'OnlineAutoClustEnsemble', 'RandomSearchStream', 'MOStream']
SINGLE = ['CluStream', 'DenStream', 'DBSTREAM', 'STREAMKMeans']
MLABELS = {
    'AutoDSC': 'AutoDSC', 'ConfStream': 'ConfStream',
    'OnlineAutoClustEnsemble': 'OAClEns', 'RandomSearchStream': 'RandSearch',
    'MOStream': 'MOStream', 'CluStream': 'CluStream', 'DenStream': 'DenStream',
    'DBSTREAM': 'DBSTREAM', 'STREAMKMeans': 'STREAMKMeans',
}
# (color, marker) -- AutoDSC emphasized via thick black line
STYLE = {
    'AutoDSC':                ('#000000', 'o'),
    'ConfStream':             ('#0072B2', 's'),
    'OnlineAutoClustEnsemble':('#E69F00', 'P'),
    'RandomSearchStream':     ('#009E73', '^'),
    'MOStream':               ('#D55E00', 'h'),
    'CluStream':              ('#CC79A7', 'v'),
    'DenStream':              ('#56B4E9', 'D'),
    'DBSTREAM':               ('#F0E442', 'X'),
    'STREAMKMeans':           ('#999999', '*'),
}
METRIC_LABEL = {'ari': 'ARI', 'cmm': 'CMM', 'nmi': 'NMI', 'purity': 'Purity'}
DS_NAME = {
    'real_pendigits': 'Pendigits', 'real_insects': 'Insects', 'real_kdd99': 'KDD99',
    'real_shuttle': 'Shuttle', 'real_vehicle': 'Vehicle',
    'synth_blobs_expanding': 'Expanding', 'synth_blobs_sudden': 'Sudden',
    'synth_blobs_gradual': 'Gradual', 'synth_blobs_merge_split': 'Merge/Split',
    'synth_rbf_fast': 'RBF-Fast',
}


def stage_means(dataset, model, metric, stages):
    """Seed-averaged mean score within each of `stages` equal stream stages."""
    series = []
    for f in glob.glob(os.path.join(RESULTS_DIR, f'{model}_{dataset}_*.json')):
        wm = json.load(open(f)).get('window_metric', {})
        if wm.get(metric):
            series.append(np.asarray(wm[metric], dtype=float))
    if not series:
        return None
    n = min(len(s) for s in series)
    mean_traj = np.vstack([s[:n] for s in series]).mean(axis=0)
    # split the window trajectory into `stages` contiguous chunks, average each
    return np.array([chunk.mean() for chunk in np.array_split(mean_traj, stages)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['real_pendigits', 'synth_blobs_expanding'])
    ap.add_argument('--metrics', nargs='+', default=['ari', 'cmm'])
    ap.add_argument('--stages', type=int, default=5)
    ap.add_argument('--single', choices=['all', 'best'], default='all',
                    help="'all' shows every fixed algorithm; 'best' shows only the "
                         "single best-performing fixed algorithm")
    ap.add_argument('--zoom', action='store_true',
                    help='auto-scale each y-axis to its data range instead of [0,1]')
    ap.add_argument('--out', default='benchmark_stages')
    args = ap.parse_args()
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # choose which models to plot
    models = list(MODELS)
    if args.single == 'best':
        # rank single algorithms by mean ARI over the shown datasets, keep the best
        scores = {}
        for m in SINGLE:
            vals = []
            for ds in args.datasets:
                y = stage_means(ds, m, 'ari', args.stages)
                if y is not None:
                    vals.append(y.mean())
            scores[m] = np.mean(vals) if vals else -1
        best_single = max(scores, key=scores.get)
        models = AUTOML + [best_single]
        print(f"Best single algorithm: {best_single} "
              f"(mean ARI {scores[best_single]:.3f})")

    panels = [(ds, m) for ds in args.datasets for m in args.metrics]
    x = np.arange(1, args.stages + 1)
    plt.rcParams.update({'font.size': 13, 'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 3.4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (ds, metric) in zip(axes, panels):
        panel_vals = []
        for m in models:
            y = stage_means(ds, m, metric, args.stages)
            if y is None:
                continue
            panel_vals.append(y)
            color, mk = STYLE[m]
            lw = 3.0 if m == 'AutoDSC' else 1.8
            ms = 11 if m == 'AutoDSC' else 8
            ax.plot(x, y, '-', marker=mk, markersize=ms, linewidth=lw,
                    color=color, label=MLABELS[m], markeredgecolor='black',
                    markeredgewidth=0.6, zorder=(6 if m == 'AutoDSC' else 3))
        ax.set_xlabel(f'Stages of {DS_NAME.get(ds, ds)} Stream',
                      fontsize=14, fontweight='bold')
        ax.set_ylabel(METRIC_LABEL[metric], fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        if args.zoom and panel_vals:
            allv = np.concatenate(panel_vals)
            lo, hi = allv.min(), allv.max()
            pad = (hi - lo) * 0.08 + 1e-3
            ax.set_ylim(lo - pad, hi + pad)
        else:
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        ax.tick_params(axis='both', labelsize=12)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
        # 2-border style: keep left + bottom only, no internal gridlines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(models), frameon=False,
               bbox_to_anchor=(0.5, 1.12), columnspacing=1.0, handletextpad=0.4,
               prop={'weight': 'bold', 'size': 13})
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    pdf = os.path.join(CHARTS_DIR, f'{args.out}.pdf')
    png = os.path.join(CHARTS_DIR, f'{args.out}.png')
    plt.savefig(pdf, bbox_inches='tight')
    plt.savefig(png, dpi=300, bbox_inches='tight')
    print(f'Saved {pdf}\nSaved {png}')


if __name__ == '__main__':
    main()
