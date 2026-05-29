#!/usr/bin/env python3
"""
Drift-behaviour chart: two representative synthetic streams.

(a) Merge-Split -- a single merge-then-split event mid-stream where every
    adaptive method's ARI dips and then recovers. AutoDSC holds the highest
    stable quality on either side and recovers with the strongest baselines.
(b) Expanding -- continuous expansion drift. AutoDSC reaches near-perfect ARI
    almost immediately and sustains it, while ConfStream converges slowly and
    OnlineAutoClust / MOStream plateau below.

Per-window ARI, seed-averaged. Visualizes the recovery / sustained-adaptation
behaviour predicted by the theory (Lemma 2 / Corollary 1) with no new runs.

Usage:
    python plot_recovery.py
Output:
    charts/recovery.pdf / .png
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

# adaptive methods shown (canonical names/colors/markers from style.py);
# fixed single algorithms never recover and are summarized by Default(Best)
# elsewhere, so we show the 4 adaptive comparators here.
import style as S
METHODS = [
    (m, S.METHOD_PREFIX[m], S.METHOD_COLOR[m], ls, S.METHOD_MARKER[m])
    for m, ls in [('AutoDSC', '-'), ('ConfStream', '--'),
                  ('OACE', '-.'), ('MOStream', ':')]
]

# panel: (dataset, title, drift descriptor, ylim)
#   drift = ('region', lo, hi, label)        -> grey band (discrete drift event)
#   drift = ('continuous', label)            -> left->right gradient (gradual drift)
PANELS = [
    ('synth_blobs_merge_split', 'Merge-Split',
     ('region', 46, 63, 'merge–split drift'), (0.60, 1.02)),
    ('synth_blobs_expanding',   'Expanding',
     ('continuous', 'within-cluster variance grows →'), (0.84, 1.01)),
]


def mean_curve(prefix, dataset, metric='ari'):
    curves = []
    for f in glob.glob(os.path.join(RESULTS_DIR, f'{prefix}_{dataset}_*.json')):
        c = json.load(open(f))['window_metric'].get(metric, [])
        if c:
            curves.append(c)
    if not curves:
        return None
    n = min(len(c) for c in curves)
    return np.array([c[:n] for c in curves]).mean(axis=0)


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    plt.rcParams.update({'font.size': 13, 'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.4))

    STAGES = 20   # divide the stream into equal stages (like the benchmark chart)

    def to_stages(c):
        return np.array([ch.mean() for ch in np.array_split(c, STAGES)])

    def win_to_stage(w):     # window index (out of 100) -> stage x-coordinate
        return w / 100.0 * STAGES + 0.5

    for idx, (ax, (dataset, title, drift, ylim)) in enumerate(zip(axes, PANELS)):
        if drift[0] == 'region':
            _, lo, hi, _ = drift
            ax.axvspan(win_to_stage(lo), win_to_stage(hi),
                       color='#cccccc', alpha=0.45, zorder=0)
        elif drift[0] == 'continuous':
            grad = np.linspace(0, 1, 256).reshape(1, -1)
            ax.imshow(grad, extent=[0.5, STAGES + 0.5, ylim[0], ylim[1]],
                      aspect='auto', cmap='Greys', alpha=0.30, zorder=0,
                      vmin=-0.2, vmax=1.4)

        for name, prefix, color, ls, mk in METHODS:
            c = mean_curve(prefix, dataset)
            if c is None:
                continue
            y = to_stages(c)
            x = np.arange(1, STAGES + 1)
            lw = 3.4 if name == 'AutoDSC' else 2.0
            ms = 7 if name == 'AutoDSC' else 5
            ax.plot(x, y, linestyle=ls, marker=mk, markersize=ms, linewidth=lw,
                    color=color, markeredgecolor='black',
                    markeredgewidth=0.4, label=name, zorder=3)

        ax.set_xlabel(f'Stages of {title} Stream', fontsize=15, fontweight='bold')
        ax.set_ylim(*ylim)
        ax.set_xlim(0.5, STAGES + 0.5)
        ax.set_xticks(np.arange(5, STAGES + 1, 5))
        ax.set_ylabel(r'ARI $\uparrow$', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(METHODS),
               frameon=False, bbox_to_anchor=(0.5, 1.03), columnspacing=1.4,
               handletextpad=0.4, prop={'weight': 'bold', 'size': 12})
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(CHARTS_DIR, f'recovery.{ext}'),
                    dpi=300, bbox_inches='tight')
    print('Saved charts/recovery.pdf / .png')


if __name__ == '__main__':
    main()
