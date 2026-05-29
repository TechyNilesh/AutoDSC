#!/usr/bin/env python3
"""
Drift-behaviour chart: two representative synthetic streams, as individual panels.

(a) Merge-Split -- a single merge-then-split event mid-stream where every
    adaptive method's ARI dips and then recovers. AutoDSC holds the highest
    stable quality on either side and recovers with the strongest baselines.
(b) Expanding -- continuous expansion drift. AutoDSC reaches near-perfect ARI
    almost immediately and sustains it, while ConfStream converges slowly and
    OnlineAutoClust / MOStream plateau below.

Per-window ARI, seed-averaged. Each panel is saved as its own vector PDF under
charts/recovery/, with a single shared legend.pdf placed above them.

Usage:
    python plot_recovery.py
Output:
    charts/recovery/merge_split.pdf / .png
    charts/recovery/expanding.pdf / .png
    charts/recovery/legend.pdf / .png   (shared legend)
"""
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import style as S

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, '..', 'benchmark')
GROUP = 'recovery'

# adaptive methods shown (canonical names/colors/markers from style.py); fixed
# single algorithms never recover and are summarized by Default(Best) elsewhere.
ADAPTIVE = ['AutoDSC', 'ConfStream', 'OACE', 'MOStream']
LINESTYLES = {'AutoDSC': '-', 'ConfStream': '--', 'OACE': '-.', 'MOStream': ':'}

# panel: (dataset, title, drift descriptor, ylim, out-name)
PANELS = [
    ('synth_blobs_merge_split', 'Merge-Split',
     ('region', 46, 63), (0.60, 1.02), 'merge_split'),
    ('synth_blobs_expanding', 'Expanding',
     ('continuous',), (0.84, 1.01), 'expanding'),
]
STAGES = 20


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


def to_stages(c):
    return np.array([ch.mean() for ch in np.array_split(c, STAGES)])


def win_to_stage(w):
    return w / 100.0 * STAGES + 0.5


def main():
    S.set_rc()
    for dataset, title, drift, ylim, out in PANELS:
        fig, ax = plt.subplots(figsize=(7.0, 3.7))
        if drift[0] == 'region':
            _, lo, hi = drift
            ax.axvspan(win_to_stage(lo), win_to_stage(hi),
                       color='#cccccc', alpha=0.45, zorder=0)
        elif drift[0] == 'continuous':
            grad = np.linspace(0, 1, 256).reshape(1, -1)
            ax.imshow(grad, extent=[0.5, STAGES + 0.5, ylim[0], ylim[1]],
                      aspect='auto', cmap='Greys', alpha=0.30, zorder=0,
                      vmin=-0.2, vmax=1.4)

        for name in ADAPTIVE:
            c = mean_curve(S.METHOD_PREFIX[name], dataset)
            if c is None:
                continue
            y = to_stages(c)
            x = np.arange(1, STAGES + 1)
            ax.plot(x, y, linestyle=LINESTYLES[name], marker=S.METHOD_MARKER[name],
                    markersize=S.ms(name) - 2, linewidth=S.lw(name),
                    color=S.METHOD_COLOR[name], markeredgecolor='black',
                    markeredgewidth=0.4, zorder=3)

        ax.set_xlabel('Stages of Stream', fontsize=15, fontweight='bold')
        ax.set_ylabel(r'ARI $\uparrow$', fontsize=15, fontweight='bold')
        ax.set_ylim(*ylim)
        ax.set_xlim(0.5, STAGES + 0.5)
        ax.set_xticks(np.arange(5, STAGES + 1, 5))
        ax.tick_params(axis='both', labelsize=12)
        S.bold_ticks(ax)
        S.clean_spines(ax)
        fig.tight_layout()
        S.save_fig(fig, GROUP, out)
        print(f'Saved charts/{GROUP}/{out}.pdf / .png')

    # shared legend (line style matches the panels)
    handles = S.method_handles(ADAPTIVE, linestyles=[LINESTYLES[m] for m in ADAPTIVE])
    S.save_legend(handles, GROUP, 'legend', ncol=len(ADAPTIVE))
    print(f'Saved charts/{GROUP}/legend.pdf / .png')


if __name__ == '__main__':
    main()
