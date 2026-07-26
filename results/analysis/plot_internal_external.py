#!/usr/bin/env python3
"""
Label-free selection validity for AutoDSC, rendered as clean grouped bar charts
(MOStream-style: full frame, horizontal dashed grid, hatched bars, bold fonts)
on two sample workloads (one real, one drifting synthetic).

(a) Selection quality: windows are bucketed into terciles of the incumbent's
    *label-free* internal multi-CVI score (Low/Med/High); each bar is the mean
    true ARI of that bucket. ARI rises with the internal score, so the score
    AutoDSC selects on ranks windows by their (unseen) external quality ---
    the premise of label-free selection. (Spearman rho annotated.)
(b) Cluster count: AutoDSC never sees the true number of clusters, yet the
    incumbent's mean per-window cluster count lands close to the ground-truth k
    (exactly on the controlled synthetic stream; in range on the harder real
    stream). Addresses the classic "Finding k" problem.

Reads:  extra/validity.json   (from run_extra_analysis.py)
Output: charts/validity/{selection,pred_k,legend}.pdf/.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import style as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'extra', 'validity.json')

# two sample workloads -> display name, bar color, hatch (MOStream-style)
PANEL = {
    'real_pendigits':     (S.DATASET_NAME['real_pendigits'], '#0072B2', '//'),
    'synth_blobs_sudden': (S.DATASET_NAME['synth_blobs_sudden'], '#E69F00', 'xx'),
}
BAR_KW = dict(edgecolor='black', linewidth=1.0)
ERR_KW = dict(ecolor='black', elinewidth=1.1, capsize=3, capthick=1.1)


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float('nan')
    return float(np.corrcoef(np.argsort(np.argsort(a)),
                            np.argsort(np.argsort(b)))[0, 1])


def _frame(ax):
    ax.set_facecolor('white')
    ax.figure.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.tick_params(labelsize=12)
    S.bold_ticks(ax)


def _load():
    with open(DATA) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
def selection_panel(data):
    """Mean external ARI per internal-score tercile, grouped by dataset."""
    S.set_rc()
    dsets = [d for d in PANEL if d in data]
    buckets = ['Low', 'Med', 'High']
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    x = np.arange(len(buckets))
    nb = len(dsets)
    width = 0.8 / nb
    for j, ds in enumerate(dsets):
        rows = data[ds]['rows']
        s = np.array([r['composite'] for r in rows])
        a = np.array([r['ari'] for r in rows])
        q1, q2 = np.quantile(s, [1 / 3, 2 / 3])
        masks = [s <= q1, (s > q1) & (s <= q2), s > q2]
        means = [a[m].mean() if m.any() else 0 for m in masks]
        # standard error of the mean (cleaner than std for a mean estimate)
        sems = [a[m].std() / np.sqrt(m.sum()) if m.any() else 0 for m in masks]
        name, color, hatch = PANEL[ds]
        ax.bar(x + (j - (nb - 1) / 2) * width, means, width, yerr=sems,
               color=color, hatch=hatch, error_kw=ERR_KW, label=name, **BAR_KW)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, fontsize=12, fontweight='bold')
    ax.set_xlabel('Internal multi-CVI score (tercile)', fontsize=13.5, fontweight='bold')
    ax.set_ylabel(r'External ARI $\uparrow$', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.18)
    ax.legend(loc='upper center', ncol=len(dsets), fontsize=10, frameon=False,
              bbox_to_anchor=(0.5, 1.0), prop={'weight': 'bold', 'size': 10})
    _frame(ax)
    plt.tight_layout()
    S.save_fig(fig, 'validity', 'selection')


def pred_k_panel(data):
    """Mean predicted #clusters vs true k, grouped by dataset."""
    S.set_rc()
    dsets = [d for d in PANEL if d in data]
    cats = ['AutoDSC pred.', 'True k']
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    x = np.arange(len(cats))
    nb = len(dsets)
    width = 0.8 / nb
    for j, ds in enumerate(dsets):
        pk = np.array([r['pred_k'] for r in data[ds]['rows']], float)
        name, color, hatch = PANEL[ds]
        vals = [pk.mean(), data[ds]['true_k']]
        errs = [pk.std() / np.sqrt(pk.size), 0]
        ax.bar(x + (j - (nb - 1) / 2) * width, vals, width, yerr=errs,
               color=color, hatch=hatch, error_kw=ERR_KW, label=name, **BAR_KW)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of clusters', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(d['true_k'] for d in data.values()) + 4)
    _frame(ax)
    plt.tight_layout()
    S.save_fig(fig, 'validity', 'pred_k')


def save_validity_legend(data):
    """Standalone legend shared by both panels (dataset identity)."""
    handles = [Patch(facecolor=color, hatch=hatch, edgecolor='black',
                     linewidth=1.0, label=PANEL[ds][0])
               for ds, (_, color, hatch) in
               ((d, PANEL[d]) for d in PANEL if d in data)]
    S.save_legend(handles, 'validity', ncol=len(handles), width=5.5)


def main():
    data = _load()
    selection_panel(data)
    pred_k_panel(data)
    save_validity_legend(data)
    # remove superseded panels from earlier runs
    for old in ('scatter', 'corr_bars'):
        for ext in ('pdf', 'png'):
            p = os.path.join(S.CHARTS_DIR, 'validity', f'{old}.{ext}')
            if os.path.exists(p):
                os.remove(p)
    print('Saved charts/validity/selection.*, pred_k.* and legend.*')


if __name__ == '__main__':
    main()
