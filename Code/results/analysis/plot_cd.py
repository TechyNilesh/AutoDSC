#!/usr/bin/env python3
"""
Critical Difference (CD) diagrams for AutoDSC vs baselines, colored per method.

Uses `autorank` (Friedman + Nemenyi) to compute mean ranks and the critical
difference, then draws a Demsar-style CD diagram with each method's line and
label colored to match the benchmark-stages chart. Methods joined by a thick
bar are not significantly different.

Usage:
    python plot_cd.py
Output:
    charts/cd_ari.pdf / .png  and  charts/cd_cmm.pdf / .png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from autorank import autorank
import style as S

CHARTS_DIR = S.CHARTS_DIR
# colors keyed by canonical display label (== canonical name)
COLOR = {S.METHOD_NAME[m]: S.METHOD_COLOR[m] for m in S.METHODS}


# Nemenyi q_alpha (alpha=0.05, infinite df, studentized range / sqrt(2))
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164}


def nemenyi_cd(k, n):
    """Demsar critical difference: q * sqrt(k(k+1)/(6n))."""
    return _Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))


def build_frame(metric):
    data = {S.METHOD_NAME[m]: [S.scalar(m, ds, metric) for ds in S.DATASETS]
            for m in S.METHODS}
    return pd.DataFrame(data, index=[S.DATASET_NAME[d] for d in S.DATASETS])


def cd_diagram(ranks, cd, ax):
    """Demsar CD diagram. ranks: dict {label: meanrank}. cd: critical diff."""
    methods = sorted(ranks, key=ranks.get)        # best (lowest rank) first
    avg = [ranks[m] for m in methods]
    k = len(methods)
    lowv, highv = 1, int(np.ceil(max(avg)))
    highv = max(highv, k)

    # geometry
    textspace = 2.2
    scale = 9.0
    def rankpos(r):
        return textspace + scale * (highv - r) / (highv - lowv)

    ax.set_xlim(0, textspace * 2 + scale)
    ax.set_ylim(0, 1)
    ax.axis('off')
    top = 0.85
    # top rank axis
    ax.plot([rankpos(lowv), rankpos(highv)], [top, top], 'k-', lw=1.6)
    for r in range(lowv, highv + 1):
        ax.plot([rankpos(r)] * 2, [top, top + 0.03], 'k-', lw=1.6)
        ax.text(rankpos(r), top + 0.06, str(r), ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    # split methods left/right
    half = (k + 1) // 2
    right_idx = list(range(half))          # best ranks -> right side
    left_idx = list(range(half, k))        # worst ranks -> left side

    def draw(idx_list, side):
        for j, i in enumerate(idx_list):
            m = methods[i]
            rp = rankpos(avg[i])
            col = COLOR.get(m, '#333333')
            if side == 'right':
                y = top - 0.10 - j * 0.13
                xend = textspace * 2 + scale - 0.2
                ax.plot([rp, rp], [top, y], color=col, lw=2.4)
                ax.plot([rp, xend], [y, y], color=col, lw=2.4)
                ax.text(xend + 0.1, y, m, ha='left', va='center',
                        fontsize=14, fontweight="bold", color=col)
            else:
                y = top - 0.10 - j * 0.13
                xend = 0.2
                ax.plot([rp, rp], [top, y], color=col, lw=2.4)
                ax.plot([rp, xend], [y, y], color=col, lw=2.4)
                ax.text(xend - 0.1, y, m, ha='right', va='center',
                        fontsize=14, fontweight="bold", color=col)

    draw(right_idx, 'right')
    draw(left_idx, 'left')

    # CD bar (top-left reference)
    cdy = top + 0.06
    x0 = rankpos(lowv)
    ax.plot([x0, x0 + scale * cd / (highv - lowv)], [cdy + 0.08, cdy + 0.08],
            'k-', lw=2.6)
    ax.text(x0 + scale * cd / (highv - lowv) / 2, cdy + 0.11,
            f'CD = {cd:.2f}', ha='center', va='bottom',
            fontsize=12, fontweight='bold')

    # cliques: maximal groups within CD
    ybar = top - 0.05
    groups = []
    for i in range(k):
        for jj in range(k - 1, i, -1):
            if avg[jj] - avg[i] <= cd:
                groups.append((i, jj)); break
    # de-duplicate nested groups
    filt = [g for g in groups if not any(g != h and h[0] <= g[0] and g[1] <= h[1]
                                          for h in groups)]
    for n, (i, j) in enumerate(filt):
        yy = ybar - n * 0.045
        ax.plot([rankpos(avg[i]) + 0.05, rankpos(avg[j]) - 0.05], [yy, yy],
                'k-', lw=4.0, solid_capstyle='round')

    # crop the y-range to the content actually drawn (no empty bottom band)
    rows = (k + 1) // 2
    y_low = top - 0.10 - (rows - 1) * 0.13
    if filt:
        y_low = min(y_low, ybar - (len(filt) - 1) * 0.045)
    ax.set_ylim(y_low - 0.07, top + 0.22)


def make_cd(metric, out):
    df = build_frame(metric)
    res = autorank(df, alpha=0.05, verbose=False, order='descending')
    ranks = res.rankdf['meanrank'].to_dict()
    cd = getattr(res, 'cd', None)
    if cd is None or (isinstance(cd, float) and np.isnan(cd)):
        cd = nemenyi_cd(k=df.shape[1], n=df.shape[0])   # manual fallback
    cd = float(cd)
    S.set_rc()
    fig, ax = plt.subplots(figsize=(9.0, 2.4))
    cd_diagram(ranks, cd, ax)
    plt.tight_layout()
    S.save_fig(fig, 'cd', out)
    print(f'[{metric}] CD={cd:.3f}  best={min(ranks, key=ranks.get)} ({min(ranks.values()):.2f})')


def main():
    make_cd('ari', 'cd_ari')
    make_cd('cmm', 'cd_cmm')


if __name__ == '__main__':
    main()
