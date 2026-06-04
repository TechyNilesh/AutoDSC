#!/usr/bin/env python3
"""
Hero figure (Figure 1): AutoDSC wins on BOTH quality and efficiency on a single
representative real dataset (default: PenDigits).

Two individual panels + one shared standalone legend, MOStream-style:
  (a) quality    -- per-window ARI over the stream; AutoDSC (thick red) rides on
                    top of every baseline throughout.
  (b) efficiency -- throughput (x, log) vs peak memory (y, log) on the same
                    dataset, with MARKER SIZE proportional to ARI, so the
                    speed / footprint / quality trade-off is in one panel:
                    AutoDSC is the largest marker (best ARI) in the low-memory
                    region, while the only faster methods are far smaller
                    (much lower quality).

Names / colors / markers / order come from style.py.

Usage:
    python plot_hero.py                      # PenDigits
    python plot_hero.py --dataset synth_blobs_expanding
Output:
    charts/hero/quality.pdf / .png
    charts/hero/efficiency.pdf / .png
    charts/hero/legend.pdf / .png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import style as S

GROUP = 'hero'


def panel_quality(dataset):
    """Per-window ARI over the stream; AutoDSC emphasized on top."""
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    ws = S.window_size(dataset)
    for m in S.METHODS:
        c = S.curve(m, dataset, 'ari')
        if c is None:
            continue
        x = np.arange(1, len(c) + 1) * ws / 1000.0
        ax.plot(x, c, '-', marker=S.METHOD_MARKER[m], markersize=S.ms(m),
                linewidth=S.lw(m), color=S.METHOD_COLOR[m],
                label=S.METHOD_NAME[m], markeredgecolor='black',
                markeredgewidth=0.6, zorder=(6 if m == S.EMPHASIS else 3))
    ax.set_xlabel(S.STREAM_XLABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(r'ARI $\uparrow$', fontsize=15, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    S.bold_ticks(ax)
    S.clean_spines(ax)
    fig.tight_layout()
    S.save_fig(fig, GROUP, 'quality')
    print(f'Saved charts/{GROUP}/quality.pdf / .png')


def panel_efficiency(dataset):
    """Throughput vs memory on `dataset`; marker size encodes ARI (quality)."""
    # per-method ARI-label placement: (dx, dy in points, anchor)
    OFFSET = {
        'AutoDSC':       (-9, 12, 'right'),
        'ConfStream':    (12, -17, 'left'),
        'OACE':          (0, 12, 'center'),
        'MOStream':      (0, 13, 'center'),
        'RandomSearch':  (0, -19, 'center'),
        S.DEFAULT_BEST:  (0, 13, 'center'),
    }
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    pts = []
    for m in S.METHODS:
        tp = S.scalar(m, dataset, 'throughput_ips')
        mem = S.scalar(m, dataset, 'memory_mb')
        ari = S.scalar(m, dataset, 'ari')
        if np.isnan(tp) or np.isnan(mem):
            continue
        pts.append((m, tp, mem, ari))

    aris = np.array([p[3] for p in pts])
    amin, amax = np.nanmin(aris), np.nanmax(aris)

    def size(ari):
        # map ARI -> marker area; keep a visible floor so low-ARI points still show
        t = 0.0 if amax == amin else (ari - amin) / (amax - amin)
        return 110 + 760 * t

    # draw smaller (lower-ARI) markers LAST so they are never hidden behind the
    # large AutoDSC marker when two methods share an efficiency region.
    for m, tp, mem, ari in sorted(pts, key=lambda p: p[3], reverse=True):
        is_auto = m == S.EMPHASIS
        z = 5 if is_auto else 6  # small markers on top of the big AutoDSC one
        ax.scatter(tp, mem, s=size(ari) * (1.18 if is_auto else 1.0),
                   marker=S.METHOD_MARKER[m], color=S.METHOD_COLOR[m],
                   alpha=0.92, edgecolor='black',
                   linewidth=1.8 if is_auto else 0.8, zorder=z)
        dx, dy, ha = OFFSET.get(m, (0, 12, 'center'))
        txt = f'ARI {ari:.2f}' if is_auto else f'{ari:.2f}'
        ax.annotate(txt, (tp, mem), textcoords='offset points',
                    xytext=(dx, dy), ha=ha,
                    fontsize=11 if is_auto else 10,
                    fontweight='bold', color=S.METHOD_COLOR[m], zorder=8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Throughput (instances/s) $\uparrow$', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Memory (MB) $\downarrow$', fontsize=13, fontweight='bold')
    ax.set_xlim(ax.get_xlim()[0] * 0.45, ax.get_xlim()[1] * 2.6)
    ax.set_ylim(ax.get_ylim()[0] * 0.6, ax.get_ylim()[1] * 1.7)
    ax.tick_params(axis='both', labelsize=11)
    S.bold_ticks(ax)
    S.clean_spines(ax)
    ax.grid(axis='both', which='major', alpha=0.16, linewidth=0.6)

    # shade the favorable low-memory region and label it
    xl, yl = ax.get_xlim(), ax.get_ylim()
    mem_auto = [p[2] for p in pts if p[0] == S.EMPHASIS]
    ry1 = (max(mem_auto) * 1.35) if mem_auto else yl[1]
    ax.add_patch(Rectangle((xl[0], yl[0]), xl[1] - xl[0], ry1 - yl[0],
                           facecolor='#9ed99a', alpha=0.28, edgecolor='none',
                           zorder=0))
    ax.annotate('low-memory region', (xl[1] * 0.92, yl[0] * 1.18),
                ha='right', va='bottom', fontsize=9.5, style='italic',
                color='#2e7d32', zorder=1)
    ax.annotate('marker size $\\propto$ ARI', (0.035, 0.93),
                xycoords='axes fraction', fontsize=9.5, style='italic',
                color='#444444', zorder=7)

    fig.tight_layout()
    S.save_fig(fig, GROUP, 'efficiency')
    print(f'Saved charts/{GROUP}/efficiency.pdf / .png')


def _seed_values(method, dataset, key):
    """All per-seed scalar values for a method on a dataset (Default(Best) resolved)."""
    import glob, json, os
    if method == S.DEFAULT_BEST:
        p = S.best_default_prefix(dataset)
    else:
        p = S.METHOD_PREFIX[method]
    if not p:
        return []
    out = []
    for f in glob.glob(os.path.join(S.RESULTS_DIR, f'{p}_{dataset}_*.json')):
        v = json.load(open(f)).get(key)
        if v is not None and not np.isnan(v):
            out.append(float(v))
    return out


def panel_efficiency_box(key='throughput_ips', name='efficiency_box_tp',
                         datasets=None, logy=True):
    """Box plot of an efficiency metric per method across datasets+seeds."""
    datasets = datasets or S.DATASETS
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    data, positions, colors = [], [], []
    for i, m in enumerate(S.METHODS):
        vals = []
        for d in datasets:
            vals += _seed_values(m, d, key)
        data.append(vals)
        positions.append(i)
        colors.append(S.METHOD_COLOR[m])
    bp = ax.boxplot(data, positions=positions, widths=0.62, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black', linewidth=1.6),
                    whiskerprops=dict(linewidth=1.1), capprops=dict(linewidth=1.1))
    for patch, c, m in zip(bp['boxes'], colors, S.METHODS):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
        patch.set_edgecolor('black')
        patch.set_linewidth(2.0 if m == S.EMPHASIS else 1.0)
    if logy:
        ax.set_yscale('log')
    ax.set_xticks(positions)
    ax.set_xticklabels([S.METHOD_NAME[m] for m in S.METHODS], rotation=35,
                       ha='right', fontsize=10)
    ylabel = {'throughput_ips': r'Throughput (instances/s) $\uparrow$',
              'memory_mb': r'Memory (MB) $\downarrow$'}[key]
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.tick_params(axis='y', labelsize=11)
    S.bold_ticks(ax)
    S.clean_spines(ax)
    ax.grid(axis='y', which='major', alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    S.save_fig(fig, GROUP, name)
    print(f'Saved charts/{GROUP}/{name}.pdf / .png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='real_pendigits')
    args = ap.parse_args()
    S.set_rc()
    panel_quality(args.dataset)
    panel_efficiency(args.dataset)
    panel_efficiency_box('throughput_ips', 'efficiency_box_tp')
    panel_efficiency_box('memory_mb', 'efficiency_box_mem')
    S.save_legend(S.method_handles(S.METHODS), GROUP, 'legend',
                  ncol=len(S.METHODS))
    print(f'Saved charts/{GROUP}/legend.pdf / .png')


if __name__ == '__main__':
    main()
