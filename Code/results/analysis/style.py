#!/usr/bin/env python3
"""
Canonical naming / color / marker / order spec for ALL AutoDSC charts.

Single source of truth -- see ../../../CLAUDE.md. Every plot script imports from
here; nothing about names, colors, markers, order, or dataset labels is
hard-coded in individual scripts.

Also provides data-access helpers that resolve the synthetic "Default(Best)"
method (per-dataset best River default algorithm) transparently.
"""
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, '..', 'benchmark')
CHARTS_DIR = os.path.join(HERE, 'charts')

# ----------------------------------------------------------------------------
# Methods (canonical order). Default(Best) has prefix None -> resolved per data.
# ----------------------------------------------------------------------------
DEFAULT_BEST = 'Default(Best)'
METHODS = ['AutoDSC', 'ConfStream', 'OACE', 'MOStream', 'RandomSearch', DEFAULT_BEST]

METHOD_PREFIX = {
    'AutoDSC':       'AutoDSC',
    'ConfStream':    'ConfStream',
    'OACE':          'OnlineAutoClustEnsemble',
    'MOStream':      'MOStream',
    'RandomSearch':  'RandomSearchStream',
    DEFAULT_BEST:    None,   # computed
}
METHOD_NAME = {m: m for m in METHODS}        # display label == canonical name
METHOD_COLOR = {
    'AutoDSC':      '#D55E00',   # vermilion-red (ours, emphasized)
    'ConfStream':   '#0072B2',   # blue
    'OACE':         '#E69F00',   # orange
    'MOStream':     '#CC79A7',   # purple
    'RandomSearch': '#009E73',   # green
    DEFAULT_BEST:   '#555555',   # dark grey
}
METHOD_MARKER = {
    'AutoDSC': 'o', 'ConfStream': 's', 'OACE': '^',
    'MOStream': 'h', 'RandomSearch': 'D', DEFAULT_BEST: 'X',
}
METHOD_GROUP = {
    'AutoDSC': 'AutoML', 'ConfStream': 'AutoML', 'OACE': 'AutoML',
    'MOStream': 'Rule-based', 'RandomSearch': 'Baselines', DEFAULT_BEST: 'Baselines',
}
EMPHASIS = 'AutoDSC'   # always thick

# ----------------------------------------------------------------------------
# Single fixed algorithms (full names) -- only used by the evolution chart.
# ----------------------------------------------------------------------------
SINGLE_PREFIXES = ['CluStream', 'DenStream', 'DBSTREAM', 'STREAMKMeans']
SINGLE_NAME = {'CluStream': 'CluStream', 'DenStream': 'DenStream',
               'DBSTREAM': 'DBSTREAM', 'STREAMKMeans': 'StreamKMeans'}
SINGLE_COLOR = {'CluStream': '#0072B2', 'DenStream': '#E69F00',
                'DBSTREAM': '#009E73', 'STREAMKMeans': '#CC79A7'}

# ----------------------------------------------------------------------------
# Datasets (canonical order: real first, then synthetic).
# ----------------------------------------------------------------------------
REAL = ['real_insects', 'real_kdd99', 'real_pendigits', 'real_shuttle', 'real_vehicle']
SYNTH = ['synth_blobs_expanding', 'synth_blobs_gradual', 'synth_blobs_merge_split',
         'synth_blobs_sudden', 'synth_rbf_fast']
DATASETS = REAL + SYNTH
DATASET_NAME = {
    'real_insects': 'Insects', 'real_kdd99': 'KDD99', 'real_pendigits': 'PenDigits',
    'real_shuttle': 'Shuttle', 'real_vehicle': 'Vehicle',
    'synth_blobs_expanding': 'Expanding', 'synth_blobs_gradual': 'Gradual',
    'synth_blobs_merge_split': 'Merge-Split', 'synth_blobs_sudden': 'Sudden',
    'synth_rbf_fast': 'Fast-RBF',
}
DATASET_GROUP = {d: ('Real' if d in REAL else 'Synthetic') for d in DATASETS}

METRIC_LABEL = {'ari': 'ARI', 'cmm': 'CMM', 'nmi': 'NMI', 'purity': 'Purity'}

# ----------------------------------------------------------------------------
# Style helpers
# ----------------------------------------------------------------------------
def set_rc():
    plt.rcParams.update({'font.size': 13, 'font.family': 'DejaVu Sans'})

def bold_ticks(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')

def clean_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

def lw(method):
    return 3.2 if method == EMPHASIS else 2.0

def ms(method):
    return 11 if method == EMPHASIS else 8

# ----------------------------------------------------------------------------
# Output organization + shared standalone legends (MOStream-style: one folder
# per figure group, individual panel PDFs, and a separate legend.pdf reused
# above the subfigures that share it).
# ----------------------------------------------------------------------------
def group_dir(group):
    """charts/<group>/ ; created on demand."""
    d = os.path.join(CHARTS_DIR, group)
    os.makedirs(d, exist_ok=True)
    return d

def save_fig(fig, group, name):
    """Save one panel as vector PDF + 300-dpi PNG into charts/<group>/."""
    d = group_dir(group)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(d, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def method_handles(methods, linestyles=None):
    """Proxy legend handles for canonical methods (independent of any axes)."""
    from matplotlib.lines import Line2D
    hs = []
    for i, m in enumerate(methods):
        ls = linestyles[i] if linestyles else '-'
        hs.append(Line2D([0], [0], color=METHOD_COLOR[m], marker=METHOD_MARKER[m],
                         linestyle=ls, linewidth=lw(m), markersize=ms(m) - 2,
                         markeredgecolor='black', markeredgewidth=0.5,
                         label=METHOD_NAME[m]))
    return hs

def save_legend(handles, group, name='legend', ncol=None, width=None):
    """Save a standalone horizontal bold legend as its own vector PDF + PNG."""
    labels = [h.get_label() for h in handles]
    ncol = ncol or len(handles)
    width = width or max(4.0, 1.7 * len(handles))
    lf = plt.figure(figsize=(width, 0.5))
    lf.legend(handles, labels, loc='center', ncol=ncol, frameon=False,
              columnspacing=1.2, handletextpad=0.4,
              prop={'weight': 'bold', 'size': 13})
    save_fig(lf, group, name)

# ----------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------
def _files(prefix, dataset):
    return glob.glob(os.path.join(RESULTS_DIR, f'{prefix}_{dataset}_*.json'))

def _seed_mean_scalar(prefix, dataset, key):
    vals = [json.load(open(f)).get(key, np.nan) for f in _files(prefix, dataset)]
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan

def _seed_mean_curve(prefix, dataset, metric):
    curves = []
    for f in _files(prefix, dataset):
        c = json.load(open(f)).get('window_metric', {}).get(metric, [])
        if c:
            curves.append(np.asarray(c, dtype=float))
    if not curves:
        return None
    n = min(len(c) for c in curves)
    return np.vstack([c[:n] for c in curves]).mean(axis=0)

def best_default_prefix(dataset, select_metric='ari'):
    """River default algorithm with the best seed-mean `select_metric` on dataset."""
    scored = [(p, _seed_mean_scalar(p, dataset, select_metric)) for p in SINGLE_PREFIXES]
    scored = [(p, s) for p, s in scored if not np.isnan(s)]
    return max(scored, key=lambda t: t[1])[0] if scored else None

def scalar(method, dataset, key, select_metric='ari'):
    """Seed-mean scalar for a canonical method (Default(Best) resolved per dataset)."""
    if method == DEFAULT_BEST:
        p = best_default_prefix(dataset, select_metric)
        return _seed_mean_scalar(p, dataset, key) if p else np.nan
    return _seed_mean_scalar(METHOD_PREFIX[method], dataset, key)

def curve(method, dataset, metric, select_metric='ari'):
    """Seed-mean per-window curve for a canonical method (Default(Best) resolved)."""
    if method == DEFAULT_BEST:
        p = best_default_prefix(dataset, select_metric)
        return _seed_mean_curve(p, dataset, metric) if p else None
    return _seed_mean_curve(METHOD_PREFIX[method], dataset, metric)

# Canonical x-axis label for all staged stream charts (instance count, thousands).
STREAM_XLABEL = '#Instances (in thousands)'

def window_size(dataset):
    """Per-window instance count used in the benchmark runs (default 1000)."""
    for m in METHODS:
        p = METHOD_PREFIX.get(m)
        if not p:
            continue
        for f in _files(p, dataset):
            ws = json.load(open(f)).get('window_metric', {}).get('window_size')
            if ws:
                return int(ws)
    return 1000

def stage_thousands(dataset, stages, method='AutoDSC', metric='ari'):
    """Cumulative #instances (in thousands) at the END of each of `stages` equal
    -window bins, matching np.array_split binning of the seed-mean curve. Use as
    the x positions for staged stream charts so the axis reads in instances
    processed rather than abstract stage index, while keeping the staged shape."""
    c = curve(method, dataset, metric)
    nwin = len(c) if c is not None else stages
    sizes = [len(ch) for ch in np.array_split(np.arange(nwin), stages)]
    return np.cumsum(sizes) * window_size(dataset) / 1000.0

def scalar_overall(method, key, select_metric='ari'):
    """Mean over all datasets of the per-dataset seed-mean scalar."""
    vals = [scalar(method, d, key, select_metric) for d in DATASETS]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan
