#!/usr/bin/env python3
"""
Instrumented AutoDSC runs for the three supplementary analyses requested for the
paper (MOStream-style: each is computed on one or two sample workloads rather
than the full suite). Re-uses the *real* AutoDSC implementation and the shared
StreamEvaluator so numbers are consistent with the main benchmark.

Produces JSON under analysis/extra/ consumed by:
  - plot_internal_external.py  (label-free selection validity)
  - plot_overhead.py           (per-window runtime decomposition)
  - plot_dim_sweep.py          (dimensionality scalability)

Usage:
    python run_extra_analysis.py            # all three
    python run_extra_analysis.py validity overhead dim
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from river import stream

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SCRIPTS = os.path.join(CODE_ROOT, 'experiments', 'scripts')
sys.path.insert(0, os.path.join(CODE_ROOT, 'src', 'autodsc'))
sys.path.insert(0, SCRIPTS)                      # for utils.metrics
sys.path.insert(0, os.path.join(CODE_ROOT, 'datasets'))

from AutoDSC import AutoDSC                       # noqa: E402
from utils.metrics import StreamEvaluator         # noqa: E402
from generate_synth_blobs import generate_blobs_drift  # noqa: E402

OUT = os.path.join(HERE, 'extra')
os.makedirs(OUT, exist_ok=True)
WINDOW = 1000
POP = 10


# --------------------------------------------------------------------------- #
# core runner: returns per-window external metrics + AutoDSC internal trace
# --------------------------------------------------------------------------- #
def run_once(df, seed, window=WINDOW, pop=POP):
    X = df.copy()
    y = X.pop('class')
    model = AutoDSC(window_size=window, population_size=pop, random_seed=seed)
    ev = StreamEvaluator(window=window)
    for xrow, yval in stream.iter_pandas(X, y):
        xv = list(xrow.values())
        pred = model.predict_one(xv)
        model.learn_one(xv)
        if pred is None or pred == -1:
            continue
        ev.update(yval, pred, xv)
    res = ev.finalize(model='AutoDSC', dataset='_', verbose=False)
    return res, model


def _align_internal_external(res, model):
    """Pair per-window incumbent internal scores with that window's external ARI.
    The trace stores the 1-based window index; ev arrays are 0-based append
    order, so external metric for window i lives at index i-1."""
    ext_ari = res['window_metric']['ari']
    ext_cmm = res['window_metric']['cmm']
    rows = []
    t = model.trace
    for k in range(len(t['composite'])):
        idx = t['window_idx'][k] - 1
        if not (0 <= idx < len(ext_ari)):
            continue
        rows.append({
            'composite': t['composite'][k],
            'silhouette': t['silhouette'][k],
            'calinski': t['calinski'][k],
            'davies': t['davies'][k],
            'pred_k': t['pred_k'][k],
            'ari': ext_ari[idx],
            'cmm': ext_cmm[idx],
        })
    return rows


# --------------------------------------------------------------------------- #
# 1+2. validity (internal CVI vs external ARI) AND overhead (runtime split) are
#      computed from the SAME runs over two sample workloads (one real, one
#      drifting synthetic), so the 100k stream is processed only once per seed.
# --------------------------------------------------------------------------- #
def analysis_validity_overhead(seeds=tuple(range(1, 11))):
    jobs = {
        'real_pendigits': dict(loader=lambda: _load_csv('real_pendigits'), true_k=10),
        'synth_blobs_sudden': dict(
            loader=lambda: generate_blobs_drift('sudden', n_samples=100_000,
                                                n_features=5, n_classes=5, seed=11),
            true_k=5),
    }
    validity, overhead = {}, {}
    for name, spec in jobs.items():
        df = spec['loader']()
        rows = []
        acc = {'learn': [], 'score': [], 'search': []}
        windows = []
        for s in seeds:
            res, model = run_once(df, seed=s)
            rows.extend(_align_internal_external(res, model))
            for k in acc:
                acc[k].append(model.timing[k])
            windows.append(model.windows_processed)
            print(f'[val+ovh] {name} seed={s}: {len(model.trace["composite"])} windows '
                  f'timing={ {k: round(v, 3) for k, v in model.timing.items()} }')
        validity[name] = {'true_k': spec['true_k'], 'rows': rows}
        overhead[name] = {
            'learn': float(np.mean(acc['learn'])),
            'score': float(np.mean(acc['score'])),
            'search': float(np.mean(acc['search'])),
            'learn_std': float(np.std(acc['learn'])),
            'score_std': float(np.std(acc['score'])),
            'search_std': float(np.std(acc['search'])),
            'windows': float(np.mean(windows)),
        }
    _dump('validity.json', validity)
    _dump('overhead.json', overhead)


# --------------------------------------------------------------------------- #
# 3. dimensionality sweep on a controlled blob stream
# --------------------------------------------------------------------------- #
def analysis_dim(seeds=tuple(range(1, 11)), dims=(5, 10, 20, 40, 80), n=20_000):
    out = {'dims': list(dims), 'n_samples': n, 'results': {}}
    for d in dims:
        ari, thr, mem = [], [], []
        for s in seeds:
            df = generate_blobs_drift('sudden', n_samples=n, n_features=d,
                                      n_classes=5, seed=10 + s)
            res, _ = run_once(df, seed=s)
            ari.append(res['ari'])
            thr.append(res['throughput_ips'])
            mem.append(res['memory_mb'])
            print(f'[dim] d={d} seed={s}: ARI={res["ari"]:.3f} '
                  f'thr={res["throughput_ips"]:.0f} mem={res["memory_mb"]:.1f}')
        out['results'][str(d)] = {
            'ari': float(np.mean(ari)), 'ari_std': float(np.std(ari)),
            'throughput': float(np.mean(thr)), 'throughput_std': float(np.std(thr)),
            'memory': float(np.mean(mem)), 'memory_std': float(np.std(mem)),
        }
    _dump('dim_sweep.json', out)


# --------------------------------------------------------------------------- #
def _load_csv(name):
    return pd.read_csv(os.path.join(CODE_ROOT, 'datasets', name + '.csv'))


def _dump(fname, obj):
    path = os.path.join(OUT, fname)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f'  -> wrote {path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('which', nargs='*', default=['valovh', 'dim'])
    args = ap.parse_args()
    if 'valovh' in args.which or 'validity' in args.which or 'overhead' in args.which:
        analysis_validity_overhead()
    if 'dim' in args.which:
        analysis_dim()
    print('done.')
