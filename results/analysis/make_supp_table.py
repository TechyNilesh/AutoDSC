#!/usr/bin/env python3
"""
Emit ready-to-paste LaTeX for the supplementary-analyses summary tables from the
10-seed JSON produced by run_extra_analysis.py. Replaces the three figures with
compact tables to save space.

Usage:  python make_supp_table.py   (after run_extra_analysis.py finishes)
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, 'extra')
DS = {'real_pendigits': 'PenDigits', 'synth_blobs_sudden': 'Sudden'}


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(np.argsort(np.argsort(a)),
                            np.argsort(np.argsort(b)))[0, 1])


def main():
    val = json.load(open(os.path.join(EX, 'validity.json')))
    ovh = json.load(open(os.path.join(EX, 'overhead.json')))
    dim = json.load(open(os.path.join(EX, 'dim_sweep.json')))
    order = [k for k in DS if k in val]

    # ---- per-stream metrics ----
    rho, pk_mean, pk_std, true_k = {}, {}, {}, {}
    for k in order:
        rows = val[k]['rows']
        rho[k] = _spearman([r['composite'] for r in rows], [r['ari'] for r in rows])
        pk = np.array([r['pred_k'] for r in rows], float)
        pk_mean[k], pk_std[k] = pk.mean(), pk.std()
        true_k[k] = val[k]['true_k']

    ms = {}
    for k in order:
        w = max(1.0, ovh[k]['windows'])
        ms[k] = {c: 1000.0 * ovh[k][c] / w for c in ('learn', 'score', 'search')}
    share = {k: 100.0 * ovh[k]['search'] / (ovh[k]['learn'] + ovh[k]['score'] + ovh[k]['search'])
             for k in order}

    cols = ' & '.join(DS[k] for k in order)
    print(r'% ===== Table: supplementary analyses (per-stream) =====')
    print(r'\begin{table}[t]')
    print(r'\centering')
    print(r'\caption{Supplementary analyses on a real (PenDigits) and a drifting '
          r'synthetic (Sudden) stream (mean over 10 seeds). AutoDSC never observes '
          r'labels or the cluster count $k$ during the stream.}')
    print(r'\label{tab:supplementary}')
    print(r'\begin{tabular}{l c c}')
    print(r'\hline')
    print(fr'\textbf{{Measure}} & \textbf{{{DS[order[0]]}}} & \textbf{{{DS[order[1]]}}} \\')
    print(r'\hline')
    print(r'\multicolumn{3}{l}{\textit{Label-free selection validity}} \\')
    print('Spearman $\\rho$(internal score, ARI) & '
          + ' & '.join(f'{rho[k]:.2f}' for k in order) + r' \\')
    print('Predicted clusters (mean) & '
          + ' & '.join(f'{pk_mean[k]:.1f}' for k in order) + r' \\')
    print('True cluster count $k$ & '
          + ' & '.join(f'{true_k[k]}' for k in order) + r' \\')
    print(r'\hline')
    print(r'\multicolumn{3}{l}{\textit{Per-window runtime (ms)}} \\')
    print('Train + predict & '
          + ' & '.join(f'{ms[k]["learn"]:.0f}' for k in order) + r' \\')
    print('Multi-CVI scoring & '
          + ' & '.join(f'{ms[k]["score"]:.0f}' for k in order) + r' \\')
    print('PGSSO search & '
          + ' & '.join(f'{ms[k]["search"]:.2f}' for k in order) + r' \\')
    print('PGSSO search share (\\%) & '
          + ' & '.join(f'{share[k]:.2f}' for k in order) + r' \\')
    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{table}')
    print()

    # ---- dimensionality sweep ----
    dims = dim['dims']
    R = dim['results']
    ari = [R[str(d)]['ari'] for d in dims]
    thr = [R[str(d)]['throughput'] for d in dims]
    print(r'% ===== Table: dimensionality sweep =====')
    print(r'\begin{table}[t]')
    print(r'\centering')
    print(fr"\caption{{Dimensionality scalability of AutoDSC on a controlled "
          fr"five-cluster blob stream (mean over 10 seeds): quality is stable "
          fr"while throughput degrades only gracefully as the feature count grows.}}")
    print(r'\label{tab:dimscaling}')
    print(r'\begin{tabular}{l ' + 'c ' * len(dims) + r'}')
    print(r'\hline')
    print(r'\textbf{Dimensionality} & ' + ' & '.join(str(d) for d in dims) + r' \\')
    print(r'\hline')
    print(r'ARI $\uparrow$ & ' + ' & '.join(f'{v:.3f}' for v in ari) + r' \\')
    print(r'Throughput (ips) $\uparrow$ & ' + ' & '.join(f'{v:.0f}' for v in thr) + r' \\')
    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{table}')
    # also report ARI range for the prose
    print(f'\n% ARI range: [{min(ari):.3f}, {max(ari):.3f}]; '
          f'throughput {thr[0]:.0f} -> {thr[-1]:.0f} ips')


if __name__ == '__main__':
    main()
