"""Cluster Mapping Measure (CMM) — Python port of Sesame's CMM::Evaluate.

Faithful to intellistream/Sesame (mostream branch) src/Evaluation/CMM.cpp:
  AnalyseGT  -> per-GT-cluster kNN connectivity of each point
  CalcMatch  -> map each found cluster -> ground-truth cluster
  CalcError  -> connectivity-weighted misplaced / noise / missed penalties
  CMM = 1 - totalError / totalErrorMax        (in [0,1], higher better)

Notes / simplifications matching our setup:
  - knnNeighbourhood k = 2 (Sesame default)
  - point weight = 1 (no time decay)
  - noise label is -1 (our streams have none, but handled for generality)
"""
import numpy as np
from collections import defaultdict


def _cluster_knn_connectivity(idxs, X, k=2):
    """For one GT cluster: per-point avg-of-2NN dist, cluster mean/dev, per-point conn."""
    m = len(idxs)
    pts = X[idxs]
    if m == 1:
        return {idxs[0]: 1.0}, 0.0, np.sqrt(1e-50)
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    Ds = np.sort(D, axis=1)
    avg = (Ds[:, 0] + Ds[:, 1]) / 2.0 if m >= 3 else Ds[:, 0]
    mean = avg.mean()
    var = (avg ** 2).mean() - mean * mean
    var = var if var > 0 else 1e-50
    dev = np.sqrt(var)
    upper = mean + dev
    conn = np.where(avg < upper, 1.0, np.where(avg > 0, upper / avg, 1.0))
    return {idxs[j]: float(conn[j]) for j in range(m)}, float(mean), float(dev)


def _calc_match(true, pred, gt_labels):
    """Map each found cluster fc -> ground-truth cluster (Sesame CalcMatch)."""
    mapFC = defaultdict(lambda: defaultdict(int))
    mapGT = defaultdict(lambda: defaultdict(int))
    sumsFC = defaultdict(int)
    for t, p in zip(true, pred):
        if p != -1 and t != -1:
            mapFC[p][t] += 1
            sumsFC[p] += 1
        if t != -1:
            mapGT[t][t] += 1
    match = {}
    for fc, hcmap in mapFC.items():
        nonzero = [hc for hc, c in hcmap.items() if c != 0]
        if len(nonzero) == 1:
            match[fc] = nonzero[0]
            continue
        # greedy: GT cluster minimizing positive frequency difference
        best, min_diff = -1, np.inf
        for hc0 in gt_labels:
            err = 0
            for hc1 in gt_labels:
                fd = mapFC[fc][hc1] - mapGT[hc0][hc1]
                if fd > 0:
                    err += fd
            if err < min_diff:
                min_diff, best = err, hc0
        match[fc] = best
    return match


def _calc_conn(i, j, X, gt_arr, own_mean, own_dev):
    """Connectivity of point i w.r.t. cluster j (Sesame CalcConn)."""
    pts = X[gt_arr[j]]
    d = np.sort(np.linalg.norm(pts - X[i], axis=1))
    avg = (d[0] + d[1]) / 2.0 if len(d) >= 2 else d[0]
    upper = own_mean + own_dev
    if avg < upper:
        return 1.0
    return upper / avg if avg > 0 else 1.0


def cmm_score(true_labels, pred_labels, X):
    """Compute CMM for one (true, pred) labelling over feature matrix X."""
    true = np.asarray(true_labels)
    pred = np.asarray(pred_labels)
    X = np.asarray(X, dtype=float)
    n = len(true)
    if n == 0:
        return 1.0

    gt = defaultdict(list)
    for i, t in enumerate(true):
        if t != -1:
            gt[t].append(i)
    gt_arr = {c: np.array(idx) for c, idx in gt.items()}
    gt_labels = sorted(gt.keys())

    conn = np.ones(n)
    knn_mean, knn_dev = {}, {}
    for c, idx in gt_arr.items():
        cc, mean, dev = _cluster_knn_connectivity(idx, X)
        knn_mean[c], knn_dev[c] = mean, dev
        for i, v in cc.items():
            conn[i] = v

    match = _calc_match(true, pred, gt_labels)
    gt_set = {c: set(idx.tolist()) for c, idx in gt_arr.items()}

    total_err, total_max = 0.0, 0.0
    for i in range(n):
        fc, hc = pred[i], true[i]
        total_max += 1.0          # weight = 1
        err = 0.0
        real_hc = match.get(fc, -1)
        if fc != -1 and hc != -1:
            if real_hc == hc:
                err = 0.0
            elif real_hc == -1:
                err = 1.0
            elif i in gt_set.get(real_hc, ()):
                err = 0.00001
            else:
                cc = _calc_conn(i, real_hc, X, gt_arr, knn_mean[hc], knn_dev[hc])
                err = (1.0 - cc) * conn[i]
        elif fc != -1 and hc == -1:        # noise assigned to a cluster
            if real_hc == -1:
                err = 1.0
            elif i in gt_set.get(real_hc, ()):
                err = 0.00001
            else:
                err = 1.0 - _calc_conn(i, real_hc, X, gt_arr,
                                       knn_mean.get(hc, 0.0), knn_dev.get(hc, 0.0))
        elif fc == -1 and hc != -1:        # missed: should be clustered
            err = conn[i]
        total_err += err
    return 1.0 - total_err / total_max if total_max > 0 else 1.0
