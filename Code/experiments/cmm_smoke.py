"""Smoke test: Purity / NMI / CMM, computed window-wise AND global-wise,
for several models on a small (subsampled) dataset."""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "scripts", "utils"))
sys.path.insert(0, os.path.join(ROOT, "src", "autodsc"))
sys.path.insert(0, os.path.join(ROOT, "src", "baselines", "ConfStream"))
from cmm import cmm_score
import stream_clustering as sc

DATASET = sys.argv[1] if len(sys.argv) > 1 else "real_pendigits"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
W = 1000


def purity(true, pred, X=None):
    from collections import defaultdict
    cl = defaultdict(lambda: defaultdict(int))
    for t, p in zip(true, pred):
        cl[p][t] += 1
    return sum(max(c.values()) for c in cl.values()) / len(true)


def nmi(true, pred, X=None):
    return normalized_mutual_info_score(true, pred)


def windowed(fn, true, pred, X, w=W):
    vals = []
    for s in range(0, len(true), w):
        tt, pp, xx = true[s:s + w], pred[s:s + w], X[s:s + w]
        if len(tt) < 10 or len(set(pp)) <= 1:
            continue
        vals.append(fn(tt, pp, xx))
    return float(np.mean(vals)) if vals else 0.0


def build_models():
    from AutoDSC import AutoDSC
    m = {
        "CluStream": sc.CluStream(),
        "DenStream": sc.DenStream(),
        "DBSTREAM": sc.DBStream(),
        "KMeans": sc.KMeans(),
        "STREAMKMeans": sc.StreamKMeans(),
        "AutoDSC": AutoDSC(),
    }
    try:
        from ConfStream import ConfStream
        m["ConfStream"] = ConfStream()
    except Exception as e:
        print("(ConfStream skipped:", e, ")")
    return m


def run_model(model, X):
    preds = []
    for row in X:
        x = list(row)
        try:
            model.learn_one(x)
            p = model.predict_one(x)
        except Exception:
            p = 0
        preds.append(int(p) if p is not None else 0)
    return preds


def main():
    df = pd.read_csv(os.path.join(ROOT, "datasets", DATASET + ".csv"))
    y = df.pop("class").to_numpy()
    X = df.to_numpy(dtype=float)
    if len(X) > N:
        X, y = X[:N], y[:N]
    print(f"Dataset={DATASET}  n={len(X)}  features={X.shape[1]}  true_k={len(set(y))}\n")

    print("{:<14} | {:^23} | {:^23}".format("", "WINDOW-WISE", "GLOBAL"))
    print("{:<14} | {:>7}{:>8}{:>8} | {:>7}{:>8}{:>8}".format(
        "Model", "Pur", "NMI", "CMM", "Pur", "NMI", "CMM"))
    print("-" * 66)
    for name, model in build_models().items():
        preds = run_model(model, X)
        yl = list(y)
        wp = windowed(purity, yl, preds, X)
        wn = windowed(nmi, yl, preds, X)
        wc = windowed(cmm_score, yl, preds, X)
        gp = purity(yl, preds)
        gn = nmi(yl, preds)
        gc = cmm_score(yl, preds, X)
        print("{:<14} | {:>7.3f}{:>8.3f}{:>8.3f} | {:>7.3f}{:>8.3f}{:>8.3f}".format(
            name, wp, wn, wc, gp, gn, gc))


if __name__ == "__main__":
    main()
