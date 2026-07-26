"""Shared streaming evaluator: records GLOBAL and WINDOWED versions of
ARI, NMI, Purity, CMM, plus latency / throughput / memory.

Usage in a run script:
    ev = StreamEvaluator(window=progress_interval)
    for x, y in stream:
        pred = model.predict_one(x); model.learn_one(x)   # or model-specific
        ev.update(y, pred, x_list)
    results = ev.finalize(model="CluStream", dataset=ds, params={...})
    json.dump(results, ...)

Global metrics are computed on a reservoir sample (default 5000 pts) so CMM /
silhouette stay tractable on million-row streams (mirrors Sesame's <=60k cap).
"""
import time
import numpy as np
import psutil
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
try:
    from cmm import cmm_score
except ImportError:
    from utils.cmm import cmm_score


def purity_score(true, pred):
    cl = defaultdict(lambda: defaultdict(int))
    for t, p in zip(true, pred):
        cl[p][t] += 1
    n = len(true)
    return sum(max(c.values()) for c in cl.values()) / n if n else 0.0


def _quad(true, pred, X):
    """ARI, NMI, Purity, CMM for one labelling block."""
    if len(set(pred)) <= 1 or len(true) < 10:
        return (0.0, 0.0, purity_score(true, pred) if true else 0.0, 0.0)
    ari = adjusted_rand_score(true, pred)
    nmi = normalized_mutual_info_score(true, pred)
    pur = purity_score(true, pred)
    try:
        cmm = cmm_score(true, pred, np.asarray(X, dtype=float))
    except Exception:
        cmm = 0.0
    return (ari, nmi, pur, cmm)


class StreamEvaluator:
    """Window-wise evaluation only. The reported score for each metric is the
    mean over windows (computed from the per-window lists). No global/cumulative
    computation — keeps CMM/silhouette cheap (only on small windows)."""

    def __init__(self, window=1000):
        self.window = window
        self.wt, self.wp, self.wx = [], [], []          # current window buffers
        self.win = {k: [] for k in ("ari", "nmi", "purity", "cmm")}
        self.times, self.mems, self.ccounts = [], [], []
        self.n = 0
        self.unique = set()
        self.eval_overhead = 0.0   # time spent computing metrics (excluded from runtime)
        self.start = time.time()
        self.start_mem = psutil.Process().memory_info().rss

    def update(self, y, pred, x):
        self.n += 1
        self.unique.add(int(pred))
        self.wt.append(int(y)); self.wp.append(int(pred)); self.wx.append(x)
        if len(self.wt) == self.window:
            self._flush()

    def _flush(self):
        if not self.wt:
            return
        t0 = time.time()
        ari, nmi, pur, cmm = _quad(self.wt, self.wp, self.wx)
        self.win["ari"].append(ari); self.win["nmi"].append(nmi)
        self.win["purity"].append(pur); self.win["cmm"].append(cmm)
        self.eval_overhead += time.time() - t0          # exclude metric calc from runtime
        self.ccounts.append(len(self.unique))
        self.times.append(time.time() - self.start - self.eval_overhead)
        self.mems.append(abs((psutil.Process().memory_info().rss - self.start_mem) / 1024 / 1024))
        self.wt, self.wp, self.wx = [], [], []

    def finalize(self, model, dataset, params=None, run_count=None, seed=42, verbose=True):
        self._flush()  # leftover partial window
        runtime = (time.time() - self.start) - self.eval_overhead   # algorithm time only
        mem = abs((psutil.Process().memory_info().rss - self.start_mem) / 1024 / 1024)
        latency_us = (runtime / self.n * 1e6) if self.n else 0.0
        throughput = (self.n / runtime) if runtime > 0 else 0.0
        mean = lambda a: float(np.mean(a)) if a else 0.0
        res = {
            "dataset": dataset, "model": model,
            # reported score = mean over windows
            "ari": mean(self.win["ari"]),
            "nmi": mean(self.win["nmi"]),
            "purity": mean(self.win["purity"]),
            "cmm": mean(self.win["cmm"]),
            # efficiency
            "latency_us": latency_us, "throughput_ips": throughput,
            "memory_mb": mem, "total_runtime": runtime,
            "active_clusters": len(self.unique), "instances": self.n,
            "model_params": params or {},
            "window_metric": {
                "window_size": self.window,
                "ari": self.win["ari"], "nmi": self.win["nmi"],
                "purity": self.win["purity"], "cmm": self.win["cmm"],
                "cluster_counts": self.ccounts,
                "runtimes": self.times, "memory_usages": self.mems,
            },
        }
        if run_count is not None:
            res["run_count"] = run_count
            res["seed"] = seed
        if verbose:
            print(f"{model} on {dataset}: ARI={res['ari']:.4f} NMI={res['nmi']:.4f} "
                  f"Pur={res['purity']:.4f} CMM={res['cmm']:.4f} "
                  f"thr={throughput:.0f}/s lat={latency_us:.1f}us mem={mem:.1f}MB", flush=True)
        return res
