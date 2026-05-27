"""
Generate paper charts in MOStream visual style from benchmark_v3 results.

MOStream style reference (arXiv 2309.04799v2):
  - Grouped bar charts: one hatched+colored bar per method, black edges,
    light dotted horizontal gridlines, large sans-serif fonts, no title,
    scientific notation on throughput, separate top legend.
  - Line charts: distinct markers (o/s/D/^/v), shaded phase regions.

Output: Code/results/charts/*.pdf (+ .png previews)
"""
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "benchmark_v3")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "charts")
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = ["ari", "nmi", "purity", "cmm"]      # all metrics loaded
COMPOSITE_METRICS = ["ari", "nmi", "cmm"]      # composite = ARI+NMI+CMM (no Purity)

# ---- Display config (order: baselines first, AutoDSC last/highlighted) ----
MODEL_ORDER = [
    "CluStream", "DBSTREAM", "DenStream", "KMeans",
    "MOStream", "ConfStream", "OnlineAutoClustEnsemble",
    "RandomSearchStream", "AutoDSC",
]
MODEL_LABEL = {
    "CluStream": "CluStream", "DBSTREAM": "DBSTREAM", "DenStream": "DenStream",
    "KMeans": "KMeans", "MOStream": "MOStream", "ConfStream": "ConfStream",
    "OnlineAutoClustEnsemble": "OAC-Ens", "RandomSearchStream": "RandSearch",
    "AutoDSC": "AutoDSC",
}
# MOStream-like palette; AutoDSC gets the standout lime-green MOStream used for itself
MODEL_COLOR = {
    "CluStream": "#4C9BD0",  # blue
    "DBSTREAM": "#B7B7B7",   # gray
    "DenStream": "#3C2F4A",  # dark purple
    "KMeans": "#C0392B",     # red
    "MOStream": "#D9C9A3",   # tan
    "ConfStream": "#8B5A2B", # brown
    "OnlineAutoClustEnsemble": "#E67E22",  # orange
    "RandomSearchStream": "#7Fa9d6",       # light blue
    "AutoDSC": "#A0C53F",    # lime green (highlight)
}
MODEL_HATCH = {
    "CluStream": "||", "DBSTREAM": "//", "DenStream": "..", "KMeans": "xx",
    "MOStream": "++", "ConfStream": "oo", "OnlineAutoClustEnsemble": "**",
    "RandomSearchStream": "\\\\", "AutoDSC": "",
}

DATASET_ORDER = [
    "real_insects", "real_kdd99", "real_pendigits", "real_shuttle", "real_vehicle",
    "synth_blobs_expanding", "synth_blobs_gradual", "synth_blobs_merge_split",
    "synth_blobs_sudden", "synth_rbf_fast",
]
DATASET_LABEL = {
    "real_insects": "Insects", "real_kdd99": "KDD99", "real_pendigits": "Pendigits",
    "real_shuttle": "Shuttle", "real_vehicle": "Vehicle",
    "synth_blobs_expanding": "B-Exp", "synth_blobs_gradual": "B-Grad",
    "synth_blobs_merge_split": "B-Merge", "synth_blobs_sudden": "B-Sud",
    "synth_rbf_fast": "RBF-Fast",
}

# ---- Global MOStream-like rcParams ----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.linewidth": 1.2,
    "hatch.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load():
    """Return nested dict: data[dataset][model][metric] = list over runs,
    plus extra[dataset][model] = {throughput, memory, latency} (run means),
    plus windows[dataset][model] = list-of-runs windowed composite arrays."""
    data = {}
    extra = {}
    windows = {}
    for f in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        d = json.load(open(f))
        ds, m = d["dataset"], d["model"]
        data.setdefault(ds, {}).setdefault(m, {k: [] for k in METRICS})
        for k in METRICS:
            data[ds][m][k].append(d[k])
        extra.setdefault(ds, {}).setdefault(m, {"throughput": [], "memory": [], "latency": []})
        extra[ds][m]["throughput"].append(d.get("throughput_ips", np.nan))
        extra[ds][m]["memory"].append(d.get("memory_mb", np.nan))
        extra[ds][m]["latency"].append(d.get("latency_us", np.nan))
        wm = d.get("window_metric", {})
        if all(k in wm for k in COMPOSITE_METRICS):
            comp = np.mean([wm[k] for k in COMPOSITE_METRICS], axis=0)
            windows.setdefault(ds, {}).setdefault(m, []).append(np.asarray(comp))
    return data, extra, windows


def composite(data, ds, m):
    return np.mean([np.mean(data[ds][m][k]) for k in COMPOSITE_METRICS])


def models_present(data):
    present = set()
    for ds in data:
        present |= set(data[ds].keys())
    return [m for m in MODEL_ORDER if m in present]


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, name + ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", name + ".pdf/.png")


def make_legend(models):
    """Standalone legend (MOStream keeps legend separate above subfigures)."""
    handles = [Patch(facecolor=MODEL_COLOR[m], hatch=MODEL_HATCH[m],
                     edgecolor="black", label=MODEL_LABEL[m]) for m in models]
    fig = plt.figure(figsize=(12, 0.6))
    fig.legend(handles=handles, ncol=len(models), loc="center",
               frameon=False, fontsize=14, handlelength=1.6,
               columnspacing=1.0, handletextpad=0.4)
    save(fig, "legend_methods")


def bar_grouped(data, models, metric_fn, ylabel, name, sci=False):
    """Grouped bars: x=datasets, one bar per method (MOStream General style)."""
    datasets = [d for d in DATASET_ORDER if d in data]
    nM = len(models)
    x = np.arange(len(datasets))
    w = 0.85 / nM
    fig, ax = plt.subplots(figsize=(13, 3.4))
    for i, m in enumerate(models):
        vals = [metric_fn(data, ds, m) for ds in datasets]
        ax.bar(x + i * w - 0.425 + w / 2, vals, w,
               color=MODEL_COLOR[m], hatch=MODEL_HATCH[m],
               edgecolor="black", linewidth=0.7, label=MODEL_LABEL[m])
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in datasets], fontsize=15)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.yaxis.grid(True, linestyle=":", color="0.6", linewidth=0.8)
    ax.set_axisbelow(True)
    if sci:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    save(fig, name)


def _window_mean(windows, ds, m):
    runs = windows[ds][m]
    L = min(len(r) for r in runs)
    return np.mean([r[:L] for r in runs], axis=0), L


WINDOW_SIZE = 1000  # instances per window (window_metric.window_size)


def temporal(windows, ds, name, drift_window=None):
    """Windowed composite over the stream (MOStream Dynamic style), using the
    group view: Best(Default) + AutoML methods. AutoDSC stands out with a thick
    line + high z-order; the others are thinner."""
    markers = {"AutoDSC": "o", "ConfStream": "D", "MOStream": "s",
               "OnlineAutoClustEnsemble": "^", "RandomSearchStream": "v",
               "Best(Default)": "P"}
    avail = windows.get(ds, {})
    show = [m for m in GROUP_ORDER
            if m == "Best(Default)" or m in avail]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for m in show:
        if m == "Best(Default)":
            singles = [_window_mean(windows, ds, s)[0]
                       for s in SINGLE_ALGOS if s in avail]
            L = min(len(s) for s in singles)
            arr = np.max([s[:L] for s in singles], axis=0)
        else:
            arr, L = _window_mean(windows, ds, m)
        xs = np.arange(L) * WINDOW_SIZE / 1000.0  # instances in thousands
        every = max(1, L // 14)
        is_ours = (m == "AutoDSC")
        ax.plot(xs, arr, marker=markers.get(m, "o"), markevery=every,
                markersize=9 if is_ours else 5,
                linewidth=3.5 if is_ours else 1.4,
                color=GROUP_COLOR[m], label=GROUP_LABEL[m],
                alpha=1.0 if is_ours else 0.75,
                markeredgecolor="black" if is_ours else None,
                markeredgewidth=0.8 if is_ours else 0,
                zorder=6 if is_ours else 3)
    if drift_window is not None:
        dx = drift_window * WINDOW_SIZE / 1000.0
        ax.axvline(dx, color="red", linestyle="--", linewidth=1.6, zorder=2)
        ax.annotate("drift", xy=(dx, 0.30), xytext=(6, 0),
                    textcoords="offset points", color="red", fontsize=13)
    ax.set_xlabel(r"Instances ($\times 10^3$)", fontsize=18)
    ax.set_ylabel("Composite Score", fontsize=18)
    ax.grid(True, linestyle=":", color="0.7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10.5, ncol=2, loc="lower right", frameon=True,
              framealpha=0.9)
    save(fig, name)


def time_memory(data, extra):
    """Composite vs memory scatter (efficiency trade-off) using the group view:
    Best(Default) + AutoML methods. Labels offset to avoid overlap."""
    datasets = list(data.keys())
    models = [m for m in GROUP_ORDER
              if m == "Best(Default)" or any(m in data[d] for d in datasets)]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for m in models:
        comps, mems = [], []
        for ds in datasets:
            if m == "Best(Default)":
                best_s = max((s for s in SINGLE_ALGOS if s in data[ds]),
                             key=lambda s: composite(data, ds, s))
                comps.append(composite(data, ds, best_s))
                mems.append(np.nanmean(extra[ds][best_s]["memory"]))
            elif m in data[ds]:
                comps.append(composite(data, ds, m))
                mems.append(np.nanmean(extra[ds][m]["memory"]))
        mx, my = np.nanmean(mems), np.mean(comps)
        is_ours = (m == "AutoDSC")
        ax.scatter(mx, my, s=360 if is_ours else 240,
                   color=GROUP_COLOR[m], edgecolor="black",
                   linewidth=1.6 if is_ours else 1.0,
                   marker="*" if is_ours else "o",
                   zorder=5 if is_ours else 3, label=GROUP_LABEL[m])
    ax.set_xlabel("Memory (MB)", fontsize=18)
    ax.set_ylabel("Mean Composite Score", fontsize=18)
    ax.grid(True, linestyle=":", color="0.7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, loc="lower right", frameon=True, framealpha=0.95,
              scatterpoints=1, handletextpad=0.3, labelspacing=0.3)
    save(fig, "time_memory_tradeoff")


# ---- AutoML-group view: Best(Default) + AutoML methods ----
SINGLE_ALGOS = ["CluStream", "DBSTREAM", "DenStream", "KMeans"]
GROUP_ORDER = ["Best(Default)", "MOStream", "ConfStream",
               "OnlineAutoClustEnsemble", "RandomSearchStream", "AutoDSC"]
GROUP_LABEL = dict(MODEL_LABEL, **{
    "Best(Default)": "Best(Default)",
    "OnlineAutoClustEnsemble": "OACEnsemble",
    "RandomSearchStream": "RandSearch",
})
GROUP_COLOR = dict(MODEL_COLOR, **{"Best(Default)": "#9AA7B0"})  # steel gray
GROUP_HATCH = dict(MODEL_HATCH, **{"Best(Default)": "//"})


def group_composite(data, ds, m):
    if m == "Best(Default)":
        return max(composite(data, ds, s) for s in SINGLE_ALGOS if s in data[ds])
    return composite(data, ds, m)


def bar_group_view(data, name):
    """Composite bars: Best(Default) + AutoML methods (oracle single-algo)."""
    datasets = [d for d in DATASET_ORDER if d in data]
    models = [m for m in GROUP_ORDER
              if m == "Best(Default)" or any(m in data[d] for d in datasets)]
    x = np.arange(len(datasets))
    w = 0.85 / len(models)
    fig, ax = plt.subplots(figsize=(13, 3.6))
    for i, m in enumerate(models):
        vals = [group_composite(data, ds, m) for ds in datasets]
        ax.bar(x + i * w - 0.425 + w / 2, vals, w,
               color=GROUP_COLOR[m], hatch=GROUP_HATCH[m],
               edgecolor="black", linewidth=0.7, label=GROUP_LABEL[m])
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in datasets], fontsize=15)
    ax.set_ylabel("Composite Score", fontsize=18)
    ax.yaxis.grid(True, linestyle=":", color="0.6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=len(models), fontsize=12, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 1.22),
              handlelength=1.4, columnspacing=1.0, handletextpad=0.4)
    save(fig, name)


def throughput_temporal(name, dataset, raw_files_dir=None):
    """Per-window throughput (instances/sec) over the stream vs instances,
    MOStream Dynamic style. Log y-axis (single algos are ~100x faster than
    AutoML methods). AutoDSC highlighted."""
    import glob
    rdir = raw_files_dir or RESULTS_DIR
    show = ["KMeans", "MOStream", "RandomSearchStream", "ConfStream",
            "OnlineAutoClustEnsemble", "AutoDSC"]
    markers = {"AutoDSC": "o", "ConfStream": "D", "MOStream": "s",
               "OnlineAutoClustEnsemble": "^", "RandomSearchStream": "v",
               "KMeans": "P"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for m in show:
        runs = []
        for f in glob.glob(os.path.join(rdir, f"{m}_{dataset}_*.json")):
            d = json.load(open(f))
            wm = d.get("window_metric", {})
            rts = wm.get("runtimes")          # cumulative runtime per window
            ws = wm.get("window_size", 1000)
            if rts:
                # running throughput = cumulative instances / cumulative runtime
                tp = [((i + 1) * ws) / r if r > 0 else np.nan
                      for i, r in enumerate(rts)]
                runs.append(np.asarray(tp))
        if not runs:
            continue
        L = min(len(r) for r in runs)
        arr = np.nanmean([r[:L] for r in runs], axis=0)
        xs = np.arange(L) * 1000 / 1000.0  # instances in thousands (ws=1000)
        every = max(1, L // 14)
        is_ours = (m == "AutoDSC")
        ax.plot(xs, arr, marker=markers.get(m, "o"), markevery=every,
                markersize=9 if is_ours else 5,
                linewidth=3.5 if is_ours else 1.4,
                color=MODEL_COLOR[m], label=MODEL_LABEL[m],
                alpha=1.0 if is_ours else 0.8,
                markeredgecolor="black" if is_ours else None,
                markeredgewidth=0.8 if is_ours else 0,
                zorder=6 if is_ours else 3)
    ax.set_yscale("log")
    ax.set_xlabel(r"Instances ($\times 10^3$)", fontsize=18)
    ax.set_ylabel("Throughput (ips)", fontsize=18)
    ax.grid(True, which="both", linestyle=":", color="0.7", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.28), frameon=False, columnspacing=1.0,
              handletextpad=0.4)
    save(fig, name)


def sensitivity_charts(sens_dir, dataset_filter, tag):
    """MOStream dual-axis sensitivity: accuracy (composite ARI+CMM, left) +
    efficiency (throughput, right) vs each parameter value, averaged over the
    selected datasets and seeds. One chart per parameter."""
    import glob
    PARAMS = {"window_size": [500, 1000, 1500, 2000],
              "population_size": [10, 15, 20, 25, 30],
              "gamma": [0.01, 0.05, 0.1, 0.2, 0.3],
              "diversity_threshold": [0.05, 0.1, 0.15, 0.2, 0.25]}
    DEFAULT = {"window_size": 1000, "population_size": 20,
               "gamma": 0.1, "diversity_threshold": 0.1}
    PLABEL = {"window_size": "Window Size", "population_size": "Population Size",
              "gamma": r"$\gamma$", "diversity_threshold": "Diversity Threshold"}
    # index files by (param tuple) -> list of (comp, throughput)
    store = {}
    for f in glob.glob(os.path.join(sens_dir, "*.json")):
        d = json.load(open(f))
        if d["dataset"] not in dataset_filter:
            continue
        mp = d["model_params"]
        key = (mp["window_size"], mp["population_size"], mp["gamma"],
               mp["diversity_threshold"])
        comp = np.mean([d["ari"], d["cmm"]])
        store.setdefault(key, []).append((comp, d["throughput_ips"]))

    def at(pname, v):
        cfg = dict(DEFAULT); cfg[pname] = v
        key = (cfg["window_size"], cfg["population_size"], cfg["gamma"],
               cfg["diversity_threshold"])
        vals = store.get(key, [])
        if not vals:
            return None, None
        return np.mean([x[0] for x in vals]), np.mean([x[1] for x in vals])

    ACC = "#2C6FBB"   # blue, accuracy
    EFF = "#A0C53F"   # lime, efficiency
    for pname, pvals in PARAMS.items():
        accs, effs = zip(*[at(pname, v) for v in pvals])
        if any(a is None for a in accs):
            continue
        x = list(range(len(pvals)))
        fig, ax1 = plt.subplots(figsize=(5.2, 3.6))
        ax2 = ax1.twinx()
        l1, = ax1.plot(x, accs, marker="^", markersize=9, linewidth=2.5,
                       color=ACC, label="Composite")
        l2, = ax2.plot(x, effs, marker="s", markersize=9, linewidth=2.5,
                       color=EFF, label="Throughput")
        ax1.set_xticks(x); ax1.set_xticklabels([str(v) for v in pvals])
        ax1.set_xlabel(PLABEL[pname], fontsize=18)
        ax1.set_ylabel("Composite", color=ACC, fontsize=17)
        ax2.set_ylabel("Throughput (ips)", color=EFF, fontsize=17)
        ax1.tick_params(axis="y", labelcolor=ACC)
        ax2.tick_params(axis="y", labelcolor=EFF)
        ax1.grid(True, linestyle=":", color="0.8", linewidth=0.7)
        ax1.set_axisbelow(True)
        save(fig, f"sensitivity_{tag}_{pname}")


def cd_diagram(data, name, metric="composite"):
    """Critical Difference diagram (Friedman + Nemenyi) via autorank.
    Rows = datasets (samples), columns = methods. Higher score = better."""
    import pandas as pd
    from autorank import autorank, plot_stats

    datasets = [d for d in DATASET_ORDER if d in data]
    models = [m for m in GROUP_ORDER
              if m == "Best(Default)" or any(m in data[d] for d in datasets)]

    def group_metric(ds, m):
        if m == "Best(Default)":
            cand = [s for s in SINGLE_ALGOS if s in data[ds]]
            if metric == "composite":
                return max(composite(data, ds, s) for s in cand)
            best_s = max(cand, key=lambda s: composite(data, ds, s))
            return np.mean(data[ds][best_s][metric])
        if metric == "composite":
            return composite(data, ds, m)
        return np.mean(data[ds][m][metric])

    rows = {ds: {GROUP_LABEL[m]: group_metric(ds, m) for m in models}
            for ds in datasets}
    df = pd.DataFrame.from_dict(rows, orient="index")

    res = autorank(df, alpha=0.05, verbose=False, order="descending",
                   force_mode="nonparametric")
    with plt.rc_context({"font.size": 13, "axes.titlesize": 13,
                         "xtick.labelsize": 13}):
        fig, ax = plt.subplots(figsize=(9, 2.6))
        plot_stats(res, ax=ax, allow_insignificant=True)
    save(fig, name)
    print(f"  [{metric}] Friedman p={res.pvalue:.3g}; "
          f"ranks: " + ", ".join(
              f"{k}={v:.2f}" for k, v in
              res.rankdf['meanrank'].sort_values().items()))


def main():
    data, extra, windows = load()
    models = models_present(data)
    print("methods:", models)
    print("datasets:", [d for d in DATASET_ORDER if d in data])

    make_legend(models)
    bar_grouped(data, models, composite, "Composite Score",
                "general_composite_bar")
    bar_group_view(data, "composite_group_view")
    bar_grouped(data, models,
                lambda dd, ds, m: np.nanmean(extra[ds][m]["throughput"]),
                "Throughput (ips)", "general_throughput_bar", sci=True)
    temporal(windows, "synth_blobs_expanding", "temporal_blobs_expanding")
    temporal(windows, "real_shuttle", "temporal_shuttle")
    time_memory(data, extra)
    cd_diagram(data, "cd_ari", metric="ari")
    cd_diagram(data, "cd_composite", metric="composite")
    throughput_temporal("throughput_blobs_expanding", "synth_blobs_expanding")
    throughput_temporal("throughput_shuttle", "real_shuttle")


if __name__ == "__main__":
    main()
