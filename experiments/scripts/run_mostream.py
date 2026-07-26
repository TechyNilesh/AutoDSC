import numpy as np
import pandas as pd
import random
import time
import psutil
import json
import os
import argparse
import sys
from river import stream
from river import metrics
from river import preprocessing
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from utils.purityscore import Purity
from utils.metrics import StreamEvaluator

# MOStream is the self-optimizing "Benne" from intellistream/Sesame (pysame bindings)
from pysame import Benne, BenneObj

import warnings
warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sesame_kmeanspp(points, k, seed, max_iter=300):
    """Faithful port of Sesame's offline KMeans refinement
    (src/Algorithm/OfflineRefinement/KMeans.cpp): k-means++ initialization
    (D^2-weighted center selection) followed by Lloyd iterations until the
    point-to-center assignment stops changing. Single initialization, seeded.
    Returns the k macro-cluster centers."""
    rng = np.random.RandomState(seed)
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n <= k:
        return pts.copy()
    # k-means++ init: first center random, rest by D^2 weighting
    centers = [pts[rng.randint(n)]]
    for _ in range(1, k):
        d2 = np.min([np.sum((pts - c) ** 2, axis=1) for c in centers], axis=0)
        s = d2.sum()
        probs = d2 / s if s > 0 else np.full(n, 1.0 / n)
        centers.append(pts[rng.choice(n, p=probs)])
    centers = np.array(centers)
    # Lloyd iterations until assignments stabilize
    prev = None
    for _ in range(max_iter):
        d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        assign = np.argmin(d, axis=1)
        if prev is not None and np.array_equal(assign, prev):
            break
        prev = assign
        for j in range(k):
            grp = pts[assign == j]
            if len(grp):
                centers[j] = grp.mean(axis=0)
    return centers


def load_streaming_dataset(dataset_name):
    try:
        if os.path.exists(dataset_name):
            dataset_path = dataset_name
        else:
            dataset_path = os.path.join(project_root, "datasets", dataset_name + ".csv")
        df = pd.read_csv(dataset_path)
        X = df.copy()
        y = X.pop("class")
        return stream.iter_pandas(X, y)
    except FileNotFoundError:
        print(f"Dataset {dataset_name} not found at {dataset_path}")
        return None
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {str(e)}")
        return None


def run_MOStream(dataset_name, run_count=None, seed=42, progress_interval=1000,
                 output_dir="results/benchmark", verbose=True,
                 n_clusters=5, distance_threshold=0.5, queue_size_threshold=1000):
    """Run MOStream (Benne, Accuracy objective) on a dataset.

    MOStream's API is batch (partial_fit/predict), so we process the stream in
    windows of `progress_interval`: partial_fit each window, then predict it.
    This matches our windowed evaluation and Sesame's own batch evaluation.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n===Running MOStream on {dataset_name}===")

    try:
        dataset = load_streaming_dataset(dataset_name)
        if dataset is None:
            print(f"Failed to load dataset {dataset_name}")
            return None

        params = {"n_clusters": n_clusters, "distance_threshold": distance_threshold,
                  "queue_size_threshold": queue_size_threshold, "objective": "accuracy"}
        # MOStream/Benne is a fixed-radius method and crashes (micro-cluster
        # explosion) on raw multi-scale data; standardize features online.
        scaler = preprocessing.StandardScaler()

        ev = StreamEvaluator(window=progress_interval)
        # Collect the whole (standardized) stream. Benne must be fit ONCE on the
        # full stream — repeated partial_fit calls crash (segfault).
        all_X, all_true = [], []
        for X, y in dataset:
            scaler.learn_one(X)
            Xs = scaler.transform_one(X)
            all_X.append([Xs[k] for k in X.keys()])
            all_true.append(y)

        model = Benne(n_clusters=n_clusters, dim=len(all_X[0]),
                      distance_threshold=distance_threshold,
                      queue_size_threshold=queue_size_threshold, obj=BenneObj.accuracy)
        model.partial_fit(all_X)
        micro = np.asarray([int(p) for p in model.predict(all_X)])
        # Benne's predict() returns micro-cluster IDs (hundreds/thousands), not
        # the n_clusters macro assignment. We replicate Sesame's own offline
        # KMeans refinement (src/Algorithm/OfflineRefinement/KMeans.cpp): run
        # k-means++ on the online (micro) centers to obtain n_clusters macro
        # centers, then assign every point to its nearest macro center
        # (UtilityFunctions::groupByCenters). Without this, MOStream is evaluated
        # on raw micro IDs, which trivially inflates CMM and destroys ARI.
        Xarr = np.asarray(all_X, dtype=float)
        uniq = np.unique(micro)
        if len(uniq) > n_clusters:
            micro_centers = np.array([Xarr[micro == u].mean(axis=0) for u in uniq])
            macro_centers = _sesame_kmeanspp(micro_centers, n_clusters, seed)
            # groupByCenters: each point -> nearest macro center
            d = np.linalg.norm(Xarr[:, None, :] - macro_centers[None, :, :], axis=2)
            preds = np.argmin(d, axis=1).tolist()
        else:
            preds = micro.tolist()
        for yt, p, xx in zip(all_true, preds, all_X):
            ev.update(yt, p, xx)

        results = ev.finalize(model="MOStream", dataset=dataset_name,
                              params=params, run_count=run_count, seed=seed, verbose=verbose)
        file_name = (f"MOStream_{dataset_name}_{run_count}.json" if run_count is not None
                     else f"MOStream_{dataset_name}.json")
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        if verbose:
            print(f"MOStream completed on {dataset_name} -> {output_file}")
        return results

    except Exception as e:
        print(f"Error running MOStream on {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MOStream (Benne) on a dataset")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run_count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_interval", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results/benchmark")
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--distance_threshold", type=float, default=0.5)
    parser.add_argument("--queue_size_threshold", type=int, default=1000)
    parser.add_argument("--quiet", action="store_true", default=False)
    args = parser.parse_args()

    run_MOStream(dataset_name=args.dataset, run_count=args.run_count, seed=args.seed,
                 progress_interval=args.progress_interval, output_dir=args.output_dir,
                 verbose=not args.quiet, n_clusters=args.n_clusters,
                 distance_threshold=args.distance_threshold,
                 queue_size_threshold=args.queue_size_threshold)
