#!/usr/bin/env python3
"""
MOA-style RandomRBFGeneratorDrift for stream-clustering benchmarks.

Each class is represented by one or more RBF centroids placed in [0,1]^d.
Points are drawn as Gaussian samples around a (weighted) chosen centroid and
labelled by that centroid's class. Concept drift is injected by translating a
subset of centroids along fixed random unit vectors at a constant speed.

Separation is controlled by `std` (small => well-separated, high ARI achievable),
matching the separable RBF settings used in the stream-clustering literature.
"""

import numpy as np
import pandas as pd
import argparse


def generate_rbf_drift(n_samples=100_000, n_features=5, n_classes=5,
                       centroids_per_class=1, std=0.05, speed=0.0,
                       n_drift_centroids=None, seed=42):
    rng = np.random.default_rng(seed)
    n_centroids = n_classes * centroids_per_class

    centers = rng.uniform(0.0, 1.0, size=(n_centroids, n_features))
    classes = np.repeat(np.arange(n_classes), centroids_per_class)
    weights = rng.uniform(0.5, 1.0, size=n_centroids)
    weights /= weights.sum()

    # Drift: assign fixed random unit directions to a subset of centroids
    if n_drift_centroids is None:
        n_drift_centroids = n_centroids
    drift_idx = rng.choice(n_centroids, size=n_drift_centroids, replace=False)
    directions = rng.normal(size=(n_centroids, n_features))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    drift_mask = np.zeros(n_centroids, dtype=bool)
    drift_mask[drift_idx] = True

    X = np.empty((n_samples, n_features), dtype=np.float64)
    y = np.empty(n_samples, dtype=int)

    cur = centers.copy()
    for t in range(n_samples):
        # advance drifting centroids, bounce off [0,1] walls to stay in-range
        if speed > 0:
            cur[drift_mask] += directions[drift_mask] * speed
            over = cur > 1.0
            under = cur < 0.0
            cur[over] = 2.0 - cur[over]; directions[over] *= -1
            cur[under] = -cur[under]; directions[under] *= -1
        c = rng.choice(n_centroids, p=weights)
        pt = cur[c] + rng.normal(0.0, std, size=n_features)
        X[t] = np.clip(pt, 0.0, 1.0)
        y[t] = classes[c]

    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["class"] = y
    return df


# Per-instance speed is tiny; scale so total drift over the stream is meaningful.
PRESETS = {
    # gradual, continuous drift across the whole stream
    "synth_rbf_gradual": dict(speed=3e-6, std=0.05, seed=42),
    # faster drift (clusters traverse more of the space)
    "synth_rbf_fast":    dict(speed=8e-6, std=0.05, seed=7),
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--n_samples", type=int, default=100_000)
    args = ap.parse_args()

    import os
    for name, kw in PRESETS.items():
        df = generate_rbf_drift(n_samples=args.n_samples, **kw)
        path = os.path.join(args.out_dir, name + ".csv")
        df.to_csv(path, index=False)
        print(f"wrote {path}  rows={len(df)} classes={df['class'].nunique()}")
