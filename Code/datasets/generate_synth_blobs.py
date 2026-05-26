#!/usr/bin/env python3
"""
Generator for the synthetic Gaussian-blob stream-clustering benchmarks.

Five clusters live in [0,1]^d. Points are Gaussian samples around per-class
centroids; the stream injects one of five drift mechanisms (verified to match
the shipped synth_blobs_* datasets):

  - sudden      : centroids jump abruptly at a few change points
  - gradual     : centroids translate smoothly at a constant rate
  - incremental : centroids drift outward then return (triangle profile)
  - expanding   : centroids fixed, within-cluster variance grows over time
  - merge_split : clusters converge toward the global mean, then re-separate

Reproducible (seeded). Output schema: f0..f{d-1}, class  ->  matches the suite.

NOTE: regenerating overwrites with statistically-equivalent (not byte-identical)
data, since the original RNG seed of the shipped files is unknown. Only rerun
the benchmark after regenerating if you intend to refresh the suite.
"""

import os
import argparse
import numpy as np
import pandas as pd


def _base_centroids(rng, n_classes, n_features, min_sep=0.35):
    """Sample reasonably separated centroids in [0,1]^d."""
    centers = []
    while len(centers) < n_classes:
        c = rng.uniform(0.15, 0.85, size=n_features)
        if all(np.linalg.norm(c - e) >= min_sep for e in centers):
            centers.append(c)
    return np.array(centers)


def generate_blobs_drift(drift_type, n_samples=100_000, n_features=5,
                         n_classes=5, std=0.04, seed=42):
    rng = np.random.default_rng(seed)
    base = _base_centroids(rng, n_classes, n_features)
    global_mean = base.mean(axis=0)

    # Fixed per-class drift directions (unit vectors) for translation drifts
    directions = rng.normal(size=(n_classes, n_features))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Drift schedule magnitudes calibrated to the shipped datasets
    max_shift = 0.40           # peak centroid translation
    sudden_points = [0.3, 0.45, 0.75]   # fractional change points
    sudden_steps = [0.23, 0.33, 0.44]   # cumulative shift after each point
    expand_std_end = 0.077     # final blob std for 'expanding'
    merge_frac = 0.20          # how far clusters collapse toward global mean

    X = np.empty((n_samples, n_features))
    y = np.empty(n_samples, dtype=int)

    for t in range(n_samples):
        p = t / n_samples
        cur = base.copy()
        cur_std = std

        if drift_type == "gradual":
            cur = base + directions * (max_shift * p)

        elif drift_type == "sudden":
            shift = 0.0
            for cp, st in zip(sudden_points, sudden_steps):
                if p >= cp:
                    shift = st
            cur = base + directions * shift

        elif drift_type == "incremental":
            tri = 1.0 - abs(2.0 * p - 1.0)          # 0 ->1 ->0
            cur = base + directions * (0.64 * tri)

        elif drift_type == "expanding":
            cur_std = std + (expand_std_end - std) * p   # variance grows

        elif drift_type == "merge_split":
            tri = 1.0 - abs(2.0 * p - 1.0)          # converge then re-separate
            cur = base + (global_mean - base) * (merge_frac * tri)

        else:
            raise ValueError(f"unknown drift_type: {drift_type}")

        c = rng.integers(n_classes)
        X[t] = np.clip(cur[c] + rng.normal(0.0, cur_std, n_features), 0.0, 1.0)
        y[t] = c

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["class"] = y
    return df


PRESETS = {
    "synth_blobs_sudden":      dict(drift_type="sudden",      seed=11),
    "synth_blobs_gradual":     dict(drift_type="gradual",     seed=22),
    "synth_blobs_incremental": dict(drift_type="incremental", seed=33),
    "synth_blobs_expanding":   dict(drift_type="expanding",   seed=44),
    "synth_blobs_merge_split": dict(drift_type="merge_split", seed=55),
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--n_samples", type=int, default=100_000)
    args = ap.parse_args()

    for name, kw in PRESETS.items():
        df = generate_blobs_drift(n_samples=args.n_samples, **kw)
        path = os.path.join(args.out_dir, name + ".csv")
        df.to_csv(path, index=False)
        print(f"wrote {path}  rows={len(df)} classes={df['class'].nunique()}")
