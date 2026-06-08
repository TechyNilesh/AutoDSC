"""
Single-run script to capture the per-window exploration rate (alpha_t) of AutoDSC
on the Sudden stream and save it to JSON for plotting.

Usage:
    python run_alpha_trace.py
Output:
    Code/results/analysis/alpha_trace_sudden.json
"""

import os, sys, json
import numpy as np
import pandas as pd
from river import stream

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'src', 'autodsc'))
from AutoDSC import AutoDSC

DATASETS = ['synth_blobs_sudden', 'synth_blobs_expanding', 'synth_blobs_merge_split']
SEED = 42
OUT_DIR = os.path.join(project_root, 'results', 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

for ds_name in DATASETS:
    dataset_path = os.path.join(project_root, 'datasets', f'{ds_name}.csv')
    df = pd.read_csv(dataset_path)
    X = df.copy()
    y = X.pop('class')

    model = AutoDSC(window_size=1000, population_size=10, random_seed=SEED)

    for (xi, _), _ in zip(stream.iter_pandas(X, y), range(len(df))):
        x_values = list(xi.values())
        model.predict_one(x_values)
        model.learn_one(x_values)

    result = {
        'dataset': ds_name,
        'window_size': model.window_size,
        'seed': SEED,
        'window_idx': model.trace['window_idx'],
        'alpha': model.trace['alpha'],
        'composite': model.trace['composite'],
    }

    out_file = os.path.join(OUT_DIR, f'alpha_trace_{ds_name}.json')
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved {len(result['window_idx'])} windows -> {out_file}")
