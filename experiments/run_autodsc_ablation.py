import numpy as np
import pandas as pd
import random
import json
import os
import argparse
import sys
from river import stream

# Path setup: this file lives in experiments/
this_dir = os.path.dirname(os.path.abspath(__file__))            # experiments
code_root = os.path.dirname(this_dir)                            # repo root
scripts_dir = os.path.join(this_dir, 'scripts')                  # experiments/scripts
sys.path.append(code_root)
sys.path.append(os.path.join(code_root, 'src', 'autodsc'))       # for AutoDSC
sys.path.append(scripts_dir)                                     # for utils.*
sys.path.append(os.path.join(scripts_dir, 'ablation'))           # for AutoDSC_ablation

from utils.purityscore import Purity
from utils.metrics import StreamEvaluator
from AutoDSC_ablation import AutoDSCAblation, ABLATIONS

import warnings
warnings.filterwarnings("ignore")


def load_streaming_dataset(dataset_name):
    """Load streaming dataset from CSV file."""
    try:
        if os.path.exists(dataset_name):
            dataset_path = dataset_name
        else:
            dataset_path = os.path.join(code_root, "datasets", dataset_name + ".csv")
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join("datasets", dataset_name + ".csv")
        df = pd.read_csv(dataset_path)
        X = df.copy()
        y = X.pop("class")
        return stream.iter_pandas(X, y)
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {str(e)}")
        return None


def run_ablation(dataset_name, ablation, run_count=None, seed=42, progress_interval=1000,
                 output_dir="results/ablation", verbose=True,
                 window_size=1000, population_size=10):
    """Run an AutoDSC ablation variant on a dataset."""
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    model_name = f"AutoDSC_Ablation_{ablation}"
    print(f"\n===Running {model_name} on {dataset_name}===")

    try:
        dataset = load_streaming_dataset(dataset_name)
        if dataset is None:
            print(f"Failed to load dataset {dataset_name}")
            return None

        model = AutoDSCAblation(
            window_size=window_size,
            population_size=population_size,
            random_seed=seed,
            ablation=ablation,
        )

        ev = StreamEvaluator(window=progress_interval)
        for X, y in dataset:
            x_values = list(X.values())
            pred = model.predict_one(x_values)
            model.learn_one(x_values)
            if pred is None or pred == -1:
                continue
            ev.update(y, pred, x_values)

        results = ev.finalize(
            model=model_name, dataset=dataset_name,
            params={"window_size": window_size, "population_size": population_size,
                    "ablation": ablation},
            run_count=run_count, seed=seed, verbose=verbose)
        try:
            results["AutoDSC_summary"] = model.get_performance_summary()
        except Exception:
            pass

        file_name = (f"{model_name}_{dataset_name}_{run_count}.json" if run_count is not None
                     else f"{model_name}_{dataset_name}.json")
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        if verbose:
            print(f"{model_name} completed on {dataset_name} -> {output_file}")
        return results

    except Exception as e:
        print(f"Error running {model_name} on {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run an AutoDSC ablation variant on a dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--ablation', type=str, required=True, choices=ABLATIONS,
                        help='Ablation variant to run')
    parser.add_argument('--run_count', type=int, default=None, help='Run count identifier')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--progress_interval', type=int, default=1000,
                        help='Interval for progress updates')
    parser.add_argument('--output_dir', type=str, default="results/ablation",
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Disable verbose output')
    parser.add_argument('--window_size', type=int, default=1000, help='Window size')
    parser.add_argument('--population_size', type=int, default=10, help='Population size')

    args = parser.parse_args()

    result = run_ablation(
        dataset_name=args.dataset,
        ablation=args.ablation,
        run_count=args.run_count,
        seed=args.seed,
        progress_interval=args.progress_interval,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        window_size=args.window_size,
        population_size=args.population_size,
    )

    sys.exit(0 if result is not None else 1)
