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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter
from utils.purityscore import Purity
from utils.metrics import StreamEvaluator


# Add project root to path to import AutoDSC
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'autodsc'))
from AutoDSC import AutoDSC

import warnings
warnings.filterwarnings("ignore")


def load_streaming_dataset(dataset_name):
    """Load streaming dataset from CSV file."""
    try:
        # Dataset path can be a full path or relative
        if os.path.exists(dataset_name):
            dataset_path = dataset_name
        else:
            # Try project root datasets folder
            dataset_path = os.path.join(project_root, "datasets", dataset_name + ".csv")

        # Check if file exists
        if not os.path.exists(dataset_path):
            # Try alternative paths
            dataset_path = os.path.join("datasets", dataset_name + ".csv")
        
        # Load CSV file
        df = pd.read_csv(dataset_path)
        
        # Separate features and labels
        X = df.copy()
        y = X.pop("class")

        # Create stream dataset
        dataset = stream.iter_pandas(X, y)

        return dataset
    
    except FileNotFoundError:
        print(f"Dataset {dataset_name} not found. Tried paths:")
        print(f"  - {dataset_name}")
        print(f"  - {os.path.join(project_root, 'datasets', dataset_name + '.csv')}")
        print(f"  - datasets/{dataset_name}.csv")
        return None
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {str(e)}")
        return None


def run_AutoDSC(dataset_name, run_count=None, seed=42, progress_interval=1000,
                        output_dir="results/benchmark", verbose=True,
                        window_size=1000, population_size=20,
                        gamma=0.1, diversity_threshold=0.1,
                        model_name="AutoDSC"):
    """Run AutoDSC algorithm on a dataset."""

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n===Running AutoDSC on {dataset_name}===")

    try:

        dataset = load_streaming_dataset(dataset_name)

        if dataset is None:
            print(f"Failed to load dataset {dataset_name}")
            return None

        # Set AutoDSC parameters
        AutoDSC_params = {
            'window_size': window_size,                 # Window size for processing
            'population_size': population_size,         # Size of configuration archive
            'gamma': gamma,
            'diversity_threshold': diversity_threshold,
            'random_seed': seed
        }

        # Initialize model
        model = AutoDSC(**AutoDSC_params)

        ev = StreamEvaluator(window=progress_interval)
        for X, y in dataset:
            x_values = list(X.values())
            pred = model.predict_one(x_values)
            model.learn_one(x_values)
            if pred is None or pred == -1:
                continue
            ev.update(y, pred, x_values)

        results = ev.finalize(model=model_name, dataset=dataset_name,
                              params={"window_size": window_size, "population_size": population_size,
                                      "gamma": gamma, "diversity_threshold": diversity_threshold},
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
            print(f"AutoDSC completed on {dataset_name} -> {output_file}")
        return results

    except Exception as e:
        print(f"Error running AutoDSC on {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AutoDSC on a dataset')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Name of the dataset to run AutoDSC on')
    parser.add_argument('--run_count', type=int, default=None,
                       help='Run count identifier')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--progress_interval', type=int, default=1000,
                       help='Interval for progress updates')
    parser.add_argument('--output_dir', type=str, default="results/benchmark",
                       help='Output directory')
    parser.add_argument('--quiet', action='store_true', default=False,
                       help='Disable verbose output')
    
    # AutoDSC parameters
    parser.add_argument('--window_size', type=int, default=1000,
                       help='Window size for processing')
    parser.add_argument('--population_size', type=int, default=20,
                       help='Size of configuration archive')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma parameter')
    parser.add_argument('--diversity_threshold', type=float, default=0.1,
                       help='Diversity threshold')
    parser.add_argument('--model_name', type=str, default="AutoDSC",
                       help='Custom model name for output files')

    args = parser.parse_args()

    # Run AutoDSC with provided arguments
    result = run_AutoDSC(
        dataset_name=args.dataset,
        run_count=args.run_count,
        seed=args.seed,
        progress_interval=args.progress_interval,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        window_size=args.window_size,
        population_size=args.population_size,
        gamma=args.gamma,
        diversity_threshold=args.diversity_threshold,
        model_name=args.model_name
    )

    if result is None:
        print(f"\nAutoDSC run failed!")
        sys.exit(1)
    else:
        print(f"\nAutoDSC run completed successfully!")
        sys.exit(0)
