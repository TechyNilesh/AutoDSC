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


# Add project root to path to import ConfStream
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'baselines', 'ConfStream'))
from ConfStream import ConfStream

import warnings
warnings.filterwarnings("ignore")


def load_streaming_dataset(dataset_name):
    """Load streaming dataset from CSV file."""
    try:
        # Dataset path can be a full path or relative
        if os.path.exists(dataset_name):
            dataset_path = dataset_name
        else:
            dataset_path = os.path.join(project_root, "datasets", dataset_name + ".csv")

        # Load CSV file
        df = pd.read_csv(dataset_path)

        X = df.copy()
        y = X.pop("class")

        dataset = stream.iter_pandas(X, y)

        return dataset

    except FileNotFoundError:
        print(f"Dataset {dataset_name} not found at {dataset_path}")
        return None
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {str(e)}")
        return None


def run_confstream(dataset_name, run_count=None, seed=42, progress_interval=1000, 
                   output_dir="results/benchmark", verbose=True):
    """Run ConfStream algorithm on a dataset."""
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n===Running ConfStream on {dataset_name}===")
    
    try:
        
        dataset = load_streaming_dataset(dataset_name)
        
        if dataset is None:
            print(f"Failed to load dataset {dataset_name}")
            return None
                
        # Set ConfStream parameters
        confstream_params = {
            'ensemble_size': 20,  # Number of ensemble members
            'window_size': 1000, # Size of the window for evaluating performance
            'fading_factor': 0.05,
            'reset_probability': 0.01,
            'test_ensemble_size': 10,
            'random_seed': seed
        }
        
        # Initialize model
        model = ConfStream(**confstream_params)
        
        ev = StreamEvaluator(window=progress_interval)
        for X, y in dataset:
            x_values = list(X.values())
            pred = model.predict_one(x_values)
            model.learn_one(x_values)
            if pred is None or pred == -1:
                continue
            ev.update(y, pred, x_values)

        results = ev.finalize(model="ConfStream", dataset=dataset_name,
                              params=confstream_params, run_count=run_count,
                              seed=seed, verbose=verbose)
        file_name = (f"ConfStream_{dataset_name}_{run_count}.json" if run_count is not None
                     else f"ConfStream_{dataset_name}.json")
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        if verbose:
            print(f"ConfStream completed on {dataset_name} -> {output_file}")
        return results
        
    except Exception as e:
        print(f"Error running ConfStream on {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ConfStream on a dataset')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Name of the dataset to run ConfStream on')
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
    
    args = parser.parse_args()
    
    # Run ConfStream with provided arguments
    result = run_confstream(
        dataset_name=args.dataset, 
        run_count=args.run_count, 
        seed=args.seed, 
        progress_interval=args.progress_interval, 
        output_dir=args.output_dir, 
        verbose=not args.quiet
    )
    
    if result is None:
        print(f"\nConfStream run failed!")
        sys.exit(1)
    else:
        print(f"\nConfStream run completed successfully!")
        sys.exit(0)