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
from utils.purityscore import Purity
from utils.metrics import StreamEvaluator

from stream_clustering import KMeans

import warnings
warnings.filterwarnings("ignore")

# Get project root for dataset paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def run_KMeans(dataset_name, run_count=None, seed=42, progress_interval=1000, 
               output_dir="results/benchmark", verbose=True):
    """Run KMeans algorithm on a dataset."""
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n===Running KMeans on {dataset_name}===")
    
    try:
        
        dataset = load_streaming_dataset(dataset_name)
        
        if dataset is None:
            print(f"Failed to load dataset {dataset_name}")
            return None
                
        # Set KMeans parameters (matching Rust implementation)
        kmeans_params = {
            "n_clusters": 8,
            "halflife": 0.5,
            "mu": 0.0,
            "sigma": 1.0,
            "p": 2.0,
            "seed": seed
        }

        # Initialize model using stream_clustering.KMeans
        model = KMeans(**kmeans_params)
        
        ev = StreamEvaluator(window=progress_interval)
        for X, y in dataset:
            X = list(X.values())
            pred = model.predict_one(X)
            model.learn_one(X)
            if pred is None or pred == -1:
                continue
            ev.update(y, pred, X)

        results = ev.finalize(model="KMeans", dataset=dataset_name,
                              params=kmeans_params, run_count=run_count,
                              seed=seed, verbose=verbose)
        file_name = (f"KMeans_{dataset_name}_{run_count}.json" if run_count is not None
                     else f"KMeans_{dataset_name}.json")
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        if verbose:
            print(f"KMeans completed on {dataset_name} -> {output_file}")
        return results
        
    except Exception as e:
        print(f"Error running KMeans on {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run KMeans on a dataset')
    parser.add_argument('--dataset', type=str, help='name of the dataset to run KMeans on')
    parser.add_argument('--run_count', type=int, default=None, help='Run count identifier')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--progress_interval', type=int, default=1000, help='Interval for progress updates')
    parser.add_argument('--output_dir', type=str, default="results/benchmark", help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Disable verbose output', default=False)
    
    args = parser.parse_args()
    
    if args.dataset is None:
        print("Please provide a dataset name using --dataset argument.")
        sys.exit(1)
    
    # Run KMeans with provided arguments
    run_KMeans(dataset_name=args.dataset, 
               run_count=args.run_count, 
               seed=args.seed, 
               progress_interval=args.progress_interval, 
               output_dir=args.output_dir, 
               verbose=not args.quiet)
