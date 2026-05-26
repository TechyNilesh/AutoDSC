#!/usr/bin/env python3
"""
Run OnlineAutoClust on streaming datasets
Similar to run_confstream.py but for OnlineAutoClust algorithms
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import random
import json
import time
import psutil
import warnings
from river import stream
from river import metrics
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter
from utils.purityscore import Purity
from utils.metrics import StreamEvaluator


# Add project root to path to import OnlineAutoClust
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'baselines', 'OnlineAutoClust'))
from OnlineAutoClust import OnlineAutoClustVanilla, OnlineAutoClustEnsemble
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


def run_onlineautoclust(dataset_name, algorithm_type="ensemble", run_count=None, seed=42, 
                       progress_interval=1000, output_dir="results/benchmark", verbose=True,
                       window_size=1000, t_max=60, delta=0.1, top_n=5, ensemble_k=10):
    """Run OnlineAutoClust algorithm on a dataset."""
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n===Running OnlineAutoClust ({algorithm_type}) on {dataset_name}===")
    
    try:
        # Load dataset
        dataset = load_streaming_dataset(dataset_name)
        
        if dataset is None:
            print(f"Failed to load dataset: {dataset_name}")
            return None
                
        # Set OnlineAutoClust parameters
        autoclust_params = {
            'window_size': window_size,
            't_max': t_max,
            'delta': delta,
            'top_n': top_n,
            'ensemble_k': ensemble_k
        }
        
        # Initialize model
        if algorithm_type.lower() == "vanilla":
            model = OnlineAutoClustVanilla(**autoclust_params)
            algorithm_name = "OnlineAutoClustVanilla"
        elif algorithm_type.lower() == "ensemble":
            model = OnlineAutoClustEnsemble(**autoclust_params)
            algorithm_name = "OnlineAutoClustEnsemble"
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}. Use 'vanilla' or 'ensemble'")
        
        if verbose:
            print(f"Initialized {algorithm_name} with parameters:")
            for key, value in autoclust_params.items():
                print(f"  {key}: {value}")
            print(f"Random seed: {seed}")
        
        # Initialize incremental metrics
        purity_metric = Purity()
        fm_metric = metrics.FowlkesMallows()
        rand_metric = metrics.Rand()
        # Track metrics at intervals
        purity_scores = []
        fm_scores = []
        rand_scores = []
        adjusted_rand_scores = []
        completeness_scores = []
        # Windowed label collection for batch ARI/NMI (avoids cluster ID instability)
        window_true_labels = []
        window_pred_labels = []
        times = []
        memory_usages = []
        cluster_counts = []
        
        # Process stream
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        # Collect data for batch processing
        stream_data = []
        true_labels = []
        
        sample_count = 0
        for x, y in dataset:
            stream_data.append(x)
            true_labels.append(y)
            sample_count += 1
            
            if verbose and sample_count % progress_interval == 0:
                print(f"Loaded {sample_count} samples...")
        
        if verbose:
            print(f"Total samples loaded: {sample_count}")
            print("Processing stream through OnlineAutoClust...")
        
        # Process the stream through OnlineAutoClust (timed as algorithm work)
        ev = StreamEvaluator(window=progress_interval)
        results = model.process_stream(stream_data)

        # Predict every point with the final model; feed unified evaluator
        last_model = results[-1] if (results and results[-1] is not None) else None
        for i, x in enumerate(stream_data):
            x_array = [x[k] for k in sorted(x.keys())] if isinstance(x, dict) else list(x)
            if last_model is not None:
                try:
                    pred = last_model.predict_one(x_array)
                except Exception:
                    pred = 0
            else:
                pred = 0
            ev.update(true_labels[i], pred, x_array)

        results_dict = ev.finalize(model=algorithm_name, dataset=dataset_name,
                                   params=autoclust_params, run_count=run_count,
                                   seed=seed, verbose=verbose)
        results_dict["windows_processed"] = len(results)
        output_filename = (f"{algorithm_name}_{dataset_name}_{run_count}.json" if run_count is not None
                           else f"{algorithm_name}_{dataset_name}.json")
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        if verbose:
            print(f"{algorithm_name} completed on {dataset_name} -> {output_path}")
        return results_dict
        
    except Exception as e:
        print(f"Error running OnlineAutoClust on {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run OnlineAutoClust on a dataset')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Name of the dataset to run OnlineAutoClust on')
    parser.add_argument('--algorithm', type=str, default='ensemble', 
                       choices=['vanilla', 'ensemble'],
                       help='Type of OnlineAutoClust algorithm (vanilla or ensemble)')
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
    
    # OnlineAutoClust specific parameters
    parser.add_argument('--window_size', type=int, default=1000,
                       help='Window size for processing')
    parser.add_argument('--t_max', type=int, default=60,
                       help='Maximum time for AutoML phase')
    parser.add_argument('--delta', type=float, default=0.1,
                       help='Drift detection threshold')
    parser.add_argument('--top_n', type=int, default=5,
                       help='Number of top models to keep')
    parser.add_argument('--ensemble_k', type=int, default=10,
                       help='Size of backup ensemble (for ensemble version)')
    
    args = parser.parse_args()
    
    # Run OnlineAutoClust with provided arguments
    result = run_onlineautoclust(
        dataset_name=args.dataset, 
        algorithm_type=args.algorithm,
        run_count=args.run_count, 
        seed=args.seed, 
        progress_interval=args.progress_interval, 
        output_dir=args.output_dir, 
        verbose=not args.quiet,
        window_size=args.window_size,
        t_max=args.t_max,
        delta=args.delta,
        top_n=args.top_n,
        ensemble_k=args.ensemble_k
    )
    
    if result is None:
        sys.exit(1)
    else:
        print(f"\nOnlineAutoClust run completed successfully!")
        sys.exit(0)
