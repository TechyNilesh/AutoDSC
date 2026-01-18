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
from collections import Counter
from utils.purityscore import Purity


# Add project root to path to import OnlineAutoClust
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'algorithms', 'OnlineAutoClust'))
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
                       progress_interval=1000, output_dir="benchmark_results", verbose=True,
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
        # Add new metrics
        adjusted_rand_metric = metrics.AdjustedRand()
        completeness_metric = metrics.Completeness()

        # Track metrics at intervals
        purity_scores = []
        fm_scores = []
        rand_scores = []
        # Add lists for new metrics
        adjusted_rand_scores = []
        completeness_scores = []
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
        
        # Process the stream through OnlineAutoClust
        results = model.process_stream(stream_data)
        
        # Generate predictions and update metrics incrementally
        predicted_labels = []
        unique_clusters = set()
        
        if results and results[-1] is not None:
            last_model = results[-1]
            if verbose:
                print("Generating predictions and updating metrics...")
            
            for i, x in enumerate(stream_data):
                try:
                    # Convert dict to array for prediction
                    if isinstance(x, dict):
                        x_array = [x[k] for k in sorted(x.keys())]
                    else:
                        x_array = list(x)
                    
                    pred = last_model.predict_one(x_array)
                    predicted_labels.append(pred)
                    unique_clusters.add(pred)
                    
                    # Update incremental metrics
                    purity_metric.update(true_labels[i], pred)
                    fm_metric.update(true_labels[i], pred)
                    rand_metric.update(true_labels[i], pred)
                    # Update new metrics
                    adjusted_rand_metric.update(true_labels[i], pred)
                    completeness_metric.update(true_labels[i], pred)
                    
                    # Calculate metrics at intervals
                    if (i + 1) % progress_interval == 0:
                        # Get current metrics
                        current_purity = purity_metric.get()
                        current_fm = fm_metric.get()
                        current_rand = rand_metric.get()
                        current_adjusted_rand = adjusted_rand_metric.get()
                        current_completeness = completeness_metric.get()

                        purity_scores.append(current_purity)
                        fm_scores.append(current_fm)
                        rand_scores.append(current_rand)
                        adjusted_rand_scores.append(current_adjusted_rand)
                        completeness_scores.append(current_completeness)
                        
                        # Count number of active clusters
                        active_clusters = len(unique_clusters)
                        cluster_counts.append(active_clusters)
                        
                        # Time and memory usage
                        instance_time = time.time() - start_time
                        times.append(instance_time)
                        instance_memory = psutil.Process().memory_info().rss - start_memory
                        memory_usage = abs(instance_memory / 1024 / 1024)  # MB
                        memory_usages.append(memory_usage)
                        
                        if verbose:
                            if run_count is not None:
                                print(f"{algorithm_name} on {dataset_name} (Run {run_count}): Processed {i+1} instances, "
                                      f"Clusters: {active_clusters}, "
                                      f"Purity: {current_purity:.4f}, Fowlkes-Mallows: {current_fm:.4f}, "
                                      f"Rand: {current_rand:.4f}, Adjusted Rand: {current_adjusted_rand:.4f}, "
                                      f"Completeness: {current_completeness:.4f}, "
                                      f"Time: {instance_time:.2f} seconds, Memory: {memory_usage:.2f} MB", flush=True)
                            else:
                                print(f"{algorithm_name} on {dataset_name}: Processed {i+1} instances, "
                                      f"Clusters: {active_clusters}, "
                                      f"Purity: {current_purity:.4f}, Fowlkes-Mallows: {current_fm:.4f}, "
                                      f"Rand: {current_rand:.4f}, Adjusted Rand: {current_adjusted_rand:.4f}, "
                                      f"Completeness: {current_completeness:.4f}, "
                                      f"Time: {instance_time:.2f} seconds, Memory: {memory_usage:.2f} MB", flush=True)
                        
                except Exception as e:
                    if verbose:
                        print(f"Error processing sample {i}: {e}")
                    predicted_labels.append(0)  # Default prediction
                    unique_clusters.add(0)
                    
                    # Still update metrics with default prediction
                    purity_metric.update(true_labels[i], 0)
                    fm_metric.update(true_labels[i], 0)
                    rand_metric.update(true_labels[i], 0)
                    # Update new metrics
                    adjusted_rand_metric.update(true_labels[i], 0)
                    completeness_metric.update(true_labels[i], 0)
                
        else:
            print("Warning: No valid model found. Using default predictions.")
            predicted_labels = [0] * len(stream_data)
            unique_clusters = {0}
            
            # Update metrics with default predictions
            for i, y in enumerate(true_labels):
                purity_metric.update(y, 0)
                fm_metric.update(y, 0)
                rand_metric.update(y, 0)
                # Update new metrics
                adjusted_rand_metric.update(y, 0)
                completeness_metric.update(y, 0)
        
        # Calculate final metrics
        final_purity = purity_metric.get()
        final_fm = fm_metric.get()
        final_rand = rand_metric.get()
        final_adjusted_rand = adjusted_rand_metric.get()
        final_completeness = completeness_metric.get()
        final_cluster_count = len(unique_clusters)
        
        # Calculate processing time and memory usage
        end_time = time.time()
        total_runtime = end_time - start_time
        total_memory = psutil.Process().memory_info().rss - start_memory
        total_memory_usage = abs(total_memory / 1024 / 1024)  # MB
        
        # Store results in standardized format
        results_dict = {
            "dataset": dataset_name,
            "model": algorithm_name,
            "purity_score": final_purity,
            "fowlkes_mallows_score": final_fm,
            "rand_score": final_rand,
            "adjusted_rand_score": final_adjusted_rand,
            "completeness_score": final_completeness,
            "active_clusters": final_cluster_count,
            "total_runtime": total_runtime,
            "total_memory_usage": total_memory_usage,
            "instances": sample_count,
            "model_params": autoclust_params,
            "window_metric": {
                "window_size": progress_interval,
                "purity_scores": purity_scores,
                "fowlkes_mallows_scores": fm_scores,
                "rand_scores": rand_scores,
                "cluster_counts": cluster_counts,
                "runtimes": times,
                "memory_usages": memory_usages,
                "adjusted_rand_scores": adjusted_rand_scores,
                "completeness_scores": completeness_scores
            },
            "additional_info": {
                "seed": seed,
                "run_count": run_count,
                "windows_processed": len(results),
                "models_created": len([r for r in results if r is not None]),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Save results
        output_filename = f"{algorithm_name}_{dataset_name}"
        if run_count is not None:
            output_filename += f"_run{run_count}"
        output_filename += ".json"
        
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        if verbose:
            print(f"\nResults Summary:")
            print(f"  Dataset: {dataset_name}")
            print(f"  Algorithm: {algorithm_name}")
            print(f"  Samples processed: {sample_count}")
            print(f"  Windows processed: {len(results)}")
            print(f"  Valid models created: {len([r for r in results if r is not None])}")
            print(f"  Active clusters: {final_cluster_count}")
            print(f"  Purity Score: {final_purity:.4f}")
            print(f"  Fowlkes-Mallows Score: {final_fm:.4f}")
            print(f"  Rand Index: {final_rand:.4f}")
            print(f"  Adjusted Rand Index: {final_adjusted_rand:.4f}")
            print(f"  Completeness Score: {final_completeness:.4f}")
            print(f"  Processing time: {total_runtime:.2f} seconds")
            print(f"  Memory usage: {total_memory_usage:.2f} MB")
            print(f"  Results saved to: {output_path}")
            
            # Test some sample predictions if we have a valid model
            if results and results[-1] is not None:
                print(f"\nSample predictions from final model:")
                test_samples = stream_data[:min(5, len(stream_data))]
                for i, sample in enumerate(test_samples):
                    try:
                        if isinstance(sample, dict):
                            x_array = [sample[k] for k in sorted(sample.keys())]
                        else:
                            x_array = list(sample)
                        pred = results[-1].predict_one(x_array)
                        true_label = true_labels[i]
                        print(f"  Sample {i+1}: Predicted={pred}, True={true_label}")
                    except Exception as e:
                        print(f"  Sample {i+1}: Prediction failed - {e}")
        
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
    parser.add_argument('--output_dir', type=str, default="benchmark_results", 
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
