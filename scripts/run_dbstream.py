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
from utils.purityscore import Purity

from stream_clustering import DBStream

import warnings
warnings.filterwarnings("ignore")


def load_streaming_dataset(dataset_name):
    """Load streaming dataset from CSV file."""
    try:
        # Dataset path can be a full path or relative
        dataset_path = "stream_datasets/" + dataset_name + ".csv"
        # Load CSV file
        df = pd.read_csv(dataset_path)
        
        X = df.copy()
        y = X.pop("class")

        dataset = stream.iter_pandas(X, y)

        return dataset
    
    except FileNotFoundError:
        print(f"Dataset {dataset_name} not found.")
        return None
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {str(e)}")
        return None


def run_DBSTREAM(dataset_name, run_count=None, seed=42, progress_interval=1000, 
                 output_dir="benchmark_results", verbose=True):
    """Run DBSTREAM algorithm on a dataset."""
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n===Running DBSTREAM on {dataset_name}===")
    
    try:
        
        dataset = load_streaming_dataset(dataset_name)
        
        if dataset is None:
            print(f"Failed to load dataset {dataset_name}")
            return None
                
        # Set DBSTREAM parameters (matching Rust implementation)
        dbstream_params = {
            "clustering_threshold": 1.0,
            "fading_factor": 0.01,
            "cleanup_interval": 1000.0,
            "intersection_factor": 0.3,
            "minimum_weight": 1.0
        }

        # Initialize model using stream_clustering.DBStream
        model = DBStream(**dbstream_params)
        
        # Track metrics using incremental purity score, Fowlkes-Mallows score, and Rand index
        purity_metric = Purity()
        fm_metric = metrics.FowlkesMallows()
        rand_metric = metrics.Rand()
        # Add new metrics
        adjusted_rand_metric = metrics.AdjustedRand()
        completeness_metric = metrics.Completeness()
        purity_scores = []
        fm_scores = []
        rand_scores = []
        # Add lists for new metrics
        adjusted_rand_scores = []
        completeness_scores = []
        times = []
        memory_usages = []
        cluster_counts = []  # Track number of clusters at each interval
        processed_instances = 0
        unique_clusters = set()  # Track unique cluster assignments

        # Start timing
        start_time = time.time()
        
        # Start memory usage
        start_memory = psutil.Process().memory_info().rss
        
        # Process the stream
        for X, y in dataset:

            X = list(X.values())

            # Train the model
            model.learn_one(X)

            # Get prediction
            pred = model.predict_one(X)
            unique_clusters.add(pred)  # Add to set of unique clusters

            # Update incremental purity score, Fowlkes-Mallows score, and Rand index
            purity_metric.update(y, pred)
            fm_metric.update(y, pred)
            rand_metric.update(y, pred)
            # Update new metrics
            adjusted_rand_metric.update(y, pred)
            completeness_metric.update(y, pred)
            
            processed_instances += 1
            
            # Calculate metrics at intervals
            if processed_instances % progress_interval == 0:
                
                # Get current metrics from incremental metrics
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
                
                # Incremental Time and Memory Usage update
                instance_time = time.time() - start_time  # seconds
                times.append(instance_time)
                instance_memory = psutil.Process().memory_info().rss - start_memory
                memory_usage = abs(instance_memory / 1024 / 1024)  # MB
                memory_usages.append(memory_usage)
                
                if verbose:
                    if run_count is not None:
                        print(f"DBSTREAM on {dataset_name} (Run {run_count}): Processed {processed_instances} instances, "
                              f"Clusters: {active_clusters}, "
                              f"Purity: {current_purity:.4f}, Fowlkes-Mallows: {current_fm:.4f}, "
                              f"Rand: {current_rand:.4f}, Adjusted Rand: {current_adjusted_rand:.4f}, "
                              f"Completeness: {current_completeness:.4f}, "
                              f"Time: {instance_time:.2f} seconds, Memory: {memory_usage:.2f} MB", flush=True)
                    else:
                        print(f"DBSTREAM on {dataset_name}: Processed {processed_instances} instances, "
                              f"Clusters: {active_clusters}, "
                              f"Purity: {current_purity:.4f}, Fowlkes-Mallows: {current_fm:.4f}, "
                              f"Rand: {current_rand:.4f}, Adjusted Rand: {current_adjusted_rand:.4f}, "
                              f"Completeness: {current_completeness:.4f}, "
                              f"Time: {instance_time:.2f} seconds, Memory: {memory_usage:.2f} MB", flush=True)
        
        # Calculate final metrics
        final_purity = purity_metric.get()
        final_fm = fm_metric.get()
        final_rand = rand_metric.get()
        final_adjusted_rand = adjusted_rand_metric.get()
        final_completeness = completeness_metric.get()
        
        # Count final number of active clusters
        final_cluster_count = len(unique_clusters)
        
        total_runtime = time.time() - start_time
        memory_usage = psutil.Process().memory_info().rss - start_memory
        memory_usage = abs(memory_usage / 1024 / 1024)  # MB
        
        # Store results
        results = {
            "dataset": dataset_name,
            "model": "DBSTREAM",
            "purity_score": final_purity,
            "fowlkes_mallows_score": final_fm,
            "rand_score": final_rand,
            "adjusted_rand_score": final_adjusted_rand,
            "completeness_score": final_completeness,
            "active_clusters": final_cluster_count,
            "total_runtime": total_runtime,
            "total_memory_usage": memory_usage,
            "instances": processed_instances,
            "model_params": dbstream_params,
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
            }
        }
        
        # Generate file name based on run count
        if run_count is not None:
            results["run_count"] = run_count
            results["seed"] = seed
            file_name = f"{results['model']}_{dataset_name}_{run_count}.json"
        else:
            file_name = f"{results['model']}_{dataset_name}.json"
        
        # Save results to file
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        if verbose:
            print(f"DBSTREAM completed on {dataset_name}")
            print(f"Final Active Clusters: {final_cluster_count}")
            print(f"Final Purity Score: {final_purity:.4f}")
            print(f"Final Fowlkes-Mallows Score: {final_fm:.4f}")
            print(f"Final Rand Index: {final_rand:.4f}")
            print(f"Final Adjusted Rand Index: {final_adjusted_rand:.4f}")
            print(f"Final Completeness Score: {final_completeness:.4f}")
            print(f"Runtime: {total_runtime:.2f} seconds")
            print(f"Memory: {memory_usage:.2f} MB")
            print(f"Processed {processed_instances} instances")
            print(f"Results saved to {output_file}")
        
        return results
        
    except Exception as e:
        print(f"Error running DBSTREAM on {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run DBSTREAM on a dataset')
    parser.add_argument('--dataset', type=str, help='name of the dataset to run DBSTREAM on')
    parser.add_argument('--run_count', type=int, default=None, help='Run count identifier')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--progress_interval', type=int, default=1000, help='Interval for progress updates')
    parser.add_argument('--output_dir', type=str, default="benchmark_results", help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Disable verbose output', default=False)
    
    args = parser.parse_args()
    
    if args.dataset is None:
        print("Please provide a dataset name using --dataset argument.")
        sys.exit(1)
    
    # Run DBSTREAM with provided arguments
    run_DBSTREAM(dataset_name=args.dataset, 
                 run_count=args.run_count, 
                 seed=args.seed, 
                 progress_interval=args.progress_interval, 
                 output_dir=args.output_dir, 
                 verbose=not args.quiet)
