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
from collections import Counter
from utils.purityscore import Purity


# Add project root to path to import AutoDSC
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'algorithms', 'AutoDSC'))
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
        AutoDSC_summaries = []  # Store summary at each window

        # AutoDSC-specific tracking
        exploration_rates = []
        composite_scores = []
        elite_population_sizes = []

        # Start timing
        start_time = time.time()

        # Start memory usage
        start_memory = psutil.Process().memory_info().rss

        # Process the stream
        for X, y in dataset:

            # Convert dictionary to list of values
            x_values = list(X.values())

            # Train the model
            model.learn_one(x_values)

            # Get prediction
            pred = model.predict_one(x_values)
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

                # Get AutoDSC-specific metrics
                optimizer_stats = model.optimizer.get_exploration_stats()
                exploration_rates.append(optimizer_stats['exploration_rate'])
                elite_population_sizes.append(optimizer_stats['elite_population_size'])
                composite_scores.append(optimizer_stats['best_score'])

                # Get and store summary at this window
                window_summary = model.get_performance_summary()
                AutoDSC_summaries.append(window_summary)

                # Incremental Time and Memory Usage update
                instance_time = time.time() - start_time  # seconds
                times.append(instance_time)
                instance_memory = psutil.Process().memory_info().rss - start_memory
                memory_usage = abs(instance_memory / 1024 / 1024)  # MB
                memory_usages.append(memory_usage)

                if verbose:
                    if run_count is not None:
                        print(f"AutoDSC on {dataset_name} (Run {run_count}): Processed {processed_instances} instances, "
                              f"Clusters: {active_clusters}, "
                              f"Purity: {current_purity:.4f}, Fowlkes-Mallows: {current_fm:.4f}, "
                              f"Rand: {current_rand:.4f}, Adjusted Rand: {current_adjusted_rand:.4f}, "
                              f"Completeness: {current_completeness:.4f}, "
                              f"Exploration Rate: {optimizer_stats['exploration_rate']:.4f}, "
                              f"Composite Score: {optimizer_stats['best_score']:.4f}, "
                              f"Time: {instance_time:.2f} seconds, Memory: {memory_usage:.2f} MB", flush=True)
                    else:
                        print(f"AutoDSC on {dataset_name}: Processed {processed_instances} instances, "
                              f"Clusters: {active_clusters}, "
                              f"Purity: {current_purity:.4f}, Fowlkes-Mallows: {current_fm:.4f}, "
                              f"Rand: {current_rand:.4f}, Adjusted Rand: {current_adjusted_rand:.4f}, "
                              f"Completeness: {current_completeness:.4f}, "
                              f"Exploration Rate: {optimizer_stats['exploration_rate']:.4f}, "
                              f"Composite Score: {optimizer_stats['best_score']:.4f}, "
                              f"Time: {instance_time:.2f} seconds, Memory: {memory_usage:.2f} MB", flush=True)

        # Calculate final metrics
        final_purity = purity_metric.get()
        final_fm = fm_metric.get()
        final_rand = rand_metric.get()
        final_adjusted_rand = adjusted_rand_metric.get()
        final_completeness = completeness_metric.get()

        # Count final number of active clusters
        final_cluster_count = len(unique_clusters)

        # Get final AutoDSC performance summary
        final_summary = model.get_performance_summary()

        total_runtime = time.time() - start_time
        memory_usage = psutil.Process().memory_info().rss - start_memory
        memory_usage = abs(memory_usage / 1024 / 1024)  # MB

        # Store results
        results = {
            "dataset": dataset_name,
            "model": model_name,
            "purity_score": final_purity,
            "fowlkes_mallows_score": final_fm,
            "rand_score": final_rand,
            "adjusted_rand_score": final_adjusted_rand,
            "completeness_score": final_completeness,
            "active_clusters": final_cluster_count,
            "total_runtime": total_runtime,
            "total_memory_usage": memory_usage,
            "instances": processed_instances,
            "model_params": AutoDSC_params,
            "AutoDSC_summary": final_summary,
            "window_metric": {
                "window_size": progress_interval,
                "purity_scores": purity_scores,
                "fowlkes_mallows_scores": fm_scores,
                "rand_scores": rand_scores,
                "cluster_counts": cluster_counts,
                "runtimes": times,
                "memory_usages": memory_usages,
                "adjusted_rand_scores": adjusted_rand_scores,
                "completeness_scores": completeness_scores,
                "exploration_rates": exploration_rates,
                "composite_scores": composite_scores,
                "elite_population_sizes": elite_population_sizes,
                "AutoDSC_summaries": AutoDSC_summaries
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
            print(f"AutoDSC completed on {dataset_name}")
            print(f"Final Active Clusters: {final_cluster_count}")
            print(f"Final Purity Score: {final_purity:.4f}")
            print(f"Final Fowlkes-Mallows Score: {final_fm:.4f}")
            print(f"Final Rand Index: {final_rand:.4f}")
            print(f"Final Adjusted Rand Index: {final_adjusted_rand:.4f}")
            print(f"Final Completeness Score: {final_completeness:.4f}")
            print(f"Final Exploration Rate: {final_summary['optimizer_stats']['exploration_rate']:.4f}")
            print(f"Final Composite Score: {final_summary['best_configuration']['scores']['composite_score']:.4f}")
            print(f"Best Algorithm: {final_summary['best_configuration']['algorithm']}")
            print(f"Runtime: {total_runtime:.2f} seconds")
            print(f"Memory: {memory_usage:.2f} MB")
            print(f"Processed {processed_instances} instances")
            print(f"Results saved to {output_file}")

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
