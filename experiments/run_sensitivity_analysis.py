import concurrent.futures
import subprocess
import random
import argparse
import os
import itertools

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# How to run this script
# python run_sensitivity_analysis.py --mode parameter --param elite_size --dataset adult
# python run_sensitivity_analysis.py --mode dataset --dataset adult
# python run_sensitivity_analysis.py --mode full --dataset adult
# python run_sensitivity_analysis.py --mode random --dataset adult --configs 10

# List of dataset names from datasets folder
dataset_name_list = [
    'adult',
    'electricity', 
    'forest_cover',
    'hyperplane_high_gradual_drift',
    'insects',
    'movingRBF',
    'moving_squares',
    'new_airlines',
    'sea_high_abrupt_drift',
    'sea_high_mixed_drift',
    'shuttle',
    'synth_RandomRBFDrift',
    'synth_agrawal',
    'vehicle_sensIT'
]

# Parameter ranges for sensitivity analysis
parameter_ranges = {
    'elite_size': [5, 10, 15, 20, 25, 30],
    'window_size': [500, 1000, 1500, 2000],
    'exploration_rate': [0.1, 0.3, 0.5, 0.7, 0.9],
    'population_size': [10, 15, 20, 25, 30],
    'gamma': [0.01, 0.05, 0.1, 0.2, 0.3],
    'diversity_threshold': [0.05, 0.1, 0.15, 0.2, 0.25],
    'sample_ratio': [0.1, 0.15, 0.2, 0.25, 0.3]
}

# Default parameter values
default_params = {
    'elite_size': 10,
    'window_size': 1000,
    'exploration_rate': 0.5,
    'population_size': 20,
    'gamma': 0.1,
    'diversity_threshold': 0.1,
    'sample_ratio': 0.2
}

def run_autodsc_with_params(dataset_name, params, run_count=None, seed=42, 
                                   progress_interval=1000, output_dir="sensitivity_results", 
                                   analysis_type="default"):
    """
    Function to execute AutoStreamClust with specific parameter configuration.
    """
    print(f"Running AutoStreamClust on {dataset_name} with params: {params}")
    
    # Create descriptive model name based on analysis type and parameters
    if analysis_type == "single_param":
        # For single parameter analysis, highlight the varied parameter
        changed_params = {k: v for k, v in params.items() if v != default_params.get(k)}
        if changed_params:
            param_str = "_".join([f"{k}{v}" for k, v in sorted(changed_params.items())])
            model_name = f"AutoStreamClust_SensitivityAnalysis_{param_str}"
        else:
            model_name = "AutoStreamClust_SensitivityAnalysis_Default"
    elif analysis_type == "multi_param":
        # For multi-parameter analysis, include key parameters
        key_params = ['elite_size', 'exploration_rate', 'gamma', 'diversity_threshold']
        param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items()) if k in key_params])
        model_name = f"AutoStreamClust_MultiParam_{param_str}"
    elif analysis_type == "random_search":
        # For random search, create a hash-like identifier
        param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items())])
        model_name = f"AutoStreamClust_RandomSearch_{hash(param_str) % 10000}"
    else:
        # Fallback to original naming
        param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items()) if k in ['elite_size', 'exploration_rate', 'gamma']])
        model_name = f"AutoStreamClust_{param_str}"
    
    run_script = os.path.join(SCRIPT_DIR, 'scripts', 'run_autodsc.py')
    command = ['python', run_script, '--dataset', dataset_name]
    
    # Add parameter arguments
    for param_name, param_value in params.items():
        command += [f'--{param_name}', str(param_value)]
    
    # Add custom model name
    command += ['--model_name', model_name]
    
    if run_count is not None:
        command += ['--run_count', str(run_count)]
    
    command += ['--seed', str(seed)]
    command += ['--progress_interval', str(progress_interval)]
    command += ['--output_dir', output_dir]

    # Execute the command using Popen and print output in real-time
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in process.stdout:
        print(line, end='')  # Print each line of stdout in real-time

    process.wait()  # Wait for the process to complete

    # Check if there were any errors and print them
    if process.returncode != 0:
        print(f"Dataset {dataset_name}: Error\n{process.stderr.read()}")
        return False
    return True

def single_parameter_sensitivity(dataset_name, param_name, progress_interval=1000, 
                                output_dir="sensitivity_results"):
    """Run sensitivity analysis for a single parameter."""
    print(f"*** Running single parameter sensitivity analysis for {param_name} on {dataset_name} ***")
    
    if param_name not in parameter_ranges:
        print(f"Invalid parameter name: {param_name}")
        return
    
    # Create output directory for this parameter
    param_output_dir = os.path.join(output_dir, f"{dataset_name}_{param_name}_sensitivity")
    os.makedirs(param_output_dir, exist_ok=True)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.8))) as executor:
        futures = []
        
        for param_value in parameter_ranges[param_name]:
            # Create parameter configuration
            params = default_params.copy()
            params[param_name] = param_value
            
            # Use fixed seed for reproducibility
            seed = 42
            
            futures.append(executor.submit(
                run_autodsc_with_params,
                dataset_name, params, None, seed, progress_interval, param_output_dir, "single_param"
            ))
        
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in sensitivity analysis: {e}")

def multi_parameter_sensitivity(dataset_name, progress_interval=1000, 
                               output_dir="sensitivity_results"):
    """Run sensitivity analysis varying multiple parameters."""
    print(f"*** Running multi-parameter sensitivity analysis on {dataset_name} ***")
    
    # Create output directory
    multi_output_dir = os.path.join(output_dir, f"{dataset_name}_multi_param_sensitivity")
    os.makedirs(multi_output_dir, exist_ok=True)
    
    # Select fewer values for multi-parameter analysis to keep it manageable
    reduced_ranges = {
        'elite_size': [5, 10, 20],
        'exploration_rate': [0.3, 0.5, 0.7],
        'gamma': [0.05, 0.1, 0.2],
        'diversity_threshold': [0.05, 0.1, 0.2]
    }
    
    # Generate all parameter combinations
    param_names = list(reduced_ranges.keys())
    param_values = list(reduced_ranges.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"Total combinations to test: {len(combinations)}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.6))) as executor:
        futures = []
        
        for i, combination in enumerate(combinations):
            # Create parameter configuration
            params = default_params.copy()
            for param_name, param_value in zip(param_names, combination):
                params[param_name] = param_value
            
            # Use fixed seed for reproducibility
            seed = 42
            
            # Create unique run identifier for multi-param
            run_id = f"combo_{i+1}"
            
            futures.append(executor.submit(
                run_autodsc_with_params,
                dataset_name, params, run_id, seed, progress_interval, multi_output_dir, "multi_param"
            ))
        
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in multi-parameter sensitivity analysis: {e}")

def random_parameter_search(dataset_name, progress_interval=1000, 
                           output_dir="sensitivity_results", configs=10):
    """Run random parameter search for sensitivity analysis."""
    print(f"*** Running random parameter search on {dataset_name} with {configs} configurations ***")
    
    # Create output directory
    random_output_dir = os.path.join(output_dir, f"{dataset_name}_random_search")
    os.makedirs(random_output_dir, exist_ok=True)
    
    # Set seed for reproducible random parameter generation
    random.seed(42)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.8))) as executor:
        futures = []
        
        for config_id in range(1, configs + 1):
            # Generate random parameter configuration
            params = {}
            for param_name, param_range in parameter_ranges.items():
                params[param_name] = random.choice(param_range)
            
            # Use fixed seed for each configuration for reproducibility
            seed = 42
            
            futures.append(executor.submit(
                run_autodsc_with_params,
                dataset_name, params, config_id, seed, progress_interval, random_output_dir, "random_search"
            ))
        
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in random parameter search: {e}")

def full_sensitivity_analysis(dataset_name, progress_interval=1000, 
                             output_dir="sensitivity_results"):
    """Run complete sensitivity analysis for all parameters."""
    print(f"*** Running full sensitivity analysis on {dataset_name} ***")
    
    # Run single parameter sensitivity for each parameter
    for param_name in parameter_ranges.keys():
        single_parameter_sensitivity(dataset_name, param_name, progress_interval, output_dir)
    
    # Run multi-parameter sensitivity
    multi_parameter_sensitivity(dataset_name, progress_interval, output_dir)
    
    # Run random search
    random_parameter_search(dataset_name, progress_interval, output_dir, configs=20)

def all_datasets_sensitivity(param_name, progress_interval=1000, 
                            output_dir="sensitivity_results"):
    """Run sensitivity analysis for one parameter across all datasets."""
    print(f"*** Running {param_name} sensitivity analysis across all datasets ***")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.4))) as executor:
        futures = []
        
        for dataset in dataset_name_list:
            futures.append(executor.submit(
                single_parameter_sensitivity,
                dataset, param_name, progress_interval, output_dir
            ))
        
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in all datasets sensitivity: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sensitivity analysis for AutoStreamClust parameters.")
    parser.add_argument('--mode', choices=['parameter', 'dataset', 'full', 'random', 'all_datasets'], 
                       required=True, help="Choose the sensitivity analysis mode")
    parser.add_argument('--param', help="Parameter name for single parameter sensitivity")
    parser.add_argument('--dataset', help="Dataset name for analysis")
    parser.add_argument('--configs', type=int, default=10, help="Number of configurations for random search")
    parser.add_argument('--progress_interval', type=int, default=1000, 
                       help="Interval for progress updates")
    parser.add_argument('--output_dir', type=str, default="sensitivity_results", 
                       help="Directory to save results")

    args = parser.parse_args()

    if args.mode == 'parameter':
        if not args.param or not args.dataset:
            print("Please provide both --param and --dataset for parameter mode")
        else:
            single_parameter_sensitivity(args.dataset, args.param, args.progress_interval, 
                                       args.output_dir)
    
    elif args.mode == 'dataset':
        if not args.dataset:
            print("Please provide --dataset for dataset mode")
        else:
            multi_parameter_sensitivity(args.dataset, args.progress_interval, 
                                      args.output_dir)
    
    elif args.mode == 'full':
        if not args.dataset:
            print("Please provide --dataset for full mode")
        else:
            full_sensitivity_analysis(args.dataset, args.progress_interval, args.output_dir)
    
    elif args.mode == 'random':
        if not args.dataset:
            print("Please provide --dataset for random mode")
        else:
            random_parameter_search(args.dataset, args.progress_interval, 
                                  args.output_dir, args.configs)
    
    elif args.mode == 'all_datasets':
        if not args.param:
            print("Please provide --param for all_datasets mode")
        else:
            all_datasets_sensitivity(args.param, args.progress_interval, 
                                   args.output_dir)

    print("\nSensitivity analysis completed!")