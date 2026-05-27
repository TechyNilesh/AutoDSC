import concurrent.futures
import subprocess
import argparse
import os
import sys

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# How to run this script
# python run_ablation_study.py --dataset insects --ablation drift_detection
# python run_ablation_study.py --dataset insects --ablation elite_guidance
# python run_ablation_study.py --dataset insects --ablation adaptive_exploration_rate
# python run_ablation_study.py --dataset insects --ablation diversity_maintenance
# python run_ablation_study.py --dataset insects --ablation all
# python run_ablation_study.py --dataset insects --ablation full_comparison

# Finalized suite (representative datasets for ablation)
dataset_name_list = [
    'real_forest_cover',
    'real_insects',
    'real_pendigits',
    'real_shuttle',
    'real_vehicle',
    'synth_blobs_gradual',
    'synth_blobs_merge_split',
    'synth_blobs_sudden',
    'synth_moving_squares',
    'synth_rbf_gradual',
]

# Ablation configurations
# Each ablation disables a specific component using proxy parameter settings,
# since AutoDSC does not expose per-component disable flags via CLI.
ablation_configs = {
    'drift_detection': {
        'description': 'Ablate drift detection response: gamma=0.01 makes exploration rate '
                       'nearly static so drift signals have no adaptive effect',
        'params': {
            'gamma': 0.01
        }
    },
    'elite_guidance': {
        'description': 'Ablate elite guidance (PGSSO): population_size=2 forces elite_size=1, '
                       'collapsing the elite pool to a single entry so guidance is negligible',
        'params': {
            'population_size': 2
        }
    },
    'adaptive_exploration_rate': {
        'description': 'Ablate adaptive exploration rate: gamma=1.0 maximises the adjustment '
                       'magnitude, effectively switching to a highly reactive (non-smooth) rate',
        'params': {
            'gamma': 1.0
        }
    },
    'diversity_maintenance': {
        'description': 'Disable diversity maintenance in population',
        'params': {
            'diversity_threshold': 0.01  # Minimum allowed value (clamped by AutoDSC)
        }
    }
}

# Default AutoDSC parameters for baseline
default_params = {
    'window_size': 1000,
    'population_size': 20,
    'gamma': 0.1,
    'diversity_threshold': 0.1
}


def run_ablation_experiment(dataset_name, ablation_name, params, seed=42,
                            progress_interval=1000, output_dir="ablation_results"):
    """
    Run AutoDSC with specific ablation configuration.
    """
    print(f"Running ablation study: {ablation_name} on {dataset_name}")
    print(f"Parameters: {params}")

    # Create model name based on ablation
    if ablation_name == 'baseline':
        model_name = "AutoDSC_Baseline"
    else:
        model_name = f"AutoDSC_Ablation_{ablation_name}"

    run_script = os.path.join(SCRIPT_DIR, 'scripts', 'run_autodsc.py')
    command = [sys.executable, run_script, '--dataset', dataset_name]

    # Add default parameters
    for param_name, param_value in default_params.items():
        if param_name not in params:  # Don't override ablation params
            command += [f'--{param_name}', str(param_value)]

    # Add ablation-specific parameters
    for param_name, param_value in params.items():
        if isinstance(param_value, bool):
            if param_value:
                command += [f'--{param_name}']
        else:
            command += [f'--{param_name}', str(param_value)]

    command += ['--model_name', model_name]
    command += ['--seed', str(seed)]
    command += ['--progress_interval', str(progress_interval)]
    command += ['--output_dir', output_dir]

    # Execute the command
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in process.stdout:
        print(line, end='')

    process.wait()

    if process.returncode != 0:
        print(f"Error in ablation {ablation_name} on {dataset_name}:\n{process.stderr.read()}")
        return False
    return True


def run_single_ablation(dataset_name, ablation_name, progress_interval=1000,
                        output_dir="ablation_results"):
    """Run a single ablation study on a dataset."""
    print(f"*** Running {ablation_name} ablation on {dataset_name} ***")

    if ablation_name not in ablation_configs:
        print(f"Invalid ablation name: {ablation_name}")
        print(f"Available ablations: {list(ablation_configs.keys())}")
        return

    # Create output directory
    ablation_output_dir = os.path.join(output_dir, f"{dataset_name}_{ablation_name}_ablation")
    os.makedirs(ablation_output_dir, exist_ok=True)

    config = ablation_configs[ablation_name]
    print(f"Ablation: {config['description']}")

    run_ablation_experiment(
        dataset_name,
        ablation_name,
        config['params'],
        output_dir=ablation_output_dir
    )


def run_all_ablations(dataset_name, progress_interval=1000, output_dir="ablation_results"):
    """Run all ablation studies on a dataset."""
    print(f"*** Running all ablation studies on {dataset_name} ***")

    # Create output directory
    all_output_dir = os.path.join(output_dir, f"{dataset_name}_all_ablations")
    os.makedirs(all_output_dir, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.8))) as executor:
        futures = []

        for ablation_name, config in ablation_configs.items():
            futures.append(executor.submit(
                run_ablation_experiment,
                dataset_name, ablation_name, config['params'],
                42, progress_interval, all_output_dir
            ))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in ablation study: {e}")


def run_full_comparison(dataset_name, progress_interval=1000, output_dir="ablation_results"):
    """Run baseline + all ablations for full comparison."""
    print(f"*** Running full ablation comparison on {dataset_name} ***")

    # Create output directory
    comparison_output_dir = os.path.join(output_dir, f"{dataset_name}_full_comparison")
    os.makedirs(comparison_output_dir, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.8))) as executor:
        futures = []

        # Run baseline (full AutoDSC)
        futures.append(executor.submit(
            run_ablation_experiment,
            dataset_name, 'baseline', {},
            42, progress_interval, comparison_output_dir
        ))

        # Run all ablations
        for ablation_name, config in ablation_configs.items():
            futures.append(executor.submit(
                run_ablation_experiment,
                dataset_name, ablation_name, config['params'],
                42, progress_interval, comparison_output_dir
            ))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in ablation study: {e}")


def run_ablation_all_datasets(ablation_name, progress_interval=1000,
                              output_dir="ablation_results"):
    """Run one ablation study across all datasets."""
    print(f"*** Running {ablation_name} ablation across all datasets ***")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.4))) as executor:
        futures = []

        for dataset in dataset_name_list:
            futures.append(executor.submit(
                run_single_ablation,
                dataset, ablation_name, progress_interval, output_dir
            ))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in ablation study: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ablation study for AutoDSC components.")
    parser.add_argument('--ablation',
                       choices=['drift_detection', 'elite_guidance', 'adaptive_exploration_rate',
                               'diversity_maintenance', 'all', 'full_comparison'],
                       required=True,
                       help="Choose which component to ablate")
    parser.add_argument('--dataset', help="Dataset name for analysis")
    parser.add_argument('--all_datasets', action='store_true',
                       help="Run ablation on all datasets")
    parser.add_argument('--progress_interval', type=int, default=1000,
                       help="Interval for progress updates")
    parser.add_argument('--output_dir', type=str, default="ablation_results",
                       help="Directory to save results")

    args = parser.parse_args()

    if args.all_datasets:
        if args.ablation in ['all', 'full_comparison']:
            print("Running all ablations on all datasets...")
            for dataset in dataset_name_list:
                if args.ablation == 'all':
                    run_all_ablations(dataset, args.progress_interval, args.output_dir)
                else:
                    run_full_comparison(dataset, args.progress_interval, args.output_dir)
        else:
            run_ablation_all_datasets(args.ablation, args.progress_interval, args.output_dir)
    else:
        if not args.dataset:
            print("Please provide --dataset or use --all_datasets")
        elif args.ablation == 'all':
            run_all_ablations(args.dataset, args.progress_interval, args.output_dir)
        elif args.ablation == 'full_comparison':
            run_full_comparison(args.dataset, args.progress_interval, args.output_dir)
        else:
            run_single_ablation(args.dataset, args.ablation, args.progress_interval,
                               args.output_dir)

    print("\nAblation study completed!")
