import concurrent.futures
import subprocess
import argparse
import os
import sys

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# How to run this script
# python run_ablation_study.py --dataset real_insects --ablation minimal_search
# python run_ablation_study.py --dataset real_insects --ablation small_population
# python run_ablation_study.py --dataset real_insects --ablation fast_window
# python run_ablation_study.py --dataset real_insects --ablation all
# python run_ablation_study.py --dataset real_insects --ablation full_comparison

# Finalized suite (representative datasets for ablation)
dataset_name_list = [
    'real_insects',
    'real_kdd99',
    'real_pendigits',
    'real_shuttle',
    'real_vehicle',
    'synth_blobs_expanding',
    'synth_blobs_gradual',
    'synth_blobs_merge_split',
    'synth_blobs_sudden',
    'synth_rbf_fast',
]

# Search-strategy ablation configurations.
# AutoDSC now exposes only stream/search scale parameters. These ablations are
# therefore runnable public-API variants rather than internal component toggles.
ablation_configs = {
    'minimal_search': {
        'description': 'Minimal search pressure: keep only two configurations '
                       '(incumbent plus one challenger).',
        'params': {
            'population_size': 2,
        }
    },
    'small_population': {
        'description': 'Reduced search population: fewer challengers than baseline.',
        'params': {
            'population_size': 10,
        }
    },
    'large_population': {
        'description': 'Larger search population: more challengers and higher cost.',
        'params': {
            'population_size': 30,
        }
    },
    'fast_window': {
        'description': 'Shorter search window: faster adaptation but noisier CVI scores.',
        'params': {
            'window_size': 500,
        }
    },
    'slow_window': {
        'description': 'Longer search window: more stable CVI scores but slower adaptation.',
        'params': {
            'window_size': 1500,
        }
    }
}

# Default AutoDSC parameters for baseline
default_params = {
    'window_size': 1000,
    'population_size': 15,
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
    parser = argparse.ArgumentParser(description="Ablation study for AutoDSC search-scale variants.")
    parser.add_argument('--ablation',
                       choices=list(ablation_configs.keys()) + ['all', 'full_comparison'],
                       required=True,
                       help="Choose which search-strategy variant to run")
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
