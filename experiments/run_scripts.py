import concurrent.futures
import subprocess
import random
import argparse
import os

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_PATH = os.path.join(SCRIPT_DIR, 'scripts')

# How to run this script
# python run_scripts.py --mode dataset --name adult
# python run_scripts.py --mode model --name clustream
# python run_scripts.py --mode all
# python run_scripts.py --mode random

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
    'synth_RandomRBFDrift',
    'vehicle_sensIT'
]

# List of clustering models from scripts folder
models = [
    'clustream',
    'dbstream',
    'denstream',
    'kmeans',
    'streamkmeans',
    'confstream',
    'onlineautoclust',
    'randomsearch',
    'AutoDSC'
]

# List of run counts for each model
run_counts = range(1, 11)  # 10 runs for each model

def run_script(dataset_name, model_name, run_count=None, random_seed=False, progress_interval=1000, output_dir="benchmark_results"):
    """
    Function to execute a given script for a specified dataset name, model name, and run count.
    """
    if random_seed:
        seed = random.randint(1, 101)
    else:
        seed = 42

    print(f"Running {model_name.upper()} on {dataset_name} dataset with {run_count} runs and seed {seed}")

    command = ['python']
    if model_name == 'clustream':
        command += [os.path.join(SCRIPTS_PATH, 'run_clustream.py'), '--dataset', dataset_name]
    elif model_name == 'confstream':
        command += [os.path.join(SCRIPTS_PATH, 'run_confstream.py'), '--dataset', dataset_name]
    elif model_name == 'dbstream':
        command += [os.path.join(SCRIPTS_PATH, 'run_dbstream.py'), '--dataset', dataset_name]
    elif model_name == 'denstream':
        command += [os.path.join(SCRIPTS_PATH, 'run_denstream.py'), '--dataset', dataset_name]
    elif model_name == 'hybridstream':
        command += [os.path.join(SCRIPTS_PATH, 'run_hybridstream.py'), '--dataset', dataset_name]
    elif model_name == 'kmeans':
        command += [os.path.join(SCRIPTS_PATH, 'run_kmeans.py'), '--dataset', dataset_name]
    elif model_name == 'odac':
        command += [os.path.join(SCRIPTS_PATH, 'run_odac.py'), '--dataset', dataset_name]
    elif model_name == 'onlineautoclust':
        command += [os.path.join(SCRIPTS_PATH, 'run_onlineautoclust.py'), '--dataset', dataset_name]
    elif model_name == 'randomsearch':
        command += [os.path.join(SCRIPTS_PATH, 'run_randomsearch.py'), '--dataset', dataset_name]
    elif model_name == 'streamkmeans':
        command += [os.path.join(SCRIPTS_PATH, 'run_streamkmeans.py'), '--dataset', dataset_name]
    elif model_name == 'mctsstream':
        command += [os.path.join(SCRIPTS_PATH, 'run_mctsstream.py'), '--dataset', dataset_name]
    elif model_name == 'AutoDSC':
        command += [os.path.join(SCRIPTS_PATH, 'run_autodsc.py'), '--dataset', dataset_name]
    else:
        print('Invalid model name')
        return

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


def run_all_datasets_for_model(model, progress_interval=1000, output_dir="benchmark_results"):
    print(f"*** Running all datasets for {model.upper()} model ***")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.8))) as executor:
        futures = []
        for dataset in dataset_name_list:
            futures.append(executor.submit(run_script, dataset, model, 
                                           None, False, progress_interval, output_dir))
        executor.shutdown(wait=True)

def run_all_models_for_dataset(dataset, progress_interval=1000, output_dir="benchmark_results"):
    print(f"*** Running all models for {dataset.upper()} dataset ***")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.8))) as executor:
        futures = []
        for model in models:
            futures.append(executor.submit(run_script, dataset, model, 
                                          None, False, progress_interval, output_dir))
        executor.shutdown(wait=True)

def run_all_models_and_datasets(progress_interval=1000, output_dir="benchmark_results"):
    print("*** Running all models and datasets concurrently ***")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.8))) as executor:
        tasks = [(dataset, model) for dataset in dataset_name_list for model in models]
        futures = []
        for task in tasks:
            futures.append(executor.submit(run_script, task[0], task[1], 
                                          None, False, progress_interval, output_dir))
        executor.shutdown(wait=True)

def run_all_with_random(progress_interval=1000, output_dir="benchmark_results"):
    print("*** Running all models and datasets with multiple 10 runs concurrently ***")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.8))) as executor:
        tasks = [(dataset, model, run)
                 for run in run_counts for dataset in dataset_name_list for model in models]
        futures = []
        for task in tasks:
            futures.append(executor.submit(run_script, task[0], task[1], task[2], 
                                          True, progress_interval, output_dir))
        executor.shutdown(wait=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run clustering models on stream datasets.")
    parser.add_argument('--mode', choices=['dataset', 'model', 'all', 'random'], required=True, 
                        help="Choose the mode to run: dataset, model, all, random")
    parser.add_argument('--name', help="Name of the dataset or model to run in specific mode")
    parser.add_argument('--progress_interval', type=int, default=1000, 
                        help="Interval for progress updates")
    parser.add_argument('--output_dir', type=str, default="benchmark_results", 
                        help="Directory to save results")

    args = parser.parse_args()

    if args.mode == 'dataset':
        if not args.name:
            print("Please provide the name of the dataset using --name")
        else:
            run_all_models_for_dataset(args.name, args.progress_interval, args.output_dir)
    elif args.mode == 'model':
        if not args.name:
            print("Please provide the name of the model using --name")
        else:
            run_all_datasets_for_model(args.name, args.progress_interval, args.output_dir)
    elif args.mode == 'all':
        run_all_models_and_datasets(args.progress_interval, args.output_dir)
    elif args.mode == 'random':
        run_all_with_random(args.progress_interval, args.output_dir)
