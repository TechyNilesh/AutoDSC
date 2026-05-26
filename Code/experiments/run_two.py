"""Run only AutoDSC + DBSTREAM, 10 runs each, across all datasets, in parallel.

Used to refresh just these two methods in an existing benchmark directory
(the other baselines were already run). Reuses run_scripts helpers.

Usage:
  python run_two.py --output_dir /home/nv78/AutoDSC/results/benchmark_v2
"""
import concurrent.futures
import argparse
import os

from run_scripts import run_script, dataset_name_list, run_counts

MODELS = ["AutoDSC", "dbstream"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--progress_interval", type=int, default=1000)
    args = parser.parse_args()

    print(f"*** Running {MODELS} on {len(dataset_name_list)} datasets x {len(run_counts)} runs ***")
    tasks = [(d, m, r) for r in run_counts for d in dataset_name_list for m in MODELS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.8))) as ex:
        futures = [
            ex.submit(run_script, d, m, r, True, args.progress_interval, args.output_dir)
            for (d, m, r) in tasks
        ]
        ex.shutdown(wait=True)
    print("*** AutoDSC + DBSTREAM runs complete ***")
