import os
import sys
import subprocess
import csv
import json

json_path = '/Users/stnava/code/syntx/benchmark_results_tvf.json'
pairs_file = '/Users/stnava/code/syntx/examples/pairs.csv'

def main():
    print(f"Loading existing results from {json_path}...", flush=True)
    with open(pairs_file, 'r') as f:
        all_pairs = list(csv.DictReader(f))

    for idx in range(len(all_pairs)):
        # Check if already processed
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                res = next((item for item in data if item.get('pair_idx') == idx), {})
                if res.get('tvf_dice', 0.0) > 0:
                    continue

        print(f"\n==============================================", flush=True)
        print(f"Launching isolated subprocess for Pair {idx}...", flush=True)
        print(f"==============================================", flush=True)
        
        # Launch worker in isolated subprocess
        worker_path = os.path.join(os.path.dirname(__file__), 'benchmark_worker.py')
        result = subprocess.run([sys.executable, worker_path, str(idx)])
        
        if result.returncode != 0:
            print(f"Error: Worker failed for pair {idx} with exit code {result.returncode}", flush=True)

if __name__ == '__main__':
    main()
