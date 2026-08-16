import os
import json
import glob
from syntx.viz.reports import create_benchmark_report

def generate_report():
    syn_files = [f for f in glob.glob("results/pair_*_syn.json") if not f.endswith("_ants_syn.json")]
    ants_files = glob.glob("results/pair_*_ants_syn.json")
    
    syn_results = {}
    for f in syn_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
            if data.get('status') == 'SUCCESS':
                syn_results[data.get('pair_idx')] = data
        except Exception:
            pass
            
    ants_results = {}
    for f in ants_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
            if data.get('status') == 'SUCCESS':
                ants_results[data.get('pair_idx')] = data
        except Exception:
            pass

    create_benchmark_report(
        syn_results=syn_results,
        ants_results=ants_results,
        total_pairs=90,
        output_html="results/90pair_report.html"
    )

if __name__ == "__main__":
    generate_report()
    print("Comparative report generated at results/90pair_report.html")
