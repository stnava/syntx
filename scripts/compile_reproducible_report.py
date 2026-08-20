#!/usr/bin/env python
"""
Compile Reproducible 90-Pair HTML Report & Markdown Summaries
===============================================================
Executes the tested and proven `syntx.viz.create_population_benchmark_report`
infrastructure to generate interactive HTML reports with Plotly scatterplots
and persistent Markdown summary tables.
"""

import os
import sys
import argparse
from syntx.viz import create_population_benchmark_report

def main():
    parser = argparse.ArgumentParser(description="Compile reproducible 90-pair benchmark HTML report.")
    parser.add_argument("--eval_dir", type=str, default="results/reproducible_eval", help="Directory with pair JSON results.")
    parser.add_argument("--summary_json", type=str, default="results/reproducible_90pair_master_summary.json", help="Master summary JSON file.")
    parser.add_argument("--out_html", type=str, default="docs/reproducible_90pair_report.html", help="Output HTML report path.")
    args = parser.parse_args()

    # Determine results source
    results_src = args.summary_json if os.path.exists(args.summary_json) else args.eval_dir
    
    print(f"Compiling population benchmark report from: {results_src}")
    html_path = create_population_benchmark_report(
        results_source=results_src,
        output_html=args.out_html,
        title="Syntx Dirichlet-Shield TVF vs ANTs C++ — 90-Pair Mindboggle Benchmark Report"
    )
    print(f"Report compiled successfully: {html_path}")

if __name__ == "__main__":
    main()
