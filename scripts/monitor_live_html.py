#!/usr/bin/env python
"""
scripts/monitor_live_html.py — Real-Time Background Monitor for Benchmark Dashboard
===================================================================================

Monitors the results directory for newly completed pair JSON files across all models
and regenerates results/multimodel_live_comparison.html in real time.
"""

import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from syntx.benchmark.html_report import generate_live_html_report


def main():
    results_dir = os.path.join(PROJECT_ROOT, "results")
    device = "mps"
    out_html = os.path.join(results_dir, "multimodel_live_comparison.html")
    
    print(f"[monitor] Watching {results_dir} for completed registration results...")
    print(f"[monitor] Target dashboard: {out_html}")
    
    last_count = -1
    
    while True:
        try:
            # Count total completed json files across all models
            json_count = 0
            for root, _, files in os.walk(results_dir):
                for f in files:
                    if f.startswith("pair_") and f.endswith(".json"):
                        json_count += 1
            
            if json_count != last_count:
                generate_live_html_report(
                    results_dir=results_dir,
                    device=device,
                    models=["syn", "gaussian", "syngs", "tvf", "greedy"],
                    total_pairs=90,
                    out_html=out_html,
                    refresh_seconds=10,
                )
                print(f"[monitor] Updated dashboard at {time.strftime('%H:%M:%S')} (detected {json_count} completed records)", flush=True)
                last_count = json_count
        except Exception as e:
            print(f"[monitor] Error updating dashboard: {e}", flush=True)
            
        time.sleep(5)


if __name__ == "__main__":
    main()
