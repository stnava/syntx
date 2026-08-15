#!/usr/bin/env python
import sys
import os
import argparse
import json
import time
import tempfile
import traceback
import logging

from syntx.benchmark import evaluate_pair

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("syntx.benchmark.single_pair")

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)

def write_result_atomic(result: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(out_path)), prefix=".result_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(result, f, indent=2, default=str)
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        raise

def main():
    parser = argparse.ArgumentParser(description="Run a single Mindboggle pair registration benchmark.")
    parser.add_argument("--pair-idx", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["syn", "tvf", "ants_syn"])
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--config", type=str, default="docs/provenance/run_config.json")
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--save-artifacts", action="store_true", help="Save warped images, labels, and deformation fields")
    parser.add_argument("--pairs-csv", type=str, default="examples/pairs.csv")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    out_json = args.out_json or f"results/pair_{args.pair_idx:03d}_{args.model}.json"
    result = {"pair_idx": args.pair_idx, "model": args.model, "device": args.device, "status": "RUNNING", "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    try:
        config = load_config(args.config)
        out_dir = "results" if args.save_artifacts else None
        metrics = evaluate_pair(args.pair_idx, args.model, args.device, config, args.pairs_csv, args.data_dir, out_dir=out_dir)
        result.update(metrics)
        result["config"] = config.get(f"{args.model}_config", {})
        result["status"] = "SUCCESS"
        logger.info(f"RESULT | Pair {args.pair_idx} | Dice={result['dice_sym']:.4f} | Fold={result['folding_pct']:.3f}% | Time={result['runtime_seconds']:.1f}s")
    except Exception as e:
        result.update({"status": "FAILED", "error": str(e), "traceback": traceback.format_exc()})
        logger.error(f"FAILED pair {args.pair_idx}: {e}")

    result["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_result_atomic(result, out_json)
    
    if result["status"] == "SUCCESS":
        print(f"DONE:{args.pair_idx}:{args.model}:{args.device}:{result['dice_sym']:.4f}:{result['folding_pct']:.4f}")
    else:
        print(f"FAIL:{args.pair_idx}:{args.model}:{args.device}:{result.get('error', 'unknown')}")
        
    sys.exit(0 if result["status"] == "SUCCESS" else 1)

if __name__ == "__main__":
    main()
