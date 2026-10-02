#!/usr/bin/env python3
"""
Confirm the 2-D TVF tune winner on the Mindboggle 3-D pairs (77, 44, 0) against the current TVF
defaults, with SyN at its canonical defaults as the reference. Standard evaluator; run with
NOTHING else on the GPU (concurrent GPU jobs corrupt MPS results).

    python scripts/confirm_tvf_2d_winner_3d.py [--out results/confirm_tvf_3d] [--device cpu]
"""
import argparse
import json
import os

from syntx.benchmark.tune import METHODS, mindboggle_evaluator

CONFIGS = [
    ("syn", "defaults", {}),
    ("tvf", "defaults", {}),
    ("tvf", "2-D winner", {"energy_weight": 4.5e-4, "grad_step": 2.0, "multipoint_loss": [0.0, 1.0]}),
    ("tvf", "multipoint [0,1] only", {"multipoint_loss": [0.0, 1.0]}),
]
KEYS = ("dice_sym", "folding_pct", "jac_min", "inv_interior_max_mm", "inv_max_mm", "fd_folding_pct", "time_s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/confirm_tvf_3d")
    ap.add_argument("--pairs", type=int, nargs="+", default=[77, 44, 0])
    ap.add_argument("--device", default=None, help="evaluation device (default: the evaluator's choice)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["method"], r["label"], r["pair"]) for r in map(json.loads, open(path))}
    for method, label, ov in CONFIGS:
        ev = mindboggle_evaluator(METHODS[method], **({"device": a.device} if a.device else {}))
        for p in a.pairs:
            if (method, label, p) in done:
                continue
            m, _ = ev(p, dict(ov))
            row = {"method": method, "label": label, "overrides": ov, "pair": p, "metrics": m}
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{method:4s} {label:22s} pair {p:2d}: " + ", ".join(
                f"{k} {m.get(k, float('nan')):.4f}" for k in KEYS), flush=True)


if __name__ == "__main__":
    main()
