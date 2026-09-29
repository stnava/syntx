#!/usr/bin/env python3
"""
SyNGS run-to-run repeatability on Mindboggle pairs 77/44/0 (3-D, standard benchmark path),
RegAdam relative eps (new default) vs the previous absolute eps (adam_eps_rel=0).
Two repeats per (pair, setting); writes a summary table and per-run JSON with provenance.

    python scripts/check_syngs_repeatability.py [--out results/syngs_repeatability]
"""
import argparse
import json
import os

import pandas as pd

from syntx.benchmark.evaluate import evaluate_mindboggle_pair

SETTINGS = {"relative_eps_default": {}, "absolute_eps_previous": {"adam_eps_rel": 0.0}}
KEYS = ["syntx_dice_sym", "syntx_fold", "syntx_min_jac", "syntx_inv_interior_max", "syntx_inv_max", "syntx_time"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/syngs_repeatability")
    ap.add_argument("--pairs", type=int, nargs="+", default=[77, 44, 0])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for name, ov in SETTINGS.items():
        for pair in a.pairs:
            for rep in range(2):
                rec = evaluate_mindboggle_pair(pair, "syngs", generate_report=False, verbose=False, **ov)
                with open(os.path.join(a.out, f"{name}_p{pair}_r{rep}.json"), "w") as f:
                    json.dump(rec, f, default=str)
                row = {"setting": name, "pair": pair, "rep": rep, **{k: rec.get(k) for k in KEYS}}
                rows.append(row)
                print(row, flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(a.out, "runs.csv"), index=False)
    df = pd.DataFrame(rows)
    spread = df.groupby(["setting", "pair"]).agg(
        dice_rep0=("syntx_dice_sym", "first"), dice_rep1=("syntx_dice_sym", "last"),
        fold_max=("syntx_fold", "max"), inv_max=("syntx_inv_max", "max")).reset_index()
    spread["dice_abs_diff"] = (spread.dice_rep1 - spread.dice_rep0).abs()
    spread.to_csv(os.path.join(a.out, "spread.csv"), index=False)
    print(spread.to_string(index=False))


if __name__ == "__main__":
    main()
