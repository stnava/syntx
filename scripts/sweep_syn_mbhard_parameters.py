#!/usr/bin/env python3
"""
syntx.syn parameter sweep on mbhard (Mindboggle pair 44), standard benchmark path.

Each variant runs evaluate_mindboggle_pair(44, "sobolev", **overrides): canonical
syntx.syn defaults (docs/provenance/best_parameters.json) plus the listed overrides.
Every per-variant JSON carries the full provenance manifest (syntx.provenance) --
commit, uncommitted diff, this script's text, environment, and the parameters
syntx.syn actually resolved. A summary CSV/Markdown table is written at the end.

    python scripts/sweep_syn_mbhard_parameters.py [--out results/syn_param_sweep_mbhard] [--only NAME ...]
"""

import argparse
import json
import math
import os
import time

import pandas as pd

from syntx.benchmark.evaluate import evaluate_mindboggle_pair
from syntx.provenance import assert_manifest_complete, resolved_parameters

PAIR = 44  # mbhard

# name -> (description, overrides on top of the canonical syntx.syn defaults)
VARIANTS = {
    "canonical": ("canonical defaults: step 0.25, flow 3.0, fast_smooth off, alpha 1.5, in-loop inverse 10", {}),
    "fast_smooth_on": ("canonical + fast_smooth on (= the 2026-09-20 record's stated SyN settings)",
                       {"fast_smooth": True}),
    "drift_0920_committed": ("config.py at 3e862c9: step 0.35, flow 2.5, fast_smooth on, in-loop inverse 6",
                             {"grad_step": 0.35, "flow_sigma": 2.5, "fast_smooth": True, "in_loop_inv_steps": 6}),
    "in_loop_6": ("canonical + in-loop inverse 6 (pre-5.4.46 default)", {"in_loop_inv_steps": 6}),
    "alpha_0866": ("canonical + sobolev_alpha 0.866 (pre-5.4.46 implicit default)", {"sobolev_alpha": 0.866}),
    "step_035": ("canonical + grad_step 0.35", {"grad_step": 0.35}),
    "no_stationary_boundary": ("canonical + stationary_boundary off (pre-5.4.45)", {"stationary_boundary": False}),
    "canonical_repeat": ("canonical again: run-to-run (MPS) noise", {}),
}

METRICS = {
    "syntx_dice_sym": "dice_sym", "syntx_dice_fixed": "dice_fixed", "syntx_dice_moving": "dice_moving",
    "syntx_fold": "folding_pct", "syntx_min_jac": "jac_min", "syntx_max_jac": "jac_max",
    "syntx_inv_mean": "inv_mean_mm", "syntx_inv_p95": "inv_p95_mm", "syntx_inv_max": "inv_max_mm",
    "syntx_inv_interior_mean": "inv_interior_mean_mm", "syntx_inv_interior_max": "inv_interior_max_mm",
    "syntx_time": "time_s",
}


def _clean(v):
    return None if isinstance(v, float) and not math.isfinite(v) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/syn_param_sweep_mbhard")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for name, (desc, overrides) in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {desc}", flush=True)
        rec = evaluate_mindboggle_pair(
            PAIR, "sobolev", generate_report=not args.no_report,
            report_out_dir=os.path.join(args.out, name), verbose=False, **overrides)
        assert_manifest_complete(rec["provenance"])
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(rec, f, indent=2, default=str)
        p = resolved_parameters(rec["provenance"])
        fk, ma = p["fit_kwargs"], p["model_attributes"]
        row = {"variant": name, "description": desc, "overrides": json.dumps(overrides)}
        row.update({out: _clean(rec.get(k)) for k, out in METRICS.items()})
        row.update({
            "resolved_grad_step": fk.get("cfl_voxels"),
            "resolved_flow_sigma_var": (ma.get("fluid_sigma") or 0) ** 2,
            "resolved_fast_smooth": bool(fk.get("fast_smooth")),
            "resolved_sobolev_alpha": fk.get("sobolev_alpha"),
            "resolved_in_loop_inv_steps": ma.get("in_loop_inv_steps"),
            "resolved_stationary_boundary": ma.get("stationary_boundary"),
            "load_avg_1m_at_end": rec["provenance"]["environment"]["load_average"][0],
            "commit": rec["provenance"]["code"]["git"]["commit"][:10],
            "dirty": rec["provenance"]["code"]["git"]["dirty"],
            "report_html": rec.get("report_html"),
        })
        rows.append(row)
        print("   " + ", ".join(f"{k}={row[k]:.4f}" if isinstance(row[k], float) else f"{k}={row[k]}"
                                for k in ("dice_sym", "folding_pct", "inv_interior_max_mm", "time_s")),
              flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "summary.csv"), index=False)

    df = pd.DataFrame(rows)
    cols = ["variant", "dice_sym", "dice_fixed", "dice_moving", "folding_pct", "jac_min", "jac_max",
            "inv_mean_mm", "inv_p95_mm", "inv_interior_mean_mm", "inv_interior_max_mm", "inv_max_mm",
            "time_s", "load_avg_1m_at_end"]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(df[cols].to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n" + df[["variant", "description"]].to_markdown(index=False) + "\n")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
