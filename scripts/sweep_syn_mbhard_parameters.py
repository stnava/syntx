#!/usr/bin/env python3
"""
syntx.syn parameter sweep on mbhard (Mindboggle pair 44), standard benchmark path.

Each variant runs evaluate_mindboggle_pair(44, "sobolev", **overrides): canonical
syntx.syn defaults (docs/provenance/best_parameters.json) plus the listed overrides.
Every per-variant JSON carries the full provenance manifest (syntx.provenance) --
commit, uncommitted diff, this script's text, environment, and the parameters
syntx.syn actually resolved. A summary CSV/Markdown table is written at the end.

    python scripts/sweep_syn_mbhard_parameters.py [--set default|regularization] [--out DIR] [--only NAME ...]
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
    # RegAdam: max per-step displacement = grad_step * optimizer_lr when optimizer_lr != 1e-3,
    # else grad_step**2 (the 1e-3 default is a sentinel meaning "use grad_step").
    "regadam_lr1": ("canonical + optimizer reg_adam, optimizer_lr 1.0 (max step = grad_step, like CFL)",
                    {"optimizer": "reg_adam", "optimizer_lr": 1.0}),
    "regadam_default_lr": ("canonical + optimizer reg_adam, default optimizer_lr (max step = grad_step**2)",
                           {"optimizer": "reg_adam"}),
}

# Stage-1 regularisation sweep (pair 44, affine held constant). Canonical = sobolev,
# alpha 1.5, flow 3.0 (variance), fast_smooth off, total_sigma 0.
REGULARIZATION = {
    "reg_canonical_anchor": ("canonical (same-session anchor)", {}),
    "sobolev_a1.0": ("sobolev alpha 1.0", {"sobolev_alpha": 1.0}),
    "sobolev_a2.0": ("sobolev alpha 2.0", {"sobolev_alpha": 2.0}),
    "sobolev_a3.0": ("sobolev alpha 3.0", {"sobolev_alpha": 3.0}),
    "sobolev_flow2.0": ("sobolev alpha 1.5, flow_sigma 2.0 (narrower post-filter)", {"flow_sigma": 2.0}),
    "sobolev_flow4.5": ("sobolev alpha 1.5, flow_sigma 4.5 (wider post-filter)", {"flow_sigma": 4.5}),
    "sobolev_total0.5": ("sobolev canonical + total_sigma 0.5", {"total_sigma": 0.5}),
    "sobolev_total1.0": ("sobolev canonical + total_sigma 1.0", {"total_sigma": 1.0}),
    "gaussian_flow2.0": ("gaussian, flow_sigma 2.0", {"regularizer": "gaussian", "flow_sigma": 2.0}),
    "gaussian_flow3.0": ("gaussian, flow_sigma 3.0", {"regularizer": "gaussian"}),
    "gaussian_flow4.5": ("gaussian, flow_sigma 4.5", {"regularizer": "gaussian", "flow_sigma": 4.5}),
    "dsti1_a0.866": ("dsti1, alpha default sqrt(flow)/2 = 0.866", {"regularizer": "dsti1"}),
    "dsti1_a1.5": ("dsti1, alpha 1.5", {"regularizer": "dsti1", "sobolev_alpha": 1.5}),
    "dsti1_a2.5": ("dsti1, alpha 2.5", {"regularizer": "dsti1", "sobolev_alpha": 2.5}),
    "dsti_a1.5": ("dsti, alpha 1.5", {"regularizer": "dsti", "sobolev_alpha": 1.5}),
    "compact_flow3.0": ("compact (erf) gaussian, flow_sigma 3.0", {"regularizer": "compact"}),
    "bspline": ("bspline regulariser (antstorch), defaults", {"regularizer": "bspline"}),
}

# Stage 2: refine the two clean stage-1 gains (sobolev flow_sigma 4.5: +0.0006; dsti1 alpha
# 2.5: +0.0010), repeat both to separate them from run-to-run noise, combine with step 0.35,
# and try a much smaller total_sigma (0.5 / 1.0 collapsed Dice to 0.50 / 0.46).
REGULARIZATION_STAGE2 = {
    "s2_canonical_anchor": ("canonical (same-session anchor)", {}),
    "s2_sobolev_flow4.5_repeat": ("sobolev flow_sigma 4.5 (repeat)", {"flow_sigma": 4.5}),
    "s2_sobolev_flow6.0": ("sobolev flow_sigma 6.0", {"flow_sigma": 6.0}),
    "s2_sobolev_flow4.5_a1.25": ("sobolev flow_sigma 4.5, alpha 1.25", {"flow_sigma": 4.5, "sobolev_alpha": 1.25}),
    "s2_sobolev_flow4.5_a1.0": ("sobolev flow_sigma 4.5, alpha 1.0", {"flow_sigma": 4.5, "sobolev_alpha": 1.0}),
    "s2_sobolev_flow4.5_step0.35": ("sobolev flow_sigma 4.5, grad_step 0.35", {"flow_sigma": 4.5, "grad_step": 0.35}),
    "s2_dsti1_a2.5_repeat": ("dsti1 alpha 2.5 (repeat)", {"regularizer": "dsti1", "sobolev_alpha": 2.5}),
    "s2_dsti1_a2.0": ("dsti1 alpha 2.0", {"regularizer": "dsti1", "sobolev_alpha": 2.0}),
    "s2_dsti1_a3.0": ("dsti1 alpha 3.0", {"regularizer": "dsti1", "sobolev_alpha": 3.0}),
    "s2_dsti1_a2.5_step0.35": ("dsti1 alpha 2.5, grad_step 0.35",
                               {"regularizer": "dsti1", "sobolev_alpha": 2.5, "grad_step": 0.35}),
    "s2_dsti1_a2.5_flow4.5": ("dsti1 alpha 2.5, flow_sigma 4.5",
                              {"regularizer": "dsti1", "sobolev_alpha": 2.5, "flow_sigma": 4.5}),
    "s2_sobolev_total0.1": ("sobolev canonical + total_sigma 0.1", {"total_sigma": 0.1}),
}

VARIANT_SETS = {"default": VARIANTS, "regularization": REGULARIZATION,
                "regularization_stage2": REGULARIZATION_STAGE2}

# "Better than canonical": Dice gain above run-to-run noise (0.0002 measured) with no loss
# of topology or inverse consistency.
BETTER = {"dice_gain": 0.0005, "folding_pct": 0.0005, "inv_interior_max_mm": 1.0}
AFFINE_CACHE = "results/canonical_affines/pair_044_pt7_affine.mat"

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
    ap.add_argument("--set", default="default", choices=sorted(VARIANT_SETS))
    ap.add_argument("--baseline-dice", type=float, default=None,
                    help="canonical Dice for the 'better' flag (default: this run's first canonical variant)")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import hashlib

    def affine_sha():
        if not os.path.exists(AFFINE_CACHE):
            return None
        with open(AFFINE_CACHE, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    affine0 = affine_sha()
    print(f"affine cache {AFFINE_CACHE} sha256={affine0}", flush=True)
    baseline = args.baseline_dice

    rows = []
    for name, (desc, overrides) in VARIANT_SETS[args.set].items():
        if args.only and name not in args.only:
            continue
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {desc}", flush=True)
        try:
            rec = evaluate_mindboggle_pair(
                PAIR, "sobolev", generate_report=not args.no_report,
                report_out_dir=os.path.join(args.out, name), verbose=False, **overrides)
        except Exception as e:  # keep sweeping; record the failure
            print(f"   FAILED: {type(e).__name__}: {e}", flush=True)
            rows.append({"variant": name, "description": desc, "overrides": json.dumps(overrides),
                         "error": f"{type(e).__name__}: {e}"})
            pd.DataFrame(rows).to_csv(os.path.join(args.out, "summary.csv"), index=False)
            continue
        assert_manifest_complete(rec["provenance"])
        if affine_sha() != affine0:
            raise RuntimeError(f"affine cache changed during {name}; affine not held constant")
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
            "resolved_regularizer": fk.get("regularizer"),
            "resolved_total_sigma_var": (ma.get("elastic_sigma") or 0) ** 2,
            "resolved_optimizer": fk.get("optimizer_type"),
            "affine_sha256": affine0,
            "load_avg_1m_at_end": rec["provenance"]["environment"]["load_average"][0],
            "commit": rec["provenance"]["code"]["git"]["commit"][:10],
            "dirty": rec["provenance"]["code"]["git"]["dirty"],
            "changed_during_run": rec["provenance"].get("changed_during_run"),
            "report_html": rec.get("report_html"),
        })
        if baseline is None and name in ("canonical", "reg_canonical_anchor", "s2_canonical_anchor"):
            baseline = row["dice_sym"]
        if baseline is not None:
            row["dice_gain_vs_canonical"] = row["dice_sym"] - baseline
            row["better_than_canonical"] = bool(
                row["dice_gain_vs_canonical"] > BETTER["dice_gain"]
                and (row["folding_pct"] or 0) <= BETTER["folding_pct"]
                and (row["inv_interior_max_mm"] or 0) <= BETTER["inv_interior_max_mm"])
        rows.append(row)
        print("   " + ", ".join(f"{k}={row[k]:.4f}" if isinstance(row[k], float) else f"{k}={row[k]}"
                                for k in ("dice_sym", "folding_pct", "inv_interior_max_mm", "time_s")),
              flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "summary.csv"), index=False)

    df = pd.DataFrame(rows)
    if "error" in df.columns:  # failed variants stay in summary.csv, not in the table
        df = df[df["error"].isna()]
    cols = ["variant", "dice_sym", "dice_fixed", "dice_moving", "folding_pct", "jac_min", "jac_max",
            "inv_mean_mm", "inv_p95_mm", "inv_interior_mean_mm", "inv_interior_max_mm", "inv_max_mm",
            "time_s", "load_avg_1m_at_end"] + [c for c in ("dice_gain_vs_canonical", "better_than_canonical")
                                               if c in df.columns]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(df[cols].to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n" + df[["variant", "description"]].to_markdown(index=False) + "\n")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
