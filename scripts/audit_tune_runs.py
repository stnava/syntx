#!/usr/bin/env python3
"""
Re-run cached tune evaluations and check they reproduce bit-for-bit.

syngs / tvf / syn / greedy are deterministic on an otherwise idle GPU (v5.4.67), so any cached
evaluation must reproduce its metrics exactly. A mismatch means that run was corrupted -- e.g. by
another process using the MPS GPU at the same time (observed: concurrent GPU jobs make stock MPS
ops return wrong values). Run with NOTHING else on the GPU.

    python scripts/audit_tune_runs.py results/tune_syngs_20260929_det \
        --configs '{}' '{"max_step_norm": 0.3, "alpha": 0.675}'
"""
import argparse
import json
import os
import time

from syntx.benchmark.tune import METHODS, mindboggle_evaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--configs", nargs="+", required=True, help="JSON override dicts to audit")
    ap.add_argument("--pairs", type=int, nargs="+", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(a.out_dir, "evaluations.jsonl"))]
    method = rows[0]["method"]
    run = mindboggle_evaluator(METHODS[method])
    # Overrides are relative to the defaults AT TUNE TIME; if the defaults have since changed
    # (e.g. the winner was codified), pin the tune-time values explicitly.
    pinned = {}
    res_path = os.path.join(a.out_dir, "result.json")
    if os.path.exists(res_path):
        tune_defaults = json.load(open(res_path)).get("defaults", {})
        now = METHODS[method].defaults()
        pinned = {k: v for k, v in tune_defaults.items() if now.get(k) != v}
        if pinned:
            print(f"defaults changed since the tune; pinning tune-time values {pinned}", flush=True)
    report = []
    for cfg in a.configs:
        ov = json.loads(cfg)
        cached = [r for r in rows if r["overrides"] == ov and r["rep"] == 0
                  and (a.pairs is None or r["pair"] in a.pairs)]
        for r in cached:
            t = time.time()
            m, _ = run(r["pair"], {**pinned, **ov})
            keys = ("dice_sym", "folding_pct", "jac_min", "inv_interior_max_mm", "inv_max_mm")
            same = all(m[k] == r["metrics"][k] or (m[k] != m[k] and r["metrics"][k] != r["metrics"][k]) for k in keys)
            line = (f"{'OK      ' if same else 'MISMATCH'} {json.dumps(ov)} pair {r['pair']} (cached {r.get('stage')}): "
                    + ", ".join(f"{k} {r['metrics'][k]:.4f}->{m[k]:.4f}" for k in keys[:4]) + f"  [{time.time()-t:.0f}s]")
            print(line, flush=True)
            report.append({"overrides": ov, "pair": r["pair"], "reproduced": same,
                           "cached": {k: r["metrics"][k] for k in keys}, "rerun": {k: m[k] for k in keys}})
    with open(os.path.join(a.out_dir, "audit.jsonl"), "a") as f:
        for x in report:
            f.write(json.dumps(x) + "\n")


if __name__ == "__main__":
    main()
