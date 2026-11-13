#!/usr/bin/env python3
"""
Automated parameter tuning for ``syntx.syn`` Beltrami arms on Mindboggle-101 ``(77, 44, 0)``:

1. **Stage A (``syn_beltrami``)**: ``regularizer='beltrami'``, ``cfl_project_inverse=False``
   around the R34 2-rep confirmed preset (``grad_step=0.6, dilatation_weight=0.1,
   sobolev_alpha=2.25, s=2.0``).
2. **Stage B (``syn_beltrami_proj``)**: ``regularizer='beltrami'``, Option A diffeomorphic
   constraint projection (``cfl_project_inverse=True, projection_frequency=1,
   cfl_project_tol=1.0, cfl_project_jac_floor=0.0``) enforcing BOTH zero Liouville folding
   AND non-increasing / zero 2-mm finite-difference Jacobian folding.

Built directly on ``syntx.benchmark.tune.Tuner`` and ``syntx.benchmark.evaluate.evaluate_mindboggle_pair``
(SHA-256-locked ``results/canonical_affines/pair_{pair:03d}_pt7_affine.mat``, disk-cached
``evaluations.jsonl``, 2-rep determinism & confirmation gate).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from syntx.benchmark.evaluate import clean_device_cache, evaluate_mindboggle_pair
from syntx.benchmark.tune import (
    DEFAULT_PAIRS,
    METRIC_KEYS,
    Config,
    Criteria,
    MethodSpec,
    Param,
    Tuner,
    _affine_sha,
    _nan,
    _num,
    twod_evaluator,
)


class FDGuardedTuner(Tuner):
    """``Tuner`` subclass that additionally enforces 2-mm finite-difference Jacobian topology."""

    def __init__(self, *args: Any, enforce_fd_nonfolding: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.enforce_fd_nonfolding = bool(enforce_fd_nonfolding)

    def score(self, cfg: Config) -> Tuple[float, bool, Dict[int, List[str]]]:
        gain, _, viol = super().score(cfg)
        if self.enforce_fd_nonfolding:
            base = self.baseline.summary(self.pairs)
            s = cfg.summary(self.pairs)
            for p in self.pairs:
                b_fd_fold = base[p].get("fd_folding_pct", float("nan"))
                m_fd_fold = s[p].get("fd_folding_pct", float("nan"))
                b_fd_min = base[p].get("fd_jac_min", float("nan"))
                m_fd_min = s[p].get("fd_jac_min", float("nan"))
                if not _nan(b_fd_fold):
                    fd_cap = max(0.0, float(b_fd_fold))
                    if _nan(m_fd_fold) or m_fd_fold > fd_cap + 1e-12:
                        viol[p].append(f"fd_folding {m_fd_fold:.5f}% > {fd_cap:.5f}%")
                if not _nan(b_fd_min) and b_fd_min > 0.0:
                    if _nan(m_fd_min) or not (m_fd_min > 0.0):
                        viol[p].append(
                            f"fd_jac_min {m_fd_min:.4f} <= 0 (baseline {b_fd_min:.4f} > 0)"
                        )
        return gain, not any(viol.values()), viol


def _make_preset_mindboggle_evaluator(
    base_params: Dict[str, Any],
    device: Optional[str] = None,
) -> Callable[[int, Dict[str, Any]], Tuple[Dict[str, float], Dict[str, Any]]]:
    """Wrap ``evaluate_mindboggle_pair(pair, 'sobolev', ...)`` with preset base parameters."""

    def run(pair: int, overrides: Dict[str, Any], report_dir: Optional[str] = None):
        clean_device_cache()
        call_kw = dict(base_params)
        call_kw.update(overrides)
        if device is not None:
            call_kw["device"] = device
        rec = evaluate_mindboggle_pair(
            pair,
            "sobolev",
            generate_report=False,
            report_out_dir=report_dir,
            verbose=False,
            **call_kw,
        )
        clean_device_cache()
        metrics = {k: _num(rec.get(v)) for k, v in METRIC_KEYS.items()}
        return metrics, rec

    run.affine_sha = _affine_sha  # type: ignore[attr-defined]
    return run


def _make_preset_twod_evaluator(
    spec: MethodSpec,
    base_params: Dict[str, Any],
) -> Callable[[int, Dict[str, Any]], Tuple[Dict[str, float], Optional[Dict[str, Any]]]]:
    inner = twod_evaluator(spec, device="cpu")

    def run(pair: int, overrides: Dict[str, Any], report_dir: Optional[str] = None):
        call_kw = dict(base_params)
        call_kw.update(overrides)
        return inner(pair, call_kw, report_dir=report_dir)

    run.affine_sha = inner.affine_sha  # type: ignore[attr-defined]
    return run


def _format_stage_markdown(title: str, res: Dict[str, Any], base_params: Dict[str, Any]) -> str:
    pairs = res["pairs"]
    winner = res["winner"]
    win_full = dict(base_params)
    win_full.update(winner.get("overrides") or {})
    lines = [
        f"## {title}",
        "",
        f"- **Pairs**: `{pairs}`",
        f"- **Code**: `{(res.get('code') or {}).get('commit')}`",
        f"- **Evaluations**: `{len(res.get('ranking', []))}` distinct configurations (`noise = {res['noise']:.6f}`, `margin = {res['margin']:.5f}`)",
        f"- **Base preset**: `{base_params}`",
        f"- **Winning parameters**: `{win_full}`",
        f"- **Confirmed improvement over base**: `{res['improved']}` (`mean Dice gain = {winner['gain']:+.4f}`)",
        "",
        "| rank | overrides | stage | reps | mean Dice | gain vs base | feasible | "
        + " | ".join(
            f"pair {p}: Dice / LiouvFold% / FD-Fold% / FD-Jmin / ICEintMx(mm) / s"
            for p in pairs
        )
        + " |",
        "|---:|---|---|---:|---:|---:|:---:|" + "---|" * len(pairs),
    ]
    for idx, r in enumerate(res["ranking"], start=1):
        mean_d = sum(r["per_pair"][p]["dice_sym"] for p in pairs) / len(pairs)
        cells = []
        for p in pairs:
            m = r["per_pair"][p]
            fd_f = m.get("fd_folding_pct", float("nan"))
            fd_j = m.get("fd_jac_min", float("nan"))
            fd_f_s = "nan" if (isinstance(fd_f, float) and math.isnan(fd_f)) else f"{fd_f:.5f}"
            fd_j_s = "nan" if (isinstance(fd_j, float) and math.isnan(fd_j)) else f"{fd_j:.4f}"
            cells.append(
                f"`{m['dice_sym']:.4f}` / `{m['folding_pct']:.4f}` / `{fd_f_s}` / "
                f"`{fd_j_s}` / `{m['inv_interior_max_mm']:.2f}` / `{m['time_s']:.0f}`"
            )
        lines.append(
            f"| {idx} | `{r['label']}` | {r['stage']} | {r['n_reps']} | "
            f"**{mean_d:.4f}** | `{r['gain']:+.4f}` | {'yes' if r['feasible'] else 'no'} | "
            + " | ".join(cells)
            + " |"
        )
    infeas = [r for r in res["ranking"] if not r["feasible"]]
    if infeas:
        lines += ["", "### Violations of infeasible configurations", ""]
        for r in infeas:
            v = {p: x for p, x in r["violations"].items() if x}
            lines.append(f"- `{r['label']}`: `{v}`")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auto-tune syntx.syn Beltrami & Beltrami+Proj presets.")
    p.add_argument("--pairs", type=int, nargs="+", default=list(DEFAULT_PAIRS))
    p.add_argument("--out-dir", type=str, default="results/autotune_syn_beltrami_suite_20261112")
    p.add_argument("--report-md", type=str, default="docs/provenance/tuning/syn_beltrami_proj_2026-11-12.md")
    p.add_argument("--stage", choices=["both", "beltrami", "beltrami_proj"], default="both")
    p.add_argument("--max-evals-belt", type=int, default=54)
    p.add_argument("--max-evals-proj", type=int, default=75)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--smoke", action="store_true", help="Fast 2-D CPU smoke gate (counts=2).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    smoke_tmp: Optional[tempfile.TemporaryDirectory[str]] = None
    if args.smoke:
        smoke_tmp = tempfile.TemporaryDirectory(prefix="syntx_autotune_belt_smoke_")
        root = Path(smoke_tmp.name)
        pairs = [0, 1]
        reg_iters = [2, 2]
        out_root = root / "results"
        report_md = root / "smoke_tuning_report.md"
        max_evals_belt = 8
        max_evals_proj = 8
        refine_rounds = 1
        dataset = "2d"
    else:
        pairs = [int(x) for x in args.pairs]
        reg_iters = [100, 100, 20]
        out_root = Path(args.out_dir)
        report_md = Path(args.report_md)
        max_evals_belt = int(args.max_evals_belt)
        max_evals_proj = int(args.max_evals_proj)
        refine_rounds = 2
        dataset = "mindboggle"

    try:
        t0 = time.perf_counter()
        out_root.mkdir(parents=True, exist_ok=True)
        summary_out: Dict[str, Any] = {
            "smoke": bool(args.smoke),
            "pairs": pairs,
            "reg_iterations": reg_iters,
            "stages": {},
        }
        md_sections: List[str] = [
            "# Tuning Report: `syntx.syn` Beltrami & Option A Beltrami+Proj (2026-11-12)",
            "",
        ]

        # ------------------------------------------------------------------
        # Stage A: syn_beltrami (cfl_project_inverse=False)
        # ------------------------------------------------------------------
        if args.stage in ("both", "beltrami"):
            belt_base: Dict[str, Any] = {
                "regularizer": "beltrami",
                "grad_step": 0.6,
                "dilatation_weight": 0.1,
                "sobolev_alpha": 2.25,
                "s": 2.0,
                "reg_iterations": list(reg_iters),
            }
            belt_fixed = {
                "regularizer": "beltrami",
                "s": 2.0,
                "reg_iterations": list(reg_iters),
            }
            belt_space = (
                [
                    Param("grad_step", values=(0.50, 0.70), lo=0.35, hi=0.90),
                    Param("dilatation_weight", values=(0.05, 0.15), lo=0.02, hi=0.35),
                ]
                if args.smoke
                else [
                    Param("grad_step", values=(0.50, 0.55, 0.65, 0.70), lo=0.35, hi=0.90),
                    Param("dilatation_weight", values=(0.05, 0.075, 0.15), lo=0.02, hi=0.35),
                    Param("sobolev_alpha", values=(1.90, 2.50), lo=1.50, hi=3.20),
                ]
            )
            belt_spec = MethodSpec(
                name="syn_beltrami",
                model="sobolev",
                function="syntx.syn",
                defaults=lambda bp=dict(belt_base): dict(bp),
                space=belt_space,
            )
            belt_eval = (
                _make_preset_twod_evaluator(belt_spec, belt_base)
                if args.smoke
                else _make_preset_mindboggle_evaluator(belt_base, device=args.device)
            )
            belt_criteria = Criteria(
                folding_abs=0.0005 if args.smoke else 0.0,
                folding_rel=1.0,
                inv_interior_abs=1.0 if args.smoke else 0.50,
                inv_interior_rel=1.1,
                positive_jac_if_baseline=True,
                max_pair_drop=0.0020,
                min_gain=0.0005,
            )
            tuner_a = FDGuardedTuner(
                belt_spec,
                pairs=pairs,
                space=belt_space,
                criteria=belt_criteria,
                evaluator=belt_eval,
                out_dir=str(out_root / "stage_a_syn_beltrami"),
                max_evals=max_evals_belt,
                refine_rounds=refine_rounds,
                top_k=3,
                allow_dirty=True,
                check_canonical=False,
                fixed_parameters=belt_fixed,
                dataset=dataset,
                enforce_fd_nonfolding=False,
            )
            res_a = tuner_a.run()
            win_a = dict(belt_base)
            win_a.update(res_a["winner"].get("overrides") or {})
            summary_out["stages"]["syn_beltrami"] = {
                "improved": res_a["improved"],
                "gain": res_a["winner"]["gain"],
                "winning_params": win_a,
                "per_pair": res_a["winner"]["per_pair"],
                "baseline_per_pair": res_a["baseline"],
                "noise": res_a["noise"],
                "margin": res_a["margin"],
                "out_dir": str(out_root / "stage_a_syn_beltrami"),
            }
            md_sections.append(
                _format_stage_markdown("Stage A: `syn_beltrami` (`cfl_project_inverse=False`)", res_a, belt_base)
            )

        # ------------------------------------------------------------------
        # Stage B: syn_beltrami_proj (Option A: pf=1, tol=1.0 vox, jac_floor=0.0)
        # ------------------------------------------------------------------
        if args.stage in ("both", "beltrami_proj"):
            proj_base: Dict[str, Any] = {
                "regularizer": "beltrami",
                "grad_step": 0.6,
                "dilatation_weight": 0.1,
                "sobolev_alpha": 2.25,
                "s": 2.0,
                "cfl_project_inverse": True,
                "projection_frequency": 1,
                "cfl_project_tol": 1.0,
                "cfl_project_jac_floor": 0.0,
                "reg_iterations": list(reg_iters),
            }
            proj_fixed = {
                "regularizer": "beltrami",
                "s": 2.0,
                "cfl_project_inverse": True,
                "projection_frequency": 1,
                "cfl_project_tol": 1.0,
                "cfl_project_jac_floor": 0.0,
                "reg_iterations": list(reg_iters),
            }
            proj_space = (
                [
                    Param("grad_step", values=(0.50, 0.75), lo=0.40, hi=1.15),
                    Param("sobolev_alpha", values=(1.80, 2.55), lo=1.30, hi=3.20),
                ]
                if args.smoke
                else [
                    Param("grad_step", values=(0.50, 0.68, 0.76, 0.85, 0.95), lo=0.40, hi=1.15),
                    Param("sobolev_alpha", values=(1.55, 1.80, 2.00, 2.55), lo=1.30, hi=3.20),
                    Param("dilatation_weight", values=(0.05, 0.075, 0.15, 0.22), lo=0.02, hi=0.40),
                ]
            )
            proj_spec = MethodSpec(
                name="syn_beltrami_proj",
                model="sobolev",
                function="syntx.syn",
                defaults=lambda bp=dict(proj_base): dict(bp),
                space=proj_space,
            )
            proj_eval = (
                _make_preset_twod_evaluator(proj_spec, proj_base)
                if args.smoke
                else _make_preset_mindboggle_evaluator(proj_base, device=args.device)
            )
            proj_criteria = Criteria(
                folding_abs=0.0005 if args.smoke else 0.0,
                folding_rel=1.0,
                inv_interior_abs=1.0 if args.smoke else 0.50,
                inv_interior_rel=1.1,
                positive_jac_if_baseline=True,
                max_pair_drop=0.0025,
                min_gain=0.0005,
            )
            tuner_b = FDGuardedTuner(
                proj_spec,
                pairs=pairs,
                space=proj_space,
                criteria=proj_criteria,
                evaluator=proj_eval,
                out_dir=str(out_root / "stage_b_syn_beltrami_proj"),
                max_evals=max_evals_proj,
                refine_rounds=refine_rounds,
                top_k=3,
                allow_dirty=True,
                check_canonical=False,
                fixed_parameters=proj_fixed,
                dataset=dataset,
                enforce_fd_nonfolding=(not args.smoke),
            )
            res_b = tuner_b.run()
            win_b = dict(proj_base)
            win_b.update(res_b["winner"].get("overrides") or {})
            summary_out["stages"]["syn_beltrami_proj"] = {
                "improved": res_b["improved"],
                "gain": res_b["winner"]["gain"],
                "winning_params": win_b,
                "per_pair": res_b["winner"]["per_pair"],
                "baseline_per_pair": res_b["baseline"],
                "noise": res_b["noise"],
                "margin": res_b["margin"],
                "out_dir": str(out_root / "stage_b_syn_beltrami_proj"),
            }
            md_sections.append(
                _format_stage_markdown(
                    "Stage B: `syn_beltrami_proj` (Option A: `cfl_project_inverse=True, pf=1, tol=1.0 vox`)",
                    res_b,
                    proj_base,
                )
            )

        summary_out["wall_minutes"] = (time.perf_counter() - t0) / 60.0
        sum_json = out_root / "suite_summary.json"
        sum_json.write_text(json.dumps(summary_out, indent=2, default=str), encoding="utf-8")

        report_md.parent.mkdir(parents=True, exist_ok=True)
        tmp_md = report_md.with_suffix(".md.tmp")
        tmp_md.write_text("\n".join(md_sections), encoding="utf-8")
        os.replace(tmp_md, report_md)
        print(
            f"[DONE] Suite summary -> {os.path.abspath(sum_json)} | Report -> {os.path.abspath(report_md)} "
            f"({summary_out['wall_minutes']:.2f} min)",
            flush=True,
        )
        return summary_out
    finally:
        if smoke_tmp is not None:
            smoke_tmp.cleanup()


if __name__ == "__main__":
    main()
