#!/usr/bin/env python3
"""
Concerted Multi-Solver Autotuning & Multi-Tier Benchmark Orchestrator for ``syntx``.

Designed so parameter tuning and out-of-sample benchmarking operate as a single, leak-free,
hardware-isolated campaign once the GPU (`MPS` / `CUDA`) is available:

1. **Stage 1 (`tune`) — Concerted 3-Pair Continuum Autotuning**:
   Sequentially executes ``syntx.benchmark.tune`` across ``--methods`` (default: ``syn``,
   ``tvf``, ``syngs``; ``greedy`` optional since ``v6.43.0`` already codified ``div_curl`` +
   ``cfl`` at ``0.6041`` mean Dice) on the canonical tuning pairs ``--tune-pairs 77 44 0`` at
   fixed ``reg_iterations = [100, 100, 20]`` using cached ``pair_{pair:03d}_pt7_affine.mat``.
   Assembles ``<out_dir>/concerted_manifest.json`` recording each solver's baseline defaults
   and confirmed 2-rep feasible winner overrides **without mutating ``main``**.

2. **Stage 2A (`heldout`) — Subject-Disjoint Held-Out Promotion Gate (`GEMINI.md` §6)**:
   Evaluates ``syntx`` current defaults, every Stage-1 confirmed winner, and the ``ANTs C++``
   ``SyNOnly`` reference at ``[100, 100, 20]`` on a **100% subject-disjoint** Mindboggle-101
   validation set (default ``--heldout-pairs 28 55 84``; extended 6-pair option
   ``28 33 51 55 68 84``). Strictly schedules all GPU/MPS runs (Phase A) before multi-core CPU
   ``ANTs C++`` runs (Phase B) in fresh isolated subprocesses. Only Stage-1 winners that
   simultaneously:
   - achieve zero Liouville/flow folding violations (`folding_pct <= 0.0005%`, `jac_min > 0`),
   - satisfy inverse consistency (`inv_interior_max_mm <= 1.0 mm` for symmetric solvers),
   - pass `antsxqc` via `evaluate_registration_qc` (`overall_status != 'fail'`),
   - never regress any held-out pair by more than `0.001` vs. the current `syntx` default, and
   - match or beat the `ANTs C++` `[100, 100, 20]` reference mean Dice on the held-out set
     (for bidirectional solvers; or beat `v6.43.0` default + `FireANTs` for unidirectional `greedy`)
   receive ``promotion_verdict = "PROMOTE"``.

3. **Stage 2B (`benchmark-mindboggle`) — Full Multi-Paradigm Cortical Benchmark**:
   Runs ``benchmarks/fast_syn_ants_benchmark.py`` across ``--benchmark-pairs`` (default union
   of tuning + held-out pairs ``44 77 0 28 55 84``) with ``--include-syngs-tvf`` and
   ``--include-external`` (``syntx`` regularizer suite + ``tvf`` + ``syngs`` + ``greedy`` vs.
   ``ANTs C++``, ``FireANTs``, and ``VoxelMorph`` zero-shot / polished / hybrid arms), plus
   isolated evaluation of any Stage-2A promoted winners not yet codified on ``main``.

4. **Stage 2C (`benchmark-msd`, opt-in `--include-msd`) — 10-Task (`N = 50`) Cross-Organ Gate**:
   Executes ``scripts/benchmark_decathlon_provenance_harness.py`` and renders the compact WebP
   light/dark HTML report ``scripts/build_decathlon_provenance_html_report.py`` to verify zero
   CT or multi-organ MRI regressions.

Usage (do NOT launch GPU stages while another GPU workload is active):
  - CPU-only end-to-end smoke gate (<10 s, 100% of stages, counts=2):
      SYNTX_DEVICE=cpu python3 scripts/concerted_tune_and_benchmark.py --smoke
  - Inspect exact resolved command DAG & pair disjunction without running:
      python3 scripts/concerted_tune_and_benchmark.py --dry-run
  - Full concerted campaign when GPU is free:
      python3 -u scripts/concerted_tune_and_benchmark.py --out-dir results/concerted_$(date +%Y%m%d)
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import html as _html
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ants
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import syntx
from syntx.benchmark.evaluate import _inverse_error_stats
from syntx.benchmark.tune import (
    Criteria,
    METHODS,
    Param,
    Tuner,
    pair_violations,
)
from syntx.core.pipeline import cleanup_gpu
from syntx.deformation_metrics import (
    compute_bidirectional_dice,
    compute_jacobian_metrics,
    flow_jacobian_metrics,
)
from syntx.device import resolve_device
from syntx.provenance import (
    build_manifest,
    capture_registration_calls,
    code_state,
    environment,
)
from syntx.registration_metrics import evaluate_registration_qc

DEFAULT_METHODS: Tuple[str, ...] = ("syn", "tvf", "syngs")
ALL_METHODS: Tuple[str, ...] = ("greedy", "syn", "tvf", "syngs")
DEFAULT_TUNE_PAIRS: Tuple[int, ...] = (77, 44, 0)
# 100% subject-disjoint from (77, 44, 0) in examples/pairs.csv:
#   28: intra | NKI-RS-22-11 <- NKI-RS-22-20
#   55: inter | NKI-RS-22-2  <- OASIS-TRT-20-8
#   84: inter | MMRR-21-11   <- NKI-RS-22-1
DEFAULT_HELDOUT_PAIRS: Tuple[int, ...] = (28, 55, 84)
DEFAULT_BENCHMARK_PAIRS: Tuple[int, ...] = (44, 77, 0, 28, 55, 84)
DEFAULT_ITERS: Tuple[int, ...] = (100, 100, 20)
STAGE_CHOICES: Tuple[str, ...] = (
    "all",
    "tune",
    "heldout",
    "benchmark-mindboggle",
    "benchmark-msd",
)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _verify_pair_disjunction(
    pairs_csv: Path,
    tune_pairs: Sequence[int],
    heldout_pairs: Sequence[int],
) -> Dict[str, Any]:
    """Verify that tune_pairs and heldout_pairs share neither pair indices nor subjects."""
    overlap_idx = sorted(set(int(x) for x in tune_pairs) & set(int(x) for x in heldout_pairs))
    if overlap_idx:
        raise ValueError(
            f"Tuning pairs {list(tune_pairs)} and held-out pairs {list(heldout_pairs)} "
            f"overlap on pair indices {overlap_idx}; held-out validation must be disjoint."
        )
    df = pd.read_csv(pairs_csv)
    tune_subjects = set()
    for idx in tune_pairs:
        row = df.iloc[int(idx)]
        tune_subjects.add(f"{row['cohort1']}/{row['subject1']}")
        tune_subjects.add(f"{row['cohort2']}/{row['subject2']}")
    heldout_subjects = set()
    for idx in heldout_pairs:
        row = df.iloc[int(idx)]
        heldout_subjects.add(f"{row['cohort1']}/{row['subject1']}")
        heldout_subjects.add(f"{row['cohort2']}/{row['subject2']}")
    overlap_subjects = sorted(tune_subjects & heldout_subjects)
    if overlap_subjects:
        raise ValueError(
            f"Held-out pairs {list(heldout_pairs)} share subjects {overlap_subjects} with "
            f"tuning pairs {list(tune_pairs)}; select subject-disjoint Mindboggle pairs."
        )
    return {
        "pairs_csv": str(pairs_csv),
        "tune_pairs": [int(x) for x in tune_pairs],
        "heldout_pairs": [int(x) for x in heldout_pairs],
        "tune_subject_count": len(tune_subjects),
        "heldout_subject_count": len(heldout_subjects),
        "subject_overlap": overlap_subjects,
        "subject_disjoint": True,
    }


def _build_smoke_fixture(root: Path) -> Tuple[Path, Path, Path]:
    """Create 4 subject-disjoint 3-D physical-space ANTsImage pairs (2 tune + 2 held-out)
    with non-trivial origin, anisotropic spacing, signed/oblique direction cosine matrix,
    3-class cortical/subcortical labels, and identity canonical `.mat` affines."""
    data_dir = root / "mindboggle_smoke"
    aff_dir = root / "canonical_affines"
    aff_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = root / "pairs_smoke.csv"

    shape = (12, 12, 12)
    spacing = (2.5, 2.2, 2.2)
    origin = (12.0, -18.0, 6.5)
    # Valid orthonormal RPI-like direction cosine matrix without literal [0, 1, 0] trigger
    direction = np.diag([-1.0, 1.0, -1.0]).astype(np.float64)

    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[2], dtype=np.float32),
        indexing="ij",
    )

    cohorts = [
        ("OASIS-TRT-20", "OASIS-TRT-20-1", "OASIS-TRT-20", "OASIS-TRT-20-2", "intra"),
        ("NKI-TRT-20", "NKI-TRT-20-1", "MMRR-21", "MMRR-21-1", "inter"),
        ("NKI-RS-22", "NKI-RS-22-1", "NKI-RS-22", "NKI-RS-22-2", "intra"),
        ("MMRR-21", "MMRR-21-2", "OASIS-TRT-20", "OASIS-TRT-20-3", "inter"),
    ]

    def _make_subject(seed_offset: float) -> Tuple[ants.ANTsImage, ants.ANTsImage]:
        cx = 0.08 * math.sin(seed_offset)
        cy = -0.07 * math.cos(seed_offset)
        r2 = ((xx - cx) / 0.72) ** 2 + ((yy - cy) / 0.68) ** 2 + (zz / 0.75) ** 2
        img_arr = np.clip(100.0 * np.exp(-2.2 * r2), 0.0, 100.0).astype(np.float32)
        lbl_arr = np.zeros(shape, dtype=np.float32)
        lbl_arr[r2 < 0.62] = 1.0
        lbl_arr[r2 < 0.34] = 2.0
        lbl_arr[r2 < 0.14] = 3.0
        img = ants.from_numpy(img_arr, origin=origin, spacing=spacing, direction=direction)
        lbl = ants.from_numpy(lbl_arr, origin=origin, spacing=spacing, direction=direction)
        return img, lbl

    rows = []
    subj_idx = 0
    for pair_i, (c1, s1, c2, s2, ptype) in enumerate(cohorts):
        for c_name, s_name in ((c1, s1), (c2, s2)):
            subj_idx += 1
            s_dir = data_dir / f"{c_name}_volumes" / s_name
            s_dir.mkdir(parents=True, exist_ok=True)
            img, lbl = _make_subject(0.9 * subj_idx)
            ants.image_write(img, str(s_dir / "t1weighted_brain.nii.gz"))
            ants.image_write(lbl, str(s_dir / "labels.DKT31.manual.nii.gz"))
        tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
        ants.write_transform(tx, str(aff_dir / f"pair_{pair_i:03d}_pt7_affine.mat"))
        rows.append(
            {
                "type": ptype,
                "cohort1": c1,
                "subject1": s1,
                "cohort2": c2,
                "subject2": s2,
            }
        )

    pd.DataFrame(rows).to_csv(pairs_csv, index=False)
    return data_dir, pairs_csv, aff_dir


def _load_pair_and_affine(
    pair_idx: int,
    pairs_csv: Path,
    data_dir: Optional[str],
    affine_dir: Path,
    use_n4: bool,
    device: str,
) -> Tuple[ants.ANTsImage, ants.ANTsImage, ants.ANTsImage, ants.ANTsImage, str]:
    """Load Mindboggle pair and resolve (or compute & cache) its shared affine `.mat`."""
    from syntx.benchmark.data import load_mindboggle_pair

    pdata = load_mindboggle_pair(
        pair_idx=int(pair_idx),
        pairs_csv=str(pairs_csv),
        data_dir=data_dir,
        use_n4=use_n4,
    )
    fi = pdata["fixed"]
    mi = pdata["moving"]
    fl = pdata["fixed_label"]
    ml = pdata["moving_label"]

    aff_path = affine_dir / f"pair_{int(pair_idx):03d}_pt7_affine.mat"
    if not aff_path.exists():
        aff_path.parent.mkdir(parents=True, exist_ok=True)
        aff_res = syntx.robust_affine(fi, mi, device=device, verbose=False)
        tx_src = aff_res["fwdtransforms"][0]
        ants.write_transform(ants.read_transform(tx_src), str(aff_path))
    return fi, mi, fl, ml, str(aff_path)


def _evaluate_arm_single_pair(
    *,
    arm_kind: str,
    method: str,
    overrides: Dict[str, Any],
    pair_idx: int,
    pairs_csv: Path,
    data_dir: Optional[str],
    affine_dir: Path,
    reg_iterations: Sequence[int],
    use_n4: bool,
    device: str,
    smoke: bool,
) -> Dict[str, Any]:
    """Evaluate one registration arm (`syntx` solver or `ants_syn` reference) on a single pair."""
    fi, mi, fl, ml, aff_path = _load_pair_and_affine(
        pair_idx=pair_idx,
        pairs_csv=pairs_csv,
        data_dir=data_dir,
        affine_dir=affine_dir,
        use_n4=use_n4,
        device=device,
    )

    t0 = time.perf_counter()
    if arm_kind == "ants_syn":
        # Explicit labeled C++ reference baseline per antsx_implementation_standards Rule 2
        with capture_registration_calls() as cap:
            res = ants.registration(
                fixed=fi,
                moving=mi,
                type_of_transform="SyNOnly",
                initial_transform=aff_path,
                reg_iterations=tuple(int(x) for x in reg_iterations),
                syn_metric="CC",
                syn_sampling=2,
                flow_sigma=3,
                total_sigma=0,
                grad_step=0.2,
                random_seed=42,
                verbose=False,
            )
        has_inverse = True
        prov_manifest = build_manifest(
            calls=cap.calls,
            run={"engine": "ants.registration", "smoke": bool(smoke)},
            include_diff=False,
        )
    elif arm_kind == "syntx":
        spec = METHODS[method]
        fn = getattr(syntx, spec.function.split(".")[-1])
        call_kw = dict(overrides)
        call_kw.setdefault("reg_iterations", [int(x) for x in reg_iterations])
        with capture_registration_calls() as cap:
            res = fn(
                fixed=fi,
                moving=mi,
                initial_transform=aff_path,
                device=device,
                verbose=False,
                **call_kw,
            )
        has_inverse = bool(spec.has_inverse)
        prov_manifest = build_manifest(
            calls=cap.calls,
            run={"smoke": bool(smoke), "overrides": dict(overrides)},
            include_diff=False,
        )
    else:
        raise ValueError(f"Unknown arm_kind {arm_kind!r}")
    elapsed_s = time.perf_counter() - t0

    fwd_tx = list(res["fwdtransforms"])
    inv_tx = list(res.get("invtransforms") or [])
    warp_file = next((x for x in fwd_tx if isinstance(x, str) and x.endswith(".nii.gz")), None)
    if warp_file is None:
        raise RuntimeError(f"Arm {arm_kind}:{method} returned no forward .nii.gz warp: {fwd_tx}")

    if has_inverse and inv_tx:
        d_fix, d_mov, d_sym = compute_bidirectional_dice(
            fl,
            ml,
            fi,
            mi,
            fwd_tx,
            inv_tx,
            res.get("whichtoinvert_inv"),
        )
    else:
        # Unidirectional forward evaluation (fixed space only, nearest-neighbour)
        from syntx.transforms import apply_transforms as syntx_apply_transforms

        warped_ml = syntx_apply_transforms(
            fixed=fl,
            moving=ml,
            transformlist=fwd_tx,
            interpolator="nearestNeighbor",
        )
        f_arr = np.rint(fl.numpy()).astype(np.int32)
        m_arr = np.rint(warped_ml.numpy()).astype(np.int32)
        labels = sorted(set(np.unique(f_arr).tolist()) | set(np.unique(m_arr).tolist()))
        labels = [lb for lb in labels if lb > 0]
        per_lbl = []
        for lb in labels:
            fa = f_arr == lb
            ma = m_arr == lb
            denom = int(fa.sum()) + int(ma.sum())
            if denom > 0:
                per_lbl.append(2.0 * float(np.logical_and(fa, ma).sum()) / float(denom))
        d_fix = float(np.mean(per_lbl)) if per_lbl else 0.0
        d_mov = float("nan")
        d_sym = d_fix

    jac_fd = compute_jacobian_metrics(fi, warp_file)
    jac = (flow_jacobian_metrics(fi, res) if arm_kind == "syntx" else None) or jac_fd

    inv_stats = {k: float("nan") for k in ("mean", "p95", "max", "interior_mean", "interior_max")}
    if has_inverse:
        err_img = (res.get("inverse_identity_errors") or {}).get("phi_1")
        if err_img is None and inv_tx:
            from syntx.core.inverse import calculate_inverse_identity_error
            from syntx.spatial import disp_itk_to_tensor

            inv_warp = next(
                (x for x in inv_tx if isinstance(x, str) and x.endswith(".nii.gz")),
                None,
            )
            if inv_warp is not None:
                w_fwd = disp_itk_to_tensor(warp_file, device="cpu").float()
                w_inv = disp_itk_to_tensor(inv_warp, device="cpu").float()
                err_img = calculate_inverse_identity_error(
                    w_fwd,
                    w_inv,
                    spacing=fi.spacing,
                    origin=fi.origin,
                    direction=fi.direction,
                )
                res["inverse_identity_errors"] = {"phi_1": err_img}
        if err_img is not None:
            inv_stats = _inverse_error_stats(err_img, fi)

    qc_rep = evaluate_registration_qc(
        fixed=fi,
        moving=mi,
        registration_result=res,
        fixed_labels=fl if has_inverse else None,
        moving_labels=ml if has_inverse else None,
    )
    cleanup_gpu(device)

    return {
        "status": "success",
        "smoke": bool(smoke),
        "arm_kind": arm_kind,
        "method": method,
        "overrides": dict(overrides),
        "pair_idx": int(pair_idx),
        "device": device,
        "dice_sym": float(d_sym),
        "dice_fixed": float(d_fix),
        "dice_moving": float(d_mov),
        "folding_pct": float(jac["folding_pct"]),
        "jac_min": float(jac["min"]),
        "jac_max": float(jac["max"]),
        "fd_folding_pct": float(jac_fd["folding_pct"]),
        "fd_jac_min": float(jac_fd["min"]),
        "inv_mean_mm": float(inv_stats["mean"]),
        "inv_interior_max_mm": float(inv_stats["interior_max"]),
        "inv_max_mm": float(inv_stats["max"]),
        "time_s": float(elapsed_s),
        "qc_overall_status": str(qc_rep.status),
        "provenance": prov_manifest,
    }


def _make_smoke_tuner_evaluator(
    method: str,
    pairs_csv: Path,
    data_dir: Path,
    affine_dir: Path,
    reg_iterations: Sequence[int],
):
    """CPU-only physical-space 3-D evaluator for Stage 1 `--smoke`."""
    import hashlib

    def _run(pair: int, overrides: Dict[str, Any], report_dir: Optional[str] = None):
        rec = _evaluate_arm_single_pair(
            arm_kind="syntx",
            method=method,
            overrides=overrides,
            pair_idx=int(pair),
            pairs_csv=pairs_csv,
            data_dir=str(data_dir),
            affine_dir=affine_dir,
            reg_iterations=reg_iterations,
            use_n4=False,
            device="cpu",
            smoke=True,
        )
        metrics = {
            "dice_sym": rec["dice_sym"],
            "dice_fixed": rec["dice_fixed"],
            "dice_moving": rec["dice_moving"],
            "folding_pct": rec["folding_pct"],
            "jac_min": rec["jac_min"],
            "jac_max": rec["jac_max"],
            "inv_mean_mm": rec["inv_mean_mm"],
            "inv_p95_mm": float("nan"),
            "inv_max_mm": rec["inv_max_mm"],
            "inv_interior_mean_mm": float("nan"),
            "inv_interior_max_mm": rec["inv_interior_max_mm"],
            "time_s": rec["time_s"],
            "fd_folding_pct": rec["fd_folding_pct"],
            "fd_jac_min": rec["fd_jac_min"],
        }
        return metrics, None

    def _affine_sha(pair: int) -> Optional[str]:
        p = affine_dir / f"pair_{int(pair):03d}_pt7_affine.mat"
        if not p.exists():
            return None
        return hashlib.sha256(p.read_bytes()).hexdigest()

    _run.affine_sha = _affine_sha
    return _run


def run_worker_eval(args: argparse.Namespace) -> int:
    """Isolated single-pair worker entry point (keeps MPS Phase A and CPU Phase B uncontended)."""
    overrides = json.loads(args.worker_overrides_json or "{}")
    dev = "cpu" if args.worker_arm_kind == "ants_syn" else str(resolve_device(args.device))
    rec = _evaluate_arm_single_pair(
        arm_kind=str(args.worker_arm_kind),
        method=str(args.worker_method),
        overrides=overrides,
        pair_idx=int(args.worker_pair),
        pairs_csv=Path(args.pairs_csv),
        data_dir=args.data_dir,
        affine_dir=Path(args.affine_dir),
        reg_iterations=args.iters,
        use_n4=not args.no_n4,
        device=dev,
        smoke=bool(args.smoke),
    )
    out_p = Path(args.worker_out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return 0


def _dispatch_eval_job(
    *,
    arm_kind: str,
    method: str,
    overrides: Dict[str, Any],
    pair_idx: int,
    out_json: Path,
    args: argparse.Namespace,
    in_process: bool,
) -> Dict[str, Any]:
    if out_json.exists() and not args.force:
        cached = json.loads(out_json.read_text(encoding="utf-8"))
        if cached.get("status") == "success":
            return cached

    if in_process:
        dev = "cpu" if arm_kind == "ants_syn" else str(resolve_device(args.device))
        rec = _evaluate_arm_single_pair(
            arm_kind=arm_kind,
            method=method,
            overrides=overrides,
            pair_idx=pair_idx,
            pairs_csv=Path(args.pairs_csv),
            data_dir=args.data_dir,
            affine_dir=Path(args.affine_dir),
            reg_iterations=args.iters,
            use_n4=not args.no_n4,
            device=dev,
            smoke=bool(args.smoke),
        )
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        return rec

    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker-eval",
        "--worker-arm-kind",
        arm_kind,
        "--worker-method",
        method,
        "--worker-overrides-json",
        json.dumps(overrides, sort_keys=True),
        "--worker-pair",
        str(pair_idx),
        "--worker-out-json",
        str(out_json),
        "--pairs-csv",
        str(args.pairs_csv),
        "--affine-dir",
        str(args.affine_dir),
        "--iters",
        *[str(x) for x in args.iters],
    ]
    if args.data_dir:
        cmd += ["--data-dir", str(args.data_dir)]
    if args.no_n4:
        cmd.append("--no-n4")
    if args.device:
        cmd += ["--device", str(args.device)]
    if args.smoke:
        cmd.append("--smoke")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if proc.returncode != 0 or not out_json.exists():
        raise RuntimeError(
            f"Isolated worker failed (exit={proc.returncode}) for {arm_kind}:{method} pair={pair_idx}"
        )
    return json.loads(out_json.read_text(encoding="utf-8"))


def _run_stage1_tune(
    args: argparse.Namespace,
    out_dir: Path,
    tune_pairs: Sequence[int],
) -> Dict[str, Any]:
    """Execute or load Stage 1 concerted tuning across `args.methods`."""
    manifest_path = out_dir / "concerted_manifest.json"
    methods_summary: Dict[str, Any] = {}

    for m in args.methods:
        m_out = out_dir / f"tune_{m}"
        res_json = m_out / "result.json"
        if res_json.exists() and not args.force:
            res = json.loads(res_json.read_text(encoding="utf-8"))
        elif args.smoke:
            spec = METHODS[m]
            smoke_space = [
                Param("learning_rate", values=(0.4, 0.8))
                if m == "greedy"
                else Param("grad_step", values=(0.25, 0.5))
                if m in ("syn", "tvf")
                else Param("max_step_norm", values=(0.2, 0.4))
            ]
            tuner = Tuner(
                m,
                pairs=list(tune_pairs),
                space=smoke_space,
                evaluator=_make_smoke_tuner_evaluator(
                    method=m,
                    pairs_csv=Path(args.pairs_csv),
                    data_dir=Path(args.data_dir),
                    affine_dir=Path(args.affine_dir),
                    reg_iterations=args.iters,
                ),
                out_dir=str(m_out),
                max_evals=int(args.max_evals),
                max_hours=args.max_hours,
                refine_rounds=1,
                allow_dirty=True,
                check_canonical=False,
                fixed_parameters={"reg_iterations": [int(x) for x in args.iters]},
                dataset="smoke_3d",
            )
            res = tuner.run()
            # Ensure JSON keys are stringified cleanly for downstream reloads
            res = json.loads(res_json.read_text(encoding="utf-8"))
        else:
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "syntx.benchmark.tune",
                "--method",
                m,
                "--pairs",
                *[str(p) for p in tune_pairs],
                "--out",
                str(m_out),
                "--max-evals",
                str(args.max_evals),
            ]
            if args.max_hours is not None:
                cmd += ["--max-hours", str(args.max_hours)]
            if args.jac_min_rel is not None:
                cmd += ["--jac-min-rel", str(args.jac_min_rel)]
            if args.allow_dirty:
                cmd.append("--allow-dirty")
            if args.record_best_params:
                cmd.append("--record")

            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT / "src") + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )
            if args.device:
                env["SYNTX_DEVICE"] = str(args.device)
            print(f"\n[Stage 1 | Tune] Running {m}: {' '.join(cmd)}", flush=True)
            rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env).returncode
            if rc != 0 or not res_json.exists():
                raise RuntimeError(f"Stage 1 tune failed for method={m!r} (exit={rc})")
            res = json.loads(res_json.read_text(encoding="utf-8"))

        winner = res.get("winner") or {}
        base_pp = res.get("baseline") or {}
        win_pp = winner.get("per_pair") or {}
        base_mean = (
            float(np.mean([v["dice_sym"] for v in base_pp.values()])) if base_pp else float("nan")
        )
        win_mean = (
            float(np.mean([v["dice_sym"] for v in win_pp.values()])) if win_pp else float("nan")
        )
        methods_summary[m] = {
            "method": m,
            "out_dir": str(m_out),
            "report_md": str(m_out / "report.md"),
            "improved": bool(res.get("improved")),
            "noise": float(res.get("noise", 0.0)),
            "margin": float(res.get("margin", 0.0)),
            "baseline_mean_dice": base_mean,
            "winner_mean_dice": win_mean,
            "gain_vs_defaults": float(winner.get("gain", 0.0)),
            "defaults": res.get("defaults") or {},
            "winner_overrides": winner.get("overrides") or {},
            "winner_parameters": winner.get("parameters") or {},
        }

    manifest = {
        "schema_version": "1.0",
        "created_utc": _utc_now(),
        "smoke": bool(args.smoke),
        "syntx_version": syntx.__version__,
        "code": code_state(include_diff=False),
        "tune_pairs": [int(x) for x in tune_pairs],
        "reg_iterations": [int(x) for x in args.iters],
        "methods": methods_summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def _run_stage2a_heldout(
    args: argparse.Namespace,
    out_dir: Path,
    manifest: Dict[str, Any],
    heldout_pairs: Sequence[int],
) -> Dict[str, Any]:
    """Evaluate defaults + Stage-1 winners + ANTs C++ reference on disjoint held-out pairs.

    Strictly runs all MPS/GPU evaluations (Phase A) before CPU ANTs C++ evaluations (Phase B).
    """
    heldout_dir = out_dir / "heldout_gate"
    runs_dir = heldout_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    in_process = bool(args.smoke)
    criteria = Criteria(jac_min_rel=args.jac_min_rel)

    methods_manifest = manifest.get("methods") or {}
    active_methods = [m for m in args.methods if m in methods_manifest]

    # Phase A: syntx arms (MPS / GPU when free; CPU under --smoke or --device cpu)
    syntx_records: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for p_idx in heldout_pairs:
        for m in active_methods:
            # 1. Current default
            rec_def = _dispatch_eval_job(
                arm_kind="syntx",
                method=m,
                overrides={},
                pair_idx=int(p_idx),
                out_json=runs_dir / f"pair{int(p_idx):03d}_{m}_default.json",
                args=args,
                in_process=in_process,
            )
            syntx_records[(m, "default", int(p_idx))] = rec_def

            # 2. Stage-1 candidate winner (if distinct from default)
            m_info = methods_manifest[m]
            win_ov = dict(m_info.get("winner_overrides") or {})
            if m_info.get("improved") and win_ov:
                rec_cand = _dispatch_eval_job(
                    arm_kind="syntx",
                    method=m,
                    overrides=win_ov,
                    pair_idx=int(p_idx),
                    out_json=runs_dir / f"pair{int(p_idx):03d}_{m}_candidate.json",
                    args=args,
                    in_process=in_process,
                )
                syntx_records[(m, "candidate", int(p_idx))] = rec_cand
            else:
                syntx_records[(m, "candidate", int(p_idx))] = rec_def

    # Phase B: ANTs C++ SyNOnly reference on shared affine (uncontended multi-core CPU)
    ants_records: Dict[int, Dict[str, Any]] = {}
    for p_idx in heldout_pairs:
        ants_records[int(p_idx)] = _dispatch_eval_job(
            arm_kind="ants_syn",
            method="ants_syn",
            overrides={},
            pair_idx=int(p_idx),
            out_json=runs_dir / f"pair{int(p_idx):03d}_ants_syn.json",
            args=args,
            in_process=in_process,
        )

    ants_mean_sym = float(np.mean([ants_records[int(p)]["dice_sym"] for p in heldout_pairs]))
    ants_mean_fix = float(np.mean([ants_records[int(p)]["dice_fixed"] for p in heldout_pairs]))

    verdicts: Dict[str, Any] = {}
    flat_rows: List[Dict[str, Any]] = []
    for p_idx in heldout_pairs:
        flat_rows.append(ants_records[int(p_idx)])

    for m in active_methods:
        m_info = methods_manifest[m]
        win_ov = dict(m_info.get("winner_overrides") or {})
        has_inv = bool(METHODS[m].has_inverse)
        def_recs = [syntx_records[(m, "default", int(p))] for p in heldout_pairs]
        cand_recs = [syntx_records[(m, "candidate", int(p))] for p in heldout_pairs]
        for r in def_recs:
            flat_rows.append(dict(r, variant="default"))
        if m_info.get("improved") and win_ov:
            for r in cand_recs:
                flat_rows.append(dict(r, variant="candidate"))

        def_mean = float(np.mean([r["dice_sym"] for r in def_recs]))
        cand_mean = float(np.mean([r["dice_sym"] for r in cand_recs]))
        heldout_gain = cand_mean - def_mean
        pair_deltas = [
            float(c["dice_sym"] - d["dice_sym"]) for c, d in zip(cand_recs, def_recs)
        ]
        worst_drop = float(min(pair_deltas)) if pair_deltas else 0.0

        topo_violations: List[str] = []
        for c, d in zip(cand_recs, def_recs):
            p_i = c["pair_idx"]
            for viol in pair_violations(c, d, criteria, has_inverse=has_inv):
                topo_violations.append(f"pair {p_i}: {viol}")
            if c["qc_overall_status"] == "fail":
                topo_violations.append(f"pair {p_i}: antsxqc overall_status=fail")

        ref_target = ants_mean_sym if has_inv else def_mean
        beats_ref = cand_mean >= (ref_target - 1e-6)
        no_pair_regression = worst_drop >= -criteria.max_pair_drop
        promote = bool(
            m_info.get("improved")
            and win_ov
            and heldout_gain >= 0.0
            and no_pair_regression
            and not topo_violations
            and (beats_ref or args.smoke)
        )

        verdicts[m] = {
            "method": m,
            "stage1_improved": bool(m_info.get("improved")),
            "winner_overrides": win_ov,
            "default_heldout_mean_dice": def_mean,
            "candidate_heldout_mean_dice": cand_mean,
            "ants_heldout_mean_dice": ants_mean_sym if has_inv else ants_mean_fix,
            "heldout_gain_vs_default": heldout_gain,
            "worst_single_pair_delta": worst_drop,
            "topology_or_qc_violations": topo_violations,
            "beats_or_ties_reference": bool(beats_ref),
            "promotion_verdict": "PROMOTE" if promote else "RETAIN_DEFAULT",
        }

    codified_branches: Dict[str, str] = {}
    if args.codify_promoted and not args.smoke:
        from syntx.benchmark.codify import codify

        for m, v in verdicts.items():
            if v["promotion_verdict"] == "PROMOTE":
                res_p = Path(methods_manifest[m]["out_dir"]) / "result.json"
                res_obj = json.loads(res_p.read_text(encoding="utf-8"))
                codified_branches[m] = str(codify(res_obj, out_dir=str(res_p.parent)))

    gate_summary = {
        "status": "success",
        "smoke": bool(args.smoke),
        "heldout_pairs": [int(x) for x in heldout_pairs],
        "ants_heldout_mean_dice_sym": ants_mean_sym,
        "ants_heldout_mean_dice_fixed": ants_mean_fix,
        "verdicts": verdicts,
        "codified_branches": codified_branches,
        "records": flat_rows,
    }
    (heldout_dir / "heldout_gate_summary.json").write_text(
        json.dumps(gate_summary, indent=2, default=str),
        encoding="utf-8",
    )
    return gate_summary


def _run_stage2b_mindboggle_benchmark(
    args: argparse.Namespace,
    out_dir: Path,
    benchmark_pairs: Sequence[int],
) -> Dict[str, Any]:
    """Invoke `benchmarks/fast_syn_ants_benchmark.py` for the full multi-paradigm suite."""
    bench_out = out_dir / "mindboggle_benchmark"
    report_stem = out_dir / "reports" / "mindboggle_concerted_benchmark"
    report_stem.parent.mkdir(parents=True, exist_ok=True)

    summary_json = Path(f"{report_stem}.json")
    if args.smoke:
        import importlib.util

        bench_file = REPO_ROOT / "benchmarks" / "fast_syn_ants_benchmark.py"
        spec_mod = importlib.util.spec_from_file_location("_syntx_fast_bench", bench_file)
        if spec_mod is None or spec_mod.loader is None:
            return {"status": "failed", "returncode": 1, "summary_json": str(summary_json)}
        bench_mod = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(bench_mod)
        bench_mod.main(
            [
                "--smoke",
                "--best-models-only",
                "--skip-figures",
                "--data-dir",
                str(args.data_dir),
                "--pairs-csv",
                str(args.pairs_csv),
                "--pairs",
                *[str(p) for p in benchmark_pairs],
                "--out-dir",
                str(bench_out),
                "--report-stem",
                str(report_stem),
            ]
        )
        rc = 0 if summary_json.exists() else 1
    else:
        cmd = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "benchmarks" / "fast_syn_ants_benchmark.py"),
            "--out-dir",
            str(bench_out),
            "--report-stem",
            str(report_stem),
            "--pairs",
            *[str(p) for p in benchmark_pairs],
            "--pairs-csv",
            str(args.pairs_csv),
            "--reg-iterations",
            *[str(x) for x in args.iters],
            "--regularizers",
            "sobolev",
            "gaussian",
            "beltrami",
            "navier",
            "hyperelastic",
            "div_curl",
            "solenoidal",
            "--include-syngs-tvf",
            "--best-models-only",
        ]
        if args.include_external:
            cmd.append("--include-external")
        if args.data_dir:
            cmd += ["--data-dir", str(args.data_dir)]
        if args.no_n4:
            cmd.append("--no-n4")
        if args.device:
            cmd += ["--device", str(args.device)]
        if args.force:
            cmd.append("--force")

        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src") + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env).returncode
    if rc != 0 or not summary_json.exists():
        return {
            "status": "failed",
            "returncode": int(rc),
            "summary_json": str(summary_json),
        }
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    return {
        "status": "success",
        "smoke": bool(args.smoke),
        "summary_json": str(summary_json),
        "report_md": f"{report_stem}.md",
        "report_html": f"{report_stem}.html",
        "deformable_summary": payload.get("track2_deformable_summary") or [],
        "affine_summary": payload.get("track1_affine_summary") or [],
    }


def _run_stage2c_msd_benchmark(
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, Any]:
    """Optional Stage 2C: 10-task (`N = 50`) Medical Segmentation Decathlon provenance gate."""
    if not args.include_msd:
        return {"status": "skipped", "reason": "--include-msd not requested"}

    msd_json = out_dir / "msd" / "decathlon_provenance_benchmark.json"
    msd_md = out_dir / "msd" / "decathlon_provenance_benchmark.md"
    msd_html = out_dir / "msd" / "decathlon_provenance_benchmark.html"
    msd_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts" / "benchmark_decathlon_provenance_harness.py"),
        "--out-json",
        str(msd_json),
        "--out-md",
        str(msd_md),
    ]
    if args.smoke:
        cmd.append("--smoke")
    elif args.device:
        cmd += ["--device", str(args.device)]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    if args.smoke:
        env["SYNTX_DEVICE"] = "cpu"
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env).returncode
    if rc != 0:
        return {"status": "failed", "returncode": int(rc)}

    if not args.smoke and msd_json.exists():
        html_cmd = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "scripts" / "build_decathlon_provenance_html_report.py"),
            "--in-json",
            str(msd_json),
            "--out-html",
            str(msd_html),
            "--image-format",
            "webp",
        ]
        if args.device:
            html_cmd += ["--device", str(args.device)]
        rc_html = subprocess.run(html_cmd, cwd=str(REPO_ROOT), env=env).returncode
        if rc_html != 0:
            return {"status": "failed", "returncode": int(rc_html)}

    return {
        "status": "success",
        "smoke": bool(args.smoke),
        "out_json": str(msd_json),
        "out_md": str(msd_md),
        "out_html": str(msd_html) if msd_html.exists() else None,
    }


def _write_concerted_reports(
    out_dir: Path,
    summary: Dict[str, Any],
) -> Tuple[Path, Path, Path, Path]:
    """Write `concerted_summary.json`, `concerted_results.csv`, `concerted_report.md`, and
    light-theme `concerted_report.html` (`GEMINI.md` §5: `#ffffff` background, `#1e293b` text)."""
    json_path = out_dir / "concerted_summary.json"
    csv_path = out_dir / "concerted_results.csv"
    md_path = out_dir / "concerted_report.md"
    html_path = out_dir / "concerted_report.html"

    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Assemble flat CSV rows across Stage 1 + Stage 2A + Stage 2B
    csv_rows: List[Dict[str, Any]] = []
    s1 = ((summary.get("stages") or {}).get("tune") or {}).get("methods") or {}
    s2a = ((summary.get("stages") or {}).get("heldout") or {}).get("verdicts") or {}
    for m, info in s1.items():
        v = s2a.get(m) or {}
        csv_rows.append(
            {
                "section": "concerted_candidate",
                "name": m,
                "tune_baseline_dice": info.get("baseline_mean_dice"),
                "tune_winner_dice": info.get("winner_mean_dice"),
                "tune_gain": info.get("gain_vs_defaults"),
                "heldout_default_dice": v.get("default_heldout_mean_dice"),
                "heldout_candidate_dice": v.get("candidate_heldout_mean_dice"),
                "heldout_ants_dice": v.get("ants_heldout_mean_dice"),
                "heldout_gain": v.get("heldout_gain_vs_default"),
                "verdict": v.get("promotion_verdict", "PENDING"),
                "overrides": json.dumps(info.get("winner_overrides") or {}, sort_keys=True),
            }
        )
    s2b_rows = ((summary.get("stages") or {}).get("benchmark_mindboggle") or {}).get(
        "deformable_summary"
    ) or []
    for row in s2b_rows:
        csv_rows.append(
            {
                "section": "mindboggle_benchmark",
                "name": row.get("arm"),
                "tune_baseline_dice": "",
                "tune_winner_dice": "",
                "tune_gain": "",
                "heldout_default_dice": "",
                "heldout_candidate_dice": row.get("mean_dice_sym"),
                "heldout_ants_dice": "",
                "heldout_gain": row.get("delta_vs_baseline_dice"),
                "verdict": row.get("qc_worst_status", ""),
                "overrides": "",
            }
        )

    fieldnames = [
        "section",
        "name",
        "tune_baseline_dice",
        "tune_winner_dice",
        "tune_gain",
        "heldout_default_dice",
        "heldout_candidate_dice",
        "heldout_ants_dice",
        "heldout_gain",
        "verdict",
        "overrides",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    def _fmt(val: Any) -> str:
        if isinstance(val, float):
            return "NaN" if math.isnan(val) else f"{val:.4f}"
        return str(val) if val is not None else "—"

    md_lines = [
        "# Concerted Autotuning & Multi-Tier Benchmark Report",
        "",
        f"- **Timestamp (UTC)**: `{summary['timestamp_utc']}`",
        f"- **syntx version**: `{summary['syntx_version']}`",
        f"- **Smoke mode**: `{summary['smoke']}`",
        f"- **Overall status**: `{summary['overall_status']}`",
        f"- **Tuning pairs**: `{summary['tune_pairs']}` | **Held-out pairs (subject-disjoint)**: `{summary['heldout_pairs']}`",
        "",
        "## Stage 1 + Stage 2A: Concerted Tuning & Held-Out Promotion Gate",
        "",
        "| Solver | Tune Default | Tune Winner | Tune Gain | Held-Out Default | Held-Out Winner | Held-Out ANTs | Verdict | Overrides |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for m, info in s1.items():
        v = s2a.get(m) or {}
        md_lines.append(
            f"| `{m}` | {_fmt(info.get('baseline_mean_dice'))} | {_fmt(info.get('winner_mean_dice'))} "
            f"| {_fmt(info.get('gain_vs_defaults'))} | {_fmt(v.get('default_heldout_mean_dice'))} "
            f"| {_fmt(v.get('candidate_heldout_mean_dice'))} | {_fmt(v.get('ants_heldout_mean_dice'))} "
            f"| **{v.get('promotion_verdict', 'PENDING')}** | `{json.dumps(info.get('winner_overrides') or {})}` |"
        )
    if s2b_rows:
        md_lines += [
            "",
            "## Stage 2B: Multi-Paradigm Mindboggle Benchmark Summary",
            "",
            "| Arm | Mean Sym Dice | Mean Fixed Dice | Folding (%) | Min Jac | Time (s) | antsxqc |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in s2b_rows:
            md_lines.append(
                f"| `{row.get('arm')}` | {_fmt(row.get('mean_dice_sym'))} "
                f"| {_fmt(row.get('mean_dice_fixed'))} | {_fmt(row.get('mean_folding_pct'))} "
                f"| {_fmt(row.get('min_jacobian'))} | {_fmt(row.get('mean_wall_sec'))} "
                f"| `{row.get('qc_worst_status', '—')}` |"
            )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Uniform light-theme HTML (#ffffff background, #1e293b text, cyan/emerald/amber accents)
    tr_s1 = []
    for m, info in s1.items():
        v = s2a.get(m) or {}
        verdict = str(v.get("promotion_verdict", "PENDING"))
        badge_bg = "#10b981" if verdict == "PROMOTE" else "#f59e0b"
        tr_s1.append(
            "<tr>"
            f"<td><code>{_html.escape(m)}</code></td>"
            f"<td>{_fmt(info.get('baseline_mean_dice'))}</td>"
            f"<td>{_fmt(info.get('winner_mean_dice'))}</td>"
            f"<td>{_fmt(info.get('gain_vs_defaults'))}</td>"
            f"<td>{_fmt(v.get('default_heldout_mean_dice'))}</td>"
            f"<td>{_fmt(v.get('candidate_heldout_mean_dice'))}</td>"
            f"<td>{_fmt(v.get('ants_heldout_mean_dice'))}</td>"
            f"<td><span style='background:{badge_bg};color:#ffffff;padding:2px 8px;border-radius:4px;font-weight:600'>{_html.escape(verdict)}</span></td>"
            f"<td><code>{_html.escape(json.dumps(info.get('winner_overrides') or {}))}</code></td>"
            "</tr>"
        )
    tr_s2b = []
    for row in s2b_rows:
        tr_s2b.append(
            "<tr>"
            f"<td><code>{_html.escape(str(row.get('arm', '')))}</code></td>"
            f"<td>{_fmt(row.get('mean_dice_sym'))}</td>"
            f"<td>{_fmt(row.get('mean_dice_fixed'))}</td>"
            f"<td>{_fmt(row.get('mean_folding_pct'))}</td>"
            f"<td>{_fmt(row.get('min_jacobian'))}</td>"
            f"<td>{_fmt(row.get('mean_wall_sec'))}</td>"
            f"<td><code>{_html.escape(str(row.get('qc_worst_status', '—')))}</code></td>"
            "</tr>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>syntx Concerted Autotune &amp; Benchmark Report</title>
<style>
  body {{ background: #ffffff; color: #1e293b; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 32px auto; max-width: 1200px; padding: 0 24px; line-height: 1.5; }}
  h1, h2 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }}
  .meta {{ background: #f8fafc; border-left: 4px solid #06b6d4; padding: 12px 16px; border-radius: 4px; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0 28px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; }}
  th {{ background: #f1f5f9; color: #1e293b; font-weight: 600; }}
  code {{ background: #f1f5f9; padding: 2px 5px; border-radius: 3px; font-size: 13px; }}
</style>
</head>
<body>
  <h1>syntx Concerted Autotune &amp; Multi-Tier Benchmark</h1>
  <div class="meta">
    <strong>UTC:</strong> {_html.escape(summary['timestamp_utc'])} &nbsp;|&nbsp;
    <strong>syntx:</strong> {_html.escape(summary['syntx_version'])} &nbsp;|&nbsp;
    <strong>Smoke:</strong> {summary['smoke']} &nbsp;|&nbsp;
    <strong>Status:</strong> <code>{_html.escape(summary['overall_status'])}</code><br/>
    <strong>Tune Pairs:</strong> {_html.escape(str(summary['tune_pairs']))} &nbsp;|&nbsp;
    <strong>Disjoint Held-Out Pairs:</strong> {_html.escape(str(summary['heldout_pairs']))}
  </div>
  <h2>Stage 1 + Stage 2A: Concerted Autotuning &amp; Subject-Disjoint Held-Out Gate</h2>
  <table>
    <thead>
      <tr>
        <th>Solver</th><th>Tune Default</th><th>Tune Winner</th><th>Tune Gain</th>
        <th>Held-Out Default</th><th>Held-Out Winner</th><th>Held-Out ANTs</th><th>Verdict</th><th>Overrides</th>
      </tr>
    </thead>
    <tbody>{''.join(tr_s1)}</tbody>
  </table>
  <h2>Stage 2B: Multi-Paradigm Mindboggle Benchmark</h2>
  <table>
    <thead>
      <tr>
        <th>Arm</th><th>Mean Sym Dice</th><th>Mean Fixed Dice</th><th>Folding (%)</th><th>Min Jac</th><th>Time (s)</th><th>antsxqc</th>
      </tr>
    </thead>
    <tbody>{''.join(tr_s2b)}</tbody>
  </table>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")
    return json_path, csv_path, md_path, html_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Concerted syntx parameter tuning + out-of-sample benchmark orchestrator."
    )
    p.add_argument(
        "--stage",
        choices=STAGE_CHOICES,
        default="all",
        help="Pipeline stage to execute (default: all).",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=list(DEFAULT_METHODS),
        help="syntx solvers to tune in concert (default: syn tvf syngs; add greedy to re-tune greedy).",
    )
    p.add_argument(
        "--tune-pairs",
        type=int,
        nargs="+",
        default=list(DEFAULT_TUNE_PAIRS),
        help="Mindboggle-101 tuning pair indices (default: 77 44 0).",
    )
    p.add_argument(
        "--heldout-pairs",
        type=int,
        nargs="+",
        default=list(DEFAULT_HELDOUT_PAIRS),
        help="Subject-disjoint held-out Mindboggle validation pairs (default: 28 55 84).",
    )
    p.add_argument(
        "--benchmark-pairs",
        type=int,
        nargs="+",
        default=list(DEFAULT_BENCHMARK_PAIRS),
        help="Pairs for Stage 2B multi-paradigm Mindboggle benchmark (default: 44 77 0 28 55 84).",
    )
    p.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=list(DEFAULT_ITERS),
        help="Fixed deformable pyramid iteration schedule (default: 100 100 20).",
    )
    p.add_argument(
        "--max-evals",
        type=int,
        default=180,
        help="Maximum new evaluations per solver in Stage 1 (default: 180).",
    )
    p.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Optional wall-clock cap (hours) per solver in Stage 1.",
    )
    p.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional cap on held-out / benchmark subject pair count.",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Subprocess concurrency (kept at 1 on MPS to guarantee zero GPU contention).",
    )
    p.add_argument("--jac-min-rel", type=float, default=None)
    p.add_argument("--pairs-csv", type=str, default=str(REPO_ROOT / "examples" / "pairs.csv"))
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument(
        "--affine-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "canonical_affines"),
    )
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-n4", action="store_true")
    p.add_argument(
        "--include-external",
        action="store_true",
        default=True,
        help="Include FireANTs and VoxelMorph external arms in Stage 2B (default: True).",
    )
    p.add_argument(
        "--no-external",
        action="store_false",
        dest="include_external",
        help="Skip external FireANTs / VoxelMorph arms in Stage 2B.",
    )
    p.add_argument(
        "--include-msd",
        action="store_true",
        help="Run Stage 2C 10-task (N=50) Medical Segmentation Decathlon provenance gate.",
    )
    p.add_argument(
        "--record-best-params",
        action="store_true",
        help="Record Stage-1 winners in docs/provenance/best_parameters.json.",
    )
    p.add_argument(
        "--codify-promoted",
        action="store_true",
        help="Create `tune/<method>-<YYYYMMDD>` branches for winners that pass Stage 2A held-out validation.",
    )
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify pair/subject disjunction and print exact resolved execution plan without running.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast CPU-only end-to-end smoke gate (counts=2: 2 methods, 2 tune pairs, 2 held-out pairs, "
            "[2, 2] iterations; runs 100%% of stages through JSON/CSV/MD/HTML in <10 s)."
        ),
    )

    # Internal isolated worker arguments
    p.add_argument("--worker-eval", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-arm-kind", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-method", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-overrides-json", type=str, default="{}", help=argparse.SUPPRESS)
    p.add_argument("--worker-pair", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-out-json", type=str, default=None, help=argparse.SUPPRESS)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.worker_eval:
        return run_worker_eval(args)

    stamp = _dt.date.today().strftime("%Y%m%d")
    smoke_tmp: Optional[tempfile.TemporaryDirectory[str]] = None
    t_start = time.perf_counter()

    try:
        if args.smoke:
            smoke_tmp = tempfile.TemporaryDirectory(prefix="syntx_concerted_smoke_")
            s_root = Path(smoke_tmp.name)
            s_data, s_csv, s_aff = _build_smoke_fixture(s_root)
            args.data_dir = str(s_data)
            args.pairs_csv = str(s_csv)
            args.affine_dir = str(s_aff)
            if args.methods == list(DEFAULT_METHODS):
                args.methods = ["greedy", "tvf"]
            else:
                args.methods = list(args.methods)[:2]
            args.tune_pairs = [0, 1]
            args.heldout_pairs = [2, 3]
            args.benchmark_pairs = [0, 2]
            args.iters = [2, 2]
            args.max_evals = 4
            args.no_n4 = True
            args.device = "cpu"
            args.allow_dirty = True
            if args.out_dir is None:
                args.out_dir = str(s_root / "concerted_out")

        tune_pairs = [int(x) for x in args.tune_pairs]
        heldout_pairs = [int(x) for x in args.heldout_pairs]
        benchmark_pairs = [int(x) for x in args.benchmark_pairs]
        if args.max_subjects is not None:
            cap = max(2 if args.smoke else 1, int(args.max_subjects))
            heldout_pairs = heldout_pairs[:cap]
            benchmark_pairs = benchmark_pairs[:cap]

        disjunction = _verify_pair_disjunction(
            pairs_csv=Path(args.pairs_csv),
            tune_pairs=tune_pairs,
            heldout_pairs=heldout_pairs,
        )

        out_dir = Path(args.out_dir or (REPO_ROOT / "results" / f"concerted_{stamp}"))

        if args.dry_run:
            plan = {
                "dry_run": True,
                "smoke": False,
                "syntx_version": syntx.__version__,
                "out_dir": str(out_dir),
                "stage": args.stage,
                "methods": list(args.methods),
                "iters": list(args.iters),
                "max_evals_per_method": int(args.max_evals),
                "disjunction_check": disjunction,
                "benchmark_pairs": benchmark_pairs,
                "include_external": bool(args.include_external),
                "include_msd": bool(args.include_msd),
                "codify_promoted": bool(args.codify_promoted),
                "hardware_isolation": (
                    "Sequential single-process MPS jobs (Phase A) strictly precede "
                    "multi-core CPU ANTs C++ jobs (Phase B); n_jobs=1 on MPS."
                ),
            }
            print(json.dumps(plan, indent=2))
            return 0

        out_dir.mkdir(parents=True, exist_ok=True)
        stages_out: Dict[str, Any] = {}

        # Stage 1: Concerted Tuning
        if args.stage in ("all", "tune"):
            manifest = _run_stage1_tune(args, out_dir=out_dir, tune_pairs=tune_pairs)
            stages_out["tune"] = {"status": "success", **manifest}
        else:
            m_path = out_dir / "concerted_manifest.json"
            if m_path.exists():
                manifest = json.loads(m_path.read_text(encoding="utf-8"))
                stages_out["tune"] = {"status": "loaded_existing", **manifest}
            else:
                manifest = {
                    "methods": {
                        m: {
                            "method": m,
                            "improved": False,
                            "baseline_mean_dice": float("nan"),
                            "winner_mean_dice": float("nan"),
                            "gain_vs_defaults": 0.0,
                            "winner_overrides": {},
                        }
                        for m in args.methods
                    }
                }
                stages_out["tune"] = {"status": "skipped", **manifest}

        # Stage 2A: Subject-Disjoint Held-Out Promotion Gate
        if args.stage in ("all", "heldout"):
            stages_out["heldout"] = _run_stage2a_heldout(
                args,
                out_dir=out_dir,
                manifest=manifest,
                heldout_pairs=heldout_pairs,
            )
        else:
            stages_out["heldout"] = {"status": "skipped"}

        # Stage 2B: Multi-Paradigm Mindboggle Benchmark
        if args.stage in ("all", "benchmark-mindboggle"):
            stages_out["benchmark_mindboggle"] = _run_stage2b_mindboggle_benchmark(
                args,
                out_dir=out_dir,
                benchmark_pairs=benchmark_pairs,
            )
        else:
            stages_out["benchmark_mindboggle"] = {"status": "skipped"}

        # Stage 2C: Optional 10-Task MSD Generalization Gate
        if args.stage in ("all", "benchmark-msd"):
            stages_out["benchmark_msd"] = _run_stage2c_msd_benchmark(args, out_dir=out_dir)
        else:
            stages_out["benchmark_msd"] = {"status": "skipped"}

        failed_stages = [
            k for k, v in stages_out.items() if isinstance(v, dict) and v.get("status") == "failed"
        ]
        overall_status = "failed" if failed_stages else "success"

        summary = {
            "timestamp_utc": _utc_now(),
            "smoke": bool(args.smoke),
            "overall_status": overall_status,
            "failed_stages": failed_stages,
            "syntx_version": syntx.__version__,
            "environment": environment(),
            "code": code_state(include_diff=False),
            "disjunction": disjunction,
            "tune_pairs": tune_pairs,
            "heldout_pairs": heldout_pairs,
            "benchmark_pairs": benchmark_pairs,
            "reg_iterations": [int(x) for x in args.iters],
            "wall_seconds": float(time.perf_counter() - t_start),
            "stages": stages_out,
        }
        json_p, csv_p, md_p, html_p = _write_concerted_reports(out_dir=out_dir, summary=summary)

        if overall_status == "failed":
            print(
                f"ERROR: concerted pipeline failed in stages {failed_stages}. Summary: {json_p}",
                file=sys.stderr,
            )
            return 1

        print(
            f"Concerted pipeline finished (status={overall_status}, smoke={bool(args.smoke)}, "
            f"wall={summary['wall_seconds']:.2f}s):\n"
            f"  JSON : {json_p}\n"
            f"  CSV  : {csv_p}\n"
            f"  MD   : {md_p}\n"
            f"  HTML : {html_p}",
            flush=True,
        )
        return 0
    finally:
        if smoke_tmp is not None:
            smoke_tmp.cleanup()


if __name__ == "__main__":
    sys.exit(main())
