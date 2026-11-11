#!/usr/bin/env python3
"""
Single-pair (`mbhard` / Mindboggle Pair 44) isolated-subprocess pilot for:
  - Stage 1: Separable 1-D `avg_pool3d` + cached window counts + multi-channel moment fusion
    (`p1_sep_2pass` exact 2-pass centered vs `p1_sep_1pass_5ch` single-pass 5-channel).
  - Stage 2A: In-loop half-field `project_inverse` diffeomorphic warp regularization in `cfl`
    (`p2a_proj_f1` every step, `p2a_proj_f2` every 2nd step).
  - Stage 2B: Warm-started in-loop half-field inversion cadence `(1, 2, 4)` + post-level
    inverse polish (`p2b_cadence_polish`).

Reuses the SHA-256-locked shared `robust_affine` transform from `R31`
(`results/fast_syn_ants_2026-11-11/affines/pair_044_pt7_affine.mat`) so only the deformable
stage differs (`GEMINI.md` §2). Runs strictly one MPS worker subprocess at a time (`GEMINI.md` §4),
evaluates via `compute_bidirectional_dice` and `evaluate_registration_qc` (`antsxqc` single judge),
and writes per-arm JSON artifacts immediately upon completion for live monitoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import ants
import numpy as np
import torch

import syntx
from syntx.benchmark.config import get_model_config, syn_config_to_syn_kwargs
from syntx.benchmark.data import load_mindboggle_pair
from syntx.benchmark.evaluate import _inverse_error_stats, clean_device_cache, normalize_intensity
from syntx.core.inverse import calculate_inverse_identity_error, update_inverse_field_nd_hybrid_lm
from syntx.deformation_metrics import (
    compute_bending_energy,
    compute_bidirectional_dice,
    compute_harmonic_energy,
    compute_jacobian_metrics,
    flow_jacobian_metrics,
)
from syntx.device import resolve_device
from syntx.provenance import (
    assert_manifest_complete,
    build_manifest,
    capture_registration_calls,
)
from syntx.registration_metrics import evaluate_registration_qc
from syntx.spatial import disp_itk_to_tensor, disp_tensor_to_itk
from syntx.transforms import apply_transforms as syntx_apply_transforms

DEFAULT_R31_DIR = Path("results/fast_syn_ants_2026-11-11")
DEFAULT_OUT_DIR = Path("results/pilot_lncc_inv_2026-11-11")
DEFAULT_PAIR_IDX = 44
DEFAULT_SEED = 20261111

PILOT_ARMS: Dict[str, Dict[str, Any]] = {
    "p1_sep_2pass": {
        "label": "Stage 1: Separable 1D + Cached Counts + [2+3] Fused (exact)",
        "syn_kwargs": {},
    },
    "p1_sep_1pass_5ch": {
        "label": "Stage 1: Separable 1D + Single-Pass [5ch] Moments",
        "syn_kwargs": {"lncc_single_pass": True},
    },
    "p2a_proj_f1": {
        "label": "Stage 2A: Separable LNCC + cfl_project_inverse (freq=1)",
        "syn_kwargs": {"cfl_project_inverse": True, "projection_frequency": 1},
    },
    "p2a_proj_f2": {
        "label": "Stage 2A: Separable LNCC + cfl_project_inverse (freq=2)",
        "syn_kwargs": {"cfl_project_inverse": True, "projection_frequency": 2},
    },
    "p2b_cadence_polish": {
        "label": "Stage 2B: Separable LNCC + inv_refresh_every=(1,2,4) + 30-step polish",
        "syn_kwargs": {
            "inv_refresh_every": (1, 2, 4),
            "polish_inv_steps": 30,
            "polish_inv_tol": 0.02,
        },
    },
    "p3a_1win_cadence": {
        "label": "Stage 3A: [2+3] Single-Window LNCC + inv_refresh_every=(1,2,4) + polish",
        "syn_kwargs": {
            "lncc_single_pass": True,
            "inv_refresh_every": (1, 2, 4),
            "polish_inv_steps": 30,
            "polish_inv_tol": 0.02,
        },
    },
    "p3b_1win_cadence_lvlproj": {
        "label": "Stage 3B: [2+3] Single-Window LNCC + (1,2,4) cadence + end-of-level proj (all)",
        "syn_kwargs": {
            "lncc_single_pass": True,
            "inv_refresh_every": (1, 2, 4),
            "polish_inv_steps": 30,
            "polish_inv_tol": 0.02,
            "polish_project_inverse": True,
        },
    },
    "p3c_1win_cadence_coarseproj": {
        "label": "Stage 3C: [2+3] Single-Window LNCC + (1,2,4) cadence + end-of-level proj (4x,2x)",
        "syn_kwargs": {
            "lncc_single_pass": True,
            "inv_refresh_every": (1, 2, 4),
            "polish_inv_steps": 30,
            "polish_inv_tol": 0.02,
            "polish_project_inverse": (True, True, False),
        },
    },
    "p2a_proj_f2_gaussian": {
        "label": "Stage 2A-Gauss: Gaussian (sigma=sqrt(3)) + cfl_project_inverse (freq=2)",
        "syn_kwargs": {
            **syn_config_to_syn_kwargs(get_model_config("gaussian")),
            "cfl_project_inverse": True,
            "projection_frequency": 2,
        },
    },
    "p2a_proj_f2_gaussian_1win": {
        "label": "Stage 2A-Gauss-1Win: Gaussian (sigma=sqrt(3)) + lncc_single_pass + cfl_project_inverse (freq=2)",
        "syn_kwargs": {
            **syn_config_to_syn_kwargs(get_model_config("gaussian")),
            "lncc_single_pass": True,
            "cfl_project_inverse": True,
            "projection_frequency": 2,
        },
    },
}

SMOKE_ARMS: Tuple[str, ...] = ("p1_sep_2pass", "p2a_proj_f2_gaussian_1win")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _whichtoinvert_for(inv_tx: Sequence[str], res: Dict[str, Any]) -> List[bool]:
    if "whichtoinvert_inv" in res and res["whichtoinvert_inv"] is not None:
        return [bool(x) for x in res["whichtoinvert_inv"]]
    return [isinstance(t, str) and t.endswith(".mat") for t in inv_tx]


def _ensure_ice_dict(
    res: Dict[str, Any],
    fwd_tx: Sequence[str],
    inv_tx: Sequence[str],
    fi: ants.ANTsImage,
) -> Dict[str, Any]:
    existing = res.get("inverse_identity_errors")
    if isinstance(existing, dict):
        phi1 = existing.get("phi_1", existing)
        if isinstance(phi1, dict) and phi1.get("error_map") is not None:
            return phi1

    fwd_warp = next((t for t in fwd_tx if isinstance(t, str) and t.endswith(".nii.gz")), None)
    inv_warp = next((t for t in inv_tx if isinstance(t, str) and t.endswith(".nii.gz")), None)
    if fwd_warp is None or inv_warp is None:
        return {
            "max_error": 0.0,
            "mean_error": 0.0,
            "p95": 0.0,
            "error_map": np.zeros(tuple(reversed(fi.shape)), dtype=np.float32),
        }

    w_fwd = disp_itk_to_tensor(fwd_warp, device="cpu").float()
    w_inv = disp_itk_to_tensor(inv_warp, device="cpu").float()
    ice = calculate_inverse_identity_error(
        w_fwd,
        w_inv,
        spacing=fi.spacing,
        origin=fi.origin,
        direction=fi.direction,
    )
    res["inverse_identity_errors"] = {"phi_1": ice}
    return ice


def _build_smoke_pair(out_dir: Path, seed: int) -> Tuple[Dict[str, Any], str, str]:
    """Build a deterministic 3-D physical-space phantom pair + identity `.mat` for `--smoke`."""
    rng = np.random.default_rng(seed)
    shape = (24, 24, 24)
    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[2], dtype=np.float32),
        indexing="ij",
    )
    r_f = np.sqrt((xx - 0.05) ** 2 + (yy + 0.03) ** 2 + zz**2)
    r_m = np.sqrt((xx + 0.06) ** 2 + (yy - 0.04) ** 2 + zz**2)
    arr_f = np.clip(np.exp(-3.0 * r_f**2) + 0.02 * rng.standard_normal(shape, dtype=np.float32), 0.0, 1.0)
    arr_m = np.clip(np.exp(-3.0 * r_m**2) + 0.02 * rng.standard_normal(shape, dtype=np.float32), 0.0, 1.0)

    lab_f = np.zeros(shape, dtype=np.float32)
    lab_f[r_f < 0.65] = 1.0
    lab_f[r_f < 0.35] = 2.0

    lab_m = np.zeros(shape, dtype=np.float32)
    lab_m[r_m < 0.65] = 1.0
    lab_m[r_m < 0.35] = 2.0

    origin = (-12.0, -14.0, 8.0)
    spacing = (1.5, 1.5, 1.5)
    direction = np.eye(3, dtype=np.float64)
    direction[0, 0] = -1.0
    direction[1, 1] = -1.0

    def _mk(a: np.ndarray) -> ants.ANTsImage:
        return ants.from_numpy(a.astype(np.float32), origin=origin, spacing=spacing, direction=direction)

    aff_dir = out_dir / "smoke_affines"
    aff_dir.mkdir(parents=True, exist_ok=True)
    aff_path = str(aff_dir / "smoke_identity.mat")
    tx = ants.create_ants_transform(transform_type="AffineTransform", precision="double", dimension=3)
    ants.write_transform(tx, aff_path)
    sha = _sha256_file(aff_path)
    pair_data = {
        "fixed": _mk(arr_f),
        "moving": _mk(arr_m),
        "fixed_label": _mk(lab_f),
        "moving_label": _mk(lab_m),
        "pair_type": "smoke_phantom",
        "fixed_id": "smoke_fixed",
        "moving_id": "smoke_moving",
    }
    return pair_data, aff_path, sha


def run_worker_single(args: argparse.Namespace) -> None:
    """Execute one isolated pilot arm in its own Python process."""
    clean_device_cache()
    arm = str(args.arm)
    if arm not in PILOT_ARMS:
        raise ValueError(f"Unknown pilot arm: {arm!r}. Expected one of {sorted(PILOT_ARMS)}")

    pair_idx = int(args.pair_idx)
    pairs_seed = int(args.seed) + pair_idx
    torch.manual_seed(pairs_seed)
    np.random.seed(pairs_seed)

    dev = resolve_device(args.device, task="reg")
    out_dir = Path(args.out_dir)

    if args.smoke:
        reg_iterations = [2, 2]
        pair_data, shared_aff_path, shared_aff_sha256 = _build_smoke_pair(out_dir, pairs_seed)
        fi = pair_data["fixed"]
        mi = pair_data["moving"]
        fl = pair_data["fixed_label"]
        ml = pair_data["moving_label"]
    else:
        reg_iterations = [int(x) for x in args.reg_iterations]
        r31_dir = Path(args.r31_dir)
        aff_file = r31_dir / "affines" / f"pair_{pair_idx:03d}_pt7_affine.mat"
        aff_meta_file = r31_dir / "affines" / f"pair_{pair_idx:03d}_pt7_affine_info.json"
        if not aff_file.exists() or not aff_meta_file.exists():
            raise FileNotFoundError(f"Missing cached R31 affine {aff_file} or {aff_meta_file}")
        shared_aff_path = str(aff_file)
        shared_aff_sha256 = _sha256_file(shared_aff_path)
        meta = json.loads(aff_meta_file.read_text())
        if meta.get("sha256") != shared_aff_sha256:
            raise RuntimeError(
                f"Shared affine SHA-256 mismatch: expected {meta.get('sha256')}, got {shared_aff_sha256}"
            )
        pair_data = load_mindboggle_pair(pair_idx=pair_idx, use_n4=True)
        fi = normalize_intensity(pair_data["fixed"])
        mi = normalize_intensity(pair_data["moving"])
        fl = pair_data["fixed_label"]
        ml = pair_data["moving_label"]

    arm_cfg = PILOT_ARMS[arm]
    syn_kw: Dict[str, Any] = dict(arm_cfg["syn_kwargs"])
    if args.smoke and "inv_refresh_every" in syn_kw:
        syn_kw["inv_refresh_every"] = (1, 2)
    if args.smoke and isinstance(syn_kw.get("polish_project_inverse"), (list, tuple)):
        syn_kw["polish_project_inverse"] = tuple(syn_kw["polish_project_inverse"][:2])
    syn_kw["reg_iterations"] = reg_iterations

    load_before = list(os.getloadavg()) if hasattr(os, "getloadavg") else None

    with capture_registration_calls() as cap:
        t0 = time.perf_counter()
        res = syntx.syn(
            fixed=fi,
            moving=mi,
            initial_transform=shared_aff_path,
            backend="pytorch",
            device=dev,
            seed=pairs_seed,
            verbose=False,
            **syn_kw,
        )
        runtime_seconds = time.perf_counter() - t0

    load_after = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    run_meta = {
        "arm": arm,
        "pair_idx": pair_idx,
        "seed": pairs_seed,
        "reg_iterations": reg_iterations,
        "smoke": bool(args.smoke),
        "device_requested": args.device,
        "device_resolved": dev,
        "shared_affine_sha256": shared_aff_sha256,
        "syn_kwargs": syn_kw,
    }
    manifest = build_manifest(calls=cap.calls, run=run_meta, start=cap.start_state)
    assert_manifest_complete(manifest, require_calls=True)

    fwd_tx = [str(x) for x in res["fwdtransforms"]]
    inv_tx = [str(x) for x in res["invtransforms"]]
    which_inv = _whichtoinvert_for(inv_tx, res)

    dice_fixed, dice_moving, dice_sym = compute_bidirectional_dice(
        fl, ml, fi, mi, fwd_tx, inv_tx, which_inv
    )

    warped = res.get("warpedmovout")
    if warped is None:
        warped = syntx_apply_transforms(fixed=fi, moving=mi, transformlist=fwd_tx)
    lncc_sim = float(syntx.image_compare(fi, warped, "lncc"))
    mi_sim = float(syntx.image_compare(fi, warped, "mattes_mi"))

    ice_dict = _ensure_ice_dict(res, fwd_tx, inv_tx, fi)
    inv_stats = _inverse_error_stats(ice_dict, fi)

    fwd_warp_file = next((x for x in fwd_tx if x.endswith(".nii.gz")), None)
    if fwd_warp_file is not None:
        jac_fd = compute_jacobian_metrics(fi, fwd_warp_file)
        harm_e = float(compute_harmonic_energy(fwd_warp_file, spacing=fi.spacing))
        bend_e = float(compute_bending_energy(fwd_warp_file, spacing=fi.spacing))
    else:
        jac_fd = {"folding_pct": 0.0, "min": 1.0, "max": 1.0, "mean": 1.0}
        harm_e = 0.0
        bend_e = 0.0

    flow_jac = flow_jacobian_metrics(fi, res)
    jac_primary = flow_jac if flow_jac is not None else dict(jac_fd, measure="finite_difference")

    qc_rep = evaluate_registration_qc(
        fixed=fi,
        moving=mi,
        registration_result=res,
        fixed_labels=fl,
        moving_labels=ml,
    )

    post_sha = _sha256_file(shared_aff_path)
    if post_sha != shared_aff_sha256:
        raise RuntimeError(f"Shared affine mutated during {arm}!")

    saved_artifact_dir: str | None = None
    if getattr(args, "save_warps", False):
        art_dir = out_dir / "artifacts" / f"pair_{pair_idx:03d}" / arm
        art_dir.mkdir(parents=True, exist_ok=True)
        inv_warp_file = next((x for x in inv_tx if x.endswith(".nii.gz")), None)
        if fwd_warp_file is not None and os.path.exists(fwd_warp_file):
            shutil.copy2(fwd_warp_file, art_dir / "fwd_0.nii.gz")
        if os.path.exists(shared_aff_path):
            shutil.copy2(shared_aff_path, art_dir / "fwd_1.mat")
        if inv_warp_file is not None and os.path.exists(inv_warp_file):
            shutil.copy2(inv_warp_file, art_dir / "inv_1.nii.gz")
        saved_artifact_dir = str(art_dir)

    record: Dict[str, Any] = {
        "status": "SUCCESS",
        "smoke": bool(args.smoke),
        "arm": arm,
        "arm_label": arm_cfg["label"],
        "syn_kwargs": syn_kw,
        "pair_idx": pair_idx,
        "fixed_id": pair_data["fixed_id"],
        "moving_id": pair_data["moving_id"],
        "fixed_shape": list(fi.shape),
        "device": dev,
        "load_avg_start_1m": load_before[0] if load_before else None,
        "load_avg_end_1m": load_after[0] if load_after else None,
        "shared_affine_sha256": shared_aff_sha256,
        "artifact_dir": saved_artifact_dir,
        "runtime_seconds": float(runtime_seconds),
        "dice_fixed": float(dice_fixed),
        "dice_moving": float(dice_moving),
        "dice_sym": float(dice_sym),
        "lncc_similarity": lncc_sim,
        "mattes_mi_similarity": mi_sim,
        "folding_pct": float(jac_primary["folding_pct"]),
        "min_jacobian": float(jac_primary["min"]),
        "max_jacobian": float(jac_primary.get("max", float("nan"))),
        "jacobian_measure": str(jac_primary["measure"]),
        "fd_folding_pct": float(jac_fd["folding_pct"]),
        "fd_min_jacobian": float(jac_fd["min"]),
        "harmonic_energy": harm_e,
        "bending_energy": bend_e,
        "inv_mean_mm": float(inv_stats["mean"]),
        "inv_p95_mm": float(inv_stats["p95"]),
        "inv_max_mm": float(inv_stats["max"]),
        "inv_interior_mean_mm": float(inv_stats["interior_mean"]),
        "inv_interior_max_mm": float(inv_stats["interior_max"]),
        "qc": qc_rep.to_dict(),
        "provenance": manifest,
    }

    out_json = Path(args.worker_out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = out_json.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(record, indent=2, default=str))
    os.replace(tmp_json, out_json)
    clean_device_cache()


def _format_row(r: Dict[str, Any], base_dice: float | None, base_time: float | None) -> str:
    d_sym = float(r["dice_sym"])
    rt = float(r["runtime_seconds"])
    d_delta = f"{d_sym - base_dice:+.4f}" if base_dice is not None else "   ref "
    sp = f"{base_time / rt:.2f}x" if (base_time is not None and rt > 0) else "  1.00x"
    return (
        f"{r['arm']:<22s} | {rt:6.2f}s ({sp:>6s}) | "
        f"Dice={d_sym:.4f} ({d_delta}) | "
        f"fold={r['folding_pct']:.4f}% | min|J|={r['min_jacobian']:.4f} | "
        f"ICE_p95={r['inv_p95_mm']:.4f}mm | H={r['harmonic_energy']:.4f}"
    )


def evaluate_offline_m5_entry(
    *,
    label: str,
    pair_idx: int,
    fi: ants.ANTsImage,
    mi: ants.ANTsImage,
    fl: ants.ANTsImage,
    ml: ants.ANTsImage,
    fwd_warp_path: str,
    aff_path: str,
    inv_warp_path: str,
    tolerances: Sequence[float | None],
    device: str = "cpu",
    steps: int = 25,
) -> List[Dict[str, Any]]:
    """Evaluate post-composition safeguarded LM (`M5`) across tolerances with zero registrations."""
    u_fwd = disp_itk_to_tensor(fwd_warp_path, device=device).float()
    v_inv_orig = disp_itk_to_tensor(inv_warp_path, device=device).float()
    rows: List[Dict[str, Any]] = []
    for tol in tolerances:
        stats: Dict[str, Any] = {}
        if tol is None:
            v_eval = v_inv_orig
            polish_s = 0.0
            inv_eval_path = inv_warp_path
            df, dm, ds = compute_bidirectional_dice(
                fl, ml, fi, mi, [fwd_warp_path, aff_path], [aff_path, inv_eval_path], [True, False]
            )
        else:
            if device == "mps" and torch.backends.mps.is_available():
                torch.mps.synchronize()
            t0 = time.perf_counter()
            v_eval = update_inverse_field_nd_hybrid_lm(
                u_fwd,
                v_inv_orig,
                steps=steps,
                max_error_threshold=float(tol),
                mean_error_threshold=min(float(tol) * 0.5, 0.02),
                spacing=fi.spacing,
                origin=fi.origin,
                direction=fi.direction,
                stationary_boundary=True,
                stats=stats,
            )
            if device == "mps" and torch.backends.mps.is_available():
                torch.mps.synchronize()
            polish_s = time.perf_counter() - t0
            with tempfile.TemporaryDirectory() as td:
                inv_tmp = os.path.join(td, "inv_polished.nii.gz")
                ants.image_write(disp_tensor_to_itk(v_eval.cpu(), fi), inv_tmp)
                df, dm, ds = compute_bidirectional_dice(
                    fl, ml, fi, mi, [fwd_warp_path, aff_path], [aff_path, inv_tmp], [True, False]
                )

        u_cpu = u_fwd.cpu()
        v_cpu = v_eval.cpu()
        phi1 = calculate_inverse_identity_error(
            u_cpu, v_cpu, spacing=fi.spacing, origin=fi.origin, direction=fi.direction
        )
        phi2 = calculate_inverse_identity_error(
            v_cpu, u_cpu, spacing=fi.spacing, origin=fi.origin, direction=fi.direction
        )
        s1 = _inverse_error_stats(phi1, fi)
        s2 = _inverse_error_stats(phi2, fi)
        rows.append(
            {
                "label": label,
                "pair_idx": pair_idx,
                "tol_vox": tol,
                "polish_seconds": float(polish_s),
                "lm_steps_used": int(stats.get("steps", 0)),
                "lm_converged": bool(stats.get("converged", tol is None)),
                "dice_fixed": float(df),
                "dice_moving": float(dm),
                "dice_sym": float(ds),
                "phi1_mean_mm": float(s1["mean"]),
                "phi1_p95_mm": float(s1["p95"]),
                "phi1_max_mm": float(s1["max"]),
                "phi1_interior_max_mm": float(s1["interior_max"]),
                "phi2_mean_mm": float(s2["mean"]),
                "phi2_p95_mm": float(s2["p95"]),
                "phi2_max_mm": float(s2["max"]),
                "phi2_interior_max_mm": float(s2["interior_max"]),
            }
        )
    del u_fwd, v_inv_orig
    clean_device_cache()
    return rows


def run_offline_m5_sweep(args: argparse.Namespace) -> None:
    """Run zero-registration M5 tolerance sweep on saved R31 Pair 44 & 3-subject pilot warps."""
    out_dir = Path(args.out_dir) / ("smoke" if args.smoke else "full")
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(args.device, task="reg")
    tolerances: List[float | None] = [None, 0.50] if args.smoke else [None, 0.50, 0.30, 0.20, 0.05]

    if args.smoke:
        pair_data, aff_path, _ = _build_smoke_pair(out_dir, int(args.seed))
        art = out_dir / "artifacts" / f"pair_{DEFAULT_PAIR_IDX:03d}" / SMOKE_ARMS[0]
        rows = evaluate_offline_m5_entry(
            label=SMOKE_ARMS[0],
            pair_idx=DEFAULT_PAIR_IDX,
            fi=pair_data["fixed"],
            mi=pair_data["moving"],
            fl=pair_data["fixed_label"],
            ml=pair_data["moving_label"],
            fwd_warp_path=str(art / "fwd_0.nii.gz"),
            aff_path=aff_path,
            inv_warp_path=str(art / "inv_1.nii.gz"),
            tolerances=tolerances,
            device=dev,
            steps=5,
        )
        out_file = out_dir / "m5_offline_sweep.json"
        out_file.write_text(json.dumps({"rows": rows}, indent=2))
        return

    r31_dir = Path(args.r31_dir)
    pair_cache: Dict[int, Dict[str, Any]] = {}

    def _get_pair(pidx: int) -> Dict[str, Any]:
        if pidx not in pair_cache:
            pd = load_mindboggle_pair(pair_idx=pidx, use_n4=True)
            pd["fi"] = normalize_intensity(pd["fixed"])
            pd["mi"] = normalize_intensity(pd["moving"])
            pair_cache[pidx] = pd
        return pair_cache[pidx]

    all_rows: List[Dict[str, Any]] = []

    # 1. R31 Pair 44 across all 6 deformable regularizers
    r31_regs = [
        "syn_sobolev",
        "syn_gaussian",
        "syn_hyperelastic",
        "syn_divcurl",
        "syn_navier",
        "syn_solenoidal",
    ]
    for reg_name in r31_regs:
        art_dir = r31_dir / "artifacts" / "pair_044" / f"deformable_{reg_name}_rep0"
        if not (art_dir / "fwd_0.nii.gz").exists():
            continue
        pd = _get_pair(44)
        rows = evaluate_offline_m5_entry(
            label=f"r31_{reg_name}",
            pair_idx=44,
            fi=pd["fi"],
            mi=pd["mi"],
            fl=pd["fixed_label"],
            ml=pd["moving_label"],
            fwd_warp_path=str(art_dir / "fwd_0.nii.gz"),
            aff_path=str(art_dir / "fwd_1.mat"),
            inv_warp_path=str(art_dir / "inv_1.nii.gz"),
            tolerances=tolerances,
            device=dev,
            steps=25,
        )
        all_rows.extend(rows)
        for r in rows:
            t_str = "off" if r["tol_vox"] is None else f"{r['tol_vox']:.2f}"
            print(
                f"[M5 R31 p044] {r['label']:<22s} tol={t_str:<4s} | dt={r['polish_seconds']:.3f}s | "
                f"Dice={r['dice_sym']:.4f} | phi1_max={r['phi1_max_mm']:.4f} (int={r['phi1_interior_max_mm']:.4f}) | "
                f"phi2_p95={r['phi2_p95_mm']:.4f}",
                flush=True,
            )

    # 2. Any persisted warps under out_dir / "artifacts" / pair_XXX / <arm>
    art_root = out_dir / "artifacts"
    if art_root.exists():
        for pair_dir in sorted(art_root.glob("pair_*")):
            pidx = int(pair_dir.name.split("_")[1])
            pd = _get_pair(pidx)
            for arm_dir in sorted(pair_dir.iterdir()):
                if not (arm_dir / "fwd_0.nii.gz").exists():
                    continue
                rows = evaluate_offline_m5_entry(
                    label=arm_dir.name,
                    pair_idx=pidx,
                    fi=pd["fi"],
                    mi=pd["mi"],
                    fl=pd["fixed_label"],
                    ml=pd["moving_label"],
                    fwd_warp_path=str(arm_dir / "fwd_0.nii.gz"),
                    aff_path=str(arm_dir / "fwd_1.mat"),
                    inv_warp_path=str(arm_dir / "inv_1.nii.gz"),
                    tolerances=tolerances,
                    device=dev,
                    steps=25,
                )
                all_rows.extend(rows)
                for r in rows:
                    t_str = "off" if r["tol_vox"] is None else f"{r['tol_vox']:.2f}"
                    print(
                        f"[M5 p{pidx:03d}]   {r['label']:<22s} tol={t_str:<4s} | dt={r['polish_seconds']:.3f}s | "
                        f"Dice={r['dice_sym']:.4f} | phi1_max={r['phi1_max_mm']:.4f} (int={r['phi1_interior_max_mm']:.4f}) | "
                        f"phi2_p95={r['phi2_p95_mm']:.4f}",
                        flush=True,
                    )

    out_file = out_dir / "m5_offline_sweep.json"
    tmp_f = out_file.with_suffix(".json.tmp")
    tmp_f.write_text(json.dumps({"device": dev, "rows": all_rows}, indent=2))
    os.replace(tmp_f, out_file)
    print(f"\nOffline M5 sweep complete ({len(all_rows)} evaluations) -> {out_file}", flush=True)


def run_coordinator(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) / ("smoke" if args.smoke else "full")
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    arms = list(SMOKE_ARMS) if args.smoke else [a for a in args.arms if a in PILOT_ARMS]
    if len(arms) < 1:
        raise ValueError(f"Expected >= 1 pilot arm, got {arms!r}")

    pair_list = [DEFAULT_PAIR_IDX] if args.smoke else (list(args.pairs) if args.pairs else [int(args.pair_idx)])

    for pair_idx in pair_list:
        print(f"\n=== Pair {pair_idx:03d} ({len(arms)} arm(s)) ===", flush=True)
        base_dice: float | None = None
        base_time: float | None = None
        ref_records: List[Dict[str, Any]] = []

        if not args.smoke:
            r31_syn_json = Path(args.r31_dir) / "runs" / f"deformable_pair{pair_idx:03d}_syn_sobolev_rep0.json"
            r31_ants_json = Path(args.r31_dir) / "runs" / f"deformable_pair{pair_idx:03d}_ants_syn_matched_rep0.json"
            if r31_syn_json.exists():
                r31_rec = json.loads(r31_syn_json.read_text())
                r31_rec["arm"] = "ref_r31_sobolev"
                ref_records.append(r31_rec)
                base_dice = float(r31_rec["dice_sym"])
                base_time = float(r31_rec["runtime_seconds"])
                print(_format_row(r31_rec, base_dice, base_time), flush=True)
            if r31_ants_json.exists():
                ants_rec = json.loads(r31_ants_json.read_text())
                ants_rec["arm"] = "ref_ants_syn_matched"
                ref_records.append(ants_rec)
                print(_format_row(ants_rec, base_dice, base_time), flush=True)

        completed: List[Dict[str, Any]] = []
        for idx, arm in enumerate(arms, start=1):
            out_json = runs_dir / f"pair{pair_idx:03d}_{arm}.json"
            art_check = out_dir / "artifacts" / f"pair_{pair_idx:03d}" / arm / "fwd_0.nii.gz"
            need_warps = bool(args.save_warps) and not art_check.exists()
            if args.resume and out_json.exists() and not args.smoke and not need_warps:
                rec = json.loads(out_json.read_text())
                completed.append(rec)
                print(f"[cached {idx}/{len(arms)}] " + _format_row(rec, base_dice, base_time), flush=True)
                continue

            print(f"[start {idx}/{len(arms)}] pair={pair_idx:03d} arm={arm} ({PILOT_ARMS[arm]['label']}) ...", flush=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--arm",
                arm,
                "--pair-idx",
                str(pair_idx),
                "--seed",
                str(args.seed),
                "--device",
                str(args.device),
                "--r31-dir",
                str(args.r31_dir),
                "--out-dir",
                str(out_dir),
                "--worker-out-json",
                str(out_json),
                "--reg-iterations",
                *[str(x) for x in args.reg_iterations],
            ]
            if args.smoke or args.save_warps:
                cmd.append("--save-warps")
            if args.smoke:
                cmd.append("--smoke")

            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                sys.stderr.write(f"\n[ERROR] Worker {arm} (pair {pair_idx}) failed (rc={proc.returncode}):\n{proc.stderr}\n")
                raise SystemExit(proc.returncode)

            rec = json.loads(out_json.read_text())
            if base_dice is None:
                base_dice = float(rec["dice_sym"])
                base_time = float(rec["runtime_seconds"])
            completed.append(rec)
            print(f"[done  {idx}/{len(arms)}] " + _format_row(rec, base_dice, base_time), flush=True)

            summary_payload = {
                "pair_idx": pair_idx,
                "completed_count": len(completed),
                "total_arms": len(arms),
                "reference_runs": [
                    {
                        "arm": r["arm"],
                        "runtime_seconds": r["runtime_seconds"],
                        "dice_sym": r["dice_sym"],
                        "folding_pct": r["folding_pct"],
                        "min_jacobian": r["min_jacobian"],
                        "inv_p95_mm": r["inv_p95_mm"],
                        "harmonic_energy": r["harmonic_energy"],
                    }
                    for r in ref_records
                ],
                "pilot_runs": [
                    {
                        "arm": r["arm"],
                        "arm_label": r["arm_label"],
                        "runtime_seconds": r["runtime_seconds"],
                        "dice_sym": r["dice_sym"],
                        "dice_delta_vs_r31": (float(r["dice_sym"]) - base_dice) if base_dice is not None else 0.0,
                        "speedup_vs_r31": (base_time / float(r["runtime_seconds"]))
                        if (base_time is not None and float(r["runtime_seconds"]) > 0)
                        else 1.0,
                        "folding_pct": r["folding_pct"],
                        "min_jacobian": r["min_jacobian"],
                        "inv_mean_mm": r["inv_mean_mm"],
                        "inv_p95_mm": r["inv_p95_mm"],
                        "inv_max_mm": r["inv_max_mm"],
                        "harmonic_energy": r["harmonic_energy"],
                        "bending_energy": r["bending_energy"],
                        "qc_status": r["qc"].get("status"),
                    }
                    for r in completed
                ],
            }
            summary_file = out_dir / (
                "pilot_summary.json" if pair_idx == DEFAULT_PAIR_IDX else f"pilot_summary_pair{pair_idx:03d}.json"
            )
            tmp_sum = summary_file.with_suffix(".json.tmp")
            tmp_sum.write_text(json.dumps(summary_payload, indent=2))
            os.replace(tmp_sum, summary_file)

        print(f"Pair {pair_idx:03d}: {len(completed)}/{len(arms)} arms complete.", flush=True)

    if args.smoke:
        run_offline_m5_sweep(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pilot separable 3D LNCC + half-field inversion regularization.")
    p.add_argument("--smoke", action="store_true", help="Fast 2-arm, [2, 2]-iter smoke gate on 3D physical phantom.")
    p.add_argument("--worker", action="store_true", help="Internal single-arm subprocess flag.")
    p.add_argument("--offline-m5-sweep", action="store_true", help="Run zero-registration M5 LM polish sweep on saved warps.")
    p.add_argument("--save-warps", action="store_true", help="Persist fwd_0.nii.gz, fwd_1.mat, inv_1.nii.gz per run.")
    p.add_argument("--arm", type=str, default="p1_sep_2pass", help="Single arm key for --worker.")
    p.add_argument(
        "--arms",
        nargs="+",
        default=list(PILOT_ARMS.keys()),
        help="Ordered list of pilot arms to execute.",
    )
    p.add_argument("--pair-idx", type=int, default=DEFAULT_PAIR_IDX)
    p.add_argument("--pairs", nargs="+", type=int, default=None, help="Optional list of Mindboggle pair indices.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--reg-iterations", nargs="+", type=int, default=[100, 100, 20])
    p.add_argument("--r31-dir", type=str, default=str(DEFAULT_R31_DIR))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--worker-out-json", type=str, default="")
    p.add_argument("--resume", action="store_true", help="Reuse already-completed JSONs in out-dir.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker:
        run_worker_single(args)
    elif args.offline_m5_sweep:
        run_offline_m5_sweep(args)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
