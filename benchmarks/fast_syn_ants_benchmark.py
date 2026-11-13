#!/usr/bin/env python3
"""
Fast 3-subject registration benchmark: ``syntx`` vs ``ants.registration`` (Affine & Deformable).

Evaluates three complementary Mindboggle-101 3-D pairs by default (Pair 44 ``mbhard``,
Pair 77 inter-cohort PIR->LAI anisotropic, Pair 0 intra-cohort 1 mm isotropic) at the
canonical deformable schedule ``reg_iterations = [100, 100, 20]`` across four tracks:

1. **Track 1 (Affine Head-to-Head)**:
   ``ants.registration`` (``Affine`` and ``AffineFast``) vs ``syntx.robust_affine``
   (``mode='auto'`` default and ``preset='fast'``). ``syntx_affine_default`` exports and
   SHA-256-locks the canonical shared affine seed ``aff_0`` per pair.
2. **Track 2 (Fair Deformable Regularizer Suite on Shared ``aff_0``)**:
   ``ants.registration`` (``SyNOnly`` at ``(100, 100, 20)`` and ``antsRegistrationSyNQuick[so]``)
   vs ``syntx.syn`` across six regularizer families (``sobolev``, ``gaussian``,
   ``hyperelastic``, ``div_curl``, ``navier``, ``solenoidal``), all initialized from the
   exact same ``aff_0`` so only the deformable stage differs (``GEMINI.md`` §2).
3. **Track 3 (End-to-End Pipeline)**:
   ``ants.registration(..., type_of_transform='SyN', reg_iterations=(100, 100, 20))`` vs
   ``syntx.syn(..., initial_transform=None, reg_iterations=[100, 100, 20])``.
4. **Track 4 (Fresh-Process & Same-Process Reproducibility Gate on ``mbhard``)**:
   Measures repeated-run affine parameter spread ``max |Δparam|``, brain-masked physical LPS
   correspondence spread ``max_x ||y_0(x) - y_1(x)||_2`` (mm), and ``|ΔDice_sym|`` across
   both fresh Python subprocesses and back-to-back same-process invocations (``GEMINI.md`` §6).

Every registration executes in an isolated subprocess (MPS jobs scheduled before CPU-heavy
ANTs C++ jobs so neither contends nor inherits pinned ITK thread state), carries a validated
``syntx.provenance`` manifest, delegates QC grading exclusively to ``antsxqc`` via
``syntx.registration_metrics.evaluate_registration_qc``, and renders a self-contained
light-theme HTML report using ``syntx.viz``.
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import html as _html
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ants
import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

import syntx
from syntx.benchmark.config import get_model_config, syn_config_to_syn_kwargs
from syntx.benchmark.data import check_mindboggle_data, load_mindboggle_pair
from syntx.benchmark.evaluate import _inverse_error_stats, clean_device_cache, normalize_intensity
from syntx.core.inverse import calculate_inverse_identity_error
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
    code_state,
    environment,
)
from syntx.registration_metrics import evaluate_registration_qc
from syntx.spatial import disp_itk_to_tensor, vox_to_physical
from syntx.transforms import (
    apply_transforms as syntx_apply_transforms,
)
from syntx.transforms import (
    create_jacobian_determinant_image,
)
from syntx.viz import render_label_overlay_figure, render_standard_4panel

matplotlib.use("Agg")

DEFAULT_PAIRS: Tuple[int, ...] = (44, 77, 0)
DEFAULT_REG_ITERATIONS: Tuple[int, ...] = (100, 100, 20)
DEFAULT_REGULARIZERS: Tuple[str, ...] = (
    "sobolev",
    "gaussian",
    "hyperelastic",
    "div_curl",
    "navier",
    "solenoidal",
)

AFFINE_ARMS_MPS: Tuple[str, ...] = ("syntx_affine_default", "syntx_affine_fast")
AFFINE_ARMS_CPU: Tuple[str, ...] = ("ants_affine", "ants_affine_fast")

DEFORMABLE_SYN_CONFIG_KEY: Dict[str, Optional[str]] = {
    "sobolev": None,  # canonical syntx.syn() defaults
    "gaussian": "gaussian",
    "hyperelastic": "syn_hyperelastic",
    "div_curl": "syn_divcurl",
    "navier": "syn_navier",
    "solenoidal": "syn_solenoidal",
    "dsti1": None,
    "bspline": None,
    "beltrami": "syn_beltrami",
    "beltrami_proj": "syn_beltrami",
    "poroelastic": None,
}

ARM_LABELS: Dict[str, str] = {
    "syntx_affine_default": "syntx.robust_affine (default / pt7)",
    "syntx_affine_fast": "syntx.robust_affine (preset='fast')",
    "ants_affine": "ants.registration ('Affine')",
    "ants_affine_fast": "ants.registration ('AffineFast')",
    "ants_syn_matched": "ants.registration ('SyNOnly', matched iters)",
    "ants_syn_quick": "ants.registration ('antsRegistrationSyNQuick[so]')",
    "syn_sobolev": "syntx.syn (sobolev — canonical)",
    "syn_gaussian": "syntx.syn (gaussian)",
    "syn_hyperelastic": "syntx.syn (hyperelastic, K=5.0)",
    "syn_divcurl": "syntx.syn (div_curl, β=4.0, γ=0.8)",
    "syn_navier": "syntx.syn (navier, ν=0.35)",
    "syn_solenoidal": "syntx.syn (solenoidal, div v = 0)",
    "syn_dsti1": "syntx.syn (dsti1)",
    "syn_bspline": "syntx.syn (bspline)",
    "syn_beltrami": "syntx.syn (beltrami — tuned grad_step=0.6, dil=0.1)",
    "syn_beltrami_proj": "syntx.syn (beltrami + cfl_project_inverse=True)",
    "syn_poroelastic": "syntx.syn (poroelastic)",
    "syngs_best": "syntx.syngs (sobolev α=0.448 — canonical)",
    "syngs_beltrami": "syntx.syngs (beltrami α=0.27, dil=0.075, step=0.825 — tuned)",
    "tvf_best": "syntx.tvf (beltrami α=2.5 — canonical)",
    "syntx_e2e_syn": "syntx.syn ('SyN' end-to-end: robust_affine + sobolev)",
    "ants_e2e_syn": "ants.registration ('SyN' end-to-end: Affine + SyN)",
}

DKT31_NAMES: Dict[int, str] = {
    1002: "L Caudal Ant Cingulate",
    1003: "L Caudal Mid Frontal",
    1005: "L Cuneus",
    1006: "L Entorhinal",
    1007: "L Fusiform",
    1008: "L Inferior Parietal",
    1009: "L Inferior Temporal",
    1010: "L Isthmus Cingulate",
    1011: "L Lateral Occipital",
    1012: "L Lat Orbitofrontal",
    1013: "L Lingual",
    1014: "L Med Orbitofrontal",
    1015: "L Middle Temporal",
    1016: "L Parahippocampal",
    1017: "L Paracentral",
    1018: "L Pars Opercularis",
    1019: "L Pars Orbitalis",
    1020: "L Pars Triangularis",
    1021: "L Pericalcarine",
    1022: "L Postcentral",
    1023: "L Posterior Cingulate",
    1024: "L Precentral",
    1025: "L Precuneus",
    1026: "L Rostral Ant Cingulate",
    1027: "L Rostral Mid Frontal",
    1028: "L Superior Frontal",
    1029: "L Superior Parietal",
    1030: "L Superior Temporal",
    1031: "L Supramarginal",
    1034: "L Transverse Temporal",
    1035: "L Insula",
    2002: "R Caudal Ant Cingulate",
    2003: "R Caudal Mid Frontal",
    2005: "R Cuneus",
    2006: "R Entorhinal",
    2007: "R Fusiform",
    2008: "R Inferior Parietal",
    2009: "R Inferior Temporal",
    2010: "R Isthmus Cingulate",
    2011: "R Lateral Occipital",
    2012: "R Lat Orbitofrontal",
    2013: "R Lingual",
    2014: "R Med Orbitofrontal",
    2015: "R Middle Temporal",
    2016: "R Parahippocampal",
    2017: "R Paracentral",
    2018: "R Pars Opercularis",
    2019: "R Pars Orbitalis",
    2020: "R Pars Triangularis",
    2021: "R Pericalcarine",
    2022: "R Postcentral",
    2023: "R Posterior Cingulate",
    2024: "R Precentral",
    2025: "R Precuneus",
    2026: "R Rostral Ant Cingulate",
    2027: "R Rostral Mid Frontal",
    2028: "R Superior Frontal",
    2029: "R Superior Parietal",
    2030: "R Superior Temporal",
    2031: "R Supramarginal",
    2034: "R Transverse Temporal",
    2035: "R Insula",
}


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


def _compute_per_region_symmetric_dice(
    fl: ants.ANTsImage,
    ml: ants.ANTsImage,
    fi: ants.ANTsImage,
    mi: ants.ANTsImage,
    fwd_tx: Sequence[str],
    inv_tx: Sequence[str],
    which_inv: Sequence[bool],
) -> Dict[str, float]:
    """Per-label symmetric Sørensen–Dice ('MeanOverlap') over positive labels."""
    fl_c, ml_c = fl.clone(), ml.clone()
    for lab, img in ((fl_c, fi), (ml_c, mi)):
        lab.set_origin(img.origin)
        lab.set_spacing(img.spacing)
        lab.set_direction(img.direction)

    ml_w = syntx_apply_transforms(fixed=fi, moving=ml_c, transformlist=list(fwd_tx), interpolator="nearestNeighbor")
    fl_w = syntx_apply_transforms(
        fixed=mi,
        moving=fl_c,
        transformlist=list(inv_tx),
        whichtoinvert=list(which_inv),
        interpolator="nearestNeighbor",
    )
    for out_lab, img in ((ml_w, fi), (fl_w, mi)):
        out_lab.set_origin(img.origin)
        out_lab.set_spacing(img.spacing)
        out_lab.set_direction(img.direction)

    ov_f = ants.label_overlap_measures(fl_c, ml_w)
    ov_m = ants.label_overlap_measures(ml_c, fl_w)

    def _extract(df: pd.DataFrame) -> Dict[int, float]:
        sub = df[~df["Label"].astype(str).isin(["All", "0", "0.0"])]
        col = "MeanOverlap" if "MeanOverlap" in sub.columns else "Dice"
        out: Dict[int, float] = {}
        for _, row in sub.iterrows():
            try:
                lid = int(float(row["Label"]))
                val = float(row[col])
            except (TypeError, ValueError):
                continue
            if lid > 0 and math.isfinite(val) and 0.0 <= val <= 1.0:
                out[lid] = val
        return out

    map_f = _extract(ov_f)
    map_m = _extract(ov_m)
    common = sorted(set(map_f) & set(map_m))
    return {str(k): 0.5 * (map_f[k] + map_m[k]) for k in common}


def _ensure_ice_dict(
    res: Dict[str, Any],
    fwd_tx: Sequence[str],
    inv_tx: Sequence[str],
    fi: ants.ANTsImage,
) -> Dict[str, Any]:
    """Return an inverse-consistency dict with ``error_map`` for both syntx and ANTs runs."""
    existing = res.get("inverse_identity_errors")
    if isinstance(existing, dict):
        phi1 = existing.get("phi_1", existing)
        if isinstance(phi1, dict) and phi1.get("error_map") is not None:
            return phi1

    fwd_warp = next((t for t in fwd_tx if isinstance(t, str) and t.endswith(".nii.gz")), None)
    inv_warp = next((t for t in inv_tx if isinstance(t, str) and t.endswith(".nii.gz")), None)
    if fwd_warp is None or inv_warp is None:
        # Exact invertible linear matrix (.mat only)
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


def _persist_transforms(
    fwd_tx: Sequence[str],
    inv_tx: Sequence[str],
    dest_dir: Path,
) -> Dict[str, List[str]]:
    """Copy temporary transform files into ``dest_dir`` so reproducibility & figures can read them."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied_map: Dict[str, str] = {}

    def _copy_list(tx_list: Sequence[str], role: str) -> List[str]:
        out: List[str] = []
        for idx, src in enumerate(tx_list):
            src_str = str(src)
            if src_str in copied_map:
                out.append(copied_map[src_str])
                continue
            ext = ".nii.gz" if src_str.endswith(".nii.gz") else Path(src_str).suffix
            dst = str(dest_dir / f"{role}_{idx}{ext}")
            shutil.copyfile(src_str, dst)
            copied_map[src_str] = dst
            out.append(dst)
        return out

    return {
        "fwdtransforms": _copy_list(fwd_tx, "fwd"),
        "invtransforms": _copy_list(inv_tx, "inv"),
    }


def _evaluate_physical_forward_map_on_mask(
    fi: ants.ANTsImage,
    fwd_tx: Sequence[str],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Map fixed-space foreground mask voxels to physical moving LPS coordinates (mm).

    Uses ``syntx.spatial.vox_to_physical`` and exact ITK composition
    ``y = A (x + u(x) - c) + c + t`` (verified bitwise against ``ants.apply_transforms_to_points``).
    Returns ``(y_lps_mm, affine_params_12_or_None)``.
    """
    mask = ants.get_mask(fi).numpy() > 0
    if not np.any(mask):
        mask = np.ones(fi.shape, dtype=bool)
    idx_xyz = np.argwhere(mask)
    pts_lps = vox_to_physical(fi, idx_xyz.astype(np.float64)).astype(np.float64)

    aff_params: Optional[np.ndarray] = None
    for tx_path in fwd_tx:
        if str(tx_path).endswith(".nii.gz"):
            w_img = ants.image_read(str(tx_path))
            w_arr = w_img.numpy()
            if w_arr.shape[:3] != tuple(fi.shape):
                raise ValueError(
                    f"Warp grid shape {w_arr.shape[:3]} does not match fixed shape {fi.shape}"
                )
            u_lps = w_arr[idx_xyz[:, 0], idx_xyz[:, 1], idx_xyz[:, 2]].astype(np.float64)
            pts_lps = pts_lps + u_lps
        elif str(tx_path).endswith(".mat"):
            tx = ants.read_transform(str(tx_path))
            params = np.asarray(tx.parameters, dtype=np.float64)
            fixed_params = np.asarray(tx.fixed_parameters, dtype=np.float64)
            a_mat = params[:9].reshape(3, 3)
            t_vec = params[9:12]
            c_vec = fixed_params[:3]
            pts_lps = ((pts_lps - c_vec) @ a_mat.T) + c_vec + t_vec
            aff_params = params[:12].copy()
        else:
            raise ValueError(f"Unsupported transform file in reproducibility check: {tx_path}")
    return pts_lps, aff_params


def _compare_two_transform_chains(
    fi: ants.ANTsImage,
    fwd_0: Sequence[str],
    fwd_1: Sequence[str],
) -> Dict[str, Optional[float]]:
    y0, p0 = _evaluate_physical_forward_map_on_mask(fi, fwd_0)
    y1, p1 = _evaluate_physical_forward_map_on_mask(fi, fwd_1)
    disp_norm = np.linalg.norm(y0 - y1, axis=1)
    param_spread: Optional[float] = None
    if p0 is not None and p1 is not None:
        param_spread = float(np.max(np.abs(p0 - p1)))
    return {
        "max_param_spread": param_spread,
        "max_brain_disp_spread_mm": float(np.max(disp_norm)),
        "mean_brain_disp_spread_mm": float(np.mean(disp_norm)),
        "p95_brain_disp_spread_mm": float(np.percentile(disp_norm, 95)),
    }


def _execute_single_registration(
    track: str,
    arm: str,
    fi: ants.ANTsImage,
    mi: ants.ANTsImage,
    fl: ants.ANTsImage,
    pairs_seed: int,
    reg_iterations: Sequence[int],
    device: str,
    shared_affine_path: Optional[str] = None,
    ants_aff_iterations: Optional[Sequence[int]] = None,
    syntx_aff_iterations: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Invoke exactly one registration call inside ``capture_registration_calls()``."""
    reg_iters_list = [int(x) for x in reg_iterations]
    reg_iters_tuple = tuple(reg_iters_list)
    aff_kw: Dict[str, Any] = {}
    if ants_aff_iterations is not None:
        aff_ints = tuple(int(x) for x in ants_aff_iterations)
        n_lvl = len(aff_ints)
        aff_kw = {
            "aff_iterations": aff_ints,
            "aff_shrink_factors": tuple(2**i for i in range(n_lvl - 1, -1, -1)),
            "aff_smoothing_sigmas": tuple(i for i in range(n_lvl - 1, -1, -1)),
        }
    syntx_aff_kw: Dict[str, Any] = {}
    if syntx_aff_iterations is not None:
        s_ints = [int(x) for x in syntx_aff_iterations]
        n_s = len(s_ints)
        syntx_aff_kw = {
            "levels": [2**i for i in range(n_s - 1, -1, -1)],
            "iterations": s_ints,
            "multi_start": False,
        }

    if track == "affine":
        if arm == "syntx_affine_default":
            res = syntx.robust_affine(
                fi, mi, mode="auto", device=device, seed=pairs_seed, verbose=False, **syntx_aff_kw
            )
            return res, device
        if arm == "syntx_affine_fast":
            res = syntx.robust_affine(
                fi,
                mi,
                mode="pytorch",
                preset="fast",
                device=device,
                seed=pairs_seed,
                verbose=False,
                **syntx_aff_kw,
            )
            return res, device
        if arm == "ants_affine":
            res = ants.registration(
                fixed=fi, moving=mi, type_of_transform="Affine", verbose=False, **aff_kw
            )
            return res, "cpu/itk"
        if arm == "ants_affine_fast":
            res = ants.registration(
                fixed=fi, moving=mi, type_of_transform="AffineFast", verbose=False
            )
            return res, "cpu/itk"
        raise ValueError(f"Unknown affine arm: {arm!r}")

    if track == "deformable":
        if not shared_affine_path or not os.path.exists(shared_affine_path):
            raise FileNotFoundError(
                f"Shared canonical affine missing for deformable arm {arm!r}: {shared_affine_path}"
            )
        if arm == "ants_syn_matched":
            res = ants.registration(
                fixed=fi,
                moving=mi,
                type_of_transform="SyNOnly",
                initial_transform=shared_affine_path,
                syn_metric="CC",
                syn_sampling=2,
                reg_iterations=reg_iters_tuple,
                flow_sigma=3.0,
                total_sigma=0.0,
                grad_step=0.25,
                verbose=False,
            )
            return res, "cpu/itk"
        if arm == "ants_syn_quick":
            res = ants.registration(
                fixed=fi,
                moving=mi,
                type_of_transform="antsRegistrationSyNQuick[so]",
                initial_transform=shared_affine_path,
                verbose=False,
            )
            return res, "cpu/itk"
        if arm == "syngs_best":
            res = syntx.syngs(
                fixed=fi,
                moving=mi,
                initial_transform=shared_affine_path,
                backend="pytorch",
                device=device,
                compose_polish_tol=0.30,
                reg_iterations=reg_iters_list,
                seed=pairs_seed,
                verbose=False,
            )
            return res, device
        if arm == "syngs_beltrami":
            sb_cfg = get_model_config("syngs_beltrami")
            res = syntx.syngs(
                fixed=fi,
                moving=mi,
                initial_transform=shared_affine_path,
                regularizer=str(sb_cfg["regularizer"]),
                alpha=float(sb_cfg["alpha"]),
                dilatation_weight=float(sb_cfg["dilatation_weight"]),
                max_step_norm=float(sb_cfg["max_step_norm"]),
                optimizer=str(sb_cfg["optimizer"]),
                optimizer_lr=float(sb_cfg["optimizer_lr"]),
                n_steps=int(sb_cfg["n_steps"]),
                bootstrap_mode=str(sb_cfg["bootstrap_mode"]),
                backend="pytorch",
                device=device,
                compose_polish_tol=0.30,
                reg_iterations=reg_iters_list,
                seed=pairs_seed,
                verbose=False,
            )
            return res, device
        if arm == "tvf_best":
            res = syntx.tvf(
                fixed=fi,
                moving=mi,
                initial_transform=shared_affine_path,
                backend="pytorch",
                device=device,
                compose_polish_tol=0.30,
                reg_iterations=reg_iters_list,
                affine_seed=pairs_seed,
                verbose=False,
            )
            return res, device
        if arm.startswith("syn_"):
            reg_key = arm[len("syn_") :]
            if reg_key == "divcurl":
                reg_key = "div_curl"
            if reg_key not in DEFORMABLE_SYN_CONFIG_KEY:
                raise ValueError(f"Unsupported syntx.syn regularizer arm: {arm!r}")
            cfg_block = DEFORMABLE_SYN_CONFIG_KEY[reg_key]
            if reg_key == "sobolev":
                syn_kw: Dict[str, Any] = {}
                if reg_iters_list != list(DEFAULT_REG_ITERATIONS):
                    syn_kw["reg_iterations"] = reg_iters_list
            elif cfg_block is not None:
                syn_kw = syn_config_to_syn_kwargs(get_model_config(cfg_block))
                syn_kw["reg_iterations"] = reg_iters_list
                if reg_key == "beltrami_proj":
                    syn_kw["cfl_project_inverse"] = True
            else:
                syn_kw = {"regularizer": reg_key, "reg_iterations": reg_iters_list}
                if reg_key == "poroelastic":
                    syn_kw["fixed_mask"] = ants.threshold_image(fl, 1, 10000)

            res = syntx.syn(
                fixed=fi,
                moving=mi,
                initial_transform=shared_affine_path,
                backend="pytorch",
                device=device,
                seed=pairs_seed,
                verbose=False,
                **syn_kw,
            )
            return res, device
        raise ValueError(f"Unknown deformable arm: {arm!r}")

    if track == "e2e":
        if arm == "syntx_e2e_syn":
            syn_kw = {}
            if reg_iters_list != list(DEFAULT_REG_ITERATIONS):
                syn_kw["reg_iterations"] = reg_iters_list
            res = syntx.syn(
                fixed=fi,
                moving=mi,
                initial_transform=None,
                backend="pytorch",
                device=device,
                affine_seed=pairs_seed,
                seed=pairs_seed,
                verbose=False,
                **syn_kw,
            )
            return res, device
        if arm == "ants_e2e_syn":
            res = ants.registration(
                fixed=fi,
                moving=mi,
                type_of_transform="SyN",
                syn_metric="CC",
                syn_sampling=2,
                reg_iterations=reg_iters_tuple,
                flow_sigma=3.0,
                total_sigma=0.0,
                grad_step=0.25,
                verbose=False,
                **aff_kw,
            )
            return res, "cpu/itk"
        raise ValueError(f"Unknown e2e arm: {arm!r}")

    raise ValueError(f"Unknown track: {track!r}")


def run_worker_single(args: argparse.Namespace) -> None:
    """Isolated single-registration worker process."""
    clean_device_cache()
    pair_idx = int(args.pair_idx)
    rep = int(args.rep)
    track = str(args.track)
    arm = str(args.arm)
    reg_iterations = [int(x) for x in args.reg_iterations]
    pairs_seed = int(args.seed) + pair_idx

    torch.manual_seed(pairs_seed)
    np.random.seed(pairs_seed)

    dev = resolve_device(args.device, task="reg")
    use_n4 = not args.no_n4

    pair_data = load_mindboggle_pair(
        pair_idx=pair_idx,
        pairs_csv=args.pairs_csv,
        data_dir=args.data_dir,
        use_n4=use_n4,
    )
    fi = normalize_intensity(pair_data["fixed"])
    mi = normalize_intensity(pair_data["moving"])
    fl = pair_data["fixed_label"]
    ml = pair_data["moving_label"]

    out_dir = Path(args.out_dir)
    shared_aff_path: Optional[str] = None
    shared_aff_sha256: Optional[str] = None
    if track == "deformable":
        aff_file = out_dir / "affines" / f"pair_{pair_idx:03d}_pt7_affine.mat"
        aff_meta_file = out_dir / "affines" / f"pair_{pair_idx:03d}_pt7_affine_info.json"
        if not aff_file.exists() or not aff_meta_file.exists():
            raise FileNotFoundError(
                f"Expected shared affine {aff_file} (+ info JSON) before Track 2 deformable run."
            )
        shared_aff_path = str(aff_file)
        shared_aff_sha256 = _sha256_file(shared_aff_path)
        meta = json.loads(aff_meta_file.read_text())
        if meta.get("sha256") != shared_aff_sha256:
            raise RuntimeError(
                f"Shared affine SHA-256 mismatch on pair {pair_idx}: "
                f"expected {meta.get('sha256')}, got {shared_aff_sha256}"
            )

    load_before = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    itk_threads_env = os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "default_all_cores")

    with capture_registration_calls() as cap:
        t0 = time.perf_counter()
        res, exec_device = _execute_single_registration(
            track=track,
            arm=arm,
            fi=fi,
            mi=mi,
            fl=fl,
            pairs_seed=pairs_seed,
            reg_iterations=reg_iterations,
            device=dev,
            shared_affine_path=shared_aff_path,
            ants_aff_iterations=args.ants_aff_iterations,
            syntx_aff_iterations=args.syntx_aff_iterations,
        )
        runtime_seconds = time.perf_counter() - t0

    load_after = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    run_meta = {
        "track": track,
        "arm": arm,
        "pair_idx": pair_idx,
        "rep": rep,
        "seed": pairs_seed,
        "reg_iterations": reg_iterations,
        "use_n4": use_n4,
        "smoke": bool(getattr(args, "smoke", False)),
        "device_requested": args.device,
        "device_resolved": exec_device,
        "shared_affine_sha256": shared_aff_sha256,
    }
    manifest = build_manifest(calls=cap.calls, run=run_meta, start=cap.start_state)
    assert_manifest_complete(manifest, require_calls=True)

    fwd_tx = [str(x) for x in res["fwdtransforms"]]
    inv_tx = [str(x) for x in res["invtransforms"]]
    which_inv = _whichtoinvert_for(inv_tx, res)

    # Save canonical shared affine for Track 2 if this is Track 1 syntx_affine_default (rep 0)
    if track == "affine" and arm == "syntx_affine_default" and rep == 0:
        aff_dir = out_dir / "affines"
        aff_dir.mkdir(parents=True, exist_ok=True)
        dst_aff = aff_dir / f"pair_{pair_idx:03d}_pt7_affine.mat"
        shutil.copyfile(fwd_tx[0], str(dst_aff))
        sha = _sha256_file(str(dst_aff))
        shared_aff_sha256 = sha
        (aff_dir / f"pair_{pair_idx:03d}_pt7_affine_info.json").write_text(
            json.dumps(
                {
                    "pair_idx": pair_idx,
                    "fixed_id": pair_data["fixed_id"],
                    "moving_id": pair_data["moving_id"],
                    "sha256": sha,
                    "runtime_seconds": runtime_seconds,
                },
                indent=2,
            )
        )

    # Persist transforms when requested (e.g. Pair 44 mbhard for visual figures & exact spread)
    persisted_tx: Optional[Dict[str, List[str]]] = None
    if args.persist_transforms:
        art_dir = out_dir / "artifacts" / f"pair_{pair_idx:03d}" / f"{track}_{arm}_rep{rep}"
        persisted_tx = _persist_transforms(fwd_tx, inv_tx, art_dir)

    dice_fixed, dice_moving, dice_sym = compute_bidirectional_dice(
        fl, ml, fi, mi, fwd_tx, inv_tx, which_inv
    )
    regional_dice = (
        _compute_per_region_symmetric_dice(fl, ml, fi, mi, fwd_tx, inv_tx, which_inv)
        if args.save_regional_dice
        else {}
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
    if flow_jac is not None:
        jac_primary = flow_jac
    else:
        jac_primary = dict(jac_fd, measure="finite_difference")

    qc_rep = evaluate_registration_qc(
        fixed=fi,
        moving=mi,
        registration_result=res,
        fixed_labels=fl,
        moving_labels=ml,
    )

    # Verify shared affine was not mutated during deformable execution
    if track == "deformable" and shared_aff_path is not None:
        post_sha = _sha256_file(shared_aff_path)
        if post_sha != shared_aff_sha256:
            raise RuntimeError(f"Shared affine mutated during {arm} on pair {pair_idx}!")

    record: Dict[str, Any] = {
        "status": "SUCCESS",
        "smoke": bool(getattr(args, "smoke", False)),
        "track": track,
        "arm": arm,
        "arm_label": ARM_LABELS.get(arm, arm),
        "pair_idx": pair_idx,
        "rep": rep,
        "pair_type": pair_data["pair_type"],
        "fixed_id": pair_data["fixed_id"],
        "moving_id": pair_data["moving_id"],
        "fixed_shape": list(fi.shape),
        "fixed_spacing": [float(s) for s in fi.spacing],
        "fixed_orientation": str(fi.orientation),
        "moving_shape": list(mi.shape),
        "moving_spacing": [float(s) for s in mi.spacing],
        "moving_orientation": str(mi.orientation),
        "use_n4": use_n4,
        "reg_iterations": reg_iterations,
        "device": exec_device,
        "itk_threads_env": itk_threads_env,
        "load_avg_start_1m": load_before[0] if load_before else None,
        "load_avg_end_1m": load_after[0] if load_after else None,
        "shared_affine_sha256": shared_aff_sha256,
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
        "fd_max_jacobian": float(jac_fd["max"]),
        "harmonic_energy": harm_e,
        "bending_energy": bend_e,
        "inv_mean_mm": float(inv_stats["mean"]),
        "inv_p95_mm": float(inv_stats["p95"]),
        "inv_max_mm": float(inv_stats["max"]),
        "inv_interior_mean_mm": float(inv_stats["interior_mean"]),
        "inv_interior_max_mm": float(inv_stats["interior_max"]),
        "qc": qc_rep.to_dict(),
        "persisted_transforms": persisted_tx,
        "whichtoinvert_inv": which_inv,
        "regional_dice_sym": regional_dice,
        "provenance": manifest,
    }

    out_json = Path(args.worker_out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = out_json.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(record, indent=2, default=str))
    os.replace(tmp_json, out_json)

    # Remove ephemeral /tmp warp files once persisted_transforms & metrics are safely recorded
    tmp_root = os.path.realpath(tempfile.gettempdir())
    out_root = os.path.realpath(str(out_dir))
    protected = {os.path.realpath(shared_aff_path)} if shared_aff_path and os.path.exists(shared_aff_path) else set()
    if persisted_tx:
        for plist in persisted_tx.values():
            for p_str in plist:
                if os.path.exists(p_str):
                    protected.add(os.path.realpath(p_str))
    ephemeral_candidates: List[str] = []
    for tx_key in ("fwdtransforms", "invtransforms"):
        for tx_p in res.get(tx_key, []) or []:
            if isinstance(tx_p, str):
                ephemeral_candidates.append(tx_p)
    for mp_key in ("fwd_midpoint", "inv_midpoint"):
        mp_p = res.get(mp_key)
        if isinstance(mp_p, str):
            ephemeral_candidates.append(mp_p)
    for cand in ephemeral_candidates:
        if os.path.exists(cand):
            rp = os.path.realpath(cand)
            if rp.startswith(tmp_root + os.sep) and not rp.startswith(out_root + os.sep) and rp not in protected:
                try:
                    os.remove(cand)
                except OSError:
                    pass

    clean_device_cache()


def run_worker_same_process_repro(args: argparse.Namespace) -> None:
    """Back-to-back two-run reproducibility check inside a single Python process."""
    clean_device_cache()
    pair_idx = int(args.pair_idx)
    track = str(args.track)
    arm = str(args.arm)
    reg_iterations = [int(x) for x in args.reg_iterations]
    pairs_seed = int(args.seed) + pair_idx
    dev = resolve_device(args.device, task="reg")
    use_n4 = not args.no_n4

    pair_data = load_mindboggle_pair(
        pair_idx=pair_idx,
        pairs_csv=args.pairs_csv,
        data_dir=args.data_dir,
        use_n4=use_n4,
    )
    fi = normalize_intensity(pair_data["fixed"])
    mi = normalize_intensity(pair_data["moving"])
    fl = pair_data["fixed_label"]
    ml = pair_data["moving_label"]

    out_dir = Path(args.out_dir)
    shared_aff_path: Optional[str] = None
    if track == "deformable":
        shared_aff_path = str(out_dir / "affines" / f"pair_{pair_idx:03d}_pt7_affine.mat")

    runs = []
    tx_dirs = []
    for sub_rep in (0, 1):
        torch.manual_seed(pairs_seed)
        np.random.seed(pairs_seed)
        with capture_registration_calls() as cap:
            t0 = time.perf_counter()
            res, exec_dev = _execute_single_registration(
                track=track,
                arm=arm,
                fi=fi,
                mi=mi,
                fl=fl,
                pairs_seed=pairs_seed,
                reg_iterations=reg_iterations,
                device=dev,
                shared_affine_path=shared_aff_path,
                ants_aff_iterations=args.ants_aff_iterations,
                syntx_aff_iterations=args.syntx_aff_iterations,
            )
            dt = time.perf_counter() - t0
        manifest = build_manifest(
            calls=cap.calls,
            run={"mode": "same_process_repro", "track": track, "arm": arm, "sub_rep": sub_rep},
            start=cap.start_state,
        )
        assert_manifest_complete(manifest, require_calls=True)

        fwd_tx = [str(x) for x in res["fwdtransforms"]]
        inv_tx = [str(x) for x in res["invtransforms"]]
        which_inv = _whichtoinvert_for(inv_tx, res)
        _, _, d_sym = compute_bidirectional_dice(fl, ml, fi, mi, fwd_tx, inv_tx, which_inv)
        art_dir = out_dir / "artifacts" / f"pair_{pair_idx:03d}" / f"sameproc_{track}_{arm}_{sub_rep}"
        persisted = _persist_transforms(fwd_tx, inv_tx, art_dir)
        tx_dirs.append(persisted["fwdtransforms"])
        runs.append(
            {
                "sub_rep": sub_rep,
                "dice_sym": float(d_sym),
                "runtime_seconds": float(dt),
                "device": exec_dev,
                "provenance": manifest,
            }
        )
        clean_device_cache()

    spread = _compare_two_transform_chains(fi, tx_dirs[0], tx_dirs[1])
    payload = {
        "status": "SUCCESS",
        "scope": "same_process",
        "track": track,
        "arm": arm,
        "arm_label": ARM_LABELS.get(arm, arm),
        "pair_idx": pair_idx,
        "dice_sym_rep0": runs[0]["dice_sym"],
        "dice_sym_rep1": runs[1]["dice_sym"],
        "abs_dice_sym_diff": abs(runs[0]["dice_sym"] - runs[1]["dice_sym"]),
        "runtime_rep0_s": runs[0]["runtime_seconds"],
        "runtime_rep1_s": runs[1]["runtime_seconds"],
        **spread,
        "runs": runs,
    }
    out_json = Path(args.worker_out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=str))


def _build_smoke_fixture(root: Path) -> Tuple[Path, Path]:
    """Build a 3-subject / 2-pair (counts=2) 32^3 Mindboggle fixture for ``--smoke``."""
    data_dir = root / "volumes"
    cohort_dir = data_dir / "OASIS-TRT-20_volumes"
    grid = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")

    for idx, shift in ((1, -0.08), (2, 0.08), (3, 0.04)):
        subj_dir = cohort_dir / f"OASIS-TRT-20-{idx}"
        subj_dir.mkdir(parents=True, exist_ok=True)
        r2 = (xx - shift) ** 2 + (yy + 0.5 * shift) ** 2 + zz**2
        brain = np.clip(np.exp(-3.0 * r2) * (1.0 + 0.25 * np.cos(5.0 * xx)), 0.0, None)
        brain[r2 > 0.68] = 0.0

        labels = np.zeros_like(brain, dtype=np.uint32)
        labels[(r2 <= 0.55) & (xx - shift < 0.0)] = 1002
        labels[(r2 <= 0.55) & (xx - shift >= 0.0)] = 2002

        img = ants.from_numpy(
            brain.astype(np.float32),
            origin=(-24.0, -24.0, -24.0),
            spacing=(2.0, 2.0, 2.0),
            direction=np.eye(3),
        )
        lbl = ants.from_numpy(
            labels,
            origin=(-24.0, -24.0, -24.0),
            spacing=(2.0, 2.0, 2.0),
            direction=np.eye(3),
        )
        ants.image_write(img, str(subj_dir / "t1weighted_brain.nii.gz"))
        ants.image_write(lbl, str(subj_dir / "labels.DKT31.manual.nii.gz"))

    pairs_csv = root / "pairs.csv"
    pairs_csv.write_text(
        "type,cohort1,subject1,cohort2,subject2\n"
        "inter,OASIS-TRT-20,OASIS-TRT-20-1,OASIS-TRT-20,OASIS-TRT-20-2\n"
        "intra,OASIS-TRT-20,OASIS-TRT-20-1,OASIS-TRT-20,OASIS-TRT-20-3\n",
        encoding="utf-8",
    )
    return data_dir, pairs_csv


def _run_subprocess_job(
    script_path: str,
    cli_args: List[str],
    out_json: Path,
    force: bool,
    label: str,
    in_process: bool = False,
) -> Dict[str, Any]:
    if not force and out_json.exists():
        cached = json.loads(out_json.read_text())
        if cached.get("status") == "SUCCESS":
            d_str = (
                f"Dice={cached['dice_sym']:.4f}, t={cached['runtime_seconds']:.1f}s"
                if "dice_sym" in cached
                else f"|Δu|_max={cached.get('max_brain_disp_spread_mm', float('nan')):.2e} mm"
            )
            print(f"  [RESUME] {label} -> {d_str}", flush=True)
            return cached

    t0 = time.perf_counter()
    print(f"  [START ] {label} ...", flush=True)
    if in_process:
        w_args = build_parser().parse_args(cli_args + ["--worker-out-json", str(out_json)])
        if w_args.worker_mode == "single":
            run_worker_single(w_args)
        elif w_args.worker_mode == "same_proc_repro":
            run_worker_same_process_repro(w_args)
        else:
            raise ValueError(f"Unsupported worker_mode={w_args.worker_mode!r}")
    else:
        cmd = [sys.executable, "-u", script_path] + cli_args + ["--worker-out-json", str(out_json)]
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # Ensure ITK thread state is never inherited from any parent process
        env.pop("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", None)

        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Worker failed ({label}, exit {proc.returncode}):\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
    dt = time.perf_counter() - t0
    if not out_json.exists():
        raise FileNotFoundError(f"Worker exited 0 but did not write {out_json}")
    rec = json.loads(out_json.read_text())
    if "dice_sym" in rec:
        print(
            f"  [DONE  ] {label} | Dice={rec['dice_sym']:.4f} | Fold={rec['folding_pct']:.4f}% "
            f"| ICE_p95={rec['inv_p95_mm']:.4f}mm | QC={rec['qc']['status']} "
            f"| Solver={rec['runtime_seconds']:.1f}s (wall {dt:.1f}s)",
            flush=True,
        )
    else:
        print(
            f"  [DONE  ] {label} | |ΔDice|={rec['abs_dice_sym_diff']:.2e} "
            f"| max|Δu|={rec['max_brain_disp_spread_mm']:.2e} mm (wall {dt:.1f}s)",
            flush=True,
        )
    return rec


def _compute_fresh_process_repro_row(
    fi: ants.ANTsImage,
    rec0: Dict[str, Any],
    rec1: Dict[str, Any],
) -> Dict[str, Any]:
    tx0 = (rec0.get("persisted_transforms") or {}).get("fwdtransforms")
    tx1 = (rec1.get("persisted_transforms") or {}).get("fwdtransforms")
    if not tx0 or not tx1:
        raise RuntimeError(
            f"Missing persisted transforms for fresh-process reproducibility on {rec0.get('arm')}"
        )
    spread = _compare_two_transform_chains(fi, tx0, tx1)
    return {
        "scope": "fresh_process",
        "track": rec0["track"],
        "arm": rec0["arm"],
        "arm_label": rec0["arm_label"],
        "pair_idx": rec0["pair_idx"],
        "dice_sym_rep0": rec0["dice_sym"],
        "dice_sym_rep1": rec1["dice_sym"],
        "abs_dice_sym_diff": abs(rec0["dice_sym"] - rec1["dice_sym"]),
        "runtime_rep0_s": rec0["runtime_seconds"],
        "runtime_rep1_s": rec1["runtime_seconds"],
        **spread,
    }


def _render_report_figures(
    repro_pair: int,
    pairs_csv: str,
    data_dir: Optional[str],
    use_n4: bool,
    deformable_records: List[Dict[str, Any]],
    e2e_records: List[Dict[str, Any]],
    fig_dir: Path,
    force: bool = False,
) -> Dict[str, str]:
    """Generate light-theme anatomical QC panels (via syntx.viz) and quantitative charts."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    viz_pair = repro_pair
    has_repro = any(
        r["pair_idx"] == repro_pair and r["rep"] == 0 and r.get("persisted_transforms")
        for r in deformable_records
    )
    if not has_repro:
        persisted_pairs = [
            int(r["pair_idx"])
            for r in deformable_records
            if r["rep"] == 0 and r.get("persisted_transforms")
        ]
        if persisted_pairs:
            viz_pair = persisted_pairs[0]

    by_arm = {
        r["arm"]: r
        for r in deformable_records
        if r["pair_idx"] == viz_pair and r["rep"] == 0
    }
    figs: Dict[str, str] = {}

    panel_targets = (
        ("ants_syn_matched", "4panel_ants"),
        ("tvf_best", "4panel_tvf"),
        ("syngs_beltrami", "4panel_syngs_beltrami"),
        ("syn_beltrami", "4panel_syn_beltrami"),
        ("syn_beltrami_proj", "4panel_syn_beltrami_proj"),
        ("syn_sobolev", "4panel_sobolev"),
        ("syn_hyperelastic", "4panel_hyperelastic"),
    )
    need_volume_load = False
    for arm_key, tag in panel_targets:
        rec = by_arm.get(arm_key)
        if rec and rec.get("persisted_transforms"):
            out_png = fig_dir / f"pair_{viz_pair:03d}_{tag}.png"
            if force or not out_png.exists():
                need_volume_load = True
                break
    best_syn = (
        by_arm.get("tvf_best")
        or by_arm.get("syngs_beltrami")
        or by_arm.get("syn_beltrami")
        or by_arm.get("syn_hyperelastic")
        or by_arm.get("syn_sobolev")
    )
    out_lbl_path = fig_dir / f"pair_{viz_pair:03d}_warped_dkt31_overlay.png"
    if best_syn and best_syn.get("persisted_transforms") and (force or not out_lbl_path.exists()):
        need_volume_load = True

    fi: Optional[ants.ANTsImage] = None
    mi: Optional[ants.ANTsImage] = None
    ml: Optional[ants.ANTsImage] = None
    pair_data: Optional[Dict[str, Any]] = None
    if need_volume_load and by_arm:
        pair_data = load_mindboggle_pair(
            pair_idx=viz_pair, pairs_csv=pairs_csv, data_dir=data_dir, use_n4=use_n4
        )
        fi = normalize_intensity(pair_data["fixed"])
        mi = normalize_intensity(pair_data["moving"])
        ml = pair_data["moving_label"]

    # 1. Standard 4-panel light-theme figures via syntx.viz.render_standard_4panel (cached per viz_pair)
    for arm_key, tag in panel_targets:
        rec = by_arm.get(arm_key)
        if rec is None or not rec.get("persisted_transforms"):
            continue
        out_png = str(fig_dir / f"pair_{viz_pair:03d}_{tag}.png")
        if not force and os.path.exists(out_png):
            figs[tag] = out_png
            continue
        assert fi is not None and mi is not None and pair_data is not None
        fwd_tx = rec["persisted_transforms"]["fwdtransforms"]
        inv_tx = rec["persisted_transforms"]["invtransforms"]
        fwd_warp = next((t for t in fwd_tx if t.endswith(".nii.gz")), None)
        inv_warp = next((t for t in inv_tx if t.endswith(".nii.gz")), None)
        if fwd_warp is None or inv_warp is None:
            continue
        warped = syntx_apply_transforms(fixed=fi, moving=mi, transformlist=fwd_tx)
        det_j = create_jacobian_determinant_image(fi, ants.image_read(fwd_warp), do_log=False)
        ice = calculate_inverse_identity_error(
            disp_itk_to_tensor(fwd_warp, device="cpu").float(),
            disp_itk_to_tensor(inv_warp, device="cpu").float(),
            spacing=fi.spacing,
            origin=fi.origin,
            direction=fi.direction,
        )
        render_standard_4panel(
            fixed=fi,
            warped=warped,
            warp=fwd_warp,
            detJ=det_j,
            inv_err_map=ice["error_map"],
            moving=mi,
            slice_axis=2,
            show_header=True,
            theme="light",
            lncc_val=rec["lncc_similarity"],
            mi_val=rec["mattes_mi_similarity"],
            inv_err_max=rec["inv_max_mm"],
            inv_err_mean=rec["inv_mean_mm"],
            inv_err_p95=rec["inv_p95_mm"],
            min_detJ=rec["min_jacobian"],
            jac_measure=rec["jacobian_measure"],
            title_prefix=(
                f"Pair {viz_pair:02d} ({pair_data['fixed_id']} ← {pair_data['moving_id']}) "
                f"• {rec['arm_label']} • Sym Dice {rec['dice_sym']:.4f}"
            ),
            output_path=out_png,
        )
        figs[tag] = out_png

    # 2. Warped DKT31 cortical label overlay on fixed anatomy via syntx.viz.render_label_overlay_figure
    if best_syn and best_syn.get("persisted_transforms"):
        out_lbl = str(out_lbl_path)
        if not force and os.path.exists(out_lbl):
            figs["label_overlay"] = out_lbl
        elif fi is not None and mi is not None and ml is not None:
            ml_c = ml.clone()
            ml_c.set_origin(mi.origin)
            ml_c.set_spacing(mi.spacing)
            ml_c.set_direction(mi.direction)
            ml_warped = syntx_apply_transforms(
                fixed=fi,
                moving=ml_c,
                transformlist=best_syn["persisted_transforms"]["fwdtransforms"],
                interpolator="nearestNeighbor",
            )
            render_label_overlay_figure(
                background=fi,
                labels=ml_warped,
                title=(
                    f"Pair {viz_pair:02d} Warped Moving DKT31 Labels on Fixed Anatomy "
                    f"({best_syn['arm_label']}, Sym Dice {best_syn['dice_sym']:.4f})"
                ),
                save_path=out_lbl,
                theme="light",
            )
            figs["label_overlay"] = out_lbl

    # 3. Light-theme Accuracy vs Runtime Pareto chart across completed pairs (GEMINI.md §5 palette)
    arm_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in deformable_records + e2e_records:
        if r["rep"] == 0:
            arm_groups.setdefault(f"{r['track']}::{r['arm']}", []).append(r)

    n_unique_pairs = len({int(r["pair_idx"]) for r in deformable_records if r["rep"] == 0})
    if arm_groups:
        fig = Figure(figsize=(10.5, 5.2), dpi=140, facecolor="#ffffff")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#ffffff")
        for key, recs in arm_groups.items():
            track_k, arm_k = key.split("::", 1)
            mean_t = float(np.mean([x["runtime_seconds"] for x in recs]))
            mean_d = float(np.mean([x["dice_sym"] for x in recs]))
            is_ants = arm_k.startswith("ants_")
            if track_k == "e2e":
                color = "#d97706" if is_ants else "#059669"
                marker = "D"
            elif arm_k in ("tvf_best", "syngs_beltrami", "syn_beltrami", "syn_beltrami_proj"):
                color = "#059669"
                marker = "^"
            else:
                color = "#64748b" if is_ants else "#0891b2"
                marker = "s" if is_ants else "o"
            ax.scatter(
                mean_t,
                mean_d,
                s=120,
                c=color,
                marker=marker,
                edgecolors="#1e293b",
                linewidths=1.2,
                zorder=4,
            )
            short_map = {
                "syntx_e2e_syn": "syntx E2E",
                "ants_e2e_syn": "ANTs E2E",
                "ants_syn_matched": "ANTs SyN (matched)",
                "ants_syn_quick": "ANTs SyNQuick",
                "tvf_best": "tvf (beltrami)",
                "syngs_beltrami": "syngs (beltrami)",
                "syngs_best": "syngs (sobolev)",
                "syn_beltrami": "syn (beltrami)",
                "syn_beltrami_proj": "syn (beltrami+proj)",
                "syn_sobolev": "syn (sobolev)",
                "syn_gaussian": "syn (gaussian)",
                "syn_hyperelastic": "syn (hyperelastic)",
            }
            short = short_map.get(arm_k, arm_k.replace("syn_", ""))
            offsets = {
                "syn_gaussian": (-58, -14),
                "syn_sobolev": (-58, -4),
                "syn_beltrami": (10, 6),
                "syn_beltrami_proj": (10, -14),
                "syngs_beltrami": (-68, 12),
                "syngs_best": (-62, -14),
                "tvf_best": (-62, 10),
                "syn_navier": (-45, -18),
                "syn_hyperelastic": (10, -4),
                "syntx_e2e_syn": (10, 10),
                "syn_divcurl": (10, -13),
                "syn_solenoidal": (9, -4),
                "ants_syn_matched": (-92, 11),
                "ants_e2e_syn": (-58, -16),
                "ants_syn_quick": (9, 6),
            }
            dx, dy = offsets.get(arm_k, (7, 5))
            ax.annotate(
                short,
                (mean_t, mean_d),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.5,
                color="#1e293b",
                fontweight="bold" if not is_ants else "normal",
                arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.7},
            )
        ax.margins(x=0.14, y=0.16)
        ax.set_xlabel(
            "Mean Wall-Clock Runtime per 3-D Subject Pair (seconds, lower is faster)",
            color="#1e293b",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_ylabel(
            "Mean Symmetric Cortical DKT31 Dice (higher is better)",
            color="#1e293b",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_title(
            f"Mindboggle ({n_unique_pairs} Pair{'s' if n_unique_pairs != 1 else ''}): "
            "Cortical Accuracy vs. Runtime ([100, 100, 20])",
            color="#0f172a",
            fontsize=11.5,
            fontweight="bold",
            pad=12,
        )
        ax.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#94a3b8")
        ax.tick_params(colors="#1e293b")
        fig.tight_layout()
        pareto_png = str(fig_dir / "cohort_accuracy_vs_runtime_pareto.png")
        fig.savefig(pareto_png, facecolor="#ffffff")
        figs["pareto"] = pareto_png

    # 3b. Cohort Distribution Box + Intra/Inter Strip Plot across completed pairs
    def_only_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in deformable_records:
        if r["rep"] == 0:
            def_only_groups.setdefault(r["arm"], []).append(r)
    if def_only_groups and n_unique_pairs >= 2:
        ordered_arms = sorted(
            def_only_groups.keys(),
            key=lambda a: float(np.mean([x["dice_sym"] for x in def_only_groups[a]])),
        )
        fig_d = Figure(figsize=(10.5, max(4.6, 0.58 * len(ordered_arms) + 1.4)), dpi=140, facecolor="#ffffff")
        ax_d = fig_d.add_subplot(111)
        ax_d.set_facecolor("#ffffff")
        rng_j = np.random.default_rng(42)
        short_labels = {
            "tvf_best": "syntx.tvf (beltrami)",
            "syngs_beltrami": "syntx.syngs (beltrami)",
            "syngs_best": "syntx.syngs (sobolev)",
            "syn_beltrami": "syntx.syn (beltrami)",
            "syn_beltrami_proj": "syntx.syn (beltrami + proj)",
            "syn_sobolev": "syntx.syn (sobolev)",
            "syn_gaussian": "syntx.syn (gaussian)",
            "syn_hyperelastic": "syntx.syn (hyperelastic)",
            "ants_syn_matched": "ants.registration (SyNOnly)",
            "ants_syn_quick": "ants.registration (SyNQuick)",
        }
        data_boxes = [[x["dice_sym"] for x in def_only_groups[a]] for a in ordered_arms]
        bp = ax_d.boxplot(
            data_boxes,
            vert=False,
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#0f172a", "linewidth": 1.8},
            whiskerprops={"color": "#64748b", "linewidth": 1.1},
            capprops={"color": "#64748b", "linewidth": 1.1},
        )
        for patch, a in zip(bp["boxes"], ordered_arms):
            if a.startswith("ants_"):
                patch.set_facecolor("#f1f5f9")
                patch.set_edgecolor("#64748b")
            elif "beltrami" in a or a == "tvf_best":
                patch.set_facecolor("#ecfdf5")
                patch.set_edgecolor("#059669")
            else:
                patch.set_facecolor("#ecfeff")
                patch.set_edgecolor("#0891b2")
            patch.set_linewidth(1.2)

        for idx_y, a in enumerate(ordered_arms, start=1):
            recs_a = def_only_groups[a]
            intra_vals = [x["dice_sym"] for x in recs_a if x.get("pair_type") == "intra"]
            inter_vals = [x["dice_sym"] for x in recs_a if x.get("pair_type") != "intra"]
            if intra_vals:
                jit = rng_j.uniform(-0.16, 0.16, size=len(intra_vals))
                ax_d.scatter(
                    intra_vals,
                    np.full(len(intra_vals), idx_y) + jit,
                    s=24,
                    c="#0891b2",
                    alpha=0.72,
                    edgecolors="none",
                    label="Intra-site pair" if idx_y == 1 else None,
                    zorder=5,
                )
            if inter_vals:
                jit = rng_j.uniform(-0.16, 0.16, size=len(inter_vals))
                ax_d.scatter(
                    inter_vals,
                    np.full(len(inter_vals), idx_y) + jit,
                    s=24,
                    c="#d97706",
                    alpha=0.72,
                    edgecolors="none",
                    label="Inter-site pair" if idx_y == 1 else None,
                    zorder=5,
                )
            ax_d.scatter(
                [float(np.mean([x["dice_sym"] for x in recs_a]))],
                [idx_y],
                s=52,
                c="#059669" if not a.startswith("ants_") else "#475569",
                marker="D",
                edgecolors="#1e293b",
                linewidths=0.9,
                label="Mean Dice" if idx_y == 1 else None,
                zorder=6,
            )

        ax_d.set_yticks(range(1, len(ordered_arms) + 1))
        ax_d.set_yticklabels([short_labels.get(a, a) for a in ordered_arms], fontsize=9, color="#1e293b")
        ax_d.set_xlabel(
            "Bidirectional Cortical DKT31 Sørensen–Dice",
            color="#1e293b",
            fontsize=10,
            fontweight="bold",
        )
        ax_d.set_title(
            f"Mindboggle Per-Pair Cortical Dice Distribution ({n_unique_pairs} Completed Pairs: Intra vs. Inter)",
            color="#0f172a",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        ax_d.legend(loc="lower left", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8.5)
        ax_d.grid(True, axis="x", linestyle="--", alpha=0.45, color="#cbd5e1")
        for spine in ax_d.spines.values():
            spine.set_color("#94a3b8")
        ax_d.tick_params(colors="#1e293b")
        fig_d.tight_layout()
        dist_png = str(fig_dir / "cohort_dice_distribution.png")
        fig_d.savefig(dist_png, facecolor="#ffffff")
        figs["distribution"] = dist_png

    # 4. Regional DKT31 Dice gains on viz_pair vs ANTs C++
    ants_rec = by_arm.get("ants_syn_matched")
    sob_rec = by_arm.get("syn_sobolev")
    champ_rec = (
        by_arm.get("tvf_best")
        or by_arm.get("syngs_beltrami")
        or by_arm.get("syn_beltrami")
        or by_arm.get("syn_hyperelastic")
    )
    if ants_rec and sob_rec and ants_rec.get("regional_dice_sym") and sob_rec.get("regional_dice_sym"):
        ants_reg = ants_rec["regional_dice_sym"]
        sob_reg = sob_rec["regional_dice_sym"]
        champ_reg = (
            champ_rec["regional_dice_sym"]
            if champ_rec and champ_rec.get("regional_dice_sym")
            else sob_reg
        )
        champ_lbl = champ_rec["arm"] if champ_rec and champ_rec.get("regional_dice_sym") else "syn_sobolev"
        common_ids = sorted(set(ants_reg) & set(sob_reg) & set(champ_reg))
        deltas = [
            (
                lid,
                DKT31_NAMES.get(int(lid), f"DKT {lid}"),
                (champ_reg[lid] - ants_reg[lid]) * 100.0,
                (sob_reg[lid] - ants_reg[lid]) * 100.0,
            )
            for lid in common_ids
        ]
        deltas.sort(key=lambda x: x[2], reverse=True)
        top_sel = deltas[:10] + deltas[-4:] if len(deltas) > 14 else deltas
        top_sel = list(reversed(top_sel))

        if top_sel:
            fig2 = Figure(figsize=(10.5, 5.6), dpi=140, facecolor="#ffffff")
            ax2 = fig2.add_subplot(111)
            ax2.set_facecolor("#ffffff")
            y_pos = np.arange(len(top_sel))
            h = 0.37
            ax2.barh(
                y_pos + h / 2,
                [x[2] for x in top_sel],
                height=h,
                color="#059669",
                label=f"{champ_lbl} − ANTs SyN",
            )
            ax2.barh(
                y_pos - h / 2,
                [x[3] for x in top_sel],
                height=h,
                color="#0891b2",
                label="syntx.syn (sobolev) − ANTs SyN",
            )
            ax2.axvline(0.0, color="#1e293b", linewidth=1.0)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([x[1] for x in top_sel], fontsize=8.5, color="#1e293b")
            ax2.set_xlabel(
                "Symmetric Sørensen–Dice Gain over ANTs C++ SyN (percentage points)",
                color="#1e293b",
                fontsize=10,
                fontweight="bold",
            )
            ax2.set_title(
                f"Pair {viz_pair:02d}: Regional DKT31 Cortical Dice "
                "Differential vs. ANTs C++ (Shared Affine)",
                color="#0f172a",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )
            ax2.legend(
                loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8.5
            )
            ax2.grid(True, axis="x", linestyle="--", alpha=0.45, color="#cbd5e1")
            for spine in ax2.spines.values():
                spine.set_color("#94a3b8")
            ax2.tick_params(colors="#1e293b")
            fig2.tight_layout()
            reg_png = str(fig_dir / f"pair_{viz_pair:03d}_regional_dkt31_gains.png")
            fig2.savefig(reg_png, facecolor="#ffffff")
            figs["regional_gains"] = reg_png

    return figs


def _aggregate_arm_rows(
    records: List[Dict[str, Any]],
    arm_order: Sequence[str],
    baseline_arm: str,
) -> List[Dict[str, Any]]:
    from scipy.stats import wilcoxon

    by_arm: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in records:
        if r["rep"] == 0:
            by_arm.setdefault(r["arm"], {})[int(r["pair_idx"])] = r

    base_map = by_arm.get(baseline_arm, {})
    rows: List[Dict[str, Any]] = []
    for arm in arm_order:
        pmap = by_arm.get(arm)
        if not pmap:
            continue
        recs = [pmap[p] for p in sorted(pmap)]
        d_syms = [r["dice_sym"] for r in recs]
        d_fix = [r["dice_fixed"] for r in recs]
        d_mov = [r["dice_moving"] for r in recs]
        d_intra = [r["dice_sym"] for r in recs if r.get("pair_type") == "intra"]
        d_inter = [r["dice_sym"] for r in recs if r.get("pair_type") != "intra"]
        folds = [r["folding_pct"] for r in recs]
        min_j = [r["min_jacobian"] for r in recs]
        max_j = [r["max_jacobian"] for r in recs if math.isfinite(r["max_jacobian"])]
        fd_folds = [float(r.get("fd_folding_pct", r["folding_pct"])) for r in recs]
        fd_min_j = [float(r.get("fd_min_jacobian", r["min_jacobian"])) for r in recs]
        harm = [r["harmonic_energy"] for r in recs]
        bend = [r["bending_energy"] for r in recs]
        ice_m = [r["inv_mean_mm"] for r in recs]
        ice_p95 = [r["inv_p95_mm"] for r in recs]
        ice_imax = [r["inv_interior_max_mm"] for r in recs if math.isfinite(r["inv_interior_max_mm"])]
        times = [r["runtime_seconds"] for r in recs]
        qc_fails = sum(1 for r in recs if r["qc"]["status"] == "fail")
        qc_warns = sum(1 for r in recs if r["qc"]["status"] == "warn")
        qc_passes = sum(1 for r in recs if r["qc"]["status"] == "pass")
        if qc_fails > 0 or qc_warns > 0:
            qc_summary = f"{qc_fails} fail, {qc_warns} warn / {len(recs)}"
        elif qc_passes > 0:
            qc_summary = f"{qc_passes}/{len(recs)} pass"
        else:
            diffeo_n = sum(1 for r in recs if r["qc"].get("is_diffeomorphic"))
            qc_summary = f"0/{len(recs)} fail ({diffeo_n}/{len(recs)} diffeo)"

        wins = 0
        n_cmp = 0
        deltas_pp: List[float] = []
        speedups: List[float] = []
        for p in sorted(pmap):
            r = pmap[p]
            if p in base_map:
                n_cmp += 1
                diff_pp = (r["dice_sym"] - base_map[p]["dice_sym"]) * 100.0
                deltas_pp.append(diff_pp)
                if r["dice_sym"] >= base_map[p]["dice_sym"]:
                    wins += 1
                if r["runtime_seconds"] > 0:
                    speedups.append(base_map[p]["runtime_seconds"] / r["runtime_seconds"])

        wilcoxon_p: Optional[float] = None
        if arm != baseline_arm and len(deltas_pp) >= 6 and any(abs(x) > 1e-12 for x in deltas_pp):
            try:
                _, p_val = wilcoxon(deltas_pp, alternative="two-sided")
                wilcoxon_p = float(p_val)
            except ValueError:
                wilcoxon_p = None

        iqr_sym = (
            float(np.percentile(d_syms, 75) - np.percentile(d_syms, 25))
            if len(d_syms) > 1
            else 0.0
        )

        rows.append(
            {
                "arm": arm,
                "arm_label": ARM_LABELS.get(arm, arm),
                "device": recs[0]["device"],
                "n_pairs": len(recs),
                "mean_dice_sym": float(statistics.mean(d_syms)),
                "std_dice_sym": float(statistics.stdev(d_syms)) if len(d_syms) > 1 else 0.0,
                "median_dice_sym": float(np.median(d_syms)),
                "iqr_dice_sym": iqr_sym,
                "n_intra": len(d_intra),
                "mean_dice_intra": float(statistics.mean(d_intra)) if d_intra else float("nan"),
                "std_dice_intra": float(statistics.stdev(d_intra)) if len(d_intra) > 1 else 0.0,
                "n_inter": len(d_inter),
                "mean_dice_inter": float(statistics.mean(d_inter)) if d_inter else float("nan"),
                "std_dice_inter": float(statistics.stdev(d_inter)) if len(d_inter) > 1 else 0.0,
                "mean_dice_fixed": float(statistics.mean(d_fix)),
                "mean_dice_moving": float(statistics.mean(d_mov)),
                "mean_delta_vs_ants_pp": float(statistics.mean(deltas_pp)) if deltas_pp else 0.0,
                "wins_vs_ants": f"{wins}/{n_cmp}" if n_cmp else "—",
                "wilcoxon_p_vs_ants": wilcoxon_p,
                "mean_folding_pct": float(statistics.mean(folds)),
                "worst_min_jac": float(min(min_j)),
                "mean_max_jac": float(statistics.mean(max_j)) if max_j else float("nan"),
                "mean_fd_folding_pct": float(statistics.mean(fd_folds)),
                "max_fd_folding_pct": float(max(fd_folds)),
                "worst_fd_min_jac": float(min(fd_min_j)),
                "jac_measure": recs[0]["jacobian_measure"],
                "mean_harmonic_energy": float(statistics.mean(harm)),
                "mean_bending_energy": float(statistics.mean(bend)),
                "mean_inv_mm": float(statistics.mean(ice_m)),
                "mean_inv_p95_mm": float(statistics.mean(ice_p95)),
                "max_inv_interior_mm": float(max(ice_imax)) if ice_imax else 0.0,
                "mean_time_s": float(statistics.mean(times)),
                "mean_speedup_vs_ants": float(statistics.mean(speedups)) if speedups else 1.0,
                "qc_pass_count": qc_summary,
            }
        )
    return rows


def _atomic_write_text(path: str, content: str) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _img_b64_tag(path: Optional[str], caption: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return (
        '<figure class="fig-box">'
        f'<img src="data:image/png;base64,{b64}" alt="{_html.escape(caption)}" />'
        f"<figcaption>{_html.escape(caption)}</figcaption>"
        "</figure>"
    )


def _qc_badge_html(status: str) -> str:
    colors = {
        "pass": ("#ecfdf5", "#059669"),
        "warn": ("#fffbeb", "#d97706"),
        "fail": ("#fef2f2", "#dc2626"),
        "not_assessable": ("#f1f5f9", "#475569"),
    }
    bg, fg = colors.get(status, ("#f1f5f9", "#475569"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:999px;'
        f'font-weight:700;font-size:0.76rem;text-transform:uppercase">{_html.escape(status)}</span>'
    )


def _fmt_pval(p: Optional[float]) -> str:
    if p is None or not math.isfinite(p):
        return "—"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _write_reports(
    report_stem: Path,
    summary_payload: Dict[str, Any],
    affine_rows: List[Dict[str, Any]],
    deform_rows: List[Dict[str, Any]],
    e2e_rows: List[Dict[str, Any]],
    all_records: List[Dict[str, Any]],
    repro_rows: List[Dict[str, Any]],
    figures: Dict[str, str],
) -> Tuple[str, str, str]:
    report_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = str(report_stem.with_suffix(".json"))
    md_path = str(report_stem.with_suffix(".md"))
    html_path = str(report_stem.with_suffix(".html"))

    _atomic_write_text(json_path, json.dumps(summary_payload, indent=2, default=str))

    n_total_pairs = int(summary_payload.get("total_pairs", len(summary_payload["pairs"])))
    n_done_pairs = int(summary_payload.get("completed_pairs", n_total_pairs))
    in_progress = bool(summary_payload.get("in_progress", n_done_pairs < n_total_pairs))
    eta_min = summary_payload.get("eta_minutes")
    shuffle_seed = summary_payload.get("shuffle_seed")
    pct_done = (100.0 * n_done_pairs / max(1, n_total_pairs))

    done_pair_types: Dict[int, str] = {}
    for r in all_records:
        if r["rep"] == 0 and r["track"] == "deformable":
            done_pair_types[int(r["pair_idx"])] = str(r.get("pair_type", ""))
    if not done_pair_types:
        for r in all_records:
            if r["rep"] == 0:
                done_pair_types[int(r["pair_idx"])] = str(r.get("pair_type", ""))
    n_done_intra = sum(1 for v in done_pair_types.values() if v == "intra")
    n_done_inter = sum(1 for v in done_pair_types.values() if v != "intra")

    repro_pair = int(summary_payload.get("repro_pair", 44))
    viz_pair = repro_pair
    has_repro_viz = any(
        r["track"] == "deformable" and r["pair_idx"] == repro_pair and r["rep"] == 0 and r.get("persisted_transforms")
        for r in all_records
    )
    if not has_repro_viz:
        persisted_viz = [
            int(r["pair_idx"])
            for r in all_records
            if r["track"] == "deformable" and r["rep"] == 0 and r.get("persisted_transforms")
        ]
        if persisted_viz:
            viz_pair = persisted_viz[0]

    is_cohort_mode = n_total_pairs > 3
    report_title_md = (
        f"# Mindboggle-101 Registration Benchmark (`syntx` TVF / SyNGS / SyN vs `ants.registration` — {n_done_pairs}/{n_total_pairs} Pairs)"
        if is_cohort_mode
        else "# Fast 3-Subject Registration Benchmark (`syntx` vs `ants.registration`)"
    )

    # Markdown summary
    env_d = summary_payload["environment"]
    pairs_preview = (
        f"`{n_done_pairs}/{n_total_pairs}` completed (`{n_done_intra}` intra-site, `{n_done_inter}` inter-site; shuffled seed=`{shuffle_seed}`)"
        if is_cohort_mode
        else f"`{summary_payload['pairs']}` (`44 = mbhard` RPI→LAI, `77` PIR→LAI, `0` intra PIR)"
    )
    status_line_md = (
        f"IN PROGRESS — `{n_done_pairs}/{n_total_pairs}` pairs (`{pct_done:.1f}%`), "
        f"elapsed `{summary_payload['total_wall_minutes']:.1f} min`"
        + (f", ETA `{eta_min:.1f} min`" if eta_min is not None else "")
        if in_progress
        else f"COMPLETE — `{n_done_pairs}/{n_total_pairs}` pairs (`100.0%`) in `{summary_payload['total_wall_minutes']:.1f} min`"
    )

    md_lines = [
        report_title_md,
        "",
        f"- **Status:** {status_line_md}",
        f"- **Generated (UTC):** {summary_payload['timestamp_utc']}",
        (
            f"- **syntx Version / Commit:** `v{summary_payload['syntx_version']}` "
            f"(`{summary_payload['git_commit'][:10]}`, dirty={summary_payload['git_dirty']})"
        ),
        (
            f"- **Host / Hardware:** `{env_d['platform']}` "
            f"({env_d['machine']}, {env_d['cpu_count']} cores, MPS={env_d['devices']['mps']})"
        ),
        (
            f"- **Load Average (start → current, 1m):** "
            f"`{summary_payload['load_avg_start_1m']:.2f} → {summary_payload['load_avg_end_1m']:.2f}` "
            f"(isolated sequential subprocesses, MPS before CPU)"
        ),
        f"- **Pairs Evaluated:** {pairs_preview}",
        f"- **Schedule:** `reg_iterations = {summary_payload['reg_iterations']}`",
        "",
    ]

    if affine_rows:
        md_lines.extend(
            [
                "## 1. Track 1 — Affine Registration (`ants.registration` vs `syntx.robust_affine`)",
                "",
                pd.DataFrame(affine_rows)[
                    [
                        "arm",
                        "n_pairs",
                        "mean_dice_sym",
                        "std_dice_sym",
                        "median_dice_sym",
                        "mean_delta_vs_ants_pp",
                        "wins_vs_ants",
                        "mean_time_s",
                        "mean_speedup_vs_ants",
                        "qc_pass_count",
                    ]
                ].to_markdown(index=False, floatfmt=".4f"),
                "",
            ]
        )

    if deform_rows:
        deform_md_df = pd.DataFrame(deform_rows).copy()
        deform_md_df["wilcoxon_p"] = deform_md_df["wilcoxon_p_vs_ants"].map(_fmt_pval)
        md_lines.extend(
            [
                "## 2. Track 2 — Deformable Registration & Regularizers (Shared Canonical Affine `pt7`)",
                "",
                deform_md_df[
                    [
                        "arm",
                        "n_pairs",
                        "mean_dice_sym",
                        "std_dice_sym",
                        "median_dice_sym",
                        "mean_delta_vs_ants_pp",
                        "wins_vs_ants",
                        "wilcoxon_p",
                        "mean_folding_pct",
                        "worst_min_jac",
                        "mean_fd_folding_pct",
                        "worst_fd_min_jac",
                        "mean_inv_mm",
                        "mean_inv_p95_mm",
                        "mean_time_s",
                        "mean_speedup_vs_ants",
                        "qc_pass_count",
                    ]
                ].to_markdown(index=False, floatfmt=".4f"),
                "",
            ]
        )
        if any(r.get("n_intra", 0) > 0 and r.get("n_inter", 0) > 0 for r in deform_rows):
            md_lines.extend(
                [
                    "### 2b. Intra-Site vs. Inter-Site Deformable Subgroup Breakdown",
                    "",
                    deform_md_df[
                        [
                            "arm",
                            "n_intra",
                            "mean_dice_intra",
                            "std_dice_intra",
                            "n_inter",
                            "mean_dice_inter",
                            "std_dice_inter",
                            "iqr_dice_sym",
                        ]
                    ].to_markdown(index=False, floatfmt=".4f"),
                    "",
                ]
            )

    if e2e_rows:
        md_lines.extend(
            [
                "## 3. Track 3 — End-to-End Pipeline (`initial_transform=None`)",
                "",
                pd.DataFrame(e2e_rows)[
                    [
                        "arm",
                        "mean_dice_sym",
                        "std_dice_sym",
                        "mean_delta_vs_ants_pp",
                        "wins_vs_ants",
                        "mean_folding_pct",
                        "mean_inv_p95_mm",
                        "mean_time_s",
                        "mean_speedup_vs_ants",
                        "qc_pass_count",
                    ]
                ].to_markdown(index=False, floatfmt=".4f"),
                "",
            ]
        )

    if repro_rows:
        md_lines.extend(
            [
                "## 4. Track 4 — Fresh-Process & Same-Process Reproducibility (`mbhard`, Pair 44)",
                "",
                pd.DataFrame(repro_rows)[
                    [
                        "scope",
                        "track",
                        "arm",
                        "dice_sym_rep0",
                        "dice_sym_rep1",
                        "abs_dice_sym_diff",
                        "max_param_spread",
                        "mean_brain_disp_spread_mm",
                        "max_brain_disp_spread_mm",
                    ]
                ].to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )
    _atomic_write_text(md_path, "\n".join(md_lines))

    # Build HTML KPI highlights
    aff_map = {r["arm"]: r for r in affine_rows}
    def_map = {r["arm"]: r for r in deform_rows}
    e2e_map = {r["arm"]: r for r in e2e_rows}

    sob = def_map.get("syn_sobolev", {})
    ants_d = def_map.get("ants_syn_matched", {})
    aff_s = aff_map.get("syntx_affine_default", {})
    aff_a = aff_map.get("ants_affine", {})
    e2e_s = e2e_map.get("syntx_e2e_syn", {})
    e2e_a = e2e_map.get("ants_e2e_syn", {})

    show_wilcoxon = any(r.get("wilcoxon_p_vs_ants") is not None for r in deform_rows)

    def _render_summary_table(rows: List[Dict[str, Any]], include_deform: bool = True) -> str:
        if not rows:
            return "<p>No records.</p>"
        hdr = (
            "<tr><th>Arm</th><th>Device</th><th>Mean Sym Dice ± SD</th>"
            "<th>Median (IQR)</th><th>Fixed / Moving</th><th>Δ vs ANTs (pp)</th><th>Wins</th>"
        )
        if include_deform and show_wilcoxon:
            hdr += "<th>Wilcoxon <em>p</em></th>"
        if include_deform:
            hdr += (
                "<th>Fold %</th><th>Min |J|</th><th>FD Fold % (Min |J|)</th><th>Max |J|</th>"
                "<th>Harm / Bend Energy</th><th>ICE Mean / P95 (mm)</th>"
            )
        hdr += "<th>Mean Time (s)</th><th>Speedup</th><th>antsxqc</th></tr>"
        body = []
        best_dice = max(r["mean_dice_sym"] for r in rows)
        for r in rows:
            is_best = abs(r["mean_dice_sym"] - best_dice) < 1e-9
            d_style = ' style="color:#059669;font-weight:700"' if is_best else ""
            delta = r["mean_delta_vs_ants_pp"]
            delta_color = "#059669" if delta > 0.01 else ("#dc2626" if delta < -0.01 else "#475569")
            row_html = (
                f"<tr>"
                f"<td><strong>{_html.escape(r['arm'])}</strong>"
                f"<div class='sub'>{_html.escape(r['arm_label'])} (n={r['n_pairs']})</div></td>"
                f"<td><code>{_html.escape(str(r['device']))}</code></td>"
                f"<td{d_style}>{r['mean_dice_sym']:.4f} ± {r['std_dice_sym']:.4f}</td>"
                f"<td>{r['median_dice_sym']:.4f} ({r['iqr_dice_sym']:.4f})</td>"
                f"<td>{r['mean_dice_fixed']:.4f} / {r['mean_dice_moving']:.4f}</td>"
                f"<td style='color:{delta_color};font-weight:600'>{delta:+.2f}%</td>"
                f"<td><strong>{_html.escape(r['wins_vs_ants'])}</strong></td>"
            )
            if include_deform and show_wilcoxon:
                pv = r.get("wilcoxon_p_vs_ants")
                pv_cls = " class='emerald'" if (pv is not None and pv < 0.05 and delta > 0) else ""
                row_html += f"<td{pv_cls}><code>{_html.escape(_fmt_pval(pv))}</code></td>"
            if include_deform:
                max_j_s = f"{r['mean_max_jac']:.1f}" if math.isfinite(r["mean_max_jac"]) else "—"
                row_html += (
                    f"<td>{r['mean_folding_pct']:.5f}%</td>"
                    f"<td>{r['worst_min_jac']:+.4f}</td>"
                    f"<td>{r['mean_fd_folding_pct']:.5f}% ({r['worst_fd_min_jac']:+.3f})</td>"
                    f"<td>{max_j_s}</td>"
                    f"<td>{r['mean_harmonic_energy']:.4f} / {r['mean_bending_energy']:.4f}</td>"
                    f"<td>{r['mean_inv_mm']:.4f} / {r['mean_inv_p95_mm']:.4f}</td>"
                )
            row_html += (
                f"<td><strong>{r['mean_time_s']:.1f}s</strong></td>"
                f"<td><span class='cyan'>{r['mean_speedup_vs_ants']:.2f}×</span></td>"
                f"<td>{_html.escape(r['qc_pass_count'])}</td>"
                f"</tr>"
            )
            body.append(row_html)
        return f"<table><thead>{hdr}</thead><tbody>{''.join(body)}</tbody></table>"

    subgroup_section_html = ""
    if any(r.get("n_intra", 0) > 0 and r.get("n_inter", 0) > 0 for r in deform_rows):
        sub_body = []
        best_intra = max(
            (r["mean_dice_intra"] for r in deform_rows if math.isfinite(r["mean_dice_intra"])),
            default=float("nan"),
        )
        best_inter = max(
            (r["mean_dice_inter"] for r in deform_rows if math.isfinite(r["mean_dice_inter"])),
            default=float("nan"),
        )
        for r in deform_rows:
            s_in = (
                ' style="color:#059669;font-weight:700"'
                if math.isfinite(best_intra) and abs(r["mean_dice_intra"] - best_intra) < 1e-9
                else ""
            )
            s_out = (
                ' style="color:#059669;font-weight:700"'
                if math.isfinite(best_inter) and abs(r["mean_dice_inter"] - best_inter) < 1e-9
                else ""
            )
            sub_body.append(
                f"<tr>"
                f"<td><strong>{_html.escape(r['arm'])}</strong>"
                f"<div class='sub'>{_html.escape(r['arm_label'])}</div></td>"
                f"<td{s_in}>{r['mean_dice_intra']:.4f} ± {r['std_dice_intra']:.4f} "
                f"(<code>n={r['n_intra']}</code>)</td>"
                f"<td{s_out}>{r['mean_dice_inter']:.4f} ± {r['std_dice_inter']:.4f} "
                f"(<code>n={r['n_inter']}</code>)</td>"
                f"<td>{(r['mean_dice_intra'] - r['mean_dice_inter']) * 100.0:+.2f} pp</td>"
                f"<td>{r['median_dice_sym']:.4f} (IQR {r['iqr_dice_sym']:.4f})</td>"
                f"</tr>"
            )
        subgroup_section_html = (
            "<h3>2b. Intra-Cohort vs. Cross-Cohort (Inter-Site) Cortical Dice Stratification</h3>"
            "<p>Mindboggle-101 comprises 40 intra-cohort pairs (matched scanner/protocol) and 50 "
            "cross-cohort inter-site pairs (heterogeneous acquisition geometry, contrast, and FOV).</p>"
            "<table><thead><tr><th>Arm</th><th>Intra-Site Mean ± SD</th>"
            "<th>Inter-Site Mean ± SD</th><th>Intra &minus; Inter Gap</th>"
            "<th>Cohort Median (IQR)</th></tr></thead>"
            f"<tbody>{''.join(sub_body)}</tbody></table>"
        )

    # Per-pair granular table
    pair_rows_html = []
    for r in sorted(
        [x for x in all_records if x["rep"] == 0],
        key=lambda z: ({"affine": 0, "deformable": 1, "e2e": 2}.get(z["track"], 9), z["pair_idx"], z["arm"]),
    ):
        pair_rows_html.append(
            f"<tr>"
            f"<td><code>{_html.escape(r['track'])}</code></td>"
            f"<td><strong>#{r['pair_idx']:02d}</strong> ({_html.escape(r['pair_type'])})"
            f"<div class='sub'>{_html.escape(r['fixed_id'])} ← {_html.escape(r['moving_id'])}</div></td>"
            f"<td><code>{_html.escape(r['arm'])}</code></td>"
            f"<td><strong>{r['dice_sym']:.4f}</strong> ({r['dice_fixed']:.4f} / {r['dice_moving']:.4f})</td>"
            f"<td>{r['folding_pct']:.5f}%</td>"
            f"<td>{r['min_jacobian']:+.4f}</td>"
            f"<td>{r['inv_mean_mm']:.4f} / {r['inv_p95_mm']:.4f}</td>"
            f"<td>{r['runtime_seconds']:.1f}s</td>"
            f"<td>{_qc_badge_html(r['qc']['status'])}</td>"
            f"</tr>"
        )

    # Reproducibility table
    repro_html_rows = []
    for rr in repro_rows:
        p_str = f"{rr['max_param_spread']:.3e}" if rr["max_param_spread"] is not None else "—"
        exact_zero = rr["max_brain_disp_spread_mm"] == 0.0 and rr["abs_dice_sym_diff"] == 0.0
        spread_cls = " class='emerald'" if exact_zero else " class='amber'"
        repro_html_rows.append(
            f"<tr>"
            f"<td><code>{_html.escape(rr['scope'])}</code></td>"
            f"<td><code>{_html.escape(rr['track'])}</code></td>"
            f"<td><strong>{_html.escape(rr['arm'])}</strong></td>"
            f"<td>{rr['dice_sym_rep0']:.6f}</td>"
            f"<td>{rr['dice_sym_rep1']:.6f}</td>"
            f"<td{spread_cls}>{rr['abs_dice_sym_diff']:.3e}</td>"
            f"<td>{p_str}</td>"
            f"<td{spread_cls}>{rr['mean_brain_disp_spread_mm']:.3e} mm</td>"
            f"<td{spread_cls}>{rr['max_brain_disp_spread_mm']:.3e} mm</td>"
            f"</tr>"
        )

    sob_gain = sob.get("mean_delta_vs_ants_pp", 0.0)
    sob_speed = sob.get("mean_speedup_vs_ants", 1.0)
    aff_speed = aff_s.get("mean_speedup_vs_ants", 1.0)
    e2e_speed = e2e_s.get("mean_speedup_vs_ants", 1.0)

    viz_tag = f"Pair {viz_pair:02d}" + (" (mbhard)" if viz_pair == 44 else "")
    fig_specs = [
        (
            "pareto",
            f"Accuracy vs. Runtime Pareto Frontier ({n_done_pairs} Completed Mindboggle Pairs, [100, 100, 20]).",
        ),
        (
            "distribution",
            f"Per-Pair Cortical DKT31 Dice Distribution Across {n_done_pairs} Completed Pairs (Intra-Site vs. Inter-Site).",
        ),
        (
            "regional_gains",
            f"{viz_tag} Per-Region DKT31 Cortical Symmetric Dice Gain over ANTs C++ SyN (Shared Affine).",
        ),
        (
            "4panel_tvf",
            f"{viz_tag} Standard 4-Panel QC — syntx.tvf Canonical Beltrami ([100, 100, 20], Shared Affine).",
        ),
        (
            "4panel_syngs_beltrami",
            f"{viz_tag} Standard 4-Panel QC — syntx.syngs Tuned Beltrami ([100, 100, 20], Shared Affine).",
        ),
        (
            "4panel_syn_beltrami",
            f"{viz_tag} Standard 4-Panel QC — syntx.syn Tuned Beltrami ([100, 100, 20], Shared Affine).",
        ),
        (
            "4panel_syn_beltrami_proj",
            f"{viz_tag} Standard 4-Panel QC — syntx.syn Tuned Beltrami + cfl_project_inverse=True ([100, 100, 20], Shared Affine).",
        ),
        (
            "4panel_sobolev",
            f"{viz_tag} Standard 4-Panel QC — syntx.syn Canonical Sobolev ([100, 100, 20], Shared Affine).",
        ),
        (
            "4panel_hyperelastic",
            f"{viz_tag} Standard 4-Panel QC — syntx.syn Continuum Hyperelastic ([100, 100, 20], Shared Affine).",
        ),
        (
            "4panel_ants",
            f"{viz_tag} Standard 4-Panel QC — ANTs C++ SyNOnly ([100, 100, 20], Shared Affine).",
        ),
        (
            "label_overlay",
            f"{viz_tag} Warped Moving DKT31 Manual Cortical Parcellation Overlaid on Fixed Brain.",
        ),
    ]
    rendered_figs: List[str] = []
    fig_num = 0
    for fkey, fcap in fig_specs:
        fpath = figures.get(fkey)
        if fpath and os.path.exists(fpath):
            fig_num += 1
            rendered_figs.append(_img_b64_tag(fpath, f"Figure {fig_num}: {fcap}"))
    figs_html = "".join(rendered_figs)

    ver_s = _html.escape(str(summary_payload["syntx_version"]))
    sha_s = _html.escape(summary_payload["git_commit"][:10])
    plat_s = _html.escape(env_d["platform"])
    iters_s = _html.escape(str(summary_payload["reg_iterations"]))
    aff_d_s = aff_s.get("mean_dice_sym", float("nan"))
    aff_d_a = aff_a.get("mean_dice_sym", float("nan"))
    aff_t_s = aff_s.get("mean_time_s", float("nan"))
    aff_t_a = aff_a.get("mean_time_s", float("nan"))
    sob_d = sob.get("mean_dice_sym", float("nan"))
    ants_d_v = ants_d.get("mean_dice_sym", float("nan"))
    sob_t = sob.get("mean_time_s", float("nan"))
    ants_t = ants_d.get("mean_time_s", float("nan"))
    e2e_d_s = e2e_s.get("mean_dice_sym", float("nan"))
    e2e_d_a = e2e_a.get("mean_dice_sym", float("nan"))
    e2e_t_s = e2e_s.get("mean_time_s", float("nan"))
    e2e_t_a = e2e_a.get("mean_time_s", float("nan"))

    syntx_def_rows = [r for r in deform_rows if not r["arm"].startswith("ants_")]
    best_def = (
        max(syntx_def_rows, key=lambda r: r["mean_dice_sym"])
        if syntx_def_rows
        else (deform_rows[0] if deform_rows else {})
    )
    best_def_name = _html.escape(best_def.get("arm", "—"))
    best_def_d = best_def.get("mean_dice_sym", float("nan"))
    best_def_gain = best_def.get("mean_delta_vs_ants_pp", 0.0)
    best_def_fold = best_def.get("mean_folding_pct", 0.0)
    best_def_wins = _html.escape(str(best_def.get("wins_vs_ants", "—")))

    syngs_b = def_map.get("syngs_beltrami") or def_map.get("syn_beltrami") or best_def
    syngs_b_name = _html.escape(syngs_b.get("arm", "—"))
    syngs_b_d = syngs_b.get("mean_dice_sym", float("nan"))
    syngs_b_gain = syngs_b.get("mean_delta_vs_ants_pp", 0.0)
    syngs_b_speed = syngs_b.get("mean_speedup_vs_ants", 1.0)
    syngs_b_t = syngs_b.get("mean_time_s", float("nan"))

    if e2e_rows:
        third_card_html = (
            '  <div class="card">\n'
            '    <div class="label">End-to-End (Affine + SyN)</div>\n'
            f'    <div class="value emerald">{e2e_d_s:.4f} vs {e2e_d_a:.4f}</div>\n'
            f'    <div class="note">{e2e_speed:.2f}&times; faster ({e2e_t_s:.1f}s vs {e2e_t_a:.1f}s)</div>\n'
            "  </div>"
        )
        e2e_section_html = (
            "<h2>3. Track 3 — Full End-to-End Pipeline (<code>initial_transform=None</code>)</h2>\n"
            "<p>Compares out-of-the-box <code>syntx.syn(fixed, moving)</code> "
            "(<code>robust_affine</code> + Sobolev SyN) against "
            "<code>ants.registration(fixed, moving, type_of_transform='SyN')</code>.</p>\n"
            f"{_render_summary_table(e2e_rows, include_deform=True)}"
        )
    else:
        third_card_html = (
            '  <div class="card">\n'
            f'    <div class="label">Speed-Accuracy Frontier ({syngs_b_name})</div>\n'
            f'    <div class="value emerald">{syngs_b_d:.4f} ({syngs_b_gain:+.2f}%)</div>\n'
            f'    <div class="note">{syngs_b_speed:.2f}&times; faster than ANTs ({syngs_b_t:.1f}s vs {ants_t:.1f}s)</div>\n'
            "  </div>"
        )
        e2e_section_html = ""

    progress_banner_html = ""
    if is_cohort_mode or in_progress:
        bar_color = "#0891b2" if in_progress else "#059669"
        badge_text = "LIVE BENCHMARK IN PROGRESS" if in_progress else "COHORT BENCHMARK COMPLETE"
        badge_bg = "#ecfeff" if in_progress else "#ecfdf5"
        badge_fg = "#0891b2" if in_progress else "#059669"
        eta_str = f" &bull; <strong>ETA:</strong> {eta_min:.1f} min" if (in_progress and eta_min is not None) else ""
        seed_str = f" &bull; <strong>Random Pair Seed:</strong> <code>{shuffle_seed}</code>" if shuffle_seed is not None else ""
        progress_banner_html = (
            '<div style="border:1px solid #cbd5e1;border-radius:8px;padding:14px 18px;margin-bottom:20px;background:#f8fafc;">'
            '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
            f'<div><span style="background:{badge_bg};color:{badge_fg};padding:3px 10px;border-radius:999px;'
            f'font-weight:700;font-size:0.76rem;letter-spacing:0.04em;">{badge_text}</span> '
            f'<strong style="margin-left:8px;color:#0f172a;">{n_done_pairs} / {n_total_pairs} Subject Pairs '
            f'({pct_done:.1f}%)</strong> '
            f'<span style="color:#475569;font-size:0.88rem;">({n_done_intra} intra-site, {n_done_inter} inter-site)'
            f'{seed_str}{eta_str}</span></div>'
            f'<div style="font-size:0.84rem;color:#475569;">Elapsed: <strong>{summary_payload["total_wall_minutes"]:.1f} min</strong></div>'
            '</div>'
            '<div style="width:100%;height:10px;background:#e2e8f0;border-radius:999px;margin-top:10px;overflow:hidden;">'
            f'<div style="width:{min(100.0, max(0.0, pct_done)):.2f}%;height:100%;background:{bar_color};"></div>'
            '</div>'
            '</div>'
        )

    repro_section_html = ""
    if repro_rows:
        repro_section_html = (
            "<h2>4. Track 4 — Reproducibility Gate on <code>mbhard</code> "
            "(Pair 44: Fresh Process &amp; Same Process)</h2>"
            "<p>Measures affine parameter spread <code>max |&Delta;param|</code> and exact physical "
            "LPS brain-voxel correspondence spread <code>||T_0(x) &minus; T_1(x)||_2</code> (mm) "
            "across all foreground brain voxels via <code>syntx.spatial.vox_to_physical</code>.</p>"
            "<table><thead><tr>"
            "<th>Scope</th><th>Track</th><th>Arm</th><th>Sym Dice (Rep 0)</th>"
            "<th>Sym Dice (Rep 1)</th><th>|&Delta;Sym Dice|</th><th>Max |&Delta;Affine Param|</th>"
            "<th>Mean Brain |&Delta;y| (mm)</th><th>Max Brain |&Delta;y| (mm)</th>"
            f"</tr></thead><tbody>{''.join(repro_html_rows)}</tbody></table>"
        )

    page_title = (
        f"syntx vs ants.registration — Mindboggle-101 Benchmark ({n_done_pairs}/{n_total_pairs} Pairs)"
        if is_cohort_mode
        else "syntx vs ants.registration — Fast 3-Subject Canonical Benchmark"
    )
    h1_title = (
        "syntx (TVF, SyNGS, SyN) vs. ants.registration — Mindboggle-101 Cohort Benchmark"
        if is_cohort_mode
        else "syntx vs. ants.registration — Fast 3-Subject Affine &amp; Deformable Benchmark"
    )
    pairs_meta_html = (
        f"<code>{n_done_pairs}/{n_total_pairs}</code> Mindboggle-101 pairs "
        f"({n_done_intra} intra-site, {n_done_inter} inter-site)"
        if is_cohort_mode
        else f"<code>{_html.escape(str(summary_payload['pairs']))}</code> "
        "(Pair 44 <code>mbhard</code> RPI&rarr;LAI, Pair 77 PIR&rarr;LAI, Pair 00 intra PIR)"
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{page_title}</title>
<style>
  body {{
    background: #ffffff;
    color: #1e293b;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.55;
    max-width: 1320px;
    margin: 32px auto;
    padding: 0 24px 48px;
  }}
  h1, h2, h3 {{ color: #0f172a; }}
  h1 {{ border-bottom: 2px solid #0891b2; padding-bottom: 12px; margin-bottom: 10px; }}
  h2 {{ margin-top: 34px; border-left: 4px solid #0891b2; padding-left: 10px; font-size: 1.25rem; }}
  h3 {{ margin-top: 22px; font-size: 1.06rem; color: #0f172a; }}
  .meta {{
    font-size: 0.9rem; color: #475569; background: #f8fafc;
    border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px;
  }}
  .cards {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 14px; margin: 18px 0 28px;
  }}
  .card {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px 16px; background: #ffffff; }}
  .card .label {{
    font-size: 0.78rem; color: #475569; text-transform: uppercase;
    letter-spacing: 0.04em; font-weight: 600;
  }}
  .card .value {{ font-size: 1.5rem; font-weight: 700; margin-top: 4px; }}
  .card .note {{ font-size: 0.82rem; color: #64748b; margin-top: 4px; }}
  .cyan {{ color: #0891b2; font-weight: 700; }}
  .emerald {{ color: #059669; font-weight: 700; }}
  .amber {{ color: #d97706; font-weight: 700; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0 24px; font-size: 0.86rem; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; vertical-align: middle; }}
  th {{ background: #f8fafc; color: #0f172a; font-weight: 600; }}
  tr:hover td {{ background: #f8fafc; }}
  .sub {{ font-size: 0.76rem; color: #64748b; margin-top: 2px; }}
  code {{ background: #f1f5f9; padding: 1px 5px; border-radius: 4px; font-size: 0.85em; }}
  .fig-box {{
    margin: 18px 0 26px; border: 1px solid #cbd5e1; border-radius: 8px;
    padding: 12px; background: #ffffff; text-align: center;
  }}
  .fig-box img {{ max-width: 100%; height: auto; border-radius: 4px; }}
  .fig-box figcaption {{ font-size: 0.85rem; color: #475569; margin-top: 8px; font-weight: 500; }}
</style>
</head>
<body>
<h1>{h1_title}</h1>
{progress_banner_html}
<div class="meta">
  <strong>Generated:</strong> {_html.escape(summary_payload['timestamp_utc'])} &nbsp;|&nbsp;
  <strong>syntx:</strong> <code>v{ver_s}</code> (<code>{sha_s}</code>,
  dirty={summary_payload['git_dirty']}) &nbsp;|&nbsp;
  <strong>Hardware:</strong> {plat_s} ({env_d['cpu_count']} CPU cores,
  MPS={env_d['devices']['mps']}) &nbsp;|&nbsp;
  <strong>Load Avg (1m):</strong> {summary_payload['load_avg_start_1m']:.2f} &rarr;
  {summary_payload['load_avg_end_1m']:.2f}<br/>
  <strong>Protocol:</strong> Isolated sequential subprocess per evaluation (MPS jobs before
  CPU-heavy ANTs C++ jobs; unpinned ITK multi-threading for ANTs) &bull;
  <strong>Pairs:</strong> {pairs_meta_html} &bull;
  <strong>Schedule:</strong> <code>reg_iterations = {iters_s}</code> &bull;
  <strong>Total Wall-Clock:</strong> {summary_payload['total_wall_minutes']:.2f} min
</div>

<div class="cards">
  <div class="card">
    <div class="label">Affine: syntx vs. ANTs</div>
    <div class="value cyan">{aff_d_s:.4f} vs {aff_d_a:.4f}</div>
    <div class="note">{aff_speed:.2f}&times; faster ({aff_t_s:.1f}s vs {aff_t_a:.1f}s), bitwise deterministic</div>
  </div>
  <div class="card">
    <div class="label">Deformable (Shared Affine): Sobolev vs. ANTs</div>
    <div class="value emerald">{sob_d:.4f} ({sob_gain:+.2f}%)</div>
    <div class="note">ANTs SyNOnly: {ants_d_v:.4f} &bull; {sob_speed:.2f}&times; faster
    ({sob_t:.1f}s vs {ants_t:.1f}s)</div>
  </div>
{third_card_html}
  <div class="card">
    <div class="label">Top Deformable Model ({best_def_name})</div>
    <div class="value cyan">{best_def_d:.4f} ({best_def_gain:+.2f}%)</div>
    <div class="note">Wins vs ANTs: {best_def_wins} &bull; Folding: {best_def_fold:.5f}%</div>
  </div>
</div>

<h2>1. Track 1 — Affine Registration Head-to-Head
(<code>ants.registration</code> vs <code>syntx.robust_affine</code>)</h2>
<p>Evaluated on N4-corrected, 2–98% foreground-normalized brain volumes using bidirectional
nearest-neighbor DKT31 Sørensen–Dice. <code>syntx_affine_default</code> exports the SHA-256-verified
canonical affine seed shared across all Track 2 deformable arms.</p>
{_render_summary_table(affine_rows, include_deform=False)}

<h2>2. Track 2 — Deformable Registration: TVF, SyNGS &amp; SyN Regularizer Suite (Shared Canonical Affine)</h2>
<p>All methods start from the exact same per-pair <code>syntx_affine_default</code> transform
(<code>initial_transform=aff_0</code>, single-interpolation composition) at
<code>reg_iterations = {iters_s}</code>. QC verdicts come exclusively from <code>antsxqc</code>
via <code>syntx.registration_metrics.evaluate_registration_qc</code>.</p>
{_render_summary_table(deform_rows, include_deform=True)}
{subgroup_section_html}

{e2e_section_html}

{repro_section_html}

<h2>Visual Anatomical &amp; Quantitative Evidence (<code>syntx.viz</code> Light Theme)</h2>
{figs_html}

<h2>Per-Pair Granular Results &amp; <code>antsxqc</code> Verdicts</h2>
<table>
  <thead>
    <tr>
      <th>Track</th><th>Pair</th><th>Arm</th><th>Sym Dice (Fix / Mov)</th>
      <th>Fold %</th><th>Min |J|</th><th>ICE Mean / P95 (mm)</th><th>Time</th><th>antsxqc</th>
    </tr>
  </thead>
  <tbody>
    {''.join(pair_rows_html)}
  </tbody>
</table>
</body>
</html>
"""
    _atomic_write_text(html_path, html_doc)
    return json_path, md_path, html_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fast 3-subject registration benchmark: syntx vs ants.registration (Affine & Deformable)."
    )
    p.add_argument("--pairs", type=int, nargs="+", default=list(DEFAULT_PAIRS))
    p.add_argument("--reg-iterations", type=int, nargs="+", default=list(DEFAULT_REG_ITERATIONS))
    p.add_argument("--regularizers", type=str, nargs="+", default=list(DEFAULT_REGULARIZERS))
    p.add_argument(
        "--repro-pair",
        type=int,
        default=44,
        help="Pair index for reproducibility & visual panels (44 = mbhard).",
    )
    p.add_argument("--pairs-csv", type=str, default="examples/pairs.csv")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--no-n4", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--out-dir",
        type=str,
        default=f"results/fast_syn_ants_{_dt.date.today().isoformat()}",
    )
    p.add_argument(
        "--report-stem",
        type=str,
        default=f"docs/reports/fast_syn_ants_benchmark_{_dt.date.today().isoformat()}",
    )
    p.add_argument("--force", action="store_true", help="Re-run even if cached per-run JSON exists.")
    p.add_argument("--skip-figures", action="store_true", help="Skip PNG figure rendering.")
    p.add_argument(
        "--with-reproducibility",
        action="store_true",
        help="Opt-in: run repeated rep=1 fresh-process & same-process reproducibility checks (off by default).",
    )
    p.add_argument(
        "--include-syngs-tvf",
        action="store_true",
        help="Opt-in: include syntx.syngs (sobolev) and syntx.tvf (beltrami) deformable arms.",
    )
    p.add_argument(
        "--include-syngs-beltrami",
        action="store_true",
        help="Opt-in: include tuned syntx.syngs (beltrami) deformable arm.",
    )
    p.add_argument(
        "--all-pairs",
        action="store_true",
        help="Evaluate all subject pairs in --pairs-csv (e.g. all 90 Mindboggle-101 pairs).",
    )
    p.add_argument(
        "--shuffle-pairs",
        action="store_true",
        help="Deterministically shuffle subject pair execution order using --seed (persisted in pair_order.json).",
    )
    p.add_argument(
        "--pair-by-pair",
        action="store_true",
        help="Execute MPS arms then CPU arms per subject pair and update JSON/MD/HTML reports live after every pair.",
    )
    p.add_argument(
        "--best-models-only",
        action="store_true",
        help="Evaluate primary affine & deformable models only (skips *_fast, ants_syn_quick, and Track 3 E2E).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast end-to-end smoke gate (counts=2: 2 pairs, [2, 2] iterations, 2 regularizers on CPU; "
            "executes 100%% of stages including syntx.viz figures, JSON, Markdown, and HTML)."
        ),
    )
    p.add_argument("--max-subjects", type=int, default=None, help="Optional cap on number of pairs.")
    p.add_argument("--iters", type=int, nargs="+", default=None, dest="iters_alias")
    p.add_argument(
        "--ants-aff-iterations",
        type=int,
        nargs="+",
        default=None,
        help="Optional override for ANTs affine iterations (used only by fast unit tests).",
    )
    p.add_argument(
        "--syntx-aff-iterations",
        type=int,
        nargs="+",
        default=None,
        help="Optional override for syntx affine iterations (used only by fast unit tests).",
    )

    # Internal isolated worker flags
    p.add_argument("--worker-mode", choices=["single", "same_proc_repro"], default=None)
    p.add_argument("--track", type=str, default=None)
    p.add_argument("--arm", type=str, default=None)
    p.add_argument("--pair-idx", type=int, default=None)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--persist-transforms", action="store_true")
    p.add_argument("--save-regional-dice", action="store_true")
    p.add_argument("--worker-out-json", type=str, default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    default_out_dir = f"results/fast_syn_ants_{_dt.date.today().isoformat()}"
    default_report_stem = f"docs/reports/fast_syn_ants_benchmark_{_dt.date.today().isoformat()}"
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.worker_mode == "single":
        run_worker_single(args)
        return {}
    if args.worker_mode == "same_proc_repro":
        run_worker_same_process_repro(args)
        return {}

    if args.iters_alias is not None:
        args.reg_iterations = list(args.iters_alias)

    smoke_tmp: Optional[tempfile.TemporaryDirectory[str]] = None
    in_process = False
    if args.smoke:
        smoke_tmp = tempfile.TemporaryDirectory(prefix="syntx_fast_bench_smoke_")
        smoke_root = Path(smoke_tmp.name)
        if args.data_dir is None and args.pairs_csv == "examples/pairs.csv":
            s_data_dir, s_pairs_csv = _build_smoke_fixture(smoke_root)
            args.data_dir = str(s_data_dir)
            args.pairs_csv = str(s_pairs_csv)
            args.pairs = [0, 1]
            args.repro_pair = 0
        else:
            args.pairs = list(args.pairs)[:2]
        args.all_pairs = False
        args.reg_iterations = [2, 2]
        args.regularizers = list(args.regularizers)[:2]
        args.ants_aff_iterations = [2, 2]
        args.syntx_aff_iterations = [2, 2]
        args.no_n4 = True
        args.device = "cpu"
        in_process = True
        if args.out_dir == default_out_dir:
            args.out_dir = str(smoke_root / "results")
        if args.report_stem == default_report_stem:
            args.report_stem = str(smoke_root / "reports" / "smoke_report")

    try:
        t_wall0 = time.perf_counter()
        load_start = list(os.getloadavg()) if hasattr(os, "getloadavg") else [float("nan")] * 3
        out_dir = Path(args.out_dir)
        runs_dir = out_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        script_path = os.path.abspath(__file__)

        if args.all_pairs:
            df_all = pd.read_csv(args.pairs_csv)
            pairs = list(range(len(df_all)))
        else:
            pairs = [int(x) for x in args.pairs]

        pair_order_file = out_dir / "pair_order.json"
        if args.shuffle_pairs:
            base_pairs = list(pairs)
            loaded_order = False
            if pair_order_file.exists() and not args.force:
                saved_order = json.loads(pair_order_file.read_text(encoding="utf-8"))
                if (
                    saved_order.get("seed") == int(args.seed)
                    and sorted(saved_order.get("base_pairs", [])) == sorted(base_pairs)
                ):
                    pairs = [int(x) for x in saved_order["shuffled_pairs"]]
                    loaded_order = True
            if not loaded_order:
                rng_p = random.Random(int(args.seed))
                rng_p.shuffle(pairs)
                pair_order_file.write_text(
                    json.dumps(
                        {
                            "seed": int(args.seed),
                            "base_pairs": base_pairs,
                            "shuffled_pairs": pairs,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        if args.max_subjects is not None:
            pairs = pairs[: max(1, int(args.max_subjects))]
        repro_pair = int(args.repro_pair)
        if repro_pair not in pairs:
            repro_pair = pairs[0]

        ok, data_rep = check_mindboggle_data(pairs_csv=args.pairs_csv, data_dir=args.data_dir, verbose=False)
        if not ok:
            missing_requested = {m["pair_idx"] for m in data_rep.get("missing_pairs", [])} & set(pairs)
            if missing_requested or not data_rep.get("pairs_csv_exists") or not data_rep.get("data_dir_exists"):
                raise RuntimeError(
                    f"Mindboggle dataset missing required pairs {sorted(missing_requested)}: {data_rep}"
                )

        affine_arms_mps: Tuple[str, ...] = (
            ("syntx_affine_default",) if args.best_models_only else AFFINE_ARMS_MPS
        )
        affine_arms_cpu: Tuple[str, ...] = (
            ("ants_affine",) if args.best_models_only else AFFINE_ARMS_CPU
        )
        ants_deform_arms: Tuple[str, ...] = (
            ("ants_syn_matched",) if args.best_models_only else ("ants_syn_matched", "ants_syn_quick")
        )
        run_e2e = not args.best_models_only

        syn_arms = [f"syn_{('divcurl' if r == 'div_curl' else r)}" for r in args.regularizers]
        if args.include_syngs_beltrami and "syngs_beltrami" not in syn_arms:
            syn_arms.append("syngs_beltrami")
        if args.include_syngs_tvf:
            for extra_arm in ("syngs_best", "tvf_best"):
                if extra_arm not in syn_arms:
                    syn_arms.append(extra_arm)

        cs = code_state(include_diff=False)
        env_snapshot = environment()
        rendered_4panel_pairs: set[int] = set()

        def _common_flags(pair_i: int, persist: bool = False, regional: bool = False) -> List[str]:
            fl = [
                "--pairs-csv",
                str(args.pairs_csv),
                "--seed",
                str(args.seed),
                "--out-dir",
                str(out_dir),
                "--reg-iterations",
                *[str(x) for x in args.reg_iterations],
            ]
            if args.smoke:
                fl.append("--smoke")
            if args.data_dir:
                fl += ["--data-dir", str(args.data_dir)]
            if args.no_n4:
                fl.append("--no-n4")
            if args.device:
                fl += ["--device", str(args.device)]
            if args.ants_aff_iterations is not None:
                fl += ["--ants-aff-iterations", *[str(x) for x in args.ants_aff_iterations]]
            if args.syntx_aff_iterations is not None:
                fl += ["--syntx-aff-iterations", *[str(x) for x in args.syntx_aff_iterations]]
            if persist:
                fl.append("--persist-transforms")
            if regional:
                fl.append("--save-regional-dice")
            return fl

        all_records: List[Dict[str, Any]] = []
        fresh_repro_rep1: Dict[Tuple[str, str], Dict[str, Any]] = {}
        same_proc_repro_rows: List[Dict[str, Any]] = []
        repro_rows: List[Dict[str, Any]] = []

        def _emit_snapshot(
            completed_n: int,
            total_n: int,
            in_prog: bool,
            eta_minutes: Optional[float],
        ) -> Tuple[str, str, str, Dict[str, Any]]:
            affine_records = [r for r in all_records if r["track"] == "affine"]
            deform_records = [r for r in all_records if r["track"] == "deformable"]
            e2e_records = [r for r in all_records if r["track"] == "e2e"]

            affine_rows = _aggregate_arm_rows(
                affine_records,
                list(affine_arms_mps) + list(affine_arms_cpu),
                baseline_arm="ants_affine",
            )
            deform_rows = _aggregate_arm_rows(
                deform_records,
                syn_arms + list(ants_deform_arms),
                baseline_arm="ants_syn_matched",
            )
            if args.best_models_only or args.all_pairs or args.pair_by_pair:
                deform_rows.sort(key=lambda r: r["mean_dice_sym"], reverse=True)
            e2e_rows = (
                _aggregate_arm_rows(
                    e2e_records,
                    ["syntx_e2e_syn", "ants_e2e_syn"],
                    baseline_arm="ants_e2e_syn",
                )
                if e2e_records
                else []
            )

            figures: Dict[str, str] = {}
            if not args.skip_figures and deform_records:
                active_viz = repro_pair
                has_rep_tx = any(
                    r["pair_idx"] == repro_pair and r["rep"] == 0 and r.get("persisted_transforms")
                    for r in deformable_records_for_viz(deform_records)
                )
                if not has_rep_tx:
                    pers_p = [
                        int(r["pair_idx"])
                        for r in deform_records
                        if r["rep"] == 0 and r.get("persisted_transforms")
                    ]
                    if pers_p:
                        active_viz = pers_p[0]
                force_figs = bool(args.force and active_viz not in rendered_4panel_pairs)
                figures = _render_report_figures(
                    repro_pair=repro_pair,
                    pairs_csv=args.pairs_csv,
                    data_dir=args.data_dir,
                    use_n4=not args.no_n4,
                    deformable_records=deform_records,
                    e2e_records=e2e_records,
                    fig_dir=out_dir / "figures",
                    force=force_figs,
                )
                if any(k.startswith("4panel_") for k in figures):
                    rendered_4panel_pairs.add(active_viz)

            load_cur = list(os.getloadavg()) if hasattr(os, "getloadavg") else [float("nan")] * 3
            total_wall_min = (time.perf_counter() - t_wall0) / 60.0

            compact_records = []
            for r in all_records:
                rc = dict(r)
                prov = dict(rc.get("provenance") or {})
                if "script" in prov and isinstance(prov["script"], dict):
                    prov["script"] = {k: v for k, v in prov["script"].items() if k != "text"}
                if "code" in prov and isinstance(prov["code"], dict) and isinstance(prov["code"].get("git"), dict):
                    git_c = {
                        k: v
                        for k, v in prov["code"]["git"].items()
                        if k not in ("diff", "untracked_package_sources")
                    }
                    prov["code"] = dict(prov["code"], git=git_c)
                rc["provenance"] = prov
                compact_records.append(rc)

            summary_payload: Dict[str, Any] = {
                "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "smoke": bool(args.smoke),
                "in_progress": bool(in_prog),
                "completed_pairs": int(completed_n),
                "total_pairs": int(total_n),
                "eta_minutes": float(eta_minutes) if eta_minutes is not None else None,
                "shuffle_seed": int(args.seed) if args.shuffle_pairs else None,
                "syntx_version": getattr(syntx, "__version__", "unknown"),
                "git_commit": ((cs.get("git") or {}).get("commit") or "unknown"),
                "git_dirty": bool((cs.get("git") or {}).get("dirty")),
                "environment": env_snapshot,
                "pairs": pairs,
                "repro_pair": repro_pair,
                "reg_iterations": [int(x) for x in args.reg_iterations],
                "regularizers": list(args.regularizers),
                "load_avg_start_1m": float(load_start[0]),
                "load_avg_end_1m": float(load_cur[0]),
                "total_wall_minutes": float(total_wall_min),
                "track1_affine_summary": affine_rows,
                "track2_deformable_summary": deform_rows,
                "track3_e2e_summary": e2e_rows,
                "track4_reproducibility": repro_rows,
                "figures": figures,
                "per_run_records": compact_records,
            }

            jp, mp, hp = _write_reports(
                report_stem=Path(args.report_stem),
                summary_payload=summary_payload,
                affine_rows=affine_rows,
                deform_rows=deform_rows,
                e2e_rows=e2e_rows,
                all_records=all_records,
                repro_rows=repro_rows,
                figures=figures,
            )
            return jp, mp, hp, summary_payload

        def deformable_records_for_viz(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return recs

        if args.pair_by_pair:
            print("=" * 88)
            print(
                f"  PAIR-BY-PAIR REGISTRATION BENCHMARK | total_pairs={len(pairs)} "
                f"| shuffle={args.shuffle_pairs} (seed={args.seed}) | iters={args.reg_iterations}"
            )
            print(f"  Deformable arms: {syn_arms + list(ants_deform_arms)}")
            print("=" * 88, flush=True)

            live_pair_minutes: List[float] = []
            json_p = md_p = html_p = ""
            summary_payload: Dict[str, Any] = {}
            records_map: Dict[Tuple[str, int, str, int], Dict[str, Any]] = {}
            preloaded_completed_pairs: set[int] = set()
            if not args.force:
                active_pair_specs: List[Tuple[str, str]] = (
                    [("affine", a) for a in affine_arms_mps]
                    + [("deformable", a) for a in syn_arms]
                    + ([("e2e", "syntx_e2e_syn")] if run_e2e else [])
                    + [("affine", a) for a in affine_arms_cpu]
                    + [("deformable", a) for a in ants_deform_arms]
                    + ([("e2e", "ants_e2e_syn")] if run_e2e else [])
                )
                for p_pre in pairs:
                    ants_sentinel = runs_dir / f"deformable_pair{p_pre:03d}_ants_syn_matched_rep0.json"
                    if not ants_sentinel.exists():
                        continue
                    for trk_pre, arm_pre in active_pair_specs:
                        cand_p = runs_dir / f"{trk_pre}_pair{p_pre:03d}_{arm_pre}_rep0.json"
                        if cand_p.exists():
                            c_rec = json.loads(cand_p.read_text())
                            if c_rec.get("status") == "SUCCESS":
                                records_map[(trk_pre, int(p_pre), arm_pre, 0)] = c_rec
                    preloaded_completed_pairs.add(int(p_pre))
                if records_map:
                    all_records[:] = list(records_map.values())

            def _record_put(rec_item: Dict[str, Any]) -> None:
                records_map[
                    (
                        str(rec_item["track"]),
                        int(rec_item["pair_idx"]),
                        str(rec_item["arm"]),
                        int(rec_item["rep"]),
                    )
                ] = rec_item
                all_records[:] = list(records_map.values())

            for step_idx, p_idx in enumerate(pairs, start=1):
                t_p0 = time.perf_counter()
                is_viz_target = (p_idx == repro_pair) or (step_idx == 1)
                n_cached_in_pair = 0
                print(
                    f"\n--- [PAIR {step_idx:02d}/{len(pairs):02d}] Subject Pair #{p_idx:02d} "
                    f"(MPS -> CPU sequential, live HTML update) ---",
                    flush=True,
                )

                # 1. MPS Arms for Pair p_idx (no concurrent CPU jobs)
                for arm in affine_arms_mps:
                    persist = is_viz_target and arm == "syntx_affine_default"
                    out_p = runs_dir / f"affine_pair{p_idx:03d}_{arm}_rep0.json"
                    if not args.force and out_p.exists():
                        n_cached_in_pair += 1
                    rec = _run_subprocess_job(
                        script_path,
                        [
                            "--worker-mode", "single",
                            "--track", "affine",
                            "--arm", arm,
                            "--pair-idx", str(p_idx),
                            "--rep", "0",
                            *_common_flags(p_idx, persist=persist),
                        ],
                        out_p,
                        args.force,
                        f"[{step_idx}/{len(pairs)} Pair {p_idx:02d}] {arm}",
                        in_process=in_process,
                    )
                    _record_put(rec)

                for arm in syn_arms:
                    out_p = runs_dir / f"deformable_pair{p_idx:03d}_{arm}_rep0.json"
                    if not args.force and out_p.exists():
                        n_cached_in_pair += 1
                    rec = _run_subprocess_job(
                        script_path,
                        [
                            "--worker-mode", "single",
                            "--track", "deformable",
                            "--arm", arm,
                            "--pair-idx", str(p_idx),
                            "--rep", "0",
                            *_common_flags(p_idx, persist=is_viz_target, regional=is_viz_target),
                        ],
                        out_p,
                        args.force,
                        f"[{step_idx}/{len(pairs)} Pair {p_idx:02d}] {arm}",
                        in_process=in_process,
                    )
                    _record_put(rec)

                if run_e2e:
                    out_p = runs_dir / f"e2e_pair{p_idx:03d}_syntx_e2e_syn_rep0.json"
                    if not args.force and out_p.exists():
                        n_cached_in_pair += 1
                    rec = _run_subprocess_job(
                        script_path,
                        [
                            "--worker-mode", "single",
                            "--track", "e2e",
                            "--arm", "syntx_e2e_syn",
                            "--pair-idx", str(p_idx),
                            "--rep", "0",
                            *_common_flags(p_idx, persist=is_viz_target),
                        ],
                        out_p,
                        args.force,
                        f"[{step_idx}/{len(pairs)} Pair {p_idx:02d}] syntx_e2e_syn",
                        in_process=in_process,
                    )
                    _record_put(rec)

                # 2. Multi-Core CPU C++ Arms for Pair p_idx (after MPS subprocesses exited)
                for arm in affine_arms_cpu:
                    persist = is_viz_target and arm == "ants_affine"
                    out_p = runs_dir / f"affine_pair{p_idx:03d}_{arm}_rep0.json"
                    if not args.force and out_p.exists():
                        n_cached_in_pair += 1
                    rec = _run_subprocess_job(
                        script_path,
                        [
                            "--worker-mode", "single",
                            "--track", "affine",
                            "--arm", arm,
                            "--pair-idx", str(p_idx),
                            "--rep", "0",
                            *_common_flags(p_idx, persist=persist),
                        ],
                        out_p,
                        args.force,
                        f"[{step_idx}/{len(pairs)} Pair {p_idx:02d}] {arm}",
                        in_process=in_process,
                    )
                    _record_put(rec)

                for arm in ants_deform_arms:
                    out_p = runs_dir / f"deformable_pair{p_idx:03d}_{arm}_rep0.json"
                    if not args.force and out_p.exists():
                        n_cached_in_pair += 1
                    rec = _run_subprocess_job(
                        script_path,
                        [
                            "--worker-mode", "single",
                            "--track", "deformable",
                            "--arm", arm,
                            "--pair-idx", str(p_idx),
                            "--rep", "0",
                            *_common_flags(p_idx, persist=is_viz_target, regional=is_viz_target),
                        ],
                        out_p,
                        args.force,
                        f"[{step_idx}/{len(pairs)} Pair {p_idx:02d}] {arm}",
                        in_process=in_process,
                    )
                    _record_put(rec)

                if run_e2e:
                    out_p = runs_dir / f"e2e_pair{p_idx:03d}_ants_e2e_syn_rep0.json"
                    if not args.force and out_p.exists():
                        n_cached_in_pair += 1
                    rec = _run_subprocess_job(
                        script_path,
                        [
                            "--worker-mode", "single",
                            "--track", "e2e",
                            "--arm", "ants_e2e_syn",
                            "--pair-idx", str(p_idx),
                            "--rep", "0",
                            *_common_flags(p_idx, persist=is_viz_target),
                        ],
                        out_p,
                        args.force,
                        f"[{step_idx}/{len(pairs)} Pair {p_idx:02d}] ants_e2e_syn",
                        in_process=in_process,
                    )
                    _record_put(rec)

                preloaded_completed_pairs.add(int(p_idx))
                effective_done = len(preloaded_completed_pairs)

                dt_pair_min = (time.perf_counter() - t_p0) / 60.0
                if n_cached_in_pair == 0 and dt_pair_min > (0.03 if not in_process else 0.0):
                    live_pair_minutes.append(dt_pair_min)

                rem = len(pairs) - effective_done
                if rem > 0 or step_idx < len(pairs):
                    if live_pair_minutes:
                        avg_m = float(statistics.mean(live_pair_minutes))
                    else:
                        pair_solver_s: Dict[int, float] = {}
                        for r_done in all_records:
                            if r_done.get("rep") == 0 and "runtime_seconds" in r_done:
                                pair_solver_s[int(r_done["pair_idx"])] = (
                                    pair_solver_s.get(int(r_done["pair_idx"]), 0.0)
                                    + float(r_done["runtime_seconds"])
                                )
                        overhead_mult = 1.0 if in_process else 1.34
                        avg_m = (
                            (float(statistics.mean(pair_solver_s.values())) * overhead_mult / 60.0)
                            if pair_solver_s
                            else ((time.perf_counter() - t_wall0) / 60.0) / max(1, step_idx)
                        )
                    backfill_left = max(0, effective_done - step_idx)
                    eta_m: Optional[float] = avg_m * rem + backfill_left * (1.4 if not in_process else 0.0)
                else:
                    eta_m = 0.0

                json_p, md_p, html_p, summary_payload = _emit_snapshot(
                    completed_n=effective_done,
                    total_n=len(pairs),
                    in_prog=(step_idx < len(pairs)),
                    eta_minutes=eta_m,
                )
                top_row = (
                    summary_payload["track2_deformable_summary"][0]
                    if summary_payload["track2_deformable_summary"]
                    else {}
                )
                eta_log = f"{eta_m:.1f}m" if eta_m is not None else "0.0m"
                print(
                    f"  [LIVE REPORT {step_idx:02d}/{len(pairs):02d}] "
                    f"Leader={top_row.get('arm', '—')} ({top_row.get('mean_dice_sym', float('nan')):.4f}, "
                    f"{top_row.get('mean_delta_vs_ants_pp', 0.0):+.2f}pp vs ANTs) | "
                    f"ETA={eta_log} -> {os.path.abspath(html_p)}",
                    flush=True,
                )
        else:
            print("=" * 88)
            print(
                f"  FAST 3-SUBJECT REGISTRATION BENCHMARK | pairs={pairs} | reg_iterations={args.reg_iterations}"
            )
            print("  Phase A: MPS Registrations (syntx affine, deformable regularizers, end-to-end)")
            print("=" * 88, flush=True)

            # --- PHASE A (MPS FIRST, ONE SUBPROCESS AT A TIME — GEMINI.md §4) ---
            # A1. Track 1 MPS Affines (creates & locks shared aff_0 per pair)
            for p_idx in pairs:
                for arm in affine_arms_mps:
                    persist = p_idx == repro_pair and arm == "syntx_affine_default"
                    rec = _run_subprocess_job(
                        script_path,
                        [
                            "--worker-mode", "single",
                            "--track", "affine",
                            "--arm", arm,
                            "--pair-idx", str(p_idx),
                            "--rep", "0",
                            *_common_flags(p_idx, persist=persist),
                        ],
                        runs_dir / f"affine_pair{p_idx:03d}_{arm}_rep0.json",
                        args.force,
                        f"Track1 [Pair {p_idx:02d}] {arm}",
                        in_process=in_process,
                    )
                    all_records.append(rec)

            # A2. Track 2 MPS Deformable Regularizers (shared aff_0)
            for p_idx in pairs:
                for arm in syn_arms:
                    is_repro = p_idx == repro_pair
                    rec = _run_subprocess_job(
                        script_path,
                        [
                            "--worker-mode", "single",
                            "--track", "deformable",
                            "--arm", arm,
                            "--pair-idx", str(p_idx),
                            "--rep", "0",
                            *_common_flags(p_idx, persist=is_repro, regional=is_repro),
                        ],
                        runs_dir / f"deformable_pair{p_idx:03d}_{arm}_rep0.json",
                        args.force,
                        f"Track2 [Pair {p_idx:02d}] {arm}",
                        in_process=in_process,
                    )
                    all_records.append(rec)

            # A3. Track 3 MPS End-to-End
            if run_e2e:
                for p_idx in pairs:
                    is_repro = p_idx == repro_pair
                    rec = _run_subprocess_job(
                        script_path,
                        [
                            "--worker-mode", "single",
                            "--track", "e2e",
                            "--arm", "syntx_e2e_syn",
                            "--pair-idx", str(p_idx),
                            "--rep", "0",
                            *_common_flags(p_idx, persist=is_repro),
                        ],
                        runs_dir / f"e2e_pair{p_idx:03d}_syntx_e2e_syn_rep0.json",
                        args.force,
                        f"Track3 [Pair {p_idx:02d}] syntx_e2e_syn",
                        in_process=in_process,
                    )
                    all_records.append(rec)

            # A4. Optional MPS Reproducibility on repro_pair (only when --with-reproducibility is set)
            if args.with_reproducibility:
                mps_repro_targets: List[Tuple[str, str]] = [("affine", "syntx_affine_default")]
                if "syn_sobolev" in syn_arms:
                    mps_repro_targets.append(("deformable", "syn_sobolev"))
                if run_e2e:
                    mps_repro_targets.append(("e2e", "syntx_e2e_syn"))

                for trk, arm in mps_repro_targets:
                    r1 = _run_subprocess_job(
                        script_path,
                        [
                            "--worker-mode", "single",
                            "--track", trk,
                            "--arm", arm,
                            "--pair-idx", str(repro_pair),
                            "--rep", "1",
                            *_common_flags(repro_pair, persist=True),
                        ],
                        runs_dir / f"{trk}_pair{repro_pair:03d}_{arm}_rep1.json",
                        args.force,
                        f"Track4-Fresh [Pair {repro_pair:02d}] {arm} (rep=1)",
                        in_process=in_process,
                    )
                    fresh_repro_rep1[(trk, arm)] = r1

                for trk, arm in [("affine", "syntx_affine_default")] + (
                    [("deformable", "syn_sobolev")] if "syn_sobolev" in syn_arms else []
                ):
                    sp = _run_subprocess_job(
                        script_path,
                        [
                            "--worker-mode", "same_proc_repro",
                            "--track", trk,
                            "--arm", arm,
                            "--pair-idx", str(repro_pair),
                            *_common_flags(repro_pair),
                        ],
                        runs_dir / f"sameproc_{trk}_pair{repro_pair:03d}_{arm}.json",
                        args.force,
                        f"Track4-SameProc [Pair {repro_pair:02d}] {arm}",
                        in_process=in_process,
                    )
                    same_proc_repro_rows.append({k: v for k, v in sp.items() if k != "runs"})

            print("\n" + "=" * 88)
            print("  Phase B: Multi-Core CPU C++ ants.registration Baselines (Affine, Deformable, E2E)")
            print("=" * 88, flush=True)

            # --- PHASE B (CPU ANTs C++, UNPINNED MULTI-THREADING) ---
            # B1. Track 1 ANTs Affines
            for p_idx in pairs:
                for arm in affine_arms_cpu:
                    persist = p_idx == repro_pair and arm == "ants_affine"
                    rec = _run_subprocess_job(
                        script_path,
                        [
                            "--worker-mode", "single",
                            "--track", "affine",
                            "--arm", arm,
                            "--pair-idx", str(p_idx),
                            "--rep", "0",
                            *_common_flags(p_idx, persist=persist),
                        ],
                        runs_dir / f"affine_pair{p_idx:03d}_{arm}_rep0.json",
                        args.force,
                        f"Track1 [Pair {p_idx:02d}] {arm}",
                        in_process=in_process,
                    )
                    all_records.append(rec)

            # B2. Track 2 ANTs Deformable on Shared aff_0
            for p_idx in pairs:
                for arm in ants_deform_arms:
                    is_repro = p_idx == repro_pair
                    rec = _run_subprocess_job(
                        script_path,
                        [
                            "--worker-mode", "single",
                            "--track", "deformable",
                            "--arm", arm,
                            "--pair-idx", str(p_idx),
                            "--rep", "0",
                            *_common_flags(p_idx, persist=is_repro, regional=is_repro),
                        ],
                        runs_dir / f"deformable_pair{p_idx:03d}_{arm}_rep0.json",
                        args.force,
                        f"Track2 [Pair {p_idx:02d}] {arm}",
                        in_process=in_process,
                    )
                    all_records.append(rec)

            # B3. Track 3 ANTs End-to-End
            if run_e2e:
                for p_idx in pairs:
                    is_repro = p_idx == repro_pair
                    rec = _run_subprocess_job(
                        script_path,
                        [
                            "--worker-mode", "single",
                            "--track", "e2e",
                            "--arm", "ants_e2e_syn",
                            "--pair-idx", str(p_idx),
                            "--rep", "0",
                            *_common_flags(p_idx, persist=is_repro),
                        ],
                        runs_dir / f"e2e_pair{p_idx:03d}_ants_e2e_syn_rep0.json",
                        args.force,
                        f"Track3 [Pair {p_idx:02d}] ants_e2e_syn",
                        in_process=in_process,
                    )
                    all_records.append(rec)

            # B4. Optional ANTs Fresh-Process Reproducibility (rep=1) on repro_pair
            if args.with_reproducibility:
                cpu_repro_targets: List[Tuple[str, str]] = [
                    ("affine", "ants_affine"),
                    ("deformable", "ants_syn_matched"),
                ]
                if run_e2e:
                    cpu_repro_targets.append(("e2e", "ants_e2e_syn"))
                for trk, arm in cpu_repro_targets:
                    r1 = _run_subprocess_job(
                        script_path,
                        [
                            "--worker-mode", "single",
                            "--track", trk,
                            "--arm", arm,
                            "--pair-idx", str(repro_pair),
                            "--rep", "1",
                            *_common_flags(repro_pair, persist=True),
                        ],
                        runs_dir / f"{trk}_pair{repro_pair:03d}_{arm}_rep1.json",
                        args.force,
                        f"Track4-Fresh [Pair {repro_pair:02d}] {arm} (rep=1)",
                        in_process=in_process,
                    )
                    fresh_repro_rep1[(trk, arm)] = r1

                repro_pair_data = load_mindboggle_pair(
                    pair_idx=repro_pair,
                    pairs_csv=args.pairs_csv,
                    data_dir=args.data_dir,
                    use_n4=not args.no_n4,
                )
                fi_repro = normalize_intensity(repro_pair_data["fixed"])
                rep0_lookup = {
                    (r["track"], r["arm"]): r
                    for r in all_records
                    if r["pair_idx"] == repro_pair and r["rep"] == 0
                }
                for key in [
                    ("affine", "syntx_affine_default"),
                    ("affine", "ants_affine"),
                    ("deformable", "syn_sobolev"),
                    ("deformable", "ants_syn_matched"),
                    ("e2e", "syntx_e2e_syn"),
                    ("e2e", "ants_e2e_syn"),
                ]:
                    if key in rep0_lookup and key in fresh_repro_rep1:
                        repro_rows.append(
                            _compute_fresh_process_repro_row(
                                fi_repro, rep0_lookup[key], fresh_repro_rep1[key]
                            )
                        )
                repro_rows.extend(same_proc_repro_rows)

            if not args.skip_figures:
                print("\n  [REPORT] Rendering syntx.viz light-theme figures ...", flush=True)
            json_p, md_p, html_p, summary_payload = _emit_snapshot(
                completed_n=len(pairs),
                total_n=len(pairs),
                in_prog=False,
                eta_minutes=0.0,
            )

        total_wall_min = summary_payload["total_wall_minutes"]
        print("\n" + "=" * 88)
        print(f"  BENCHMARK COMPLETE in {total_wall_min:.2f} minutes")
        print(f"  HTML Report : {os.path.abspath(html_p)}")
        print(f"  Markdown    : {os.path.abspath(md_p)}")
        print(f"  Summary JSON: {os.path.abspath(json_p)}")
        print("=" * 88, flush=True)
        return {
            "html_path": os.path.abspath(html_p),
            "md_path": os.path.abspath(md_p),
            "json_path": os.path.abspath(json_p),
            "summary": summary_payload,
        }
    finally:
        if smoke_tmp is not None:
            smoke_tmp.cleanup()


if __name__ == "__main__":
    main()
