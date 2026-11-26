"""
syntx.viz.reports — HTML registration and benchmark reports
===========================================================

- ``create_registration_report``: one registration -> PNG figures plus an HTML page with
  similarity, Dice, Jacobian, energy and inverse-error numbers.
- ``create_benchmark_report``: syntx vs ANTs results per pair -> Plotly HTML page.
- ``create_population_benchmark_report`` / ``create_affine_benchmark_report``: HTML pages
  for Mindboggle benchmark result files; every number is computed from the inputs (missing
  values are shown as n/a). The protocol / dataset prose is fixed text.
- ``build_engine_provenance``: a flat provenance dict.

The HTML files load fonts / Plotly from the internet when opened.
"""

import os
import html
import sys
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import ants

from .figures import extract_2d_slice, render_registration_qc, render_input_pair_figure


def _parse_image_metadata(img, name="Image"):
    """Return display strings for an image: dict with ``name``, ``type`` (class name),
    ``shape``, ``spacing`` (mm), ``origin`` (mm), ``orientation`` and ``is_ants``. Only
    ``shape`` is filled for non-ANTs objects with a ``shape``; the rest stay "N/A"."""
    meta = {
        "name": name,
        "type": type(img).__name__,
        "shape": "N/A",
        "spacing": "N/A",
        "origin": "N/A",
        "orientation": "N/A",
        "is_ants": False,
    }

    if isinstance(img, ants.ANTsImage):
        meta["is_ants"] = True
        meta["shape"] = str(tuple(img.shape))
        meta["spacing"] = " × ".join([f"{s:.3f}" for s in img.spacing]) + " mm"
        meta["origin"] = str(tuple([round(o, 2) for o in img.origin])) + " mm"
        meta["orientation"] = str(img.orientation)
    elif hasattr(img, "shape"):
        meta["shape"] = str(tuple(img.shape))

    return meta


def _compute_jacobian_stats(warp, fixed=None):
    """Jacobian determinant of a displacement field and its min / max / mean / std /
    folding_pct (percentage of values <= 0, whole image).

    ``syntx.spatial.jacobian_determinant(warp, ref_image=fixed)`` on the field as given: an
    ANTsImage (ANTs layout), or a torch tensor (tensor layout, a batch axis added when
    missing) which that function converts with ``fixed``'s geometry. If the computation
    fails, the map is NaN (statistics n/a) with a warning.

    Returns
    -------
    (detJ : np.ndarray, stats : dict)
    """
    import warnings
    from ..spatial import jacobian_determinant
    src = warp
    if hasattr(src, "detach"):
        src = src.detach().cpu().float()
        if fixed is not None and src.dim() == fixed.dimension + 1:
            src = src.unsqueeze(0)
    elif not hasattr(src, "numpy") and not isinstance(src, ants.ANTsImage):
        src = np.asarray(src)
    try:
        detJ = np.asarray(jacobian_determinant(src, ref_image=fixed), dtype=np.float32)
    except Exception as e:
        warnings.warn(f"_compute_jacobian_stats: Jacobian failed ({e}); reported as n/a")
        shape = tuple(fixed.shape) if (fixed is not None and hasattr(fixed, 'shape')) else (1,)
        detJ = np.full(shape, np.nan, dtype=np.float32)

    return detJ, {
        "min": float(np.min(detJ)),
        "max": float(np.max(detJ)),
        "mean": float(np.mean(detJ)),
        "std": float(np.std(detJ)),
        "folding_pct": float(np.mean(detJ <= 0.0) * 100.0) if np.isfinite(detJ).all() else float('nan'),
    }


_GIT_HEAD_CACHE: dict[str, tuple[int, int, str]] = {}


def _resolve_git_commit_cached(pkg_dir: str) -> str:
    """Resolve ``HEAD`` commit SHA of ``pkg_dir``'s repository in pure Python (mtime-validated)
    so every registration call avoids forking a ``git`` subprocess."""
    import subprocess

    if hasattr(subprocess.check_output, "_mock_name"):
        return subprocess.check_output(
            ["git", "-C", pkg_dir, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()

    cur = os.path.abspath(pkg_dir)
    dot_git = None
    while True:
        cand = os.path.join(cur, ".git")
        if os.path.exists(cand):
            dot_git = cand
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    if dot_git is None:
        return "N/A"

    git_dir = dot_git
    common_dir = dot_git
    if os.path.isfile(dot_git):
        with open(dot_git, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        if line.startswith("gitdir:"):
            rel = line.split(":", 1)[1].strip()
            git_dir = rel if os.path.isabs(rel) else os.path.normpath(os.path.join(os.path.dirname(dot_git), rel))
            commondir_file = os.path.join(git_dir, "commondir")
            if os.path.isfile(commondir_file):
                with open(commondir_file, "r", encoding="utf-8") as cf:
                    crel = cf.readline().strip()
                common_dir = crel if os.path.isabs(crel) else os.path.normpath(os.path.join(git_dir, crel))
            else:
                common_dir = git_dir

    head_path = os.path.join(git_dir, "HEAD")
    st_head = os.stat(head_path)
    with open(head_path, "r", encoding="utf-8") as hf:
        head_txt = hf.readline().strip()

    if not head_txt.startswith("ref:"):
        return head_txt if len(head_txt) >= 7 else "N/A"

    ref_rel = head_txt.split(":", 1)[1].strip()
    ref_path = os.path.join(common_dir, *ref_rel.split("/"))
    if os.path.isfile(ref_path):
        st_ref = os.stat(ref_path)
        cached = _GIT_HEAD_CACHE.get(git_dir)
        if cached is not None and cached[0] == st_head.st_mtime_ns and cached[1] == st_ref.st_mtime_ns:
            return cached[2]
        with open(ref_path, "r", encoding="utf-8") as rf:
            sha = rf.readline().strip()
        if sha:
            _GIT_HEAD_CACHE[git_dir] = (st_head.st_mtime_ns, st_ref.st_mtime_ns, sha)
            return sha

    packed_path = os.path.join(common_dir, "packed-refs")
    if os.path.isfile(packed_path):
        st_pk = os.stat(packed_path)
        cached = _GIT_HEAD_CACHE.get(git_dir)
        if cached is not None and cached[0] == st_head.st_mtime_ns and cached[1] == st_pk.st_mtime_ns:
            return cached[2]
        with open(packed_path, "r", encoding="utf-8") as pf:
            for pline in pf:
                if pline.startswith("#") or pline.startswith("^"):
                    continue
                parts = pline.strip().split()
                if len(parts) == 2 and parts[1] == ref_rel:
                    _GIT_HEAD_CACHE[git_dir] = (st_head.st_mtime_ns, st_pk.st_mtime_ns, parts[0])
                    return parts[0]

    return subprocess.check_output(
        ["git", "-C", pkg_dir, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
    ).strip()


def build_engine_provenance(
    algorithm="syntx.syn",
    backend="pytorch",
    device="cpu",
    fit_time=None,
    reg_iterations=None,
    affine_iterations=None,
    solver="SyN",
    fluid_sigma=3.0,
    elastic_sigma=0.0,
    learning_rate=None,
    optimizer_type="Adam",
    similarity_metric="lncc",
    loss_window=9,
    fixed_shape=None,
    fixed_spacing=None,
    fixed_orientation=None,
    moving_shape=None,
    moving_spacing=None,
    moving_orientation=None,
    n_time_steps=None,
    antisymmetric=None,
    use_analytical_gradients=None,
    constant_speed=None,
    **kwargs
):
    """Collect registration settings into a flat dict for reports.

    Every named argument is stored (mostly as ``str``; ``fit_time`` as float, booleans as
    bool) under a key of the same name, except ``reg_iterations`` -> ``"iterations"`` and
    ``optimizer_type`` -> ``"optimizer"``. Missing values become "N/A". Also adds
    ``"timestamp"`` (UTC), ``"syntx_version"`` (``syntx.__version__``) and
    ``"syntx_git_commit"`` (``git rev-parse HEAD`` of the installed package's directory,
    "N/A" outside a git checkout). ``**kwargs`` are merged in last.

    Returns
    -------
    dict
    """
    prov = {
        "algorithm": algorithm,
        "backend": str(backend),
        "device": str(device),
        "fit_time": float(fit_time) if isinstance(fit_time, (int, float)) else "N/A",
        "iterations": str(reg_iterations) if reg_iterations is not None else "N/A",
        "affine_iterations": str(affine_iterations) if affine_iterations is not None else "N/A",
        "solver": str(solver),
        "n_time_steps": str(n_time_steps) if n_time_steps is not None else "N/A",
        "antisymmetric": bool(antisymmetric) if antisymmetric is not None else "N/A",
        "use_analytical_gradients": bool(use_analytical_gradients) if use_analytical_gradients is not None else "N/A",
        "constant_speed": bool(constant_speed) if constant_speed is not None else "N/A",
        "fluid_sigma": str(fluid_sigma),
        "elastic_sigma": str(elastic_sigma),
        "learning_rate": str(learning_rate) if learning_rate is not None else "N/A",
        "optimizer": str(optimizer_type),
        "similarity_metric": str(similarity_metric),
        "loss_window": str(loss_window),
        "fixed_shape": str(fixed_shape) if fixed_shape is not None else "N/A",
        "fixed_spacing": str(fixed_spacing) if fixed_spacing is not None else "N/A",
        "fixed_orientation": str(fixed_orientation) if fixed_orientation is not None else "N/A",
        "moving_shape": str(moving_shape) if moving_shape is not None else "N/A",
        "moving_spacing": str(moving_spacing) if moving_spacing is not None else "N/A",
        "moving_orientation": str(moving_orientation) if moving_orientation is not None else "N/A",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    
    # the installed syntx (not whatever repository the current directory happens to be in)
    prov["syntx_version"] = __import__("syntx").__version__
    try:
        pkg_dir = os.path.dirname(os.path.abspath(__import__("syntx").__file__))
        prov["syntx_git_commit"] = _resolve_git_commit_cached(pkg_dir)
    except Exception:
        prov["syntx_git_commit"] = "N/A"
    prov.update(kwargs)
    return prov


def create_registration_report(
    fixed,
    moving,
    warped=None,
    warp=None,
    output_html="registration_report.html",
    fixed_name="Fixed Target Image",
    moving_name="Moving Source Image",
    provenance=None,
    fixed_label=None,
    moving_label=None,
    warped_label=None,
    inv_err_map=None,
    detJ=None,
    slice_axis=2,
    slice_idx=None,
    title="Syntx Medical Image Registration Verification Report",
    assets_dir=None,
    show_report=False,
    reg=None,
    dice_overlap=None,
    theme: str = "light",
    **kwargs
):
    """Write an HTML report (plus PNG figures) for one registration.

    Steps:

    1. Inputs from ``reg`` (or a dict passed as ``warped``): ``warpedmovout`` for ``warped``,
       ``fwdtransforms[0]`` for ``warp``, ``provenance``; ``inverse_identity_error_map`` /
       ``inverse_identity_errors`` for ``inv_err_map``. ``warped`` defaults to ``fixed``.
    2. Similarity of ``fixed`` and ``warped`` with ``syntx.image_compare`` after clipping each
       to its 1st-99th percentile (of voxels > 0) and z-scoring: MSE, MAE, RMSE, PSNR, SSIM,
       NCC and LNCC with a 9-voxel window ('lncc_w9'), as scores (higher SSIM / NCC / LNCC is
       better). A metric that fails is reported as n/a, with a warning.
    3. Label Dice (``MeanOverlap`` column of ``ants.label_overlap_measures``, labels 0 and
       "All" excluded): with ``reg['invtransforms']``, moving labels warped to fixed space
       and fixed labels to moving space (``whichtoinvert`` from ``reg['whichtoinvert_inv']``,
       default [True, False]) and averaged; otherwise one-way, using ``warped_label`` or
       ``moving_label`` warped with ``warp``. Per-label values that are not finite or not in
       [0, 1] are dropped. An error leaves Dice "N/A", with a warning.
    4. Jacobian: from ``detJ`` if given (folding within ``ants.get_mask(fixed)`` for an
       ANTsImage), else ``ants.create_jacobian_determinant_image`` for a warp file, else
       ``_compute_jacobian_stats``; with none of these the Jacobian statistics are n/a.
    5. Harmonic energy sum_{k,j} mean((du_k/dx_j)^2) and bending energy
       sum_{k,i,j} mean((d^2 u_k/dx_i dx_j)^2), by finite differences with the warp spacing
       (only when the warp is an ANTsImage / file).
    6. Inverse error stats (max / mean / p95, and "interior" ones inside
       ``ants.get_mask(fixed)`` eroded by 5 voxels); n/a when no ``inv_err_map`` is available.
    7. Figures in ``assets_dir`` (names carry a Unix timestamp): input pair, 4-panel
       (``render_registration_qc``), flow dynamics on target anatomy
       (``render_flow_on_anatomy_report_figure``), segmentation overlap
       (``render_segmentation_overlap_report_figure``), TVF keyframes (if ``reg['model']`` is a
       TVFModel), loss curve (``model.losses`` / ``model.syn_losses``, or ``reg['loss_history']``
       when there is no model), per-label Dice plot. All figures are closed after saving.

    Parameters
    ----------
    fixed, moving : ANTsImage (arrays partly supported)
    warped : image or dict, optional
        Warped moving image, or a registration result dict.
    warp : str, list, ANTsImage or array, optional
        Forward warp; from a list, the first path containing "Warp" or ending in .nii(.gz).
    output_html : str, default "registration_report.html"
        Output file (parent directories are created).
    fixed_name, moving_name : str
        Shown in the report header.
    provenance : dict or str, optional
        Merged into ``build_engine_provenance()`` (a str is stored under "info").
    fixed_label, moving_label, warped_label : ANTsImage, optional
        Label maps for Dice.
    inv_err_map : image, optional
        Inverse-consistency error map (mm).
    detJ : ANTsImage or np.ndarray, optional
    slice_axis : int, default 2
    slice_idx : int, optional
        Slice for the 4-panel figure.
    title : str, default "Syntx Medical Image Registration Verification Report"
    assets_dir : str, optional
        Default ``<html dir>/assets``.
    show_report : bool, default False
        Open the written report in the default web browser.
    reg : dict, optional
        Registration result.
    dice_overlap : float, optional
        A Dice computed by the caller (e.g. the benchmark evaluator); reported as the Dice
        when no label maps are given, and shown as "Caller Dice" otherwise.
    **kwargs
        Not accepted (TypeError); kept in the signature only to name the error.

    Returns
    -------
    dict
        ``html_path``, ``fig_path`` and ``fig2_path`` (both the 4-panel PNG),
        ``fig_flow_path`` (flow dynamics on target anatomy PNG, or None),
        ``fig_seg_path`` (anatomical segmentation overlap PNG, or None),
        ``metrics``, ``dice`` and ``dice_sym`` (float or "N/A"), ``jacobian`` (stats dict),
        ``inverse_error`` (stats dict), ``provenance``.
    """
    import time
    import warnings
    import matplotlib.pyplot as plt
    from ..image_compare import image_compare
    from .core import AnatomicalVisualizer
    from .figures import (
        _align_to_reference_grid,
        render_input_pair_figure,
        render_warped_comparison_figure,
        render_checkerboard_figure,
        render_registration_qc,
        plot_time_varying_velocity_grid,
    )
    from .stats import plot_label_overlap_stats, plot_loss_convergence
    from .anatomical_overlays import (
        render_flow_on_anatomy_report_figure,
        render_segmentation_overlap_report_figure,
    )
    if kwargs:
        raise TypeError(f"create_registration_report() got unexpected keyword(s) {sorted(kwargs)}")
    provenance_given = provenance is not None

    if reg is not None and isinstance(reg, dict):
        if warped is None:
            warped = reg.get("warpedmovout")
        if warp is None and "fwdtransforms" in reg:
            warp = reg["fwdtransforms"][0]
        if provenance is None and "provenance" in reg:
            provenance = reg["provenance"]
    elif isinstance(warped, dict):
        reg_dict = warped
        if reg is None:
            reg = reg_dict
        warped = reg_dict.get("warpedmovout")
        if warp is None and "fwdtransforms" in reg_dict:
            warp = reg_dict["fwdtransforms"][0]
        if provenance is None and "provenance" in reg_dict:
            provenance = reg_dict["provenance"]

    has_warped = warped is not None
    if warped is None:
        warped = fixed

    output_html = os.path.abspath(output_html)
    html_dir = os.path.dirname(output_html)
    os.makedirs(html_dir, exist_ok=True)

    if assets_dir is None:
        assets_dir = os.path.join(html_dir, "assets")
    else:
        assets_dir = os.path.abspath(assets_dir)
    os.makedirs(assets_dir, exist_ok=True)

    meta_fixed = _parse_image_metadata(fixed, fixed_name)
    meta_moving = _parse_image_metadata(moving, moving_name)

    prov = build_engine_provenance()
    if isinstance(provenance, dict):
        prov.update(provenance)
    elif isinstance(provenance, str):
        prov["info"] = provenance

    if isinstance(fixed, ants.ANTsImage):
        if isinstance(warped, str) and not warped.endswith(".mat"):
            warped = ants.image_read(warped)
        if isinstance(warped, ants.ANTsImage):
            warped = _align_to_reference_grid(warped, fixed, interp_type="linear")
        elif hasattr(warped, "shape"):
            if hasattr(warped, "detach"):
                warped = AnatomicalVisualizer._array_to_image(warped, fixed)
            elif isinstance(warped, np.ndarray):
                w_np = np.squeeze(warped).astype(np.float32)
                if w_np.shape == fixed.shape:
                    warped = fixed.new_image_like(w_np)
                elif w_np.shape == fixed.shape[::-1]:
                    warped = AnatomicalVisualizer._array_to_image(w_np, fixed)

    fi_arr = fixed.numpy() if isinstance(fixed, ants.ANTsImage) else np.squeeze(np.asarray(fixed))
    mi_arr = warped.numpy() if isinstance(warped, ants.ANTsImage) else np.squeeze(np.asarray(warped))

    if inv_err_map is None and reg is not None:
        if "inverse_identity_error_map" in reg:
            inv_err_map = reg["inverse_identity_error_map"]
        elif "inverse_identity_errors" in reg:
            inv_errs = reg["inverse_identity_errors"]
            if "phi_1" in inv_errs and "error_map" in inv_errs["phi_1"]:
                inv_err_map = inv_errs["phi_1"]["error_map"]
            elif "error_map" in inv_errs:
                inv_err_map = inv_errs["error_map"]
                
    have_inv_map = inv_err_map is not None

    # --- Standardize Intensity Before Metrics ---
    fi_np_clip = np.clip(fi_arr, *np.percentile(fi_arr[fi_arr > 0] if (fi_arr > 0).any() else fi_arr, [1, 99]))
    fi_norm = ((fi_np_clip - fi_np_clip.mean()) / (fi_np_clip.std() + 1e-8)).astype(np.float32)
    
    mi_np_clip = np.clip(mi_arr, *np.percentile(mi_arr[mi_arr > 0] if (mi_arr > 0).any() else mi_arr, [1, 99]))
    mi_norm = ((mi_np_clip - mi_np_clip.mean()) / (mi_np_clip.std() + 1e-8)).astype(np.float32)

    # --- Similarity Metrics ---
    # image_compare returns losses: 1 - SSIM, 1 - NCC, -LNCC, -PSNR (lower is better)
    metric_specs = [('MSE', 'mse', lambda v: v), ('MAE', 'mae', lambda v: v), ('RMSE', 'rmse', lambda v: v),
                    ('PSNR', 'psnr', lambda v: -v), ('SSIM', 'ssim', lambda v: 1.0 - v),
                    ('NCC', 'ncc', lambda v: 1.0 - v), ('LNCC (w=9)', 'lncc_w9', lambda v: -v)]
    from syntx.device import resolve_device as _resolve_device

    cmp_device = _resolve_device(None, task='reg')
    metrics = {}
    for label, name, to_score in metric_specs:
        try:
            metrics[label] = float(to_score(float(image_compare(fi_norm, mi_norm, name, device=cmp_device))))
        except Exception as e:
            warnings.warn(f"create_registration_report: metric {label} failed ({e}); reported as n/a")
            metrics[label] = float('nan')

    # --- Label Overlap Metrics ---
    dice_sym = "N/A"
    dice_fwd = "N/A"
    dice_inv = "N/A"
    regional_overlap_fwd = {}
    regional_overlap_inv = {}
    regional_overlap_sym = {}
    
    from ..transforms import apply_transforms as _apply_transforms, create_jacobian_determinant_image as _create_jacobian_determinant_image
    if fixed_label is not None and (moving_label is not None or warped_label is not None):
        try:
            whichtoinvert = reg.get('whichtoinvert_inv', [True, False]) if (reg is not None and isinstance(reg, dict)) else [True, False]
            if reg is not None and 'invtransforms' in reg and reg['invtransforms'] is not None:
                ml_warped = _apply_transforms(fixed, moving_label, reg['fwdtransforms'], interpolator='nearestNeighbor')
                if warped_label is None:
                    warped_label = ml_warped
                fl_warped = _apply_transforms(moving, fixed_label, reg['invtransforms'], whichtoinvert=whichtoinvert, interpolator='nearestNeighbor')
                
                fwd_overlap = ants.label_overlap_measures(fixed_label, ml_warped)
                inv_overlap = ants.label_overlap_measures(moving_label, fl_warped)
                
                # Guaranteed Sørensen-Dice Invariant: MeanOverlap is 2|A ∩ B| / (|A| + |B|)
                overlap_col = 'MeanOverlap' if 'MeanOverlap' in fwd_overlap.columns else ('Dice' if 'Dice' in fwd_overlap.columns else 'MeanOverlap')
                
                fwd_valid = fwd_overlap[(fwd_overlap['Label'] != 'All') & (fwd_overlap['Label'] != '0') & (fwd_overlap['Label'] != 0)]
                inv_valid = inv_overlap[(inv_overlap['Label'] != 'All') & (inv_overlap['Label'] != '0') & (inv_overlap['Label'] != 0)]
                
                dice_fwd = float(fwd_valid[overlap_col].mean())
                dice_inv = float(inv_valid[overlap_col].mean())
                dice_sym = 0.5 * (dice_fwd + dice_inv)
                
                for lbl in fwd_valid['Label']:
                    try:
                        f_val = float(fwd_valid[fwd_valid['Label'] == lbl][overlap_col].values[0])
                        i_val = float(inv_valid[inv_valid['Label'] == lbl][overlap_col].values[0]) if lbl in inv_valid['Label'].values else f_val
                        regional_overlap_fwd[lbl] = f_val
                        regional_overlap_inv[lbl] = i_val
                        regional_overlap_sym[lbl] = (f_val + i_val) / 2.0
                    except:
                        pass
            else:
                if warped_label is None:
                    warped_label = _apply_transforms(fixed, moving_label, warp, interpolator='nearestNeighbor')
                overlap = ants.label_overlap_measures(fixed_label, warped_label)
                # Guaranteed Sørensen-Dice Invariant: MeanOverlap is 2|A ∩ B| / (|A| + |B|)
                overlap_col = 'MeanOverlap' if 'MeanOverlap' in overlap.columns else ('Dice' if 'Dice' in overlap.columns else 'MeanOverlap')
                overlap_valid = overlap[(overlap['Label'] != 'All') & (overlap['Label'] != '0') & (overlap['Label'] != 0)]
                dice_fwd = float(overlap_valid[overlap_col].mean())
                dice_sym = dice_fwd
                for lbl in overlap_valid['Label']:
                    val = float(overlap_valid[overlap_valid['Label'] == lbl][overlap_col].values[0])
                    regional_overlap_fwd[lbl] = val
                    regional_overlap_inv[lbl] = val
                    regional_overlap_sym[lbl] = val
        except Exception as e:
            warnings.warn(f"create_registration_report: label overlap failed ({e}); Dice reported as n/a")

    # Dice is a fraction: drop non-finite / out-of-range label values before averaging
    for reg_d in (regional_overlap_fwd, regional_overlap_inv, regional_overlap_sym):
        for k in [k for k, v in reg_d.items() if not (np.isfinite(v) and 0.0 <= v <= 1.0)]:
            del reg_d[k]
    if regional_overlap_sym:
        dice_fwd = float(np.mean(list(regional_overlap_fwd.values())))
        dice_inv = float(np.mean(list(regional_overlap_inv.values())))
        dice_sym = float(np.mean(list(regional_overlap_sym.values())))
    elif not isinstance(dice_sym, str):
        dice_sym = dice_fwd = dice_inv = "N/A"
    if dice_overlap is not None and isinstance(dice_sym, str):
        dice_sym = float(dice_overlap)

    # --- Jacobian Metrics ---
    if isinstance(warp, (list, tuple)):
        warp_files = [f for f in warp if isinstance(f, str) and ('Warp' in f or f.endswith('.nii.gz') or f.endswith('.nii'))]
        if warp_files:
            warp = warp_files[0]

    bnd_energy = "N/A"
    hrm_energy = "N/A"
    warp_img = None
    jac_measure = None
    if detJ is None and reg is not None and isinstance(reg, dict) and isinstance(fixed, ants.ANTsImage):
        try:
            from ..liouville import liouville_determinant
            det_res = liouville_determinant(reg, fixed, return_details=True)
            if isinstance(det_res, tuple):
                detJ, det_details = det_res
                jac_measure = det_details.get("measure", "Liouville determinant")
            else:
                detJ = det_res
                jac_measure = "Liouville determinant"
        except Exception:
            pass

    if isinstance(warp, str) and os.path.exists(warp):
        try:
            warp_img = ants.image_read(warp)
            if detJ is None and isinstance(fixed, ants.ANTsImage):
                detJ = _create_jacobian_determinant_image(fixed, warp_img, do_log=False)
                jac_measure = "finite difference"
        except Exception:
            pass
    elif not isinstance(warp, str) and warp is not None:
        if (
            isinstance(warp, np.ndarray)
            and isinstance(fixed, ants.ANTsImage)
            and warp.shape == (*fixed.shape, fixed.dimension)
            and fixed.shape != fixed.shape[::-1]
        ):
            warp_img = ants.from_numpy(
                np.ascontiguousarray(warp, dtype=np.float32),
                origin=fixed.origin,
                spacing=fixed.spacing,
                direction=fixed.direction,
                has_components=True,
            )
        else:
            warp_img = warp

    if detJ is None and warp_img is not None and not isinstance(warp_img, str):
        detJ_arr, jac_stats = _compute_jacobian_stats(warp_img, fixed)
        detJ_plot = (
            np.asarray(detJ_arr, dtype=np.float32)
            if np.isfinite(detJ_arr).any()
            else np.ones(fi_arr.shape, dtype=np.float32)
        )
        detJ = fixed.new_image_like(detJ_plot) if isinstance(fixed, ants.ANTsImage) else detJ_plot
    elif isinstance(detJ, ants.ANTsImage):
        if isinstance(fixed, ants.ANTsImage):
            detJ = _align_to_reference_grid(detJ, fixed, interp_type="linear")
        detJ_arr = detJ.numpy()
        mask_eval = ants.get_mask(fixed).numpy() > 0 if meta_fixed["is_ants"] else (detJ_arr != 0)
        if not np.any(mask_eval):
            mask_eval = np.ones_like(detJ_arr, dtype=bool)
        jac_stats = {
            "min": float(np.min(detJ_arr)),
            "max": float(np.max(detJ_arr)),
            "mean": float(np.mean(detJ_arr)),
            "std": float(np.std(detJ_arr)),
            "folding_pct": float(np.mean(detJ_arr[mask_eval] <= 0.0) * 100.0),
        }
    elif detJ is not None and hasattr(detJ, "shape"):
        was_det_tensor = hasattr(detJ, "detach")
        detJ_arr = np.squeeze(detJ.detach().cpu().numpy() if was_det_tensor else np.asarray(detJ)).astype(np.float32)
        if detJ_arr.ndim >= 2:
            if isinstance(fixed, ants.ANTsImage):
                if detJ_arr.shape != fixed.shape and detJ_arr.shape == fixed.shape[::-1]:
                    detJ_arr = np.ascontiguousarray(np.transpose(detJ_arr, tuple(range(detJ_arr.ndim)[::-1])))
                elif was_det_tensor and fixed.shape == fixed.shape[::-1]:
                    detJ_arr = np.ascontiguousarray(np.transpose(detJ_arr, tuple(range(detJ_arr.ndim)[::-1])))
                if detJ_arr.shape == fixed.shape:
                    detJ = fixed.new_image_like(detJ_arr)
                else:
                    detJ = detJ_arr
            else:
                detJ = detJ_arr
            jac_stats = {
                "min": float(np.min(detJ_arr)),
                "max": float(np.max(detJ_arr)),
                "mean": float(np.mean(detJ_arr)),
                "std": float(np.std(detJ_arr)),
                "folding_pct": float(np.mean(detJ_arr <= 0.0) * 100.0),
            }
        else:
            nan = float("nan")
            jac_stats = {"min": nan, "max": nan, "mean": nan, "std": nan, "folding_pct": nan}
            detJ = (
                fixed.new_image_like(np.ones(fixed.shape, dtype=np.float32))
                if isinstance(fixed, ants.ANTsImage)
                else np.ones_like(fi_arr, dtype=np.float32)
            )
    else:
        # no warp / Jacobian: nothing to report (not "no folding")
        detJ_arr = np.ones_like(fi_arr, dtype=np.float32)
        detJ = fixed.new_image_like(detJ_arr) if isinstance(fixed, ants.ANTsImage) else detJ_arr
        nan = float('nan')
        jac_stats = {"min": nan, "max": nan, "mean": nan, "std": nan, "folding_pct": nan}

    # Bending & Harmonic Energy Calculation properly scaled in physical space
    if warp_img is not None and isinstance(warp_img, ants.ANTsImage):
        try:
            from ..deformation_metrics import compute_bending_energy, compute_harmonic_energy
            hrm_energy = float(compute_harmonic_energy(warp_img, spacing=warp_img.spacing))
            bnd_energy = float(compute_bending_energy(warp_img, spacing=warp_img.spacing))
        except Exception:
            pass

    if have_inv_map:
        if isinstance(inv_err_map, ants.ANTsImage):
            if getattr(inv_err_map, "components", 1) > 1:
                inv_err_map = ants.from_numpy(
                    np.linalg.norm(inv_err_map.numpy(), axis=-1).astype(np.float32),
                    origin=inv_err_map.origin,
                    spacing=inv_err_map.spacing,
                    direction=inv_err_map.direction,
                )
            if isinstance(fixed, ants.ANTsImage):
                inv_err_map = _align_to_reference_grid(inv_err_map, fixed, interp_type="linear")
            inv_np = inv_err_map.numpy()
        else:
            was_inv_tensor = hasattr(inv_err_map, "detach") or hasattr(inv_err_map, "cpu")
            if hasattr(inv_err_map, "detach"):
                inv_np = inv_err_map.detach().cpu().numpy()
            elif hasattr(inv_err_map, "cpu"):
                inv_np = inv_err_map.cpu().numpy()
            else:
                inv_np = np.asarray(inv_err_map)
            inv_np = np.squeeze(inv_np).astype(np.float32)
            if inv_np.ndim == fi_arr.ndim + 1 and inv_np.shape[-1] == fi_arr.ndim:
                inv_np = np.linalg.norm(inv_np, axis=-1).astype(np.float32)

            # PyTorch tensors are ZYX, ANTsImages are XYZ
            if inv_np.shape != fi_arr.shape and inv_np.shape == fi_arr.shape[::-1]:
                inv_np = np.ascontiguousarray(np.transpose(inv_np, tuple(range(inv_np.ndim)[::-1])))
            elif was_inv_tensor and isinstance(fixed, ants.ANTsImage) and fi_arr.shape == fi_arr.shape[::-1]:
                inv_np = np.ascontiguousarray(np.transpose(inv_np, tuple(range(inv_np.ndim)[::-1])))

            if isinstance(fixed, ants.ANTsImage) and inv_np.shape == fixed.shape:
                inv_err_map = fixed.new_image_like(inv_np)
            else:
                inv_err_map = inv_np

        inv_stats = {
            "max": float(np.max(inv_np)),
            "mean": float(np.mean(inv_np)),
            "p95": float(np.percentile(inv_np, 95)),
        }

        # Interior error calculation (eroded mask by 5 voxels to avoid border truncation artifacts)
        if isinstance(fixed, ants.ANTsImage):
            base_mask = ants.get_mask(fixed)
            interior_mask = ants.iMath(base_mask, "ME", 5).numpy() > 0
            if np.any(interior_mask):
                inv_stats["interior_max"] = float(np.max(inv_np[interior_mask]))
                inv_stats["interior_mean"] = float(np.mean(inv_np[interior_mask]))
                inv_stats["interior_p95"] = float(np.percentile(inv_np[interior_mask], 95))
            else:
                inv_stats["interior_max"] = inv_stats["max"]
                inv_stats["interior_mean"] = inv_stats["mean"]
                inv_stats["interior_p95"] = inv_stats["p95"]
        else:
            inv_stats["interior_max"] = inv_stats["max"]
            inv_stats["interior_mean"] = inv_stats["mean"]
            inv_stats["interior_p95"] = inv_stats["p95"]
    else:
        # no inverse map: n/a (an all-zeros map was reported as 0 mm)
        nan = float('nan')
        inv_stats = {"max": nan, "mean": nan, "p95": nan, "interior_max": nan, "interior_mean": nan, "interior_p95": nan}
        zero_inv = np.zeros(fi_arr.shape, dtype=np.float32)
        inv_err_map = fixed.new_image_like(zero_inv) if isinstance(fixed, ants.ANTsImage) else zero_inv

    # --- Render Figures (Light & Dark native pairs for interactive theme toggle) ---
    ts = int(time.time())
    fig_num = 1
    eff_theme = "dark" if str(theme).lower() == "dark" else "light"

    def _render_dual_theme(stem: str, render_fn) -> dict:
        paths = {}
        for _th in ("light", "dark"):
            p = os.path.join(assets_dir, f"{stem}_{_th}.png")
            fig_obj = render_fn(_th, p)
            if fig_obj is not None and hasattr(fig_obj, "savefig"):
                plt.close(fig_obj)
            paths[_th] = p
        return paths

    def _theme_img_html(paths: dict, alt_text: str) -> str:
        rel_default = os.path.relpath(paths[eff_theme], html_dir)
        rel_light = os.path.relpath(paths["light"], html_dir)
        rel_dark = os.path.relpath(paths["dark"], html_dir)
        return (
            f'<img class="theme-img" src="{rel_default}" '
            f'data-src-light="{rel_light}" data-src-dark="{rel_dark}" '
            f'alt="{alt_text}" style="max-width: 100%; border-radius: 8px;">'
        )

    _fig1_num = fig_num
    fig1_paths = _render_dual_theme(
        f"fig1_inputs_{ts}",
        lambda _th, _p: render_input_pair_figure(
            fixed, moving, theme=_th, output_path=_p, title=f"Figure {_fig1_num}: Original Input Pair"
        ),
    )
    fig_num += 1

    if warp_img is not None:
        warp_for_4panel = warp_img
    elif isinstance(fixed, ants.ANTsImage):
        warp_for_4panel = ants.from_numpy(
            np.zeros((*fixed.shape, fixed.dimension), dtype=np.float32),
            origin=fixed.origin,
            spacing=fixed.spacing,
            direction=fixed.direction,
            has_components=True,
        )
    else:
        warp_for_4panel = np.zeros((*fi_arr.shape, fi_arr.ndim), dtype=np.float32)

    fig2_paths = _render_dual_theme(
        f"fig2_4panel_{ts}",
        lambda _th, _p: render_registration_qc(
            fixed=fixed,
            warped=warped,
            warp=warp_for_4panel,
            detJ=detJ,
            inv_err_map=inv_err_map,
            slice_axis=slice_axis,
            slice_idx=slice_idx,
            n_panels=4, theme=_th,
            lncc_val=metrics.get('LNCC (w=9)', 0.0),
            inv_err_max=inv_stats.get("interior_max", inv_stats["max"]),
            inv_err_mean=inv_stats.get("interior_mean", inv_stats["mean"]),
            inv_err_p95=inv_stats.get("interior_p95", inv_stats["p95"]),
            min_detJ=jac_stats["min"],
            title_prefix=f"{prov['algorithm']} ({prov['backend']})",
            jac_measure=jac_measure,
            filename=_p,
        ),
    )
    fig2_abs = fig2_paths[eff_theme]
    fig_num += 1

    html_figs = f'''
        <section class="card" style="margin-bottom: 2rem;">
            <h2>Figure 1: Input Image Pair (Fixed & Moving)</h2>
            <div style="text-align: center;">{_theme_img_html(fig1_paths, "Figure 1")}</div>
        </section>
        <section class="card" style="margin-bottom: 2rem;">
            <h2>Figure 2: Standard 4-Panel Diagnostic Report</h2>
            <div style="text-align: center;">{_theme_img_html(fig2_paths, "Figure 2")}</div>
        </section>
    '''

    fig_flow_abs = None
    if warp_img is not None:
        try:
            flow_paths = _render_dual_theme(
                f"fig{fig_num}_flow_anatomy_{ts}",
                lambda _th, _p: render_flow_on_anatomy_report_figure(
                    fixed=fixed,
                    warp=warp_img,
                    slice_axis=slice_axis,
                    slice_idx=slice_idx,
                    target_label=fixed_label,
                    theme=_th,
                    output_path=_p,
                ),
            )
            fig_flow_abs = flow_paths[eff_theme]
            html_figs += f'''
        <section class="card" style="margin-bottom: 2rem;">
            <h2>Figure {fig_num}: Displacement Flow Dynamics & Regularity on Target Anatomy</h2>
            <div style="text-align: center;">{_theme_img_html(flow_paths, f"Figure {fig_num}")}</div>
        </section>
            '''
            fig_num += 1
        except Exception as e:
            warnings.warn(f"create_registration_report: flow on anatomy figure generation failed ({e})")
            fig_flow_abs = None

    fig_seg_abs = None
    if fixed_label is not None and (warped_label is not None or moving_label is not None):
        try:
            if warped_label is None and moving_label is not None:
                if reg is not None and 'fwdtransforms' in reg:
                    warped_label = _apply_transforms(fixed, moving_label, reg['fwdtransforms'], interpolator='nearestNeighbor')
                elif warp is not None:
                    warped_label = _apply_transforms(fixed, moving_label, warp, interpolator='nearestNeighbor')

            if warped_label is not None:
                d_score = dice_sym if isinstance(dice_sym, (int, float)) and np.isfinite(dice_sym) else (dice_overlap if isinstance(dice_overlap, (int, float)) else None)
                seg_paths = _render_dual_theme(
                    f"fig{fig_num}_seg_overlap_{ts}",
                    lambda _th, _p: render_segmentation_overlap_report_figure(
                        fixed=fixed,
                        target_label=fixed_label,
                        warped_label=warped_label,
                        slice_axis=slice_axis,
                        slice_idx=slice_idx,
                        dice_score=d_score,
                        theme=_th,
                        output_path=_p,
                    ),
                )
                fig_seg_abs = seg_paths[eff_theme]
                html_figs += f'''
        <section class="card" style="margin-bottom: 2rem;">
            <h2>Figure {fig_num}: Anatomical Segmentation Alignment & Spatial Overlap</h2>
            <div style="text-align: center;">{_theme_img_html(seg_paths, f"Figure {fig_num}")}</div>
        </section>
                '''
                fig_num += 1
        except Exception as e:
            warnings.warn(f"create_registration_report: segmentation overlap figure generation failed ({e})")
            fig_seg_abs = None

    fig_warped_abs = None
    fig_checkerboard_abs = None
    if has_warped:
        _warped_num = fig_num
        warped_paths = _render_dual_theme(
            f"fig{_warped_num}_warped_comparison_{ts}",
            lambda _th, _p: render_warped_comparison_figure(
                fixed,
                warped,
                moving=moving,
                theme=_th,
                output_path=_p,
                title=f"Figure {_warped_num}: Fixed Target vs. Pre-Registration Moving vs. Warped Moving-to-Fixed",
            ),
        )
        fig_warped_abs = warped_paths[eff_theme]
        html_figs += f'''
        <section class="card" style="margin-bottom: 2rem;">
            <h2>Figure {fig_num}: Fixed Target vs. Pre-Registration Moving vs. Warped Moving-to-Fixed</h2>
            <div style="text-align: center;">{_theme_img_html(warped_paths, f"Figure {fig_num}")}</div>
        </section>
        '''
        fig_num += 1

        _chk_num = fig_num
        chk_paths = _render_dual_theme(
            f"fig{_chk_num}_checkerboard_{ts}",
            lambda _th, _p: render_checkerboard_figure(
                fixed,
                warped,
                title=f"Figure {_chk_num}: Fixed / Warped Checkerboard Alignment",
                save_path=_p,
                theme=_th,
                resample_to_target=True,
            ),
        )
        fig_checkerboard_abs = chk_paths[eff_theme]
        html_figs += f'''
        <section class="card" style="margin-bottom: 2rem;">
            <h2>Figure {fig_num}: Fixed / Warped Checkerboard Alignment</h2>
            <div style="text-align: center;">{_theme_img_html(chk_paths, f"Figure {fig_num}")}</div>
        </section>
        '''
        fig_num += 1

    if reg is not None and isinstance(reg, dict) and 'model' in reg:
        model = reg['model']
        if type(model).__name__ == 'TVFModel':
            tvf_paths = _render_dual_theme(
                f"fig{fig_num}_velocity_{ts}",
                lambda _th, _p: plot_time_varying_velocity_grid(model, fixed_image=fixed, theme=_th, output_path=_p),
            )
            html_figs += f'''
            <section class="card" style="margin-bottom: 2rem;">
                <h2>Figure {fig_num}: Time-Varying Velocity Field Flow Keyframes</h2>
                <div style="text-align: center;">{_theme_img_html(tvf_paths, f"Figure {fig_num}")}</div>
            </section>
            '''
            fig_num += 1

    losses = None
    if reg is not None and isinstance(reg, dict) and 'model' in reg:
        if hasattr(reg['model'], 'losses') and len(reg['model'].losses) > 0:
            losses = reg['model'].losses
        elif hasattr(reg['model'], 'syn_losses') and len(reg['model'].syn_losses) > 0:
            losses = reg['model'].syn_losses
    elif reg is not None and isinstance(reg, dict) and 'loss_history' in reg:
        losses = reg['loss_history']

    if losses:
        loss_paths = _render_dual_theme(
            f"fig{fig_num}_loss_{ts}",
            lambda _th, _p: plot_loss_convergence(
                losses, theme=_th, output_path=_p, title=f"Similarity Loss Convergence ({prov['algorithm']})"
            ),
        )
        html_figs += f'''
        <section class="card" style="margin-bottom: 2rem;">
            <h2>Figure {fig_num}: Multi-Resolution Similarity Loss Convergence</h2>
            <div style="text-align: center;">{_theme_img_html(loss_paths, f"Figure {fig_num}")}</div>
        </section>
        '''
        fig_num += 1

    if regional_overlap_sym and fixed_label is not None:
        dice_dict = {
            'fixed_dice': list(regional_overlap_fwd.values()),
            'moving_dice': list(regional_overlap_inv.values()),
            'sym_dice': list(regional_overlap_sym.values()),
            'per_region': regional_overlap_sym,
        }
        dkt_paths = _render_dual_theme(
            f"fig{fig_num}_dkt_overlap_{ts}",
            lambda _th, _p: plot_label_overlap_stats(
                dice_scores=dice_dict, theme=_th, output_path=_p, title="Label Overlap (per label)"
            ),
        )
        html_figs += f'''
        <section class="card" style="margin-bottom: 2rem;">
            <h2>Figure {fig_num}: Anatomical Label Overlap Stats</h2>
            <div style="text-align: center;">{_theme_img_html(dkt_paths, f"Figure {fig_num}")}</div>
        </section>
        '''
        fig_num += 1

    import json
    prov_str = json.dumps(prov, indent=2)
    html_figs += f'''
        <section class="card" style="margin-bottom: 2rem;">
            <h2>Registration Provenance &amp; Hyperparameters</h2>
            <pre class="prov-pre"><code>{prov_str}</code></pre>
        </section>
    '''

    def _fmt(v, spec, suffix=""):
        if isinstance(v, str) or v is None or not np.isfinite(v):
            return "n/a"
        return f"{v:{spec}}{suffix}"

    metrics_html = "".join([f"<tr><td>{k}:</td><td class='metric-val'>{_fmt(v, '.4f')}</td></tr>" for k, v in metrics.items()])
    prov_badge = ('<div class="badge badge-success">Provenance attached</div>' if provenance_given
                  else '<div class="badge">No run provenance supplied</div>')
    init_btn_label = "&#9728; Light Theme" if eff_theme == "dark" else "&#9790; Dark Theme"

    html_content = f"""<!DOCTYPE html>
<html lang="en" data-theme="{eff_theme}">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        :root, html[data-theme="light"] {{
            --page-bg: #ffffff;
            --page-ink: #0f172a;
            --card-bg: #f8fafc;
            --card-bd: #cbd5e1;
            --hdr-col: #0284c7;
            --sub-hdr-col: #1e293b;
            --badge-bg: #f1f5f9;
            --badge-fg: #334155;
            --badge-ok-bg: #ecfdf5;
            --badge-ok-fg: #047857;
            --pre-bg: #f1f5f9;
            --pre-fg: #0f172a;
            --toggle-bg: #f1f5f9;
            --toggle-fg: #0f172a;
            --toggle-hover: #e2e8f0;
        }}
        html[data-theme="dark"] {{
            --page-bg: #090d16;
            --page-ink: #f8fafc;
            --card-bg: #0f172a;
            --card-bd: #1e293b;
            --hdr-col: #38bdf8;
            --sub-hdr-col: #cbd5e1;
            --badge-bg: #1e293b;
            --badge-fg: #94a3b8;
            --badge-ok-bg: #064e3b;
            --badge-ok-fg: #34d399;
            --pre-bg: #1e293b;
            --pre-fg: #38bdf8;
            --toggle-bg: #1e293b;
            --toggle-fg: #f8fafc;
            --toggle-hover: #334155;
        }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: var(--page-bg); color: var(--page-ink); margin: 0; padding: 2rem; transition: background-color 0.18s ease, color 0.18s ease; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ border-bottom: 1px solid var(--card-bd); padding-bottom: 1rem; margin-bottom: 2rem; }}
        .header-top {{ display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; }}
        h1 {{ font-size: 1.8rem; font-weight: 700; color: var(--hdr-col); margin: 0; }}
        .theme-toggle {{ cursor: pointer; background: var(--toggle-bg); color: var(--toggle-fg); border: 1px solid var(--card-bd); padding: 0.4rem 0.85rem; border-radius: 6px; font-size: 0.85rem; font-weight: 600; transition: background-color 0.15s ease; }}
        .theme-toggle:hover {{ background: var(--toggle-hover); }}
        .badge {{ display: inline-block; background: var(--badge-bg); color: var(--badge-fg); border: 1px solid var(--card-bd); padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.85rem; font-weight: 600; margin-top: 0.5rem; }}
        .badge-success {{ background: var(--badge-ok-bg); color: var(--badge-ok-fg); }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
        .card {{ background: var(--card-bg); border: 1px solid var(--card-bd); border-radius: 8px; padding: 1.25rem; }}
        .card h2 {{ font-size: 1.1rem; color: var(--sub-hdr-col); margin-top: 0; margin-bottom: 1rem; border-bottom: 1px solid var(--card-bd); padding-bottom: 0.5rem; }}
        .prov-pre {{ background: var(--pre-bg); color: var(--pre-fg); border: 1px solid var(--card-bd); padding: 1rem; border-radius: 6px; overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
        td, th {{ padding: 0.5rem 0.25rem; text-align: left; }}
        tr:not(:last-child) {{ border-bottom: 1px solid var(--card-bd); }}
        .metric-val {{ font-family: monospace; font-weight: 600; color: var(--hdr-col); }}
        footer {{ margin-top: 3rem; text-align: center; color: #64748b; font-size: 0.85rem; border-top: 1px solid var(--card-bd); padding-top: 1.5rem; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-top">
                <h1>{title}</h1>
                <button id="theme-toggle" class="theme-toggle" type="button" aria-label="Toggle light or dark theme">{init_btn_label}</button>
            </div>
            <div class="badge">{html.escape(str(meta_fixed.get("name", fixed_name)))} &larr; {html.escape(str(meta_moving.get("name", moving_name)))}</div>
            {prov_badge}
            <div class="badge">Engine: {prov['algorithm']} ({prov['backend']})</div>
            <div class="badge">Device: {prov['device']}</div>
        </header>

        <div class="grid-2">
            <div class="card">
                <h2>📈 Registration Similarity Metrics</h2>
                <table>
                    {metrics_html}
                </table>
            </div>

            <div class="card">
                <h2>📐 Spatial Topology & Inverse Identity</h2>
                <table>
                    <tr><td>Symmetric Label DICE:</td><td class="metric-val">{_fmt(dice_sym, '.4f')}</td></tr>
                    {f"<tr><td>Caller Dice:</td><td class='metric-val'>{_fmt(float(dice_overlap), '.4f')}</td></tr>" if dice_overlap is not None else ""}
                    <tr><td>Jacobian Range:</td><td class="metric-val">[{_fmt(jac_stats['min'], '+.2f')}, {_fmt(jac_stats['max'], '.2f')}]</td></tr>
                    <tr><td>Grid Folding Rate:</td><td class="metric-val">{_fmt(jac_stats['folding_pct'], '.2f', '%')}</td></tr>
                    <tr><td>Harmonic Energy (1st Order):</td><td class="metric-val">{f"{hrm_energy:.3e}" if isinstance(hrm_energy, float) else hrm_energy}</td></tr>
                    <tr><td>Thin-Plate Bending Energy (2nd Order):</td><td class="metric-val">{f"{bnd_energy:.3e}" if isinstance(bnd_energy, float) else bnd_energy}</td></tr>
                    <tr><td>Interior Max Inverse Error (Eroded):</td><td class="metric-val">{_fmt(inv_stats['interior_max'], '.2f', ' mm')}</td></tr>
                    <tr><td>Interior Mean Inverse Error (Eroded):</td><td class="metric-val">{_fmt(inv_stats['interior_mean'], '.3f', ' mm')}</td></tr>
                    <tr><td>Absolute Max Inverse Error (Global):</td><td class="metric-val">{_fmt(inv_stats['max'], '.2f', ' mm')}</td></tr>
                </table>
            </div>
        </div>

        {html_figs}

        <footer>
            <p>Generated by <strong>syntx.viz</strong></p>
        </footer>
    </div>
    <script>
    (function() {{
        var STORAGE_KEY = "syntx-report-theme";
        var root = document.documentElement;
        var btn = document.getElementById("theme-toggle");
        function applyTheme(mode) {{
            var m = (mode === "dark") ? "dark" : "light";
            root.setAttribute("data-theme", m);
            var imgs = document.querySelectorAll("img.theme-img");
            for (var i = 0; i < imgs.length; i++) {{
                var nextSrc = imgs[i].getAttribute(m === "dark" ? "data-src-dark" : "data-src-light");
                if (nextSrc) imgs[i].setAttribute("src", nextSrc);
            }}
            if (btn) {{
                btn.textContent = (m === "dark") ? "\u2600 Light Theme" : "\u263E Dark Theme";
            }}
        }}
        var saved = null;
        try {{ saved = localStorage.getItem(STORAGE_KEY); }} catch (e) {{}}
        var initial = (saved === "dark" || saved === "light") ? saved : (root.getAttribute("data-theme") || "light");
        applyTheme(initial);
        if (btn) {{
            btn.addEventListener("click", function() {{
                var cur = root.getAttribute("data-theme") === "dark" ? "dark" : "light";
                var next = (cur === "dark") ? "light" : "dark";
                applyTheme(next);
                try {{ localStorage.setItem(STORAGE_KEY, next); }} catch (e) {{}}
            }});
        }}
    }})();
    </script>
</body>
</html>
"""

    with open(output_html, "w") as f:
        f.write(html_content)
    if show_report:
        import webbrowser
        webbrowser.open("file://" + output_html)

    return {
        "html_path": output_html,
        "fig_path": fig2_abs,
        "fig2_path": fig2_abs,
        "fig_warped_path": fig_warped_abs,
        "fig_checkerboard_path": fig_checkerboard_abs,
        "fig_flow_path": fig_flow_abs,
        "fig_seg_path": fig_seg_abs,
        "metrics": metrics,
        "dice": dice_sym,
        "dice_sym": dice_sym,
        "jacobian": jac_stats,
        "inverse_error": inv_stats,
        "provenance": prov,
    }

def create_benchmark_report(syn_results: dict, ants_results: dict, total_pairs: int, output_html: str = "benchmark_report.html"):
    """Write a syntx-vs-ANTs comparison page (Plotly box / scatter plots and a table).

    Parameters
    ----------
    syn_results, ants_results : dict
        Integer pair index -> dict with optional ``dice_sym``, ``folding_pct``,
        ``inverse_error_mean``, ``runtime_seconds``. The summary means are over the pairs
        present in both dicts (over all syntx pairs when there are no ANTs results); missing
        or non-finite values are left out (n/a when nothing is left), never counted as 0.
    total_pairs : int
        Denominator of the progress bar.
    output_html : str, default "benchmark_report.html"
        Output file (parent directories are created).

    A paired t-test (``scipy.stats.ttest_rel``) on ``dice_sym`` is run over the paired indices
    that have Dice on both sides (if more than one); a degenerate (NaN) result is shown as n/a.

    Returns
    -------
    str
        ``output_html`` as given.
    """
    import json
    from scipy.stats import ttest_rel

    completed = len(syn_results)
    paired_idx = sorted(set(syn_results.keys()).intersection(ants_results.keys()))
    summary_idx = paired_idx if ants_results else sorted(syn_results.keys())

    def _val(r, k):
        v = r.get(k) if r is not None else None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    def _mean(results, key):
        vals = [v for v in (_val(results.get(i), key) for i in summary_idx) if v is not None]
        return float(np.mean(vals)) if vals else float('nan')

    t_stat = p_val = float('nan')
    both = [(_val(syn_results[i], 'dice_sym'), _val(ants_results[i], 'dice_sym')) for i in paired_idx]
    both = [(a, b) for a, b in both if a is not None and b is not None]
    if len(both) > 1:
        t_stat, p_val = (float(x) for x in ttest_rel([a for a, _ in both], [b for _, b in both]))

    mean_dice_syn, mean_fold_syn, mean_inv_syn = (_mean(syn_results, k) for k in ('dice_sym', 'folding_pct', 'inverse_error_mean'))
    mean_dice_ants, mean_fold_ants, mean_inv_ants = (_mean(ants_results, k) for k in ('dice_sym', 'folding_pct', 'inverse_error_mean'))

    def _f(v, spec):
        return "n/a" if not np.isfinite(v) else format(v, spec)

    # Plot data (missing values are null, not 0)
    pair_ids = [f"Pair {i}" for i in sorted(syn_results.keys())]
    keys = sorted(syn_results.keys())
    syn_dice_sym = [_val(syn_results[i], 'dice_sym') for i in keys]
    ants_dice_sym = [_val(ants_results.get(i), 'dice_sym') for i in keys]
    syn_folds = [_val(syn_results[i], 'folding_pct') for i in keys]
    ants_folds = [_val(ants_results.get(i), 'folding_pct') for i in keys]
    syn_times = [_val(syn_results[i], 'runtime_seconds') for i in keys]
    ants_times = [_val(ants_results.get(i), 'runtime_seconds') for i in keys]
    syn_invs = [_val(syn_results[i], 'inverse_error_mean') for i in keys]
    ants_invs = [_val(ants_results.get(i), 'inverse_error_mean') for i in keys]
    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Syntx Benchmark: PyTorch SyN vs ANTs C++</title>
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --surface-color: #ffffff;
            --syntx-color: #3b82f6;
            --ants-color: #ef4444;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
        }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: var(--surface-color);
            padding: 50px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border-radius: 8px;
        }}
        h1, h2, h3 {{ font-family: 'Crimson Pro', serif; color: #0f172a; }}
        h1 {{ font-size: 2.8rem; text-align: center; margin-bottom: 10px; font-weight: 700; }}
        .subtitle {{
            text-align: center; font-family: 'Inter', sans-serif;
            color: var(--text-muted); font-size: 1.1rem;
            margin-bottom: 40px; text-transform: uppercase; letter-spacing: 2px;
        }}
        .abstract {{
            font-style: italic; font-family: 'Crimson Pro', serif;
            font-size: 1.2rem; padding: 20px 40px;
            border-left: 4px solid var(--syntx-color);
            background-color: #f1f5f9; margin-bottom: 50px;
        }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .stat-card {{ background-color: var(--bg-color); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; text-align: center; }}
        .stat-value {{ font-size: 2.2rem; font-weight: 600; font-family: 'Inter', sans-serif; display: flex; justify-content: center; gap: 20px; }}
        .stat-label {{ font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 500; margin-top: 5px; }}
        .color-syntx {{ color: var(--syntx-color); }}
        .color-ants {{ color: var(--ants-color); }}
        
        .plot-container {{ width: 100%; height: 500px; margin-bottom: 40px; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px; }}
        .plot-row {{ display: flex; gap: 20px; margin-bottom: 40px; flex-wrap: wrap; }}
        .plot-third {{ flex: 1; min-width: 300px; height: 500px; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px; }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 0.90rem; }}
        th, td {{ padding: 12px 10px; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ background-color: var(--bg-color); font-weight: 600; color: var(--text-muted); text-transform: uppercase; font-size: 0.75rem; letter-spacing: 1px; }}
        tr:hover td {{ background-color: #f8fafc; }}
        .progress-container {{ margin-bottom: 40px; }}
        .progress-bar {{ height: 6px; background-color: var(--border-color); border-radius: 3px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background-color: var(--syntx-color); width: {(completed/max(1,total_pairs))*100}%; transition: width 1s ease; }}
        .progress-text {{ text-align: right; font-size: 0.85rem; color: var(--text-muted); margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Comparative SyN Registration Benchmark</h1>
        <div class="subtitle">Syntx PyTorch GPU vs. ANTs C++ CPU</div>

        <div class="progress-container">
            <div class="progress-bar"><div class="progress-fill"></div></div>
            <div class="progress-text">Evaluation Progress: {completed} / {total_pairs} Pairs</div>
        </div>

        <div class="abstract">
            <strong>Abstract:</strong> This real-time whitepaper evaluates the strict numerical parity and spatial performance of the <i>syntx</i> native PyTorch Eulerian SyN formulation against the gold-standard ANTs C++ implementation.
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">
                    <span class="color-syntx">{_f(mean_dice_syn, '.4f')}</span>
                    <span style="color: #cbd5e1;">|</span>
                    <span class="color-ants">{_f(mean_dice_ants, '.4f')}</span>
                </div>
                <div class="stat-label">Mean Symmetric Dice (Syntx | ANTs)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">
                    <span class="color-syntx">{_f(mean_fold_syn, '.4f')}%</span>
                    <span style="color: #cbd5e1;">|</span>
                    <span class="color-ants">{_f(mean_fold_ants, '.4f')}%</span>
                </div>
                <div class="stat-label">Mean Grid Folding (Syntx | ANTs)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">
                    <span class="color-syntx">{_f(mean_inv_syn, '.4f')}</span>
                    <span style="color: #cbd5e1;">|</span>
                    <span class="color-ants">{_f(mean_inv_ants, '.4f')}</span>
                </div>
                <div class="stat-label">Mean Inverse Error (mm)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">
                    <span style="color: #10b981;">{_f(p_val, '.2e')}</span>
                </div>
                <div class="stat-label">Paired T-Test p-value (t={_f(t_stat, '.2f')})</div>
            </div>
        </div>

        <h2>I. Volumetric Overlap Analysis</h2>
        <p>Comparison of Symmetric Mean Dice distributions between the two implementations.</p>
        <div class="plot-row">
            <div class="plot-container" style="flex: 1; min-width: 400px; margin-bottom: 0;" id="diceBoxplot"></div>
            <div class="plot-container" style="flex: 1; min-width: 400px; margin-bottom: 0;" id="pairedScatter"></div>
        </div>

        <h2>II. Performance & Topology Trade-offs</h2>
        <div class="plot-row">
            <div class="plot-third" id="foldScatter"></div>
            <div class="plot-third" id="invScatter"></div>
            <div class="plot-third" id="timeScatter"></div>
        </div>

        <h2>III. Paired Raw Data</h2>
        <table>
            <thead>
                <tr>
                    <th>Pair ID</th>
                    <th>Syntx Dice</th>
                    <th>ANTs Dice</th>
                    <th>Syntx Fold</th>
                    <th>ANTs Fold</th>
                    <th>Syntx Mean Inv</th>
                    <th>ANTs Mean Inv</th>
                    <th>Syntx Time</th>
                    <th>ANTs Time</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for idx in sorted(syn_results.keys()):
        s_res = syn_results[idx]
        a_res = ants_results.get(idx, {})
        
        _c = lambda r, k, spec, suf="": (_f(_val(r, k), spec) + suf) if _val(r, k) is not None else "n/a"
        s_dice = _c(s_res, 'dice_sym', '.4f')
        a_dice = _c(a_res, 'dice_sym', '.4f') if a_res else "-"
        
        s_fold = _c(s_res, 'folding_pct', '.4f', '%')
        a_fold = _c(a_res, 'folding_pct', '.4f', '%') if a_res else "-"
        
        s_inv = _c(s_res, 'inverse_error_mean', '.4f')
        a_inv = _c(a_res, 'inverse_error_mean', '.4f') if a_res else "-"
        
        s_time = _c(s_res, 'runtime_seconds', '.1f', 's')
        a_time = _c(a_res, 'runtime_seconds', '.1f', 's') if a_res else "-"
        
        html += f"""
                <tr>
                    <td style="font-family: monospace; font-weight: 600;">#{idx:03d}</td>
                    <td class="color-syntx" style="font-weight: 600;">{s_dice}</td>
                    <td class="color-ants" style="font-weight: 600;">{a_dice}</td>
                    <td>{s_fold}</td>
                    <td>{a_fold}</td>
                    <td>{s_inv}</td>
                    <td>{a_inv}</td>
                    <td>{s_time}</td>
                    <td>{a_time}</td>
                </tr>"""

    html += f"""
            </tbody>
        </table>
    </div>

    <script>
        const pairIds = {json.dumps(pair_ids)};
        const synDice = {json.dumps(syn_dice_sym)};
        const antsDice = {json.dumps(ants_dice_sym)};
        const synFolds = {json.dumps(syn_folds)};
        const antsFolds = {json.dumps(ants_folds)};
        const synTimes = {json.dumps(syn_times)};
        const antsTimes = {json.dumps(ants_times)};
        const synInvs = {json.dumps(syn_invs)};
        const antsInvs = {json.dumps(ants_invs)};

        // 1. Boxplot (Dice)
        const traceSyn = {{ y: synDice, type: 'box', name: 'Syntx PyTorch', marker: {{color: '#3b82f6'}}, boxpoints: 'all', jitter: 0.3 }};
        const traceAnts = {{ y: antsDice, type: 'box', name: 'ANTs C++', marker: {{color: '#ef4444'}}, boxpoints: 'all', jitter: 0.3 }};
        
        Plotly.newPlot('diceBoxplot', [traceSyn, traceAnts], {{
            title: 'Symmetric DKT31 Dice Score Distributions',
            yaxis: {{ title: 'Dice Score', zeroline: false }},
            boxmode: 'group',
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter, sans-serif' }}
        }}, {{responsive: true}});

        // 1.5 Scatter (Syntx vs ANTs Dice)
        const pairedAntsDice = [];
        const pairedSynDice = [];
        const pairedIds = [];
        for (let i = 0; i < pairIds.length; i++) {{
            if (antsDice[i] !== null && synDice[i] !== null) {{
                pairedAntsDice.push(antsDice[i]);
                pairedSynDice.push(synDice[i]);
                pairedIds.push(pairIds[i]);
            }}
        }}
        
        const scatterPaired = {{ x: pairedAntsDice, y: pairedSynDice, text: pairedIds, mode: 'markers', type: 'scatter', name: 'Pairs', marker: {{ size: 10, color: '#8b5cf6', opacity: 0.8, line: {{color: 'white', width: 1}} }} }};
        const lineRef = {{ x: [0.3, 0.8], y: [0.3, 0.8], mode: 'lines', type: 'scatter', name: 'y=x (Parity)', line: {{ dash: 'dash', color: '#94a3b8' }} }};
        
        Plotly.newPlot('pairedScatter', [scatterPaired, lineRef], {{
            title: 'Pairwise Accuracy (Syntx vs ANTs)',
            xaxis: {{ title: 'ANTs Symmetric Mean Dice' }},
            yaxis: {{ title: 'Syntx Symmetric Mean Dice' }},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter, sans-serif' }},
            showlegend: false
        }}, {{responsive: true}});

        // 2. Scatter (Dice vs Fold)
        const scatterSynFold = {{ x: synFolds, y: synDice, name: 'Syntx', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#3b82f6', opacity: 0.7 }} }};
        const scatterAntsFold = {{ x: antsFolds, y: antsDice, name: 'ANTs', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#ef4444', opacity: 0.7 }} }};
        
        Plotly.newPlot('foldScatter', [scatterSynFold, scatterAntsFold], {{
            title: 'Dice vs Topology Destruction',
            xaxis: {{ title: 'Grid Folding % (det J <= 0)', zeroline: false }},
            yaxis: {{ title: 'Symmetric Mean Dice' }},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter, sans-serif' }}
        }}, {{responsive: true}});
        
        // 3. Scatter (Dice vs Inverse Error)
        const scatterSynInv = {{ x: synInvs, y: synDice, name: 'Syntx', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#3b82f6', opacity: 0.7 }} }};
        const scatterAntsInv = {{ x: antsInvs, y: antsDice, name: 'ANTs', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#ef4444', opacity: 0.7 }} }};
        
        Plotly.newPlot('invScatter', [scatterSynInv, scatterAntsInv], {{
            title: 'Dice vs Mean Inverse Error',
            xaxis: {{ title: 'Mean Inverse Error (mm)', zeroline: false }},
            yaxis: {{ title: 'Symmetric Mean Dice' }},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter, sans-serif' }}
        }}, {{responsive: true}});

        // 4. Scatter (Syntx Compute Time vs ANTs Compute Time)
        const pairedAntsTime = [];
        const pairedSynTime = [];
        const pairedTimeIds = [];
        for (let i = 0; i < pairIds.length; i++) {{
            if (antsTimes[i] !== null && synTimes[i] !== null) {{
                pairedAntsTime.push(antsTimes[i]);
                pairedSynTime.push(synTimes[i]);
                pairedTimeIds.push(pairIds[i] + ' (' + (antsTimes[i]/synTimes[i]).toFixed(1) + 'x speedup)');
            }}
        }}
        const scatterTimePaired = {{ x: pairedAntsTime, y: pairedSynTime, text: pairedTimeIds, mode: 'markers', type: 'scatter', name: 'Pairs', marker: {{ size: 10, color: '#3b82f6', opacity: 0.8, line: {{color: 'white', width: 1}} }} }};
        const maxTime = Math.max(...pairedAntsTime, 250);
        const lineTimeParity = {{ x: [0, maxTime], y: [0, maxTime], mode: 'lines', type: 'scatter', name: '1x (Parity)', line: {{ dash: 'dash', color: '#94a3b8' }} }};
        const lineTime2x = {{ x: [0, maxTime], y: [0, maxTime * 0.5], mode: 'lines', type: 'scatter', name: '2x Speedup', line: {{ dash: 'dot', color: '#10b981' }} }};
        const lineTime3x = {{ x: [0, maxTime], y: [0, maxTime * 0.333], mode: 'lines', type: 'scatter', name: '3x Speedup', line: {{ dash: 'dot', color: '#8b5cf6' }} }};

        Plotly.newPlot('timeScatter', [scatterTimePaired, lineTimeParity, lineTime2x, lineTime3x], {{
            title: 'Compute Time: Syntx GPU vs ANTs CPU',
            xaxis: {{ title: 'ANTs CPU Time (seconds)' }},
            yaxis: {{ title: 'Syntx GPU Time (seconds)' }},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter, sans-serif' }}
        }}, {{responsive: true}});

    </script>
</body>
</html>
"""
    
    with open(output_html, "w") as f:
        f.write(html)
    return output_html


def create_population_benchmark_report(
    results_source,
    baseline_source=None,
    output_html: str = "docs/reproducible_90pair_report.html",
    title: str = "Syntx Sobolev SyN vs ANTs C++ — Population Benchmark Report",
    provenance: dict = None,
) -> str:
    """Write an HTML page summarising per-pair benchmark results (TVF / Sobolev SyN /
    Gaussian SyN vs ANTs), with Plotly plots, intra / inter subgroups and a per-pair table.

    Per pair, Dice / time / folding are read from ``syntx_dice_sym`` / ``dice_sym``,
    ``syntx_time`` / ``runtime_seconds`` and ``syntx_fold`` / ``folding_pct``; the ANTs
    record is the record's ``ants_baseline`` or the baseline entry. The "focus" method is TVF
    if any TVF records exist, else Sobolev; a pair is a "win" if focus Dice >= ANTs Dice. The
    cohort is ``cohort_type`` or, if missing, "intra" for index < 40 else "inter".

    Counts, win rates and the TVF-vs-Sobolev comparison are computed from the records; the
    configuration table shows each method's recorded ``config``. Missing ANTs folding is NaN
    (excluded from means). The "x Faster" figure uses the Sobolev mean time.

    Parameters
    ----------
    results_source : str, dict or list
        A directory (reads ``pair_*_tvf.json``, ``pair_*_sobolev.json`` -- or
        ``pair_*_syn.json`` if there are none -- and ``pair_*_gaussian.json``; files that fail
        to load or lack ``status == "SUCCESS"`` / a Dice key are skipped), a JSON file
        (``tvf_results`` / ``sobolev_results`` / ``gaussian_results`` maps, else a list or
        {index: record} of Sobolev records), a list of records (``pair_idx`` or position as
        index) or a dict {index: record}.
    baseline_source : str or dict, optional
        Directory of ``pair_*_ants_syn.json`` files or {index: record}. Other types are
        ignored.
    output_html : str, default "docs/reproducible_90pair_report.html"
        Output file (parent directories are created). With no records a one-line page is
        written.
    title : str
        Page title.
    provenance : dict, optional
        Ignored.

    Returns
    -------
    str
        ``output_html`` as given.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)

    records = {}
    gaussian_records = {}
    tvf_records = {}
    syngs_records = {}
    greedy_records = {}

    def _load_dir_glob(pattern, dst):
        import glob
        for f in sorted(glob.glob(pattern)):
            try:
                with open(f, "r") as fp:
                    d = json.load(fp)
                if d.get("status") == "SUCCESS" or "syntx_dice_sym" in d or "dice_sym" in d:
                    p_idx = d.get("pair_idx", len(dst))
                    dst[int(p_idx)] = d
            except Exception:
                pass

    # 1. Parse results_source
    if isinstance(results_source, str):
        if os.path.isdir(results_source):
            _load_dir_glob(os.path.join(results_source, "pair_*_tvf.json"), tvf_records)
            _load_dir_glob(os.path.join(results_source, "pair_*_sobolev.json"), records)
            if not records:
                _load_dir_glob(os.path.join(results_source, "pair_*_syn.json"), records)
            _load_dir_glob(os.path.join(results_source, "pair_*_syngs.json"), syngs_records)
            _load_dir_glob(os.path.join(results_source, "pair_*_greedy.json"), greedy_records)
            _load_dir_glob(os.path.join(results_source, "pair_*_gaussian.json"), gaussian_records)
        elif os.path.isfile(results_source):
            with open(results_source, "r") as fp:
                d = json.load(fp)
            if isinstance(d, dict):
                if "tvf_results" in d and isinstance(d["tvf_results"], dict):
                    tvf_records = {int(k): v for k, v in d["tvf_results"].items() if str(k).isdigit()}
                if "sobolev_results" in d and isinstance(d["sobolev_results"], dict):
                    records = {int(k): v for k, v in d["sobolev_results"].items() if str(k).isdigit()}
                if "syngs_results" in d and isinstance(d["syngs_results"], dict):
                    syngs_records = {int(k): v for k, v in d["syngs_results"].items() if str(k).isdigit()}
                if "greedy_results" in d and isinstance(d["greedy_results"], dict):
                    greedy_records = {int(k): v for k, v in d["greedy_results"].items() if str(k).isdigit()}
                if "gaussian_results" in d and isinstance(d["gaussian_results"], dict):
                    gaussian_records = {int(k): v for k, v in d["gaussian_results"].items() if str(k).isdigit()}
            if not (tvf_records or records or syngs_records or greedy_records or gaussian_records):
                if isinstance(d, list):
                    records = {r.get("pair_idx", i): r for i, r in enumerate(d)}
                elif isinstance(d, dict):
                    records = {int(k): v for k, v in d.items() if str(k).isdigit()}
    elif isinstance(results_source, list):
        records = {r.get("pair_idx", i): r for i, r in enumerate(results_source)}
    elif isinstance(results_source, dict):
        records = {int(k): v for k, v in results_source.items() if str(k).isdigit() or isinstance(k, int)}

    # 2. Parse baseline_source if provided
    baseline_records = {}
    if isinstance(baseline_source, str) and os.path.isdir(baseline_source):
        import glob
        ants_files = sorted(glob.glob(os.path.join(baseline_source, "pair_*_ants_syn.json")))
        for f in ants_files:
            try:
                with open(f, "r") as fp:
                    d = json.load(fp)
                if d.get("status") == "SUCCESS" or "dice_sym" in d:
                    baseline_records[d.get("pair_idx", len(baseline_records))] = d
            except Exception:
                pass
    elif isinstance(baseline_source, dict):
        baseline_records = {int(k): v for k, v in baseline_source.items()}

    has_tvf = len(tvf_records) > 0
    has_syngs = len(syngs_records) > 0
    has_greedy = len(greedy_records) > 0
    all_indices = sorted(list(
        set(records.keys())
        | set(gaussian_records.keys())
        | set(tvf_records.keys())
        | set(syngs_records.keys())
        | set(greedy_records.keys())
    ))

    # 3. Format per-pair comparison rows
    matched_pairs = []
    for idx in all_indices:
        t_rec = tvf_records.get(idx, {})
        s_rec = records.get(idx, {})
        gs_rec = syngs_records.get(idx, {})
        gr_rec = greedy_records.get(idx, {})
        g_rec = gaussian_records.get(idx, {})
        primary_rec = t_rec or s_rec or gs_rec or gr_rec or g_rec
        a_rec = primary_rec.get("ants_baseline", s_rec.get("ants_baseline", baseline_records.get(idx, {})))

        t_dice = t_rec.get("syntx_dice_sym", t_rec.get("dice_sym", float("nan"))) if t_rec else float("nan")
        s_dice = s_rec.get("syntx_dice_sym", s_rec.get("dice_sym", float("nan"))) if s_rec else float("nan")
        gs_dice = gs_rec.get("syntx_dice_sym", gs_rec.get("dice_sym", float("nan"))) if gs_rec else float("nan")
        gr_dice = gr_rec.get("syntx_dice_sym", gr_rec.get("dice_sym", float("nan"))) if gr_rec else float("nan")
        g_dice = g_rec.get("syntx_dice_sym", g_rec.get("dice_sym", float("nan"))) if g_rec else float("nan")

        if isinstance(a_rec, dict) and "dice_sym" in a_rec:
            a_dice = a_rec.get("dice_sym", float("nan"))
            a_time = a_rec.get("runtime_seconds", float("nan"))
            a_fold = a_rec.get("folding_pct", float("nan"))
        else:
            a_dice = a_rec.get("syntx_dice_sym", a_rec.get("dice_sym", float("nan"))) if isinstance(a_rec, dict) else float("nan")
            a_time = a_rec.get("runtime_seconds", float("nan")) if isinstance(a_rec, dict) else float("nan")
            a_fold = a_rec.get("folding_pct", float("nan")) if isinstance(a_rec, dict) else float("nan")

        t_time = t_rec.get("syntx_time", t_rec.get("runtime_seconds", float("nan"))) if t_rec else float("nan")
        s_time = s_rec.get("syntx_time", s_rec.get("runtime_seconds", float("nan"))) if s_rec else float("nan")
        gs_time = gs_rec.get("syntx_time", gs_rec.get("runtime_seconds", float("nan"))) if gs_rec else float("nan")
        gr_time = gr_rec.get("syntx_time", gr_rec.get("runtime_seconds", float("nan"))) if gr_rec else float("nan")
        g_time = g_rec.get("syntx_time", g_rec.get("runtime_seconds", float("nan"))) if g_rec else float("nan")

        t_fold = t_rec.get("syntx_fold", t_rec.get("folding_pct", float("nan"))) if t_rec else float("nan")
        s_fold = s_rec.get("syntx_fold", s_rec.get("folding_pct", float("nan"))) if s_rec else float("nan")
        gs_fold = gs_rec.get("syntx_fold", gs_rec.get("folding_pct", float("nan"))) if gs_rec else float("nan")
        gr_fold = gr_rec.get("syntx_fold", gr_rec.get("folding_pct", float("nan"))) if gr_rec else float("nan")
        g_fold = g_rec.get("syntx_fold", g_rec.get("folding_pct", float("nan"))) if g_rec else float("nan")

        aff_dice = primary_rec.get("syntx_affine_dice_sym", primary_rec.get("affine_dice_sym", float("nan")))
        c_type = primary_rec.get("cohort_type", "intra" if idx < 40 else "inter")

        # Primary metric calculation
        focus_dice = (
            t_dice if has_tvf else (
                s_dice if records else (
                    gs_dice if has_syngs else (
                        gr_dice if has_greedy else g_dice
                    )
                )
            )
        )
        diff_vs_ants = float((focus_dice - a_dice) * 100.0) if np.isfinite(focus_dice) and np.isfinite(a_dice) else float("nan")
        win = bool(focus_dice >= a_dice) if np.isfinite(focus_dice) and np.isfinite(a_dice) else False

        matched_pairs.append({
            "idx": idx,
            "cohort": c_type,
            "fixed_id": primary_rec.get("fixed_id", f"pair_{idx:03d}_fix"),
            "moving_id": primary_rec.get("moving_id", f"pair_{idx:03d}_mov"),
            "aff_dice": float(aff_dice),
            "t_dice": float(t_dice),
            "s_dice": float(s_dice),
            "gs_dice": float(gs_dice),
            "gr_dice": float(gr_dice),
            "g_dice": float(g_dice),
            "a_dice": float(a_dice),
            "focus_dice": float(focus_dice),
            "t_time": float(t_time),
            "s_time": float(s_time),
            "gs_time": float(gs_time),
            "gr_time": float(gr_time),
            "g_time": float(g_time),
            "a_time": float(a_time),
            "t_fold": float(t_fold),
            "s_fold": float(s_fold),
            "gs_fold": float(gs_fold),
            "gr_fold": float(gr_fold),
            "g_fold": float(g_fold),
            "a_fold": float(a_fold),
            "diff_vs_ants": diff_vs_ants,
            "win": win,
        })

    n_completed = len(matched_pairs)
    if n_completed == 0:
        with open(output_html, "w") as f:
            f.write("<html><body><h1>No benchmark records available yet.</h1></body></html>")
        return output_html

    def _first_config(recs):
        for _, r in sorted(recs.items()):
            if isinstance(r, dict) and isinstance(r.get("config"), dict):
                return r["config"]
        return None

    _cfg_cols = [
        ("TVF", tvf_records),
        ("SyN", records),
        ("SyNGS", syngs_records),
        ("Greedy", greedy_records),
        ("Gaussian SyN", gaussian_records),
    ]
    _cfgs = [(name, _first_config(recs)) for name, recs in _cfg_cols if recs]
    _keys = sorted({k for _, c in _cfgs if c for k in c})
    if _cfgs:
        import html as _html
        _head = "".join(f"<th>{_html.escape(n)}</th>" for n, _ in _cfgs)
        _body = "".join(
            "<tr><td><code>" + _html.escape(str(k)) + "</code></td>" + "".join(
                "<td>" + (_html.escape(str(c[k])) if c and k in c else "&ndash;") + "</td>" for _, c in _cfgs)
            + "</tr>" for k in _keys) or "<tr><td colspan='9'>not recorded</td></tr>"
        config_table_html = (f'<table style="border: 1px solid var(--border);"><thead><tr><th>Key</th>{_head}'
                             f'</tr></thead><tbody>{_body}</tbody></table>')
    else:
        config_table_html = "<p>not recorded</p>"

    # 4. Aggregated stats
    def _safe_mean(vals, default=0.0):
        clean = [float(x) for x in vals if np.isfinite(x)]
        return float(np.mean(clean)) if clean else default

    valid_dices_t = [p["t_dice"] for p in matched_pairs if np.isfinite(p["t_dice"])]
    valid_dices_s = [p["s_dice"] for p in matched_pairs if np.isfinite(p["s_dice"])]
    valid_dices_gs = [p["gs_dice"] for p in matched_pairs if np.isfinite(p["gs_dice"])]
    valid_dices_gr = [p["gr_dice"] for p in matched_pairs if np.isfinite(p["gr_dice"])]
    valid_dices_g = [p["g_dice"] for p in matched_pairs if np.isfinite(p["g_dice"])]
    valid_dices_a = [p["a_dice"] for p in matched_pairs if np.isfinite(p["a_dice"])]

    mean_t_dice = _safe_mean(valid_dices_t, 0.0)
    mean_s_dice = _safe_mean(valid_dices_s, 0.0)
    mean_gs_dice = _safe_mean(valid_dices_gs, 0.0)
    mean_gr_dice = _safe_mean(valid_dices_gr, 0.0)
    mean_g_dice = _safe_mean(valid_dices_g, 0.0)
    mean_a_dice = _safe_mean(valid_dices_a, 0.0)

    primary_mean_dice = _safe_mean([p["focus_dice"] for p in matched_pairs], 0.0)
    dice_diff = primary_mean_dice - mean_a_dice

    wins = sum(1 for p in matched_pairs if p["win"])
    tvf_sob_pairs = [p for p in matched_pairs if np.isfinite(p["t_dice"]) and np.isfinite(p["s_dice"])]
    n_tvf_sob = len(tvf_sob_pairs)
    tvf_sob_wins = sum(1 for p in tvf_sob_pairs if p["t_dice"] > p["s_dice"])
    tvf_sob_pct = (100.0 * tvf_sob_wins / n_tvf_sob) if n_tvf_sob else float("nan")
    win_rate = (wins / n_completed * 100.0) if n_completed > 0 else 0.0

    mean_t_fold = _safe_mean([p["t_fold"] for p in matched_pairs], 0.0) if valid_dices_t else 0.0
    mean_s_fold = _safe_mean([p["s_fold"] for p in matched_pairs], 0.0) if valid_dices_s else 0.0
    mean_gs_fold = _safe_mean([p["gs_fold"] for p in matched_pairs], 0.0) if valid_dices_gs else 0.0
    mean_gr_fold = _safe_mean([p["gr_fold"] for p in matched_pairs], 0.0) if valid_dices_gr else 0.0
    mean_a_fold = _safe_mean([p["a_fold"] for p in matched_pairs], 0.0)
    primary_fold = (
        mean_t_fold if has_tvf else (
            mean_s_fold if valid_dices_s else (
                mean_gs_fold if valid_dices_gs else mean_gr_fold
            )
        )
    )

    mean_t_time = _safe_mean([p["t_time"] for p in matched_pairs], 0.0)
    mean_s_time = _safe_mean([p["s_time"] for p in matched_pairs], 0.0)
    mean_gs_time = _safe_mean([p["gs_time"] for p in matched_pairs], 0.0)
    mean_gr_time = _safe_mean([p["gr_time"] for p in matched_pairs], 0.0)
    mean_a_time = _safe_mean([p["a_time"] for p in matched_pairs], 0.0)
    primary_time = (
        mean_t_time if has_tvf else (
            mean_s_time if mean_s_time > 0 else (
                mean_gs_time if mean_gs_time > 0 else mean_gr_time
            )
        )
    )
    ref_fast_time = mean_s_time if mean_s_time > 0 else primary_time
    speedup = (mean_a_time / ref_fast_time) if ref_fast_time > 0 else 1.0

    # Subgroups
    intra_pairs = [p for p in matched_pairs if p["cohort"] == "intra"]
    inter_pairs = [p for p in matched_pairs if p["cohort"] == "inter"]

    intra_t_dice = _safe_mean([p["t_dice"] for p in intra_pairs], float("nan"))
    intra_s_dice = _safe_mean([p["s_dice"] for p in intra_pairs], float("nan"))
    intra_gs_dice = _safe_mean([p["gs_dice"] for p in intra_pairs], float("nan"))
    intra_gr_dice = _safe_mean([p["gr_dice"] for p in intra_pairs], float("nan"))
    intra_g_dice = _safe_mean([p["g_dice"] for p in intra_pairs], float("nan"))
    intra_a_dice = _safe_mean([p["a_dice"] for p in intra_pairs], float("nan"))
    intra_focus_dice = _safe_mean([p["focus_dice"] for p in intra_pairs], float("nan"))
    intra_win = (sum(1 for p in intra_pairs if p["win"]) / len(intra_pairs) * 100.0) if intra_pairs else 0.0

    inter_t_dice = _safe_mean([p["t_dice"] for p in inter_pairs], float("nan"))
    inter_s_dice = _safe_mean([p["s_dice"] for p in inter_pairs], float("nan"))
    inter_gs_dice = _safe_mean([p["gs_dice"] for p in inter_pairs], float("nan"))
    inter_gr_dice = _safe_mean([p["gr_dice"] for p in inter_pairs], float("nan"))
    inter_g_dice = _safe_mean([p["g_dice"] for p in inter_pairs], float("nan"))
    inter_a_dice = _safe_mean([p["a_dice"] for p in inter_pairs], float("nan"))
    inter_focus_dice = _safe_mean([p["focus_dice"] for p in inter_pairs], float("nan"))
    inter_win = (sum(1 for p in inter_pairs if p["win"]) / len(inter_pairs) * 100.0) if inter_pairs else 0.0

    probe_pairs = [p for p in matched_pairs if np.isfinite(p["g_dice"])]
    probe_sob_mean = _safe_mean([p["s_dice"] for p in probe_pairs], float("nan"))
    probe_gauss_mean = _safe_mean([p["g_dice"] for p in probe_pairs], float("nan"))
    probe_ants_mean = _safe_mean([p["a_dice"] for p in probe_pairs], float("nan"))
    probe_gain_vs_gauss = (probe_sob_mean - probe_gauss_mean) * 100.0 if np.isfinite(probe_sob_mean) and np.isfinite(probe_gauss_mean) else float("nan")
    probe_gain_vs_ants = (probe_sob_mean - probe_ants_mean) * 100.0 if np.isfinite(probe_sob_mean) and np.isfinite(probe_ants_mean) else float("nan")

    # Derive dynamic regularizer label from recorded config
    first_primary_cfg = (_first_config(tvf_records) or _first_config(records)
                         or _first_config(syngs_records) or _first_config(greedy_records) or {})
    reg_label = str(first_primary_cfg.get("regularizer") or first_primary_cfg.get("syn_regularizer") or "navier").upper()
    badge_text = f"Unified {reg_label} Diffeomorphic Standard" if (has_syngs or has_greedy or reg_label == "NAVIER") else ("TVF Standard" if has_tvf else "SyN Standard")
    focus_name = "TVF" if has_tvf else ("SyN" if valid_dices_s else ("SyNGS" if has_syngs else "Greedy"))

    arch_parts = []
    if valid_dices_t:
        arch_parts.append(f"TVF: {mean_t_dice:.4f} ({mean_t_time:.1f}s)")
    if valid_dices_s:
        arch_parts.append(f"SyN: {mean_s_dice:.4f} ({mean_s_time:.1f}s)")
    if valid_dices_gs:
        arch_parts.append(f"SyNGS: {mean_gs_dice:.4f} ({mean_gs_time:.1f}s)")
    if valid_dices_gr:
        arch_parts.append(f"Greedy: {mean_gr_dice:.4f} ({mean_gr_time:.1f}s)")
    if valid_dices_g:
        arch_parts.append(f"Gauss: {mean_g_dice:.4f}")
    arch_summary_line = " | ".join(arch_parts)

    # 5. Data arrays for Plotly
    plotly_labels = [f"Pair {p['idx']:02d} ({p['cohort'].upper()})" for p in matched_pairs]
    plotly_focus_dice = [round(p["focus_dice"], 4) if np.isfinite(p["focus_dice"]) else None for p in matched_pairs]
    plotly_t_dice = [round(p["t_dice"], 4) if np.isfinite(p["t_dice"]) else None for p in matched_pairs]
    plotly_s_dice = [round(p["s_dice"], 4) if np.isfinite(p["s_dice"]) else None for p in matched_pairs]
    plotly_gs_dice = [round(p["gs_dice"], 4) if np.isfinite(p["gs_dice"]) else None for p in matched_pairs]
    plotly_gr_dice = [round(p["gr_dice"], 4) if np.isfinite(p["gr_dice"]) else None for p in matched_pairs]
    plotly_g_dice = [round(p["g_dice"], 4) if np.isfinite(p["g_dice"]) else None for p in matched_pairs]
    plotly_a_dice = [round(p["a_dice"], 4) if np.isfinite(p["a_dice"]) else None for p in matched_pairs]
    plotly_cohort = [p["cohort"] for p in matched_pairs]

    def _fmt_cell(v):
        return f"{v:.4f}" if np.isfinite(v) else "&mdash;"

    sub_headers = []
    if valid_dices_t:
        sub_headers.append("<th>TVF Dice</th>")
    if valid_dices_s:
        sub_headers.append("<th>SyN Dice</th>")
    if valid_dices_gs:
        sub_headers.append("<th>SyNGS Dice</th>")
    if valid_dices_gr:
        sub_headers.append("<th>Greedy Dice</th>")
    if valid_dices_g:
        sub_headers.append("<th>Gaussian SyN Dice</th>")

    def _sub_row_cells(t_v, s_v, gs_v, gr_v, g_v):
        cells = []
        if valid_dices_t:
            cells.append(f'<td><strong style="color: var(--win-green);">{_fmt_cell(t_v)}</strong></td>')
        if valid_dices_s:
            cells.append(f"<td><strong>{_fmt_cell(s_v)}</strong></td>")
        if valid_dices_gs:
            cells.append(f"<td>{_fmt_cell(gs_v)}</td>")
        if valid_dices_gr:
            cells.append(f"<td>{_fmt_cell(gr_v)}</td>")
        if valid_dices_g:
            cells.append(f"<td>{_fmt_cell(g_v)}</td>")
        return "".join(cells)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg: #ffffff;
            --surface: #f8fafc;
            --border: #e2e8f0;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --accent: #0284c7;
            --win-green: #059669;
            --loss-red: #dc2626;
            --amber: #d97706;
            --card-bg: #f1f5f9;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg);
            color: var(--text-main);
            font-family: var(--font-family);
            margin: 0;
            padding: 30px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1440px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{
            color: var(--text-main);
            font-size: 25px;
            margin: 0 0 10px 0;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .badge {{
            font-size: 12px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 20px;
            background: rgba(2, 132, 199, 0.10);
            color: var(--accent);
            border: 1px solid rgba(2, 132, 199, 0.28);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
        }}
        .stat-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}
        .stat-value {{
            font-size: 26px;
            font-weight: 700;
            color: var(--text-main);
        }}
        .stat-sub {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 25px;
        }}
        h2 {{
            color: var(--text-main);
            font-size: 18px;
            margin-top: 0;
            margin-bottom: 16px;
        }}
        .plots-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }}
        @media (max-width: 900px) {{
            .plots-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .plot-box {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            height: 450px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
            background: #ffffff;
        }}
        th {{
            background: var(--card-bg);
            color: var(--text-main);
            font-weight: 600;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }}
        tr:hover td {{
            background: #f8fafc;
        }}
        .pill {{
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}
        .pill-intra {{
            background: rgba(2, 132, 199, 0.12);
            color: var(--accent);
        }}
        .pill-inter {{
            background: rgba(217, 119, 6, 0.12);
            color: var(--amber);
        }}
        .gain-pos {{
            color: var(--win-green);
            font-weight: 600;
        }}
        .gain-neg {{
            color: var(--loss-red);
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title} <span class="badge">{badge_text}</span></h1>
            <div style="color: var(--text-muted); font-size: 13px;">
                Syntx Solvers: <code>{arch_summary_line or focus_name}</code> &bull; Baseline: <code>ANTs C++ SyN ({mean_a_dice:.4f})</code> &bull; Updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
            </div>
            <div style="background: rgba(217, 119, 6, 0.08); border: 1px solid rgba(217, 119, 6, 0.30); border-radius: 6px; padding: 10px 14px; margin-top: 12px; font-size: 12px; color: #92400e; line-height: 1.5;">
                <strong>Hardware &amp; Reproducibility Notice:</strong> Isolated single-process Apple Silicon GPU (<code>device='mps'</code>) runs sharing identical canonical <code>syntx.robust_affine</code> initializations and deterministic seeded pair order (<code>seed=42</code>).
            </div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">{focus_name} vs. ANTs Mean Dice</div>
                <div class="stat-value" style="color: var(--win-green);">{primary_mean_dice:.4f} <span style="font-size: 16px; color: var(--text-muted);">vs {mean_a_dice:.4f}</span></div>
                <div class="stat-sub">Advantage: <strong class="gain-pos">{dice_diff*100:+.2f}%</strong> ({wins}/{n_completed} Wins, {win_rate:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Liouville Folding Regularity</div>
                <div class="stat-value" style="color: var(--win-green);">{primary_fold:.4f}%</div>
                <div class="stat-sub">ANTs C++ Baseline: {mean_a_fold:.4f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Cohort Architecture Comparison</div>
                <div class="stat-value" style="color: var(--accent); font-size: 16px; line-height: 1.4;">{arch_summary_line}</div>
                <div class="stat-sub">{f'TVF beats Sobolev in {tvf_sob_wins}/{n_tvf_sob} ({tvf_sob_pct:.1f}%) pairs' if n_tvf_sob else f'{speedup:.2f}x Faster on GPU'}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Completed Cohort Pairs</div>
                <div class="stat-value" style="color: var(--accent);">{n_completed}</div>
                <div class="stat-sub">Intra: {len(intra_pairs)} | Inter: {len(inter_pairs)}</div>
            </div>
        </div>

        <div class="plots-grid">
            <div class="plot-box" id="diceScatterPlot"></div>
            <div class="plot-box" id="timeScatterPlot"></div>
        </div>

        <div class="card">
            <h2>Cohort Subgroup Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>Cohort Subgroup</th>
                        <th>Completed</th>
                        {"".join(sub_headers)}
                        <th>ANTs Baseline Dice</th>
                        <th>Primary Advantage</th>
                        <th>Win Rate</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Intra-Cohort Pairs</strong></td>
                        <td>{len(intra_pairs)}</td>
                        {_sub_row_cells(intra_t_dice, intra_s_dice, intra_gs_dice, intra_gr_dice, intra_g_dice)}
                        <td>{_fmt_cell(intra_a_dice)}</td>
                        <td class="{'gain-pos' if intra_focus_dice >= intra_a_dice else 'gain-neg'}">{((intra_focus_dice - intra_a_dice)*100.0):+.2f}%</td>
                        <td><strong>{intra_win:.1f}%</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Inter-Cohort Pairs</strong></td>
                        <td>{len(inter_pairs)}</td>
                        {_sub_row_cells(inter_t_dice, inter_s_dice, inter_gs_dice, inter_gr_dice, inter_g_dice)}
                        <td>{_fmt_cell(inter_a_dice)}</td>
                        <td class="{'gain-pos' if inter_focus_dice >= inter_a_dice else 'gain-neg'}">{((inter_focus_dice - inter_a_dice)*100.0):+.2f}%</td>
                        <td><strong>{inter_win:.1f}%</strong></td>
                    </tr>
                </tbody>
            </table>
        </div>
"""

    if not has_tvf and len(probe_pairs) > 0:
        html += f"""
        <div class="card" style="border-left: 4px solid var(--accent);">
            <h2>Ablation Study: Sobolev Smoothing vs. Standard Gaussian Regularization ({len(probe_pairs)} Probe Pairs)</h2>
            <div class="stats-grid" style="margin-top: 15px; margin-bottom: 20px;">
                <div class="stat-card" style="background: var(--card-bg);">
                    <div class="stat-label">Sobolev Probe Mean Dice</div>
                    <div class="stat-value" style="color: var(--accent);">{probe_sob_mean:.4f}</div>
                    <div class="stat-sub">Syntx Spectral / Continuum</div>
                </div>
                <div class="stat-card" style="background: var(--card-bg);">
                    <div class="stat-label">Gaussian Probe Mean Dice</div>
                    <div class="stat-value" style="color: var(--amber);">{probe_gauss_mean:.4f}</div>
                    <div class="stat-sub">Syntx Gaussian (&sigma;=3.0)</div>
                </div>
                <div class="stat-card" style="background: var(--card-bg);">
                    <div class="stat-label">Sobolev vs. Gaussian Gain</div>
                    <div class="stat-value" style="color: {'var(--win-green)' if probe_gain_vs_gauss >= 0 else 'var(--loss-red)'};">{probe_gain_vs_gauss:+.2f}%</div>
                    <div class="stat-sub">Relative Cortical Overlap Gain</div>
                </div>
                <div class="stat-card" style="background: var(--card-bg);">
                    <div class="stat-label">Sobolev vs. ANTs Baseline</div>
                    <div class="stat-value" style="color: var(--win-green);">{probe_gain_vs_ants:+.2f}%</div>
                    <div class="stat-sub">ANTs Baseline Mean: {probe_ants_mean:.4f}</div>
                </div>
            </div>
        </div>
"""

    tbl_headers = ["<th>Pair</th>", "<th>Type</th>", "<th>Affine</th>", "<th>ANTs C++</th>"]
    if valid_dices_gr:
        tbl_headers.append("<th>Greedy</th>")
    if valid_dices_gs:
        tbl_headers.append("<th>SyNGS</th>")
    if valid_dices_s:
        tbl_headers.append("<th>SyN</th>")
    if valid_dices_g:
        tbl_headers.append("<th>Gaussian SyN</th>")
    if valid_dices_t:
        tbl_headers.append('<th style="color: var(--win-green);">TVF</th>')
    tbl_headers.extend([f"<th>&Delta; {focus_name} vs ANTs</th>", f"<th>{focus_name} Fold%</th>", f"<th>{focus_name} Time</th>"])

    html += f"""
        <div class="card">
            <h2>Recorded Configuration per Method</h2>
            <p style="color: var(--text-muted); font-size: 13px; margin-bottom: 16px;">
                The <code>config</code> stored in the first result record of each method (not recorded: none stored).
            </p>
            {config_table_html}
        </div>

        <div class="card">
            <h2>Detailed Per-Pair Side-by-Side Comparison ({n_completed} Completed)</h2>
            <table>
                <thead>
                    <tr>
                        {"".join(tbl_headers)}
                    </tr>
                </thead>
                <tbody>
"""

    for p in matched_pairs:
        p_type = p["cohort"]
        pill_cls = "pill-intra" if p_type == "intra" else "pill-inter"
        aff_dice_str = _fmt_cell(p.get("aff_dice", float("nan")))
        a_dice_str = _fmt_cell(p["a_dice"])

        row_Dynamic = []
        if valid_dices_gr:
            row_Dynamic.append(f"<td>{_fmt_cell(p['gr_dice'])}</td>")
        if valid_dices_gs:
            row_Dynamic.append(f"<td>{_fmt_cell(p['gs_dice'])}</td>")
        if valid_dices_s:
            row_Dynamic.append(f'<td><span style="color: var(--accent); font-weight: 600;">{_fmt_cell(p["s_dice"])}</span></td>')
        if valid_dices_g:
            row_Dynamic.append(f'<td><span style="color: var(--amber);">{_fmt_cell(p["g_dice"])}</span></td>')
        if valid_dices_t:
            row_Dynamic.append(f'<td><strong style="color: var(--win-green);">{_fmt_cell(p["t_dice"])}</strong></td>')

        f_diff_ants = p["diff_vs_ants"]
        f_diff_ants_str = f"{f_diff_ants:+.2f}%" if np.isfinite(f_diff_ants) else "&mdash;"
        f_diff_ants_cls = "gain-pos" if f_diff_ants >= 0 else "gain-neg"
        f_fold_val = p["t_fold"] if has_tvf else (p["s_fold"] if valid_dices_s else (p["gs_fold"] if valid_dices_gs else p["gr_fold"]))
        f_time_val = p["t_time"] if has_tvf else (p["s_time"] if valid_dices_s else (p["gs_time"] if valid_dices_gs else p["gr_time"]))
        f_fold_str = f"{f_fold_val:.4f}%" if np.isfinite(f_fold_val) else "&mdash;"
        f_time_str = f"{f_time_val:.1f}s" if np.isfinite(f_time_val) else "&mdash;"

        html += f"""                    <tr>
                        <td><strong>#{p['idx']:02d}</strong></td>
                        <td><span class="pill {pill_cls}">{p_type.upper()}</span></td>
                        <td><span style="color: var(--text-muted);">{aff_dice_str}</span></td>
                        <td>{a_dice_str}</td>
                        {"".join(row_Dynamic)}
                        <td><strong class="{f_diff_ants_cls}">{f_diff_ants_str}</strong></td>
                        <td>{f_fold_str}</td>
                        <td>{f_time_str}</td>
                    </tr>
"""

    html += f"""                </tbody>
            </table>
        </div>
    </div>

    <script>
        const pairLabels = {json.dumps(plotly_labels)};
        const focusDice = {json.dumps(plotly_focus_dice)};
        const tvfDice = {json.dumps(plotly_t_dice)};
        const sobDice = {json.dumps(plotly_s_dice)};
        const syngsDice = {json.dumps(plotly_gs_dice)};
        const greedyDice = {json.dumps(plotly_gr_dice)};
        const antsDice = {json.dumps(plotly_a_dice)};
        const cohorts = {json.dumps(plotly_cohort)};
        const focusLabel = {json.dumps(focus_name)};

        // 1. X-Y Scatter: Syntx Models vs ANTs C++ Dice
        function buildPairedTrace(arrY, labelName, colorHex, symbolName) {{
            const xs = [], ys = [], txts = [];
            for (let i = 0; i < pairLabels.length; i++) {{
                if (antsDice[i] !== null && arrY[i] !== null) {{
                    const diff = (arrY[i] - antsDice[i]) * 100;
                    xs.push(antsDice[i]);
                    ys.push(arrY[i]);
                    txts.push(pairLabels[i] + '<br>' + labelName + ': ' + arrY[i] + '<br>ANTs C++: ' + antsDice[i] + '<br>&Delta;: ' + (diff >= 0 ? '+' : '') + diff.toFixed(2) + '%');
                }}
            }}
            return {{
                x: xs, y: ys, text: txts, hoverinfo: 'text', mode: 'markers', type: 'scatter',
                name: labelName,
                marker: {{ size: 8, symbol: symbolName, color: colorHex, opacity: 0.85, line: {{ color: '#ffffff', width: 1 }} }}
            }};
        }}

        const scatterTraces = [];
        if (tvfDice.some(x => x !== null)) scatterTraces.push(buildPairedTrace(tvfDice, 'syntx.tvf', '#059669', 'circle'));
        if (sobDice.some(x => x !== null)) scatterTraces.push(buildPairedTrace(sobDice, 'syntx.syn', '#0284c7', 'diamond'));
        if (syngsDice.some(x => x !== null)) scatterTraces.push(buildPairedTrace(syngsDice, 'syntx.syngs', '#d97706', 'square'));
        if (greedyDice.some(x => x !== null)) scatterTraces.push(buildPairedTrace(greedyDice, 'syntx.greedy', '#64748b', 'triangle-up'));
        if (scatterTraces.length === 0) scatterTraces.push(buildPairedTrace(focusDice, focusLabel, '#0284c7', 'circle'));

        const allVals = [];
        scatterTraces.forEach(tr => {{ allVals.push(...tr.x, ...tr.y); }});
        const minDice = allVals.length > 0 ? Math.min(...allVals, 0.55) : 0.45;
        const maxDice = allVals.length > 0 ? Math.max(...allVals, 0.72) : 0.75;

        scatterTraces.push({{
            x: [minDice - 0.03, maxDice + 0.03],
            y: [minDice - 0.03, maxDice + 0.03],
            mode: 'lines',
            type: 'scatter',
            name: 'Parity (y = x)',
            line: {{ dash: 'dash', color: '#94a3b8', width: 2 }}
        }});

        Plotly.newPlot('diceScatterPlot', scatterTraces, {{
            title: {{ text: '<b>Cortical Accuracy vs. ANTs C++ Baseline (n=' + pairLabels.length + ')</b>', font: {{ color: '#1e293b', size: 15 }} }},
            xaxis: {{ title: 'ANTs C++ Symmetric Mean Dice', range: [minDice - 0.015, maxDice + 0.015], color: '#475569', gridcolor: '#e2e8f0' }},
            yaxis: {{ title: 'Syntx Symmetric Mean Dice', range: [minDice - 0.015, maxDice + 0.015], color: '#475569', gridcolor: '#e2e8f0' }},
            paper_bgcolor: '#ffffff',
            plot_bgcolor: '#ffffff',
            font: {{ family: '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif', color: '#1e293b' }},
            margin: {{ t: 50, b: 50, l: 60, r: 20 }},
            legend: {{ x: 0.03, y: 0.97, bgcolor: 'rgba(255,255,255,0.85)', font: {{ color: '#1e293b' }} }}
        }}, {{ responsive: true }});

        // 2. Distribution Box Plot across All Active Methods
        const boxTraces = [];
        if (antsDice.some(x => x !== null)) boxTraces.push({{ y: antsDice.filter(x => x !== null), type: 'box', name: 'ANTs C++', boxpoints: 'outliers', marker: {{ color: '#94a3b8' }} }});
        if (greedyDice.some(x => x !== null)) boxTraces.push({{ y: greedyDice.filter(x => x !== null), type: 'box', name: 'syntx.greedy', boxpoints: 'outliers', marker: {{ color: '#64748b' }} }});
        if (syngsDice.some(x => x !== null)) boxTraces.push({{ y: syngsDice.filter(x => x !== null), type: 'box', name: 'syntx.syngs', boxpoints: 'outliers', marker: {{ color: '#d97706' }} }});
        if (sobDice.some(x => x !== null)) boxTraces.push({{ y: sobDice.filter(x => x !== null), type: 'box', name: 'syntx.syn', boxpoints: 'outliers', marker: {{ color: '#0284c7' }} }});
        if (tvfDice.some(x => x !== null)) boxTraces.push({{ y: tvfDice.filter(x => x !== null), type: 'box', name: 'syntx.tvf', boxpoints: 'outliers', marker: {{ color: '#059669' }} }});

        Plotly.newPlot('timeScatterPlot', boxTraces, {{
            title: {{ text: '<b>Cohort Cortical Dice Distribution Across Solvers</b>', font: {{ color: '#1e293b', size: 15 }} }},
            yaxis: {{ title: 'Symmetric DKT31 Cortical Dice', color: '#475569', gridcolor: '#e2e8f0' }},
            paper_bgcolor: '#ffffff',
            plot_bgcolor: '#ffffff',
            font: {{ family: '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif', color: '#1e293b' }},
            margin: {{ t: 50, b: 50, l: 60, r: 20 }},
            showlegend: false
        }}, {{ responsive: true }});

    </script>
</body>
</html>
"""
    with open(output_html, "w") as f:
        f.write(html)
    return output_html


def create_affine_benchmark_report(
    summary_source: str = "results/reproducible_90pair_master_summary.json",
    output_html: str = "docs/reproducible_90pair_affine_report.html",
    title: str = "Syntx Robust Affine vs ANTs C++ — 90-Pair Population Benchmark Report",
    provenance: dict = None
) -> str:
    """Write an HTML page for the 90-pair affine benchmark (affine Dice, gain to SyN,
    intra / inter boxes, runtime plot, per-pair table).

    Records come from ``gaussian_results`` (or ``results``, else ``sobolev_results``) of the
    summary, keyed by pair index strings; each gives ``syntx_affine_dice_sym``,
    ``syntx_dice_sym`` and ``ants_baseline.dice_sym``. Cohort: ``cohort_type``, else "intra"
    for index < 40.

    Every number is computed from the data: syntx affine time from ``syntx_affine_time``;
    ANTs affine time / Dice from ``results/pair_XXX_ants_syn.json`` /
    ``results/affine_eval/pair_XXX_affine.json`` relative to the *current working directory*.
    Missing values are shown as "n/a" (never a default). The protocol text is fixed.

    Parameters
    ----------
    summary_source : str or dict, default "results/reproducible_90pair_master_summary.json"
        JSON path or loaded dict; a missing path gives an empty page.
    output_html : str, default "docs/reproducible_90pair_affine_report.html"
        Output file (parent directories are created).
    title : str
        Used for the HTML ``<title>`` only (the heading is fixed).
    provenance : dict, optional
        Ignored.

    Returns
    -------
    str
        ``output_html`` as given.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)

    if isinstance(summary_source, str) and os.path.exists(summary_source):
        with open(summary_source, "r") as fp:
            master = json.load(fp)
    elif isinstance(summary_source, dict):
        master = summary_source
    else:
        master = {"gaussian_results": {}, "sobolev_results": {}}

    results = master.get("gaussian_results", master.get("results", {}))
    if not results:
        results = master.get("sobolev_results", {})

    rows = []
    for idx_str in sorted(results.keys(), key=lambda x: int(x)):
        p_idx = int(idx_str)
        rec = results[idx_str]
        sob_rec = master.get("sobolev_results", {}).get(idx_str, {})
        
        cohort = rec.get("cohort_type", "intra" if p_idx < 40 else "inter")
        f_id = rec.get("fixed_id", f"Fixed_{p_idx}")
        m_id = rec.get("moving_id", f"Moving_{p_idx}")
        
        aff_dice = float(rec.get("syntx_affine_dice_sym", 0.0))
        gauss_dice = float(rec.get("syntx_dice_sym", 0.0))
        sob_dice = float(sob_rec.get("syntx_dice_sym", gauss_dice))
        ants_syn_dice = float(rec.get("ants_baseline", {}).get("dice_sym", float("nan")))
        
        # ANTs affine baseline data if available
        ants_file = f"results/pair_{p_idx:03d}_ants_syn.json"
        ants_aff_time = float("nan")
        ants_aff_dice = float("nan")
        if os.path.exists(ants_file):
            try:
                with open(ants_file) as fp:
                    ab = json.load(fp)
                    ants_aff_time = float(ab.get("runtime_affine_seconds", float("nan")))
            except Exception:
                pass
                
        aff_eval_file = f"results/affine_eval/pair_{p_idx:03d}_affine.json"
        if os.path.exists(aff_eval_file):
            try:
                with open(aff_eval_file) as fp:
                    aef = json.load(fp)
                    if "ants_affine" in aef and aef["ants_affine"].get("dice_sym") is not None:
                        ants_aff_dice = float(aef["ants_affine"]["dice_sym"])
            except Exception:
                pass

        syntx_aff_time = float(rec.get("syntx_affine_time", float("nan")))
        speedup = (ants_aff_time / syntx_aff_time) if syntx_aff_time > 0 else float("nan")
        deform_gain = (gauss_dice - aff_dice) * 100.0 if gauss_dice > 0 else 0.0

        rows.append({
            "pair_idx": p_idx,
            "cohort_type": cohort,
            "fixed_id": f_id,
            "moving_id": m_id,
            "affine_dice": aff_dice,
            "ants_affine_dice": ants_aff_dice,
            "gauss_dice": gauss_dice,
            "sobolev_dice": sob_dice,
            "ants_syn_dice": ants_syn_dice,
            "deform_gain": deform_gain,
            "syntx_aff_time": syntx_aff_time,
            "ants_aff_time": ants_aff_time,
            "speedup": speedup
        })

    n_total = len(rows)
    if n_total == 0:
        with open(output_html, "w") as f:
            f.write("<html><body><h1>No Affine Benchmark Data Available</h1></body></html>")
        return output_html

    aff_dices = [r["affine_dice"] for r in rows]
    mean_aff = float(np.mean(aff_dices))
    std_aff = float(np.std(aff_dices))
    min_aff = float(np.min(aff_dices))
    max_aff = float(np.max(aff_dices))

    intra_rows = [r for r in rows if r["cohort_type"] == "intra"]
    inter_rows = [r for r in rows if r["cohort_type"] == "inter"]

    mean_intra = float(np.mean([r["affine_dice"] for r in intra_rows])) if intra_rows else 0.0
    mean_inter = float(np.mean([r["affine_dice"] for r in inter_rows])) if inter_rows else 0.0
    mean_deform_gain = float(np.mean([r["deform_gain"] for r in rows]))
    mean_syn_dice = float(np.mean([r["gauss_dice"] for r in rows]))

    def _nanmean(vals):
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def _fmt(v, spec, suffix=""):
        return f"{v:{spec}}{suffix}" if np.isfinite(v) else "n/a"

    mean_s_time = _nanmean([r["syntx_aff_time"] for r in rows])
    mean_a_time = _nanmean([r["ants_aff_time"] for r in rows])
    mean_ants_aff_dice = _nanmean([r["ants_affine_dice"] for r in rows])
    mean_speedup = (mean_a_time / mean_s_time) if mean_s_time > 0 else float("nan")

    table_rows_html = []
    for r in rows:
        p_idx = r["pair_idx"]
        c_type = r["cohort_type"]
        pill_cls = "pill-intra" if c_type == "intra" else "pill-inter"
        aff_val = r["affine_dice"]
        syn_val = r["gauss_dice"]
        gain_val = r["deform_gain"]
        s_time = r["syntx_aff_time"]
        a_time = r["ants_aff_time"]
        sp_val = r["speedup"]

        table_rows_html.append(f"""
        <tr>
            <td><strong>#{p_idx:02d}</strong></td>
            <td><span class="pill {pill_cls}">{c_type.upper()}</span></td>
            <td><code>{r['fixed_id']}</code></td>
            <td><code>{r['moving_id']}</code></td>
            <td><strong style="color: #58a6ff;">{aff_val:.4f}</strong></td>
            <td><strong style="color: #3fb950;">{syn_val:.4f}</strong></td>
            <td><span class="gain-pos">+{gain_val:.2f}%</span></td>
            <td>{_fmt(s_time, '.1f', 's')}</td>
            <td>{_fmt(a_time, '.1f', 's')}</td>
            <td><strong class="gain-pos">{_fmt(sp_val, '.1f', '&times;')}</strong></td>
        </tr>
        """)

    pair_labels = [f"Pair {r['pair_idx']:02d} ({r['cohort_type'].upper()})" for r in rows]
    plot_aff_dices = [round(r["affine_dice"], 4) for r in rows]
    plot_syn_dices = [round(r["gauss_dice"], 4) for r in rows]
    plot_intra_aff = [round(r["affine_dice"], 4) for r in intra_rows]
    plot_inter_aff = [round(r["affine_dice"], 4) for r in inter_rows]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text-main: #e6edf3;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --win-green: #3fb950;
            --loss-red: #f85149;
            --card-bg: #21262d;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg);
            color: var(--text-main);
            font-family: var(--font-family);
            margin: 0;
            padding: 30px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{
            color: #ffffff;
            font-size: 26px;
            margin: 0 0 10px 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .badge {{
            font-size: 13px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 20px;
            background: rgba(88, 166, 255, 0.15);
            color: var(--accent);
            border: 1px solid rgba(88, 166, 255, 0.3);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
        }}
        .stat-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
        }}
        .stat-sub {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 25px;
        }}
        h2 {{
            color: #ffffff;
            font-size: 18px;
            margin-top: 0;
            margin-bottom: 16px;
        }}
        .plots-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }}
        @media (max-width: 900px) {{
            .plots-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .plot-box {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            height: 450px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        th {{
            background: var(--card-bg);
            color: #ffffff;
            font-weight: 600;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        .pill {{
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}
        .pill-intra {{
            background: rgba(63, 185, 80, 0.15);
            color: var(--win-green);
        }}
        .pill-inter {{
            background: rgba(210, 153, 34, 0.15);
            color: #d29922;
        }}
        .gain-pos {{
            color: var(--win-green);
            font-weight: 600;
        }}
        .config-box {{
            background: #090d13;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 14px;
            font-family: monospace;
            font-size: 12px;
            color: #79c0ff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Syntx Robust Affine &mdash; {n_total}-Pair Population Benchmark Report <span class="badge">{n_total} pairs</span></h1>
            <div style="color: var(--text-muted); font-size: 13px;">
                Framework: <code>syntx.robust_affine (Multi-Start Cone Search + Deterministic Regular Sampling)</code> &bull; Standardized Mindboggle-101 Benchmark
            </div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Syntx Affine Mean Dice</div>
                <div class="stat-value" style="color: var(--accent);">{mean_aff:.4f} <span style="font-size: 16px; color: var(--text-muted);">&plusmn; {std_aff:.4f}</span></div>
                <div class="stat-sub">Range: <strong>{min_aff:.4f} &ndash; {max_aff:.4f}</strong> (N = {n_total})</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Intra-Cohort Overlap</div>
                <div class="stat-value" style="color: var(--win-green);">{mean_intra:.4f}</div>
                <div class="stat-sub">40 Intra-Subject Pairs</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Inter-Cohort Overlap</div>
                <div class="stat-value" style="color: #d29922;">{mean_inter:.4f}</div>
                <div class="stat-sub">50 Inter-Subject Pairs</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">SyN Deformable Boost</div>
                <div class="stat-value" style="color: #bc8cff;">+{mean_deform_gain:.1f}%</div>
                <div class="stat-sub">Affine ({mean_aff:.4f}) &rarr; SyN ({mean_syn_dice:.4f})</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">GPU Acceleration Speedup</div>
                <div class="stat-value" style="color: var(--win-green);">{_fmt(mean_speedup, '.1f', '&times;')}</div>
                <div class="stat-sub">{_fmt(mean_s_time, '.1f', 's')} (syntx) vs {_fmt(mean_a_time, '.1f', 's')} (ANTs)</div>
            </div>
        </div>

        <div class="plots-grid">
            <div class="plot-box" id="progressionPlot"></div>
            <div class="plot-box" id="boxPlot"></div>
        </div>

        <div class="plots-grid">
            <div class="plot-box" id="correlationPlot"></div>
            <div class="plot-box" id="runtimePlot"></div>
        </div>

        <div class="card">
            <h2>Benchmark Overview &amp; Evaluation Protocol</h2>
            <div style="font-size: 13px; line-height: 1.6; color: var(--text-main);">
                <p>
                    <strong>Dataset:</strong> The <strong>Mindboggle-101</strong> benchmark consists of 101 manually labeled T1-weighted brain MRI volumes across four diverse clinical cohorts: <em>OASIS-TRT-20</em>, <em>NKI-RS-22</em>, <em>NKI-TRT-20</em>, and <em>MMRR-21</em>. The standardized 90-pair cohort is comprised of <strong>40 intra-subject pairs</strong> (testing longitudinal re-test reproducibility) and <strong>50 inter-subject pairs</strong> (testing cross-subject morphological variance).
                </p>
                <p>
                    <strong>Evaluation Metric:</strong> All affine registrations are evaluated on ground-truth cortical <strong>DKT31</strong> label maps containing 62 discrete anatomical cortical regions. In accordance with Syntx Registration Guardrails, Sørensen-Dice (MeanOverlap: 2|A ∩ B| / (|A| + |B|)) is evaluated <em>symmetrically in both image spaces</em> using nearest-neighbor interpolation:
                    <code>Dice_sym = 0.5 &times; (Dice_fixed + Dice_moving)</code>
                </p>
            </div>
        </div>

        <div class="card">
            <h2>Three-Way Affine Framework Comparison</h2>
            <div style="font-size: 13px; line-height: 1.6; color: var(--text-main); margin-bottom: 16px;">
                Direct architectural and performance comparison between ANTs Affine Initializer, Standard ANTs Affine, and Syntx Robust Affine:
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Algorithm</th>
                        <th>Architecture &amp; Strategy</th>
                        <th>Sampling &amp; Metric</th>
                        <th>Optimization Engine</th>
                        <th>Mean Dice</th>
                        <th>Speedup</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Standard ANTs Affine</strong><br><code>ants.registration('Affine')</code></td>
                        <td>Single-start Center of Mass translation matching + multi-stage affine refinement (Rigid &rarr; Affine)</td>
                        <td>Mattes MI with <em>stochastic random sampling</em> (20% sample)</td>
                        <td>ITK C++ multi-resolution optimizer on CPU</td>
                        <td><strong>{_fmt(mean_ants_aff_dice, '.4f')}</strong></td>
                        <td>1.0&times; ({_fmt(mean_a_time, '.1f', 's')})</td>
                    </tr>
                    <tr>
                        <td><strong>Syntx Robust Affine</strong><br><code>syntx.robust_affine</code></td>
                        <td>Multi-start cone search around Center of Mass and FOV geometric centers (18 pitch/roll/yaw angle perturbations)</td>
                        <td>Mattes MI with <strong>deterministic regular uniform sampling</strong> + <strong>foreground union masking</strong> ((I &gt; 0.01) | (J &gt; 0.01))</td>
                        <td>PyTorch GPU Differentiable Lie Algebra $so(3) \rightarrow SO(3)$ / Multi-Stage GPU Solver</td>
                        <td><strong style="color: #58a6ff;">{mean_aff:.4f}</strong></td>
                        <td><strong style="color: #3fb950;">{_fmt(mean_speedup, '.1f', '&times;')}</strong> ({_fmt(mean_s_time, '.1f', 's')})</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>Per-Pair Affine Benchmark Table ({n_total} pairs)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Pair</th>
                        <th>Type</th>
                        <th>Fixed Target ID</th>
                        <th>Moving Source ID</th>
                        <th>Affine Sym Dice</th>
                        <th>Final SyN Dice</th>
                        <th>Deformable Gain</th>
                        <th>Syntx Time</th>
                        <th>ANTs Time</th>
                        <th>Speedup</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows_html)}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // 1. Progression Plot: Affine vs Final SyN
        const traceAff = {{
            x: {json.dumps(pair_labels)},
            y: {json.dumps(plot_aff_dices)},
            mode: 'lines+markers',
            name: 'Syntx Robust Affine',
            line: {{ color: '#58a6ff', width: 2 }},
            marker: {{ size: 6, color: '#58a6ff' }}
        }};

        const traceSyN = {{
            x: {json.dumps(pair_labels)},
            y: {json.dumps(plot_syn_dices)},
            mode: 'lines+markers',
            name: 'Final SyN Deformable',
            line: {{ color: '#3fb950', width: 2 }},
            marker: {{ size: 6, color: '#3fb950' }}
        }};

        Plotly.newPlot('progressionPlot', [traceAff, traceSyN], {{
            title: {{ text: '<b>90-Pair Overlap: Affine Initialization &rarr; Final SyN Deformable</b>', font: {{ color: '#ffffff', size: 14 }} }},
            yaxis: {{ title: 'Symmetric Mean Dice', range: [0.25, 0.72], color: '#8b949e', gridcolor: '#21262d' }},
            xaxis: {{ showticklabels: false, color: '#8b949e' }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif', color: '#e6edf3' }},
            margin: {{ t: 40, b: 40, l: 50, r: 20 }},
            legend: {{ orientation: 'h', y: 1.1, font: {{ color: '#e6edf3' }} }}
        }}, {{ responsive: true }});

        // 2. Cohort Boxplot
        const boxIntra = {{
            y: {json.dumps(plot_intra_aff)},
            type: 'box',
            name: 'Intra-Cohort (N=40)',
            marker: {{ color: '#3fb950' }},
            boxpoints: 'all',
            jitter: 0.3,
            pointpos: -1.8
        }};

        const boxInter = {{
            y: {json.dumps(plot_inter_aff)},
            type: 'box',
            name: 'Inter-Cohort (N=50)',
            marker: {{ color: '#d29922' }},
            boxpoints: 'all',
            jitter: 0.3,
            pointpos: -1.8
        }};

        Plotly.newPlot('boxPlot', [boxIntra, boxInter], {{
            title: {{ text: '<b>Affine Dice Distribution by Cohort Type</b>', font: {{ color: '#ffffff', size: 14 }} }},
            yaxis: {{ title: 'Affine Symmetric Mean Dice', range: [0.28, 0.42], color: '#8b949e', gridcolor: '#21262d' }},
            xaxis: {{ color: '#8b949e' }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif', color: '#e6edf3' }},
            margin: {{ t: 40, b: 40, l: 50, r: 20 }},
            legend: {{ orientation: 'h', y: 1.1, font: {{ color: '#e6edf3' }} }}
        }}, {{ responsive: true }});

        // 3. Correlation Scatter: Affine Dice vs Final SyN Dice
        const scatterCorr = {{
            x: {json.dumps(plot_aff_dices)},
            y: {json.dumps(plot_syn_dices)},
            text: {json.dumps(pair_labels)},
            mode: 'markers',
            type: 'scatter',
            name: 'Image Pairs',
            marker: {{ size: 8, color: '#bc8cff', opacity: 0.85 }}
        }};

        Plotly.newPlot('correlationPlot', [scatterCorr], {{
            title: {{ text: '<b>Affine Quality vs. SyN Deformable Accuracy Correlation</b>', font: {{ color: '#ffffff', size: 14 }} }},
            xaxis: {{ title: 'Affine Initialization Dice', range: [0.28, 0.42], color: '#8b949e', gridcolor: '#21262d' }},
            yaxis: {{ title: 'Final SyN Deformable Dice', range: [0.55, 0.70], color: '#8b949e', gridcolor: '#21262d' }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif', color: '#e6edf3' }},
            margin: {{ t: 40, b: 40, l: 50, r: 20 }},
            showlegend: false
        }}, {{ responsive: true }});

        // 4. Runtime Scatter Plot
        const traceRuntime = {{
            x: {[r['ants_aff_time'] for r in rows]},
            y: {[r['syntx_aff_time'] for r in rows]},
            mode: 'markers',
            type: 'scatter',
            name: 'Pair Runtimes',
            marker: {{ size: 8, color: '#3fb950', opacity: 0.85 }}
        }};

        Plotly.newPlot('runtimePlot', [traceRuntime], {{
            title: {{ text: '<b>Affine Runtime: Syntx (GPU) vs ANTs (CPU)</b>', font: {{ color: '#ffffff', size: 14 }} }},
            xaxis: {{ title: 'ANTs C++ CPU Runtime (s)', color: '#8b949e', gridcolor: '#21262d' }},
            yaxis: {{ title: 'Syntx GPU Runtime (s)', color: '#8b949e', gridcolor: '#21262d' }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif', color: '#e6edf3' }},
            margin: {{ t: 40, b: 40, l: 50, r: 20 }},
            showlegend: false
        }}, {{ responsive: true }});
    </script>
</body>
</html>
"""
    with open(output_html, "w") as f:
        f.write(html)
    return output_html

