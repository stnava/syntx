"""
syntx.template — Optimal Template Construction Ported from ANTsPy
=================================================================

Direct port of ``ants.build_template`` with quantitative shape residual tracking:
- Computes Euclidean (RMS L2), membrane, and bending energy of the mean deformable warp field.
- Iteration 0 shape residual quantifies the initial deformable maps to the initial template.
- Supports weighted averages, useNoRigid affine averaging, and sharpen-blending.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import List, Optional, Sequence, Union, Dict, Any

import numpy as np
import ants

__all__ = ["build_template"]


def _initial_template(image_list: List[ants.ANTsImage], weights: Sequence[float]) -> ants.ANTsImage:
    """Compute weighted initial template from image list."""
    initial_template = image_list[0] * 0
    for i in range(len(image_list)):
        temp = image_list[i] * float(weights[i])
        temp = ants.resample_image_to_target(temp, initial_template)
        initial_template = initial_template + temp
    return initial_template


def _scale_warp(warp_img: ants.ANTsImage, scale: float) -> ants.ANTsImage:
    """Scale displacement field components by a scalar."""
    return warp_img * float(scale)


def _normalize_weights(weights: Optional[Sequence[float]], n: int) -> np.ndarray:
    """Normalise scalar weights to a list summing to 1.0."""
    if weights is None:
        return np.ones(n, dtype=np.float64) / n
    w = np.asarray(weights, dtype=np.float64)
    if len(w) != n:
        raise ValueError(f"len(weights)={len(w)} must equal len(image_list)={n}")
    if np.any(w < 0):
        raise ValueError("All weights must be non-negative.")
    s = w.sum()
    if s <= 0:
        raise ValueError("Weights must have a positive sum.")
    return w / s


def _compute_shape_residual(mean_warp_img: ants.ANTsImage) -> Dict[str, float]:
    """
    Compute shape residual metrics of the mean warp field ψ̄:
    - l2_norm: RMS displacement magnitude (Euclidean, mm)
    - membrane_energy: Mean Frobenius norm of Jacobian (1st-order spatial derivative)
    - bending_energy: Mean Frobenius norm of Hessian (2nd-order spatial derivative)
    """
    arr = mean_warp_img.numpy().astype(np.float64)  # (*spatial, dim)
    spacing = mean_warp_img.spacing

    dim = arr.shape[-1]
    sp_axes = tuple(float(s) for s in spacing)

    # 1. Euclidean L2 norm (RMS displacement magnitude)
    mag_sq = np.sum(arr ** 2, axis=-1)
    l2_norm = float(np.sqrt(np.mean(mag_sq)))

    # 2. Membrane energy: mean Σ_{k,j} (∂ψ_k/∂x_j)²
    membrane = 0.0
    for k in range(dim):
        component = arr[..., k]
        for j in range(dim):
            grad = np.gradient(component, sp_axes[j], axis=j)
            membrane += float(np.mean(grad ** 2))

    # 3. Bending energy: mean Σ_{k,j,i} (∂²ψ_k/∂x_j∂x_i)²
    bending = 0.0
    for k in range(dim):
        component = arr[..., k]
        for j in range(dim):
            grad_j = np.gradient(component, sp_axes[j], axis=j)
            for i_ in range(dim):
                grad_ji = np.gradient(grad_j, sp_axes[i_], axis=i_)
                bending += float(np.mean(grad_ji ** 2))

    return {
        'l2_norm': l2_norm,
        'membrane_energy': membrane,
        'bending_energy': bending,
    }


def _template_change(old: ants.ANTsImage, new: ants.ANTsImage) -> float:
    """Mean absolute voxel-intensity change between two template images."""
    return float(np.mean(np.abs(old.numpy().astype(np.float64) -
                                 new.numpy().astype(np.float64))))


def build_template(
    initial_template: Optional[ants.ANTsImage] = None,
    image_list: Optional[List[ants.ANTsImage]] = None,
    iterations: int = 3,
    gradient_step: float = 0.2,
    blending_weight: float = 0.75,
    weights: Optional[Sequence[float]] = None,
    useNoRigid: bool = True,
    output_dir: Optional[str] = None,
    type_of_transform: str = "SyN",
    convergence_threshold: float = 0.0,
    affine_every_iteration: bool = False,
    backend: str = "pytorch",
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Build an unbiased group template -- ``syntx.build_template`` (a port of
    ``ants.build_template`` that also reports how much the template still moves).

    Each iteration registers every image to the current template, averages the warped
    images, and moves the average back by ``gradient_step`` x the mean inverse warp and the
    inverse of the average affine (so the template drifts toward the group's centre of
    shape); optionally it is sharpened. Per iteration the mean warp's size and roughness and
    the template's change are recorded::

        out = syntx.build_template(image_list=images, iterations=4)
        out['template'], out['shape_residuals'], out['convergence']

    Parameters
    ----------
    initial_template : ANTsImage, optional
        Starting template. Default: the weighted average of ``image_list`` (resampled onto
        the first image).
    image_list : list of ANTsImage
        The images (required, non-empty).
    iterations : int, default 3
    gradient_step : float in [0, 1], default 0.2
        Fraction of the mean warp applied to the template each iteration.
    blending_weight : float in (0, 1], default 0.75
        template = w * template + (1 - w) * sharpened(template); 1 = no sharpening.
    weights : sequence of float, optional
        Per-image weights (normalised to sum 1; default equal).
    useNoRigid : bool, default True
        Average the affines without their rigid part (``ants.average_affine_transform_no_rigid``,
        falling back to ``ants.average_affine_transform`` if that fails).
    output_dir : str, optional
        Where the per-iteration transforms go (default: a temporary directory).
    type_of_transform : str, default 'SyN'
        Per-image registration type.
    convergence_threshold : float, default 0.0
        Stop early when the mean warp's RMS size (mm) falls below this (0 = never).
    affine_every_iteration : bool, default False
        False: after iteration 0, 'SyN' / 'SyNTo' registrations run as 'SyNOnly'
        (deformable only), starting from each image's affine of iteration 0 (passed as
        ``initial_transform``), to avoid affine drift between iterations. True: full
        registration every iteration.
    backend : {'pytorch', 'ants'}, default 'pytorch'
        'pytorch': ``syntx.syn``; 'ants': ``ants.registration`` (``syn_metric='cc2'`` is
        mapped to 'mattes' there).
    verbose : bool, default False
    **kwargs
        Passed to the registration function.

    Returns
    -------
    dict
        ``'template'``, ``'warped_images'`` (last iteration), ``'fwdtransforms'`` /
        ``'invtransforms'`` (per image, last iteration), ``'convergence'`` (mean absolute
        template change per iteration), ``'shape_residuals'`` (per iteration: ``l2_norm``
        (RMS mm), ``membrane_energy``, ``bending_energy`` of the mean warp),
        ``'n_iterations'``, ``'weights'``, ``'work_dir'``.
    """
    if image_list is None or len(image_list) == 0:
        raise ValueError("image_list must be a non-empty list of ANTsImages.")

    n = len(image_list)
    weights = _normalize_weights(weights, n)

    if not (0.0 < blending_weight <= 1.0):
        raise ValueError(f"blending_weight must be in (0, 1], got {blending_weight}")
    if not (0.0 <= gradient_step <= 1.0):
        raise ValueError(f"gradient_step must be in [0, 1], got {gradient_step}")
    if backend not in ("pytorch", "ants"):
        raise ValueError(f"backend must be 'pytorch' or 'ants', got {backend!r}.")
    if backend == "ants" and kwargs.get("syn_metric") == "cc2":
        # 'cc2' is syntx.registration's native metric name; ants.registration has no such
        # alias and expects 'mattes'/'CC'/etc, so only remap it on the legacy ants path.
        kwargs["syn_metric"] = "mattes"

    work_dir = tempfile.mkdtemp(prefix="syntx_tmpl_") if output_dir is None else output_dir
    os.makedirs(work_dir, exist_ok=True)

    def make_outprefix(it_idx: int, k: int) -> str:
        d = os.path.join(work_dir, f"iter{it_idx:02d}_img{k:04d}")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "out")

    if initial_template is None:
        initial_template = image_list[0] * 0
        for i in range(len(image_list)):
            temp = image_list[i] * weights[i]
            temp = ants.resample_image_to_target(temp, initial_template)
            initial_template = initial_template + temp

    xavg = initial_template.clone()

    shape_residuals: List[Dict[str, float]] = []
    convergence: List[float] = []
    last_warped: List[ants.ANTsImage] = [None] * n
    last_fwdtransforms: List[List[str]] = [[] for _ in range(n)]
    last_invtransforms: List[List[str]] = [[] for _ in range(n)]
    image_affine: List[Optional[str]] = [None] * n    # each image's affine of the last full registration

    for it in range(iterations):
        if verbose:
            print(f"[build_template] Iteration {it + 1}/{iterations}")
        affinelist = []
        wavg = None
        xavgNew = None
        L = 0

        # In multi-iteration template construction, global affine pre-alignment is established
        # in iteration 0. In subsequent iterations (it >= 1), re-running unconstrained affine
        # on the running template average can introduce spurious rotation/shear drift.
        iter_tot = type_of_transform
        if it > 0 and not affine_every_iteration:
            if type_of_transform.lower() in ("syn", "synto"):
                iter_tot = "SyNOnly"

        for k in range(len(image_list)):
            if verbose:
                print(f"  Registering subject {k + 1}/{len(image_list)} ...", end=" ", flush=True)
            reg_kw = dict(kwargs)
            if iter_tot == "SyNOnly" and image_affine[k] is not None:
                reg_kw["initial_transform"] = image_affine[k]   # deformable only, from its affine
            if backend == "ants":
                w1 = ants.registration(
                    xavg,
                    image_list[k],
                    type_of_transform=iter_tot,
                    outprefix=make_outprefix(it, k),
                    **reg_kw
                )
            else:
                from .syn import registration as syn_registration

                w1 = syn_registration(
                    xavg,
                    image_list[k],
                    type_of_transform=iter_tot,
                    outprefix=make_outprefix(it, k),
                    verbose=verbose,
                    **reg_kw
                )
            L = len(w1["fwdtransforms"])
            if iter_tot != "SyNOnly":
                image_affine[k] = w1["fwdtransforms"][L - 1]
            affinelist.append(w1["fwdtransforms"][L - 1])

            last_warped[k] = w1["warpedmovout"]
            last_fwdtransforms[k] = w1["fwdtransforms"]
            last_invtransforms[k] = w1["invtransforms"]

            if k == 0:
                if L >= 2:
                    wavg = ants.image_read(w1["fwdtransforms"][0]) * weights[k]
                xavgNew = w1["warpedmovout"] * weights[k]
            else:
                if L >= 2:
                    wavg = wavg + ants.image_read(w1["fwdtransforms"][0]) * weights[k]
                xavgNew = xavgNew + w1["warpedmovout"] * weights[k]

            if verbose:
                print("done")

        # Quantify the shape residual of the mean deformable warp
        if L >= 2 and wavg is not None:
            sr = _compute_shape_residual(wavg)
        else:
            sr = {'l2_norm': 0.0, 'membrane_energy': 0.0, 'bending_energy': 0.0}
        shape_residuals.append(sr)

        if verbose:
            print(f"  Iteration {it} Shape Residual — L2: {sr['l2_norm']:.4f} mm, "
                  f"Membrane: {sr['membrane_energy']:.6f}, Bending: {sr['bending_energy']:.8f}")

        # Average affine
        if useNoRigid:
            try:
                avgaffine = ants.average_affine_transform_no_rigid(affinelist)
            except Exception:
                avgaffine = ants.average_affine_transform(affinelist)
        else:
            avgaffine = ants.average_affine_transform(affinelist)

        afffn = os.path.join(work_dir, f"avgAffine_{it}.mat")
        ants.write_transform(avgaffine, afffn)

        xavg_pre = xavg.clone()

        if L >= 2 and wavg is not None:
            wscl = (-1.0) * gradient_step
            wavgScaled = wavg * wscl
            wavgA = ants.apply_transforms(
                fixed=xavgNew,
                moving=wavgScaled,
                imagetype=1,
                transformlist=afffn,
                whichtoinvert=[1]
            )
            wavgfn = os.path.join(work_dir, f"avgWarp_{it}.nii.gz")
            ants.image_write(wavgA, wavgfn)
            xavg = ants.apply_transforms(
                fixed=xavgNew,
                moving=xavgNew,
                transformlist=[wavgfn, afffn],
                whichtoinvert=[0, 1]
            )
        else:
            xavg = ants.apply_transforms(
                fixed=xavgNew,
                moving=xavgNew,
                transformlist=[afffn],
                whichtoinvert=[1]
            )

        if blending_weight is not None and blending_weight < 1.0:
            xavg = xavg * blending_weight + ants.iMath(xavg, "Sharpen") * (1.0 - blending_weight)

        c_val = _template_change(xavg_pre, xavg)
        convergence.append(c_val)

        if verbose:
            print(f"  Template MAE change: {c_val:.6f}")

        if convergence_threshold > 0.0 and sr['l2_norm'] < convergence_threshold:
            if verbose:
                print(f"  Converged: shape residual {sr['l2_norm']:.6f} < threshold {convergence_threshold:.6f}")
            break

    return {
        "template": xavg,
        "warped_images": last_warped,
        "fwdtransforms": last_fwdtransforms,
        "invtransforms": last_invtransforms,
        "convergence": convergence,
        "shape_residuals": shape_residuals,
        "n_iterations": len(shape_residuals),
        "weights": weights,
        "work_dir": work_dir,
    }
