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
import time
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


def _correlation(a: ants.ANTsImage, b: ants.ANTsImage) -> float:
    """Pearson correlation between two images' voxel intensities -- the cheap per-image
    registration-quality signal the greedy path's adaptive affine caching checks each
    iteration against (see ``affine_drop_tolerance``). Thin wrapper over the canonical
    ``syntx.image_compare.correlation`` (added after this was found duplicated 6+ times
    across the ecosystem) -- kept as a local name since it's called throughout this
    module's iteration loop."""
    from .image_compare import correlation as _canonical_correlation
    return _canonical_correlation(a, b)


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
    initial_transforms: Optional[Sequence[Any]] = None,
    affine_drop_tolerance: Optional[float] = 0.1,
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
        Per-image registration type. 'SyN' / 'SyNOnly': symmetric, invertible (``syntx.syn``
        or, with ``backend='ants'``, ``ants.registration``) -- use when the per-image
        transforms themselves need to be invertible later (e.g. warping labels both ways).
        'Greedy' (``backend='pytorch'`` only): one-directional (``syntx.greedy``), a single
        composed field with no affine/warp split and no inverse computed -- cheaper, and the
        right choice when the template only needs each image warped INTO template space to
        accumulate an average (nothing downstream needs to invert that map). Every iteration
        re-aligns from scratch (``syntx.robust_affine``) rather than getting 'SyNOnly'-style
        drift-avoidance continuity from the previous iteration -- passing a prior composed
        field back in as ``initial_transform`` hits a real limitation in
        ``greedy_registration`` itself (it rejects a non-linear initial transform instead of
        treating it as an initial grid, despite its own docstring). ``**kwargs`` are then
        ``syntx.greedy``'s own (e.g. ``reg_iterations``, ``scales``, ``similarity_metric``,
        ``lncc_radius``), not ``syntx.syn``'s.
    convergence_threshold : float, default 0.0
        Stop early when the mean warp's RMS size (mm) falls below this (0 = never).
    affine_every_iteration : bool, default False
        False: after iteration 0, 'SyN' / 'SyNTo' registrations run as 'SyNOnly'
        (deformable only), starting from each image's affine of iteration 0 (passed as
        ``initial_transform``), to avoid affine drift between iterations -- and, for
        ``type_of_transform='Greedy'``, each image's iteration-0 ``initial_transform`` (see
        ``initial_transforms`` below) is likewise reused unchanged for every later iteration
        instead of being recomputed. True: full registration every iteration (for 'Greedy',
        this means every image re-runs ``syntx.robust_affine`` from scratch every iteration,
        since there is no cheaper 'GreedyOnly'-style deformable-only mode).
    initial_transforms : sequence, optional
        One ``initial_transform``-compatible value per image (anything ``syntx.greedy`` or
        ``ants.registration``/``syntx.syn`` accepts for that parameter: ``None``, ``False``,
        ``'identity'``, or a linear transform file/object), used for each image's iteration-0
        registration instead of the usual per-``type_of_transform`` default. The real
        motivation is ``type_of_transform='Greedy'`` against already-roughly-aligned inputs
        (e.g. images already composed into a shared space upstream): a caller can cheaply
        pre-check per image whether ``'identity'`` (skip ``robust_affine`` entirely) gives an
        acceptable registration, falling back to ``None`` (full affine search) only for the
        images that actually need it -- then pass the resulting per-image list here so that
        one-time decision, not just its cost, is what gets reused across iterations (with
        ``affine_every_iteration=False``, the default). Default ``None``: every image uses
        the same ``type_of_transform``-determined default, as before this parameter existed.
    affine_drop_tolerance : float or None, default 0.1
        ``type_of_transform='Greedy'`` only. The cached per-image affine (from
        ``initial_transforms`` or a prior iteration's ``robust_affine`` result) is reused
        unchanged across iterations (see ``affine_every_iteration``) *unless* the
        correlation between that image's warped output and the current template drops by
        more than this much versus the correlation recorded when the affine was last
        (re)computed -- in which case that one image's affine is recomputed from scratch
        (``initial_transform=None``, full ``robust_affine`` search) for this iteration, and
        the new affine + correlation become the new cached baseline. This guards against the
        template moving enough (across iterations, or because of an unusually
        poorly-pre-aligned image) to invalidate a cached affine, while still avoiding routine
        full-affine-search cost for every image on every iteration. Pass ``None`` to disable
        the check entirely (a cached affine is then trusted for the rest of the run, never
        re-verified -- the simpler, static behavior this parameter adds an adaptive
        alternative to). No effect on 'SyN' / 'SyNOnly' (no analogous mechanism yet there).
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
        ``'n_iterations'``, ``'weights'``, ``'work_dir'``, ``'elapsed_sec'`` (total wall-clock
        time of this call, including setup and validation).
    """
    t0_call = time.time()
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
    if initial_transforms is not None and len(initial_transforms) != n:
        raise ValueError(
            f"len(initial_transforms)={len(initial_transforms)} must equal "
            f"len(image_list)={n}"
        )
    if type_of_transform.lower() == "greedy" and "initial_transform" in kwargs:
        raise TypeError(
            "build_template manages each image's initial_transform itself for "
            "type_of_transform='Greedy' (see the initial_transforms= parameter, plural, for "
            "per-image control) -- passing a single initial_transform via **kwargs would be "
            "silently overwritten every iteration and is rejected here instead."
        )
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
    image_quality: List[Optional[float]] = [None] * n  # greedy only: correlation recorded when image_affine[k] was last (re)computed

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
        is_greedy = type_of_transform.lower() == "greedy"
        if it > 0 and not affine_every_iteration:
            if type_of_transform.lower() in ("syn", "synto"):
                iter_tot = "SyNOnly"

        if is_greedy and backend == "ants":
            raise ValueError(
                "type_of_transform='Greedy' is implemented by backend='pytorch' only "
                "(syntx.greedy has no backend='ants' equivalent returning a single composed, "
                "non-invertible field)."
            )

        for k in range(len(image_list)):
            if verbose:
                print(f"  Registering subject {k + 1}/{len(image_list)} ...", end=" ", flush=True)
            reg_kw = dict(kwargs)
            if is_greedy:
                # Greedy (one-directional) registration -- no affine/warp split, no inverse,
                # since the template only needs each subject warped INTO template space to
                # accumulate an average; nothing here ever needs to invert that map. Passing a
                # previous iteration's composed displacement field back in as
                # initial_transform (the SyN path's SyNOnly trick) isn't an option here: it
                # hits a real limitation in greedy_registration's own affine-detection step
                # (rejects a non-linear transform instead of treating it as an initial grid,
                # despite its docstring) -- so instead, a genuinely LINEAR affine is cached
                # and reused/adaptively re-verified across iterations below.
                #
                # Which affine to start from: iteration 0 (or affine_every_iteration=True)
                # uses the caller-supplied initial_transforms[k] (default None -> full
                # robust_affine search); later iterations reuse whatever affine ended up
                # cached from the previous iteration for this image (None is a valid cached
                # value -- it means "no affine needed", e.g. initial_transforms[k] was
                # 'identity'/False, and is reapplied unchanged, not reinterpreted as "nothing
                # cached yet") -- see affine_drop_tolerance below for when that cache gets
                # invalidated and recomputed.
                from .greedy import greedy_registration

                if it == 0 or affine_every_iteration:
                    chosen_init = initial_transforms[k] if initial_transforms is not None else None
                else:
                    chosen_init = image_affine[k]
                reg_kw["initial_transform"] = chosen_init

                w1 = greedy_registration(
                    xavg,
                    image_list[k],
                    outprefix=make_outprefix(it, k),
                    verbose=verbose,
                    **reg_kw
                )
                quality = _correlation(w1["warpedmovout"], xavg)

                # Adaptive re-affine: a cached affine (anything other than iteration 0's own
                # fresh computation) is only trusted as long as it still gives a comparable
                # registration -- if quality has dropped meaningfully since it was (re)cached,
                # recompute it from scratch for this image, this iteration, rather than
                # silently continuing with a now-stale affine for the rest of the run.
                used_cached_affine = not (it == 0 or affine_every_iteration)
                if (
                    used_cached_affine
                    and affine_drop_tolerance is not None
                    and image_quality[k] is not None
                    and quality < image_quality[k] - affine_drop_tolerance
                ):
                    if verbose:
                        print(f"\n  Subject {k + 1}: cached-affine quality dropped "
                              f"({quality:.4f} < {image_quality[k]:.4f} - {affine_drop_tolerance}) "
                              f"-- recomputing affine from scratch.", end=" ")
                    reg_kw["initial_transform"] = None
                    w1 = greedy_registration(
                        xavg,
                        image_list[k],
                        outprefix=make_outprefix(it, k),
                        verbose=verbose,
                        **reg_kw
                    )
                    quality = _correlation(w1["warpedmovout"], xavg)

                # Cache 'identity' rather than the raw None that greedy_registration returns
                # for "no affine was used" -- None fed back in as initial_transform on the
                # NEXT iteration means something different to greedy_registration itself
                # (its own "nothing given, run robust_affine" default), which would silently
                # turn a cheap identity-seeded image back into an expensive full-affine-search
                # one every subsequent iteration. 'identity' unambiguously means "skip the
                # affine step" on replay, matching what actually happened here.
                image_affine[k] = w1["affine_transform"] if w1["affine_transform"] is not None else "identity"
                image_quality[k] = quality
            elif backend == "ants":
                if iter_tot == "SyNOnly" and image_affine[k] is not None:
                    reg_kw["initial_transform"] = image_affine[k]   # deformable only, from its affine
                w1 = ants.registration(
                    xavg,
                    image_list[k],
                    type_of_transform=iter_tot,
                    outprefix=make_outprefix(it, k),
                    **reg_kw
                )
            else:
                if iter_tot == "SyNOnly" and image_affine[k] is not None:
                    reg_kw["initial_transform"] = image_affine[k]   # deformable only, from its affine
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
            if not is_greedy:
                if iter_tot != "SyNOnly":
                    image_affine[k] = w1["fwdtransforms"][L - 1]
                affinelist.append(w1["fwdtransforms"][L - 1])

            last_warped[k] = w1["warpedmovout"]
            last_fwdtransforms[k] = w1["fwdtransforms"]
            last_invtransforms[k] = w1["invtransforms"]

            has_field = is_greedy or L >= 2
            if k == 0:
                if has_field:
                    wavg = ants.image_read(w1["fwdtransforms"][0]) * weights[k]
                xavgNew = w1["warpedmovout"] * weights[k]
            else:
                if has_field:
                    wavg = wavg + ants.image_read(w1["fwdtransforms"][0]) * weights[k]
                xavgNew = xavgNew + w1["warpedmovout"] * weights[k]

            if verbose:
                print("done")

        has_field = is_greedy or L >= 2

        # Quantify the shape residual of the mean deformable warp (for greedy, the single
        # composed field stands in directly -- it already includes any affine component).
        if has_field and wavg is not None:
            sr = _compute_shape_residual(wavg)
        else:
            sr = {'l2_norm': 0.0, 'membrane_energy': 0.0, 'bending_energy': 0.0}
        shape_residuals.append(sr)

        if verbose:
            print(f"  Iteration {it} Shape Residual — L2: {sr['l2_norm']:.4f} mm, "
                  f"Membrane: {sr['membrane_energy']:.6f}, Bending: {sr['bending_energy']:.8f}")

        xavg_pre = xavg.clone()

        if is_greedy:
            # No separate affine file to average -- greedy's composed field already carries
            # any affine component, so the standard "negate and apply the scaled mean field
            # as an approximate inverse" template-recentring trick (same first-order
            # approximation ants' buildtemplateparallel / the SyN branch below both use) is
            # applied directly, with no affine composition step.
            if has_field and wavg is not None:
                wavgScaled = wavg * ((-1.0) * gradient_step)
                wavgfn = os.path.join(work_dir, f"avgWarp_{it}.nii.gz")
                ants.image_write(wavgScaled, wavgfn)
                xavg = ants.apply_transforms(
                    fixed=xavgNew,
                    moving=xavgNew,
                    transformlist=[wavgfn],
                    whichtoinvert=[0]
                )
            else:
                xavg = xavgNew
        else:
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

            if has_field and wavg is not None:
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
        "elapsed_sec": time.time() - t0_call,
    }
