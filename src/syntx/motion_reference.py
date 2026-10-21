"""syntx.motion_reference -- building a sharp, modality-agnostic reference image for
motion correction without (a) a naive whole-series temporal mean, which is itself
motion-blurred whenever real inter-frame motion exists, or (b) an expensive iterative
template-build over the whole series.

Two real, confirmed findings motivate this module (2026-10-04 session):

1. A naive ``reference="mean"`` across a CONTRAST-HETEROGENEOUS series (e.g. DWI's b0 +
   multiple diffusion-weighted shells) biases the reference toward whichever contrast is
   the numerical majority, producing the WORST registration fits for the minority class
   (confirmed on real SOCOM DWI data: b0 volumes, only 7/99 frames, showed the highest
   apparent FD of any group when registered against the full 99-volume blended mean --
   nearly 3x higher than when b0 volumes are registered among themselves instead). An
   initial fix (register every frame, both groups, to one anchor-group-only reference)
   was ALSO wrong -- it still forces every non-anchor frame through a per-frame
   CROSS-contrast registration, which the fast narrow-seed backend handles worse, not
   better (b0 FD rose to 7.28mm under that design, worse than the original 4.66mm).
   ``motion_correct_grouped`` instead builds one mean per group, aligns the two means to
   each other exactly ONCE (the only cross-contrast registration anywhere in the
   pipeline, so it can afford to be careful/robust), and keeps every per-frame
   registration same-contrast (where the fast backend is actually validated to work).

2. Even within a single-contrast series, the plain temporal mean of ALL frames is sharper
   only when real motion is near-zero; with real motion present, averaging many
   differently-positioned frames smears anatomy, which can itself inflate apparent
   registration error. ``build_low_motion_reference`` addresses this by building the
   reference from only a small, cheaply-selected low-motion subset.
"""

from __future__ import annotations

import time as _time
from typing import List, Optional, Sequence

import ants
import numpy as np

from .motion_batched import batched_rigid_register_pass, estimate_coarse_fd


def _mean_image(imgs: Sequence[ants.ANTsImage]) -> ants.ANTsImage:
    """Plain unweighted voxelwise mean of a list of same-grid images."""
    arr = np.mean(np.stack([im.numpy() for im in imgs], axis=0), axis=0)
    return imgs[0].new_image_like(arr.astype(np.float32))


def build_low_motion_reference(
    images: List[ants.ANTsImage],
    n_low_motion: int = 10,
    coarse_resolution_cap_mm: Optional[float] = 8.0,
    fd_radius: float = 50.0,
    device: str = "auto",
    num_bins: int = 18,
    verbose: bool = False,
) -> ants.ANTsImage:
    """Build a sharp reference from the ``n_low_motion`` frames with the smallest cheap,
    registration-derived FD relative to the series' own crude (unweighted) mean --
    avoids both a motion-blurred whole-series mean and an expensive full-series
    iterative template build.

    Steps: (1) one cheap coarse registration pass (``estimate_coarse_fd`` -- the same
    primitive behind ``batched_rigid_register_pass_adaptive``) ranks every frame; (2) the
    ``n_low_motion`` lowest-FD frames are registered to EACH OTHER (small N, so this
    narrow+fast pass is cheap even though it's a "real" registration, not the coarse
    ranking pass); (3) the corrected low-motion subset is averaged into the final
    reference.

    If the series has ``n_low_motion`` frames or fewer, every frame is used (no subset
    selection needed) and this reduces to "register everything to the crude mean, then
    average the corrected result" -- a single-pass sharpened mean.

    Parameters
    ----------
    images : list of ants.ANTsImage
        Candidate frames (same grid). For a grouped series (e.g. DWI b0 volumes), pass
        only that group's own frames -- this function has no modality-specific logic.
    n_low_motion : int, default 10
        Number of lowest-FD frames to build the reference from.
    coarse_resolution_cap_mm, fd_radius, device, num_bins
        See ``estimate_coarse_fd``.
    verbose : bool, default False

    Returns
    -------
    ants.ANTsImage
        The sharpened reference, on the same grid as ``images[0]``.
    """
    n = len(images)
    if n == 0:
        raise ValueError("build_low_motion_reference requires at least 1 image.")
    if n == 1:
        return images[0].clone()

    t0 = _time.time()
    crude_ref = _mean_image(images)

    if n <= n_low_motion:
        selected = images
        if verbose:
            print(f"[syntx.motion_reference] {n} frames <= n_low_motion={n_low_motion} -- "
                  f"using all of them (no subset selection).")
    else:
        coarse_fd, _, _ = estimate_coarse_fd(
            crude_ref, images, resolution_cap_mm=coarse_resolution_cap_mm, fd_radius=fd_radius,
            device=device, num_bins=num_bins,
        )
        lowest_idx = np.argsort(coarse_fd)[:n_low_motion]
        selected = [images[i] for i in lowest_idx]
        if verbose:
            print(f"[syntx.motion_reference] selected {n_low_motion}/{n} lowest-FD frames "
                  f"(coarse FD range of selection: {coarse_fd[lowest_idx].min():.3f}-"
                  f"{coarse_fd[lowest_idx].max():.3f}mm, full-series range: "
                  f"{coarse_fd.min():.3f}-{coarse_fd.max():.3f}mm).")

    selected_mean = _mean_image(selected)
    fwd, _inv, _elapsed = batched_rigid_register_pass(
        selected_mean, selected, device=device, num_bins=num_bins, verbose=False,
        seed_mode="narrow", schedule_mode="fast",
    )
    from .transforms import apply_transforms as _apply_transforms
    corrected = [
        _apply_transforms(fixed=selected_mean, moving=im, transformlist=f, interpolator="linear")
        for im, f in zip(selected, fwd)
    ]
    final_ref = _mean_image(corrected)

    if verbose:
        print(f"[syntx.motion_reference] build_low_motion_reference: {_time.time() - t0:.2f}s "
              f"total ({len(selected)} frames registered+averaged).")
    return final_ref


def _make_4d_template_from_3d(ref3d: ants.ANTsImage, n_frames: int, tr: float = 1.0) -> ants.ANTsImage:
    """Build a real-geometry 4D template for ``ants.list_to_ndimage``, instead of
    ``ants.from_numpy(np.zeros(...))`` (default header: origin (0,0,0), identity
    direction, unit spacing). ``list_to_ndimage`` copies ITS template's own header onto
    the assembled output, so a zeros-array template silently discards the real subject
    geometry from every frame stacked into it -- same convention as
    ``antsxslowflow/motion/api.py`` L326-338 and ``ANTsPyMM/antspymm/mm.py`` L2143-2154:
    3x3 spatial block from the reference 3D image, a fresh 4x4 identity otherwise (never
    copied wholesale from a stale 4D image).
    """
    spc3 = list(ants.get_spacing(ref3d))
    org3 = list(ants.get_origin(ref3d))
    dir3 = np.asarray(ants.get_direction(ref3d), dtype=float)
    mydir4d = np.eye(4, dtype=float)
    mydir4d[:3, :3] = dir3[:3, :3]
    shape4d = list(ref3d.shape) + [n_frames]
    return ants.make_image(shape4d, 0, spacing=spc3 + [tr], origin=org3 + [0.0], direction=mydir4d)


def shell_groups(bvals: np.ndarray, b0_threshold: float = 50.0, shell_tol: float = 150.0) -> np.ndarray:
    """Group label per volume for ``motion_correct_grouped``: 0 for b0 (b <= ``b0_threshold``), then 1..k for the diffusion
    shells in ascending b (b-values within ``shell_tol`` of the previous sorted b-value share a shell)."""
    b = np.asarray(bvals, dtype=float)
    labels = np.zeros(len(b), dtype=int)
    dw = np.where(b > b0_threshold)[0]
    order = dw[np.argsort(b[dw])]
    g = 1
    for k, i in enumerate(order):
        if k > 0 and b[i] - b[order[k - 1]] > shell_tol:
            g += 1
        labels[i] = g
    return labels


class GroupedMotionCorrectionResult(dict):
    """Dict subclass (attribute access) returned by ``motion_correct_grouped``.

    Keys: ``motion_corrected`` (4D, all frames in group 0's mean space), ``fd`` (series-level Jenkinson FD in acquisition
    order, every frame's total transform in group 0's space), ``fwdtransforms`` (per frame, input order: the full chain
    ``[cross_1, ..., cross_g, own]`` ready for ``ants.apply_transforms``), ``group_results`` (per-group
    ``MotionCorrectionResult``), ``group_indices`` (frame indices of each group), ``group_means`` (the corrected group means
    the cross registrations were computed on), ``cross_registrations`` (``robust_affine`` results; entry ``g - 1`` registers
    group ``g``'s mean onto group ``g - 1``'s), ``preset``.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'GroupedMotionCorrectionResult' has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


def motion_correct_grouped(
    image: ants.ANTsImage,
    groups: np.ndarray,
    n_low_motion: int = 10,
    backend: str = "pytorch_batched_adaptive",
    verbose: bool = False,
    preset: Optional[str] = None,
    cross_num_bins: int = 32,
    nuisance_covariates: Optional[np.ndarray] = None,
    nuisance_axis: Optional[np.ndarray] = None,
    pe_affine_covariates: Optional[np.ndarray] = None,
    pe_axis: Optional[np.ndarray] = None,
    reference_predictor=None,
    reference_iterations: int = 1,
    **kwargs,
) -> GroupedMotionCorrectionResult:
    """Motion-correct a contrast-heterogeneous series (e.g. DWI: b0 and one group per b-shell) with same-contrast per-frame
    registrations and one careful registration per pair of adjacent contrasts.

    (1) Each group gets its own sharp mean; (2) every frame is registered to ITS OWN group's mean (same contrast: no
    contrast-driven bias); (3) the group means are recomputed from all corrected frames and registered to each other in
    a chain, group ``g`` onto group ``g - 1``, so every registration is between the two most similar contrasts (still
    travelling-subject DWI, true bias 0: b3000 mean onto b0 directly 0.55 mm, through the b1000 -> b2000 -> b3000 chain 0.14 mm);
    (4) every frame of a later group is resampled ONCE from the original through the composed chain into group 0's space.
    This is the ``[b0_mean, dwi_mean, ...]`` multi-channel representation: all channels in one space.

    Parameters
    ----------
    image : ants.ANTsImage
        4D time series (time is the last axis).
    groups : np.ndarray, shape (n_frames,)
        Group label per frame, 0 = anchor (e.g. b0), then 1..k-1 in order of increasing contrast distance from the anchor
        (``shell_groups`` for DWI). A boolean array (True = anchor) is accepted for two groups.
    n_low_motion : int, default 10
        Forwarded to ``build_low_motion_reference`` for each group's own mean build.
    backend : str, default 'pytorch_batched_adaptive'
        Forwarded to ``syntx.motion_correction`` for each group; overridden by ``preset``.
    preset : str, optional
        ``"dwi"`` (``syntx.motion_presets``): full-precision batched backend at ``resolution_cap_mm``, b-spline final
        resampling, ``cross_num_bins`` histogram bins in the cross registrations (on a still scan the b2000 -> b1000 bias fell
        from 0.16 mm at 32 bins to 0.04 mm at 128).
    cross_num_bins : int, default 32
        MI histogram bins of the cross registrations when no preset is given.
    nuisance_covariates : np.ndarray, shape (n_frames, k), optional
        Per-frame covariates of a translation that is NOT head motion but is absorbed by the rigid fit, with
        ``nuisance_axis`` the physical unit vector it acts along. DWI: the diffusion gradient vector (``bvecs``) and the
        phase-encode axis -- eddy currents translate each volume along phase encode by an amount linear in the gradient
        vector (sub-BLAST079, 3-shell AP/PA: the per-volume PE translation is linear in g with held-out R2 0.55-0.77,
        flips sign between the AP and PA series and survives shape-only registration, so it is geometric, not a metric
        effect). Within each group the displacement of the reference's centre of mass along the axis is regressed on
        the covariates (``d = c + B @ cov``); ``B @ cov`` is the nuisance shift of that frame. The applied transforms keep
        it (the registration did align the displaced content); ``fd`` is computed with it removed so it counts motion
        only, ``fd_raw`` keeps the previous definition, and ``nuisance`` reports ``shift_mm`` per frame, ``coefficients``
        and ``r2`` per group. A group whose covariates are all zero (b0) gets no fit.
    pe_affine_covariates : np.ndarray, shape (n_frames, k), optional
        With ``pe_axis``: the eddy-current term estimated INSIDE each group's registration (``pytorch_batched`` backend,
        see ``motion_batched._batched_rigid_register_pass_core``): per frame a translation, two shears and a scale along
        the phase-encode axis, tied across the group's frames as a linear function of the covariates (DWI: the gradient
        vectors; b0 frames have zero covariates and get no term). The eddy term is applied to the data (single resampling
        through ``[cross..., eddy, rigid]``, Jacobian-modulated); ``fd`` is the Jenkinson FD of the rigid part only;
        ``fd_raw`` includes the eddy term; ``pe_affine`` holds per-group coefficient matrices and per-frame ``a``,
        ``jacobian``. This supersedes the post-hoc ``nuisance_covariates`` regression (which only models the translation).
    reference_predictor : callable, optional
        ``predictor(corrected_frames, group_index=g, frame_indices=idx) -> list of per-frame reference images on the
        group's grid, or None``. A group's frames are first registered to the group mean (same contrast on average, but
        each frame's own contrast can differ from it in a way that biases the pose -- a diffusion volume's attenuation
        pattern depends on its gradient direction); when a predictor is given, it is called with the corrected frames
        and the group is re-registered, frame by frame, to its predicted reference (``motion_correction(reference=
        <list>)``, same eddy term), ``reference_iterations`` times, re-predicting from the newly corrected frames each
        time. ``None`` from the predictor keeps the mean (the anchor/b0 group has no direction dependence). The
        predictor carries the modality knowledge (antsxdwi: a spherical-harmonic prediction at each gradient).
        ``reference_mode`` per group ('mean' | 'predicted') and the iterations are reported.
    **kwargs
        Forwarded to ``syntx.motion_correction`` (e.g. ``fd_radius``).
    """
    from .motion import _extract_rigid_parameters, motion_correction
    from .robust_affine import robust_affine
    from .transforms import apply_transforms as _apply_transforms

    groups = np.asarray(groups)
    if groups.dtype == bool:
        groups = (~groups).astype(int)
    n_frames = image.shape[-1]
    if groups.shape != (n_frames,):
        raise ValueError(f"groups must have one entry per frame ({n_frames}), got shape {groups.shape}")
    labels = np.unique(groups)
    if len(labels) < 2 or not np.array_equal(labels, np.arange(len(labels))):
        raise ValueError(f"groups must use the labels 0..k-1 (k >= 2) with every label present; got {labels.tolist()}")
    n_groups = len(labels)
    cfg = None
    if preset is not None:
        from .motion_presets import get_motion_preset

        cfg = get_motion_preset(preset)
        if preset != "dwi":
            raise ValueError(f"motion_correct_grouped takes the 'dwi' preset; {preset!r} is used through motion_correction")
        backend, cross_num_bins = cfg["backend"], cfg["cross_num_bins"]
        kwargs.setdefault("resolution_cap_mm", cfg["resolution_cap_mm"])
        for key in ("seed_mode", "schedule_mode"):
            if key in cfg:
                kwargs.setdefault(key, cfg[key])
    interp = cfg["interpolator"] if cfg is not None else "linear"
    idx = [np.where(groups == g)[0] for g in range(n_groups)]

    # Each group's own 4D container ONCE, every frame and mean sliced from it: all share EXACTLY the same grid (independently
    # built ants images drift in origin/direction and are rejected by the batched passes' grid check).
    ref3d = ants.slice_image(image, axis=3, idx=0)
    tr = float(image.spacing[3]) if image.dimension == 4 else 1.0
    img4d, imgs, means = [], [], []
    for g in range(n_groups):
        c4 = ants.list_to_ndimage(_make_4d_template_from_3d(ref3d, len(idx[g]), tr=tr),
                                  [ants.slice_image(image, axis=3, idx=int(i)) for i in idx[g]])
        img4d.append(c4)
        imgs.append([ants.slice_image(c4, axis=3, idx=i) for i in range(len(idx[g]))])
        means.append(build_low_motion_reference(imgs[g], n_low_motion=n_low_motion, verbose=verbose))
    if verbose:
        print(f"[syntx.motion_reference] motion_correct_grouped: group sizes {[len(i) for i in idx]} of {n_frames} frames.")

    # DVARS of every group is normalised against the anchor group's mean intensity (comparable across groups and subjects).
    anchor = means[0].numpy()
    anchor_fg = anchor[anchor > 0]
    ref_intensity = float(np.mean(anchor_fg)) if anchor_fg.size > 0 else None
    results = []
    reference_modes: list = []
    for g in range(n_groups):
        # group 0's corrected frames are final outputs; later groups' only feed their mean (resampled once from the originals below)
        kw = {**kwargs, "interpolator": interp if g == 0 else "linear"} if cfg is not None else dict(kwargs)
        # Every group writes its own transform files. Frame t of each group would otherwise be written to the same
        # f"{outprefix}batch_{t:04d}.mat", the last group overwriting the others; the chains below are read back from
        # these paths after all groups have run, so group g's frames would be resampled with group g+1's transforms and
        # the series FD would see one group's poses twice (found on real 3-shell DWI, sub-BLAST079, 2026-10-20).
        if kw.get("outprefix") is not None:
            kw["outprefix"] = f"{kw['outprefix']}g{g}_"
        if pe_affine_covariates is not None:
            cov_g = np.asarray(pe_affine_covariates, dtype=np.float64).reshape(n_frames, -1)[np.asarray(idx[g])]
            if np.any(cov_g != 0):
                kw["pe_affine_covariates"] = cov_g
                kw["pe_axis"] = pe_axis
        res_g = motion_correction(img4d[g], reference=means[g], backend=backend, verbose=verbose,
                                  dvars_reference_intensity=ref_intensity, **kw)
        mode_g = "mean"
        if reference_predictor is not None:
            for it in range(int(reference_iterations)):
                frames_g = [ants.slice_image(res_g.motion_corrected, axis=3, idx=i) for i in range(len(idx[g]))]
                refs = reference_predictor(frames_g, group_index=g, frame_indices=[int(i) for i in idx[g]])
                if refs is None:
                    break
                if len(refs) != len(idx[g]):
                    raise ValueError(f"reference_predictor returned {len(refs)} references for {len(idx[g])} frames of group {g}")
                kw_it = dict(kw)
                if kw_it.get("outprefix") is not None:
                    kw_it["outprefix"] = f"{kw_it['outprefix']}it{it + 1}_"
                res_g = motion_correction(img4d[g], reference=list(refs), backend=backend, verbose=verbose,
                                          dvars_reference_intensity=ref_intensity, **kw_it)
                mode_g = f"predicted (iteration {it + 1})"
        reference_modes.append(mode_g)
        results.append(res_g)

    # Corrected group means, then one careful registration per adjacent pair of contrasts.
    cmeans = [_mean_image([ants.slice_image(results[g].motion_corrected, axis=3, idx=i) for i in range(len(idx[g]))])
              for g in range(n_groups)]
    cross = [robust_affine(fixed=cmeans[g - 1], moving=cmeans[g], dof="rigid", mode="auto", verbose=verbose,
                           device=kwargs.get("device"), num_bins=cross_num_bins) for g in range(1, n_groups)]

    chains: list = [None] * n_frames
    frames_out: list = [None] * n_frames
    for g in range(n_groups):
        pre = [t for k in range(g) for t in cross[k]["fwdtransforms"]]  # [cross_1, ..., cross_g]
        for li, gi in enumerate(idx[g]):
            chains[gi] = pre + list(results[g].fwdtransforms[li])
            if g == 0:
                frames_out[gi] = ants.slice_image(results[0].motion_corrected, axis=3, idx=li)
            else:  # ONE resampling from the original through the composed chain: every interpolation blurs
                frames_out[gi] = _apply_transforms(fixed=means[0], moving=imgs[g][li], transformlist=chains[gi],
                                                   interpolator=interp)
                pe_g = results[g].get("pe_affine")
                if pe_g is not None and pe_g["jacobian"][li] != 1.0:
                    frames_out[gi] = frames_out[gi] * float(pe_g["jacobian"][li])
    motion_corrected = ants.list_to_ndimage(_make_4d_template_from_3d(frames_out[0], n_frames, tr=tr), frames_out)

    # Series-level Jenkinson FD in acquisition order: every frame's total rigid transform in group 0's space
    # (a point maps through cross_1 ... cross_g, then the frame's own transform: H = H_own @ H_cross_g @ ... @ H_cross_1).
    from .motion import jenkinson_framewise_displacement

    com = np.asarray(means[0].get_center_of_mass(), dtype=np.float64)
    mats = [[_extract_rigid_parameters(ants.read_transform(f), 3)[2] for f in chain] for chain in chains]

    def _compose(per_frame_mats):
        homog = []
        for ms in per_frame_mats:
            h = np.eye(4)
            for m in ms:
                h = m @ h
            homog.append(h)
        return homog

    fd_raw = jenkinson_framewise_displacement(_compose(mats), kwargs.get("fd_radius", 50.0), com)
    nuisance = None
    pe_affine = None
    fd = fd_raw
    if any(r.get("pe_affine") is not None for r in results):
        # the eddy PE-affine is in the chain (and in fd_raw); FD counts the rigid part only: crosses + the frame's rigid
        n_cross = [len(c["fwdtransforms"]) for c in cross]
        mats_rigid = [ms[:sum(n_cross[:g])] + [ms[-1]] for ms, g in zip(mats, [int(groups[i]) for i in range(n_frames)])]
        fd = jenkinson_framewise_displacement(_compose(mats_rigid), kwargs.get("fd_radius", 50.0), com)
        a_all = np.zeros((n_frames, 4)); jac_all = np.ones(n_frames); coef = []
        for g in range(n_groups):
            pe_g = results[g].get("pe_affine")
            cg = None if pe_g is None else pe_g["coefficients"]
            coef.append(cg[0] if isinstance(cg, list) and len(cg) == 1 else cg)  # one matrix per batch; usually one batch
            if pe_g is not None:
                a_all[np.asarray(idx[g])] = pe_g["a"]; jac_all[np.asarray(idx[g])] = pe_g["jacobian"]
        pe_affine = {"axis": [float(x) for x in np.asarray(pe_axis) / np.linalg.norm(pe_axis)], "coefficients_by_group": coef,
                     "a": a_all, "jacobian": jac_all}
    if nuisance_covariates is not None:
        if nuisance_axis is None:
            raise ValueError("nuisance_covariates needs nuisance_axis (physical unit vector the nuisance translation acts along)")
        cov = np.asarray(nuisance_covariates, dtype=np.float64).reshape(n_frames, -1)
        axis = np.asarray(nuisance_axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        shift = np.zeros(n_frames)
        coefficients, r2 = [], []
        for g in range(n_groups):
            ig = np.asarray(idx[g])
            own = np.array([mats[i][-1] for i in ig])  # the frame's own transform (last in its chain)
            d = np.einsum("nij,j->ni", own[:, :3, :3] - np.eye(3), com) + own[:, :3, 3]
            d = d @ axis  # displacement of the reference centre of mass along the axis, per frame
            X = np.c_[np.ones(len(ig)), cov[ig]]
            if not np.any(cov[ig]) or len(ig) <= X.shape[1]:
                coefficients.append(None)
                r2.append(None)
                continue
            beta, *_ = np.linalg.lstsq(X, d, rcond=None)
            fit = X[:, 1:] @ beta[1:]
            shift[ig] = fit
            ss = float(np.sum((d - d.mean()) ** 2))
            r2.append(float(1.0 - np.sum((d - X @ beta) ** 2) / ss) if ss > 0 else None)
            coefficients.append([float(b) for b in beta[1:]])
        mats_motion = []
        for i, ms in enumerate(mats):
            own = ms[-1].copy()
            own[:3, 3] = own[:3, 3] - shift[i] * axis
            mats_motion.append(ms[:-1] + [own])
        fd = jenkinson_framewise_displacement(_compose(mats_motion), kwargs.get("fd_radius", 50.0), com)
        nuisance = {"axis": [float(a) for a in axis], "shift_mm": shift, "coefficients": coefficients, "r2": r2}

    out = GroupedMotionCorrectionResult()
    out["motion_corrected"] = motion_corrected
    out["fd"] = fd
    out["fd_raw"] = fd_raw
    out["nuisance"] = nuisance
    out["pe_affine"] = pe_affine
    out["reference_modes"] = reference_modes
    out["fwdtransforms"] = chains
    out["group_results"] = results
    out["group_indices"] = idx
    out["group_means"] = cmeans
    out["cross_registrations"] = cross
    out["preset"] = preset
    return out
