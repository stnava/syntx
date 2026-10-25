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


def _invert_affine_file(path: str, out_path: str) -> str:
    """Write the inverse of an ANTs AffineTransform file (same centre), return ``out_path``."""
    tx = ants.read_transform(path)
    inv = tx.invert()
    ants.write_transform(inv, out_path)
    return out_path


def _register_group_backward(img4d_g, mean_img, kw, backend, verbose, ref_intensity, smoothing_mm, interp):
    """Register the group MEAN onto every frame (each frame is the fixed image, the mean the moving one) and return the
    result in the forward convention (``fwdtransforms`` map mean-space points into the frame, ``motion_corrected`` holds
    the frames resampled into the mean's space).

    Why backward: with the frame as the fixed image the metric's samples and Parzen marginal come from the frame's own
    contrast and the smooth mean is the image being interpolated. On the sub-BLAST079 still phantom (real contrast,
    measured noise) a b3000 volume registered forward to its shell mean is off by 0.253 mm RMS; backward 0.148 mm;
    with 2 mm smoothing of both images for the metric 0.144 (b1500 0.097), and with the tied even-l=2 bias term removed
    0.113 / 0.089 mm. The twin floor (same contrast) is 0.03-0.08.

    ``smoothing_mm`` smooths the images used by the metric only; the data are resampled from the originals.
    """
    from .motion import _extract_rigid_parameters, motion_correction
    from .transforms import apply_transforms as _apply_transforms

    n = img4d_g.shape[-1]
    frames = [ants.slice_image(img4d_g, axis=3, idx=i) for i in range(n)]
    refs = [ants.smooth_image(f, smoothing_mm) if smoothing_mm else f for f in frames]
    mean_m = ants.smooth_image(mean_img, smoothing_mm) if smoothing_mm else mean_img
    mov4d = ants.list_to_ndimage(img4d_g, [mean_m] * n)
    res_b = motion_correction(mov4d, reference=refs, backend=backend, verbose=verbose,
                              dvars_reference_intensity=ref_intensity, **kw)
    # backward chain per frame: [eddy_b, rigid_b] maps frame points -> mean points (rigid applied first). The forward
    # map is R_b^-1 o E_b^-1 = E' o R_b^-1 with E' = R_b^-1 E_b^-1 R_b, kept in the pipeline's [eddy, rigid] order.
    fwd, params, homogs, jac_f = [], [], [], []
    pe_b = res_b.get("pe_affine")
    for li in range(n):
        chain_b = list(res_b.fwdtransforms[li])
        rigid_b = chain_b[-1]
        rigid_f = _invert_affine_file(rigid_b, rigid_b.replace(".mat", "_fwd.mat"))
        chain_f = [rigid_f]
        if len(chain_b) == 2:  # an eddy affine precedes the rigid
            tb = ants.read_transform(rigid_b)
            eb = ants.read_transform(chain_b[0])
            Rb = _extract_rigid_parameters(tb, 3)[2]
            Mb = np.array(eb.parameters[:9], dtype=np.float64).reshape(3, 3)
            tbv = np.array(eb.parameters[9:12], dtype=np.float64)
            c = np.array(eb.fixed_parameters, dtype=np.float64)
            Hb_e = np.eye(4); Hb_e[:3, :3] = Mb; Hb_e[:3, 3] = tbv + c - Mb @ c  # y = M (x - c) + c + t
            Hp = np.linalg.inv(Rb) @ np.linalg.inv(Hb_e) @ Rb  # E' (affine)
            te = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
            te.set_parameters(np.concatenate([Hp[:3, :3].flatten(), Hp[:3, 3] - c + Hp[:3, :3] @ c]))
            te.set_fixed_parameters(c)
            e_path = chain_b[0].replace(".mat", "_fwd.mat")
            ants.write_transform(te, e_path)
            chain_f = [e_path, rigid_f]
        fwd.append(chain_f)
        tr_, rot_, T_h = _extract_rigid_parameters(ants.read_transform(rigid_f), 3)
        params.append(np.concatenate([tr_, rot_]))
        homogs.append(T_h)
        jac_f.append(1.0 / float(pe_b["jacobian"][li]) if pe_b is not None else 1.0)
    corrected = []
    for li in range(n):
        w = _apply_transforms(fixed=mean_img, moving=frames[li], transformlist=fwd[li], interpolator=interp)
        corrected.append(w * jac_f[li] if jac_f[li] != 1.0 else w)
    out = dict(res_b)
    out["motion_corrected"] = ants.list_to_ndimage(img4d_g, corrected)
    out["fwdtransforms"] = fwd
    out["invtransforms"] = [list(c) for c in fwd]
    mp = res_b.motion_parameters
    out["motion_parameters"] = type(mp)(np.array(params), columns=list(getattr(mp, "_columns", []) or []) or None,
                                        rad_to_deg_cols=list(getattr(mp, "_rad_cols", []) or []) or None, spatial_dim=3)
    if pe_b is not None:
        pe_f = dict(pe_b)
        pe_f["a"] = -np.asarray(pe_b["a"])  # the forward eddy coefficients are the negated backward ones (to first order)
        pe_f["jacobian"] = np.asarray(jac_f)
        pe_f["coefficients"] = [None if c is None else -np.asarray(c) for c in pe_b["coefficients"]] if isinstance(pe_b.get("coefficients"), list) else pe_b.get("coefficients")
        out["pe_affine"] = pe_f
    out["registration_direction"] = "backward"
    from .motion import MotionCorrectionResult

    return MotionCorrectionResult(out)


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


def build_bilateral_window_references(
    frames: Sequence[ants.ANTsImage] | ants.ANTsImage,
    encoding_covariates: Optional[np.ndarray] = None,
    window_size: int = 16,
    sigma_enc: float = 0.5,
) -> List[ants.ANTsImage]:
    """Build per-frame contrast-matched references using a bilateral kernel over time and encoding state:

    ``I_ref,t(x) = sum_tau K_time(t, tau; w) K_enc(e_t, e_tau) I_tau(x) / sum_tau K_time(t, tau; w) K_enc(e_t, e_tau)``

    where ``K_time(t, tau; w) = max(0, 1 - |t - tau| / w)`` restricts averaging to a local temporal window
    so slow head drift does not blur the template, and ``K_enc(e_t, e_tau) = exp(-||e_t - e_tau||^2 / (2 sigma_enc^2))``
    weights frames with matching or near-matching encoding state highest.
    """
    from .motion_batched import build_encoding_covariates

    if isinstance(frames, ants.ANTsImage):
        if frames.dimension != 4:
            raise ValueError(f"Expected 4D ANTsImage or sequence of 3D images, got {frames.dimension}D")
        frame_list = [ants.slice_image(frames, axis=3, idx=t) for t in range(frames.shape[3])]
    else:
        frame_list = list(frames)
    T = len(frame_list)
    if T == 0:
        return []
    if T == 1:
        return [frame_list[0].clone()]

    stack = np.stack([f.numpy().astype(np.float32) for f in frame_list], axis=0)  # (T, X, Y, Z)
    w = max(1.0, float(window_size))
    t_idx = np.arange(T, dtype=np.float64)
    dt = np.abs(t_idx[:, None] - t_idx[None, :])
    k_time = np.maximum(0.0, 1.0 - dt / w) + 1e-3 * np.exp(-0.5 * (dt / w) ** 2)

    if encoding_covariates is not None:
        enc_basis = build_encoding_covariates(
            encoding_covariates=encoding_covariates,
            num_frames=T,
            window_size=int(w),
        )
        E = enc_basis["E"]
        K_eff = int(enc_basis["K_eff"])
        if K_eff > 0:
            diff_e = E[:, None, :] - E[None, :, :]
            dist2 = np.sum(diff_e * diff_e, axis=-1) / float(K_eff)
            sig2 = max(float(sigma_enc) ** 2, 1e-4)
            k_enc = np.exp(-0.5 * dist2 / sig2)
        else:
            k_enc = np.ones((T, T), dtype=np.float64)
    else:
        k_enc = np.ones((T, T), dtype=np.float64)

    W = k_time * k_enc
    W = W / np.maximum(np.sum(W, axis=1, keepdims=True), 1e-12)
    ref_stack = np.tensordot(W.astype(np.float32), stack, axes=([1], [0]))
    ref_dtype = frame_list[0].dtype
    return [ants.new_image_like(frame_list[0], ref_stack[t].astype(ref_dtype)) for t in range(T)]


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
    sample_mask_fn=None,
    bias_covariates: Optional[np.ndarray] = None,
    registration_direction: str = "forward",
    metric_smoothing_mm: float = 0.0,
    encoding_covariates: Optional[np.ndarray] = None,
    window_size: Optional[int] = None,
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
    sample_mask_fn : callable, optional
        ``fn(anchor_mean) -> 3-D mask image in the anchor group's space`` (DWI: gray matter + CSF from the T1, projected
        onto the corrected b0 mean). After the first pass and the cross registrations, every group g >= 1 is registered
        AGAIN with the mask mapped into its own mean space (nearest neighbour through the inverse cross chain) as the
        solver's ``sample_mask``: the metric then sees only voxels whose signal does not depend on the frame's
        contrast. The mask must be right relative to the reference only; the moving frames' motion is the unknown and
        the sampled points are warped by it. Cross registrations are recomputed from the second pass.
    bias_covariates : np.ndarray, shape (n_frames, k), optional
        Per-frame covariates of a deterministic registration bias (DWI: even spherical-harmonic terms, l = 2, of the
        gradient direction); forwarded per group to the solver, which regresses the 6 rigid parameters on them and
        removes the explained part from the written poses (see ``motion_batched``). Zero rows (b0) are skipped.
    registration_direction : {'forward', 'backward'}
        'forward' registers every frame (moving) to its group mean (fixed). 'backward' registers the mean onto each frame
        (the frame is the fixed image) and returns the result in the forward convention -- see
        ``_register_group_backward`` for the measurement that motivates it (0.253 -> 0.148 mm RMS pose error per b3000
        volume on the still phantom). Applies to every group; the b0 group is unaffected in practice (same contrast).
    metric_smoothing_mm : float
        Gaussian smoothing (mm) of both images for the metric only, applied with 'backward' (2 mm: 0.148 -> 0.144 mm,
        and no loss under 1 mm / 1 deg motion; 3 mm hurt capture). The data are resampled from the originals.
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
        for key in ("seed_mode", "schedule_mode", "sample_fraction", "convergence_tol_vox", "max_extend_factor"):
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

    # Normalise DVARS by the anchor (b=0) mean intensity across all groups, then harmonise each non-anchor shell's
    # baseline median to the anchor shell in preset='dwi' so directional white-matter anisotropy does not create a
    # cross-shell sawtooth or b=0 V-notches when interleaved in acquisition order.
    anchor = means[0].numpy()
    anchor_fg = anchor[anchor > 0]
    ref_intensity = float(np.mean(anchor_fg)) if anchor_fg.size > 0 else None
    group_ref_intensities: list[float | None] = [ref_intensity] * n_groups
    results = []
    reference_modes: list = []
    group_kw: list = []
    for g in range(n_groups):
        ref_int_g = group_ref_intensities[g]
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
        if bias_covariates is not None:
            bc_g = np.asarray(bias_covariates, dtype=np.float64).reshape(n_frames, -1)[np.asarray(idx[g])]
            if np.any(bc_g != 0):
                kw["bias_covariates"] = bc_g
        if encoding_covariates is not None:
            ec_g = np.asarray(encoding_covariates, dtype=np.float64).reshape(n_frames, -1)[np.asarray(idx[g])]
            if np.any(np.std(ec_g, axis=0) > 1e-6):
                kw["encoding_covariates"] = ec_g
        elif preset == "dwi":
            ec_parts_g = []
            if pe_affine_covariates is not None:
                pe_g_arr = np.asarray(pe_affine_covariates, dtype=np.float64).reshape(n_frames, -1)[np.asarray(idx[g])]
                if np.any(np.std(pe_g_arr, axis=0) > 1e-6):
                    ec_parts_g.append(pe_g_arr)
            if bias_covariates is not None:
                bc_g_arr = np.asarray(bias_covariates, dtype=np.float64).reshape(n_frames, -1)[np.asarray(idx[g])]
                if np.any(np.std(bc_g_arr, axis=0) > 1e-6):
                    ec_parts_g.append(bc_g_arr)
            if ec_parts_g:
                kw["encoding_covariates"] = np.hstack(ec_parts_g) if len(ec_parts_g) > 1 else ec_parts_g[0]
        if window_size is not None:
            kw["window_size"] = int(window_size)
        elif preset == "dwi" and "window_size" in (cfg or {}):
            kw.setdefault("window_size", int(cfg["window_size"]))
        group_kw.append(kw)
        if registration_direction == "backward":
            res_g = _register_group_backward(img4d[g], means[g], kw, backend, verbose, ref_int_g, metric_smoothing_mm, interp if g == 0 else "linear")
        elif registration_direction == "forward":
            res_g = motion_correction(img4d[g], reference=means[g], backend=backend, verbose=verbose,
                                      dvars_reference_intensity=ref_int_g, **kw)
        else:
            raise ValueError(f"registration_direction must be 'forward' or 'backward', got {registration_direction!r}")
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
                                          dvars_reference_intensity=ref_int_g, **kw_it)
                mode_g = f"predicted (iteration {it + 1})"
        reference_modes.append(mode_g)
        results.append(res_g)

    # Corrected group means, then one careful registration per adjacent pair of contrasts.
    cmeans = [_mean_image([ants.slice_image(results[g].motion_corrected, axis=3, idx=i) for i in range(len(idx[g]))])
              for g in range(n_groups)]
    cross = [robust_affine(fixed=cmeans[g - 1], moving=cmeans[g], dof="rigid", mode="auto", verbose=verbose,
                           device=kwargs.get("device"), num_bins=cross_num_bins,
                           convergence_tol_vox=kwargs.get("convergence_tol_vox", 0.05), max_extend_factor=kwargs.get("max_extend_factor", 4)) for g in range(1, n_groups)]

    # Second pass on the direction-dependent groups with the metric restricted to the caller's mask (anchor space ->
    # each group's mean space through the inverse cross chain); the cross registrations are then recomputed.
    mask_pass = False
    if sample_mask_fn is not None and n_groups > 1:
        mask_anchor = sample_mask_fn(cmeans[0])
        for g in range(1, n_groups):
            chain_paths = [t for k in range(g) for t in cross[k]["fwdtransforms"]]
            mask_g = ants.apply_transforms(fixed=cmeans[g], moving=mask_anchor, transformlist=chain_paths,
                                           whichtoinvert=[True] * len(chain_paths), interpolator="nearestNeighbor")
            kw2 = dict(group_kw[g])
            kw2["sample_mask"] = mask_g
            if kw2.get("outprefix") is not None:
                kw2["outprefix"] = f"{kw2['outprefix']}m_"
            results[g] = motion_correction(img4d[g], reference=means[g], backend=backend, verbose=verbose,
                                           dvars_reference_intensity=group_ref_intensities[g], **kw2)
            reference_modes[g] = reference_modes[g] + " + masked metric"
        mask_pass = True
        cmeans = [_mean_image([ants.slice_image(results[g].motion_corrected, axis=3, idx=i) for i in range(len(idx[g]))])
                  for g in range(n_groups)]
        cross = [robust_affine(fixed=cmeans[g - 1], moving=cmeans[g], dof="rigid", mode="auto", verbose=verbose,
                               device=kwargs.get("device"), num_bins=cross_num_bins,
                           convergence_tol_vox=kwargs.get("convergence_tol_vox", 0.05), max_extend_factor=kwargs.get("max_extend_factor", 4)) for g in range(1, n_groups)]

    # Replace leading 0.0 DVARS placeholders on non-anchor shells and harmonise per-shell DVARS medians in preset='dwi'
    # so interleaved shells share a common baseline without a 2-volume sawtooth or b=0 V-notches.
    for g in range(n_groups):
        if len(idx[g]) > 1 and int(idx[g][0]) > 0:
            for dv_key in ("dvars", "dvars_post", "dvars_pre"):
                dv_arr = results[g].get(dv_key)
                if dv_arr is not None and len(dv_arr) > 1 and float(dv_arr[0]) == 0.0:
                    dv_copy = np.asarray(dv_arr, dtype=np.float64).copy()
                    dv_copy[0] = dv_copy[1]
                    results[g][dv_key] = dv_copy
    if preset == "dwi" and n_groups > 1:
        ig0 = np.asarray(idx[0], dtype=int)
        dt0 = np.diff(ig0) if len(ig0) >= 4 else np.array([], dtype=int)
        for dv_key in ("dvars", "dvars_post", "dvars_pre"):
            dv0 = results[0].get(dv_key)
            if dv0 is not None and np.any(np.asarray(dv0) > 0):
                dv0_arr = np.asarray(dv0, dtype=np.float64).copy()
                med0 = float(np.median(dv0_arr[dv0_arr > 0]))
                if dt0.size >= 3 and float(np.median(dt0)) >= 4.0 and np.any(dt0 == 1):
                    long_vals = dv0_arr[1:][dt0 >= 0.5 * float(np.median(dt0))]
                    if long_vals.size >= 2:
                        med0 = float(np.median(long_vals))
                        results[0][f"{dv_key}_shell_raw"] = dv0_arr.copy()
                        for k_short in np.where(dt0 == 1)[0]:
                            if 0.0 < dv0_arr[k_short + 1] < med0:
                                dv0_arr[k_short + 1] = med0
                        results[0][dv_key] = dv0_arr
                for g in range(1, n_groups):
                    dvg = results[g].get(dv_key)
                    if dvg is not None and np.any(np.asarray(dvg) > 0):
                        medg = float(np.median(np.asarray(dvg)[np.asarray(dvg) > 0]))
                        if medg > 0:
                            results[g][f"{dv_key}_shell_raw"] = np.asarray(dvg, dtype=np.float64).copy()
                            results[g][dv_key] = np.asarray(dvg, dtype=np.float64) * (med0 / medg)

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

    composed_all = _compose(mats)
    fd_raw = jenkinson_framewise_displacement(composed_all, kwargs.get("fd_radius", 50.0), com)
    nuisance = None
    pe_affine = None
    fd = fd_raw
    composed_rigid = composed_all
    if any(r.get("pe_affine") is not None for r in results):
        # the eddy PE-affine is in the chain (and in fd_raw); FD counts the rigid part only: crosses + the frame's rigid
        n_cross = [len(c["fwdtransforms"]) for c in cross]
        mats_rigid = [ms[:sum(n_cross[:g])] + [ms[-1]] for ms, g in zip(mats, [int(groups[i]) for i in range(n_frames)])]
        composed_rigid = _compose(mats_rigid)
        fd = jenkinson_framewise_displacement(composed_rigid, kwargs.get("fd_radius", 50.0), com)
        a_all = np.zeros((n_frames, 4)); jac_all = np.ones(n_frames); coef = []
        for g in range(n_groups):
            pe_g = results[g].get("pe_affine")
            cg = None if pe_g is None else pe_g["coefficients"]
            coef.append(cg[0] if isinstance(cg, list) and len(cg) == 1 else cg)  # one matrix per batch; usually one batch
            if pe_g is not None:
                a_all[np.asarray(idx[g])] = pe_g["a"]; jac_all[np.asarray(idx[g])] = pe_g["jacobian"]
        p_unit = np.asarray(pe_axis, dtype=np.float64) / np.linalg.norm(pe_axis)
        shift_mm_all = a_all[:, 0].copy()
        # ASL-inspired surround-difference cross-shell SE(3) debiasing + intrinsic PE-axis B0/eddy attribution:
        # In EPI DWI, respiration-induced delta-B0(t) and residual eddy currents translate volumes purely along p_unit
        # without rigid rotation or orthogonal translation. Transfer uncoupled high-frequency PE oscillations from
        # composed_rigid into shift_mm_all while gating by non-PE framewise displacement so real head jerks are kept.
        if preset == "dwi" and n_frames >= 12:
            from scipy.spatial.transform import Rotation as _Rot
            d_com = np.array([H[:3, 3] + (H[:3, :3] - np.eye(3)) @ com for H in composed_rigid], dtype=np.float64)
            rv = _Rot.from_matrix(np.array([H[:3, :3] for H in composed_rigid], dtype=np.float64)).as_rotvec()
            Y_se3 = np.c_[d_com, rv]
            # 1) Surround-difference cross-shell offset removal: u[t] = G_surr[t] @ B_shell
            u_se3 = Y_se3[1:-1] - 0.5 * (Y_se3[:-2] + Y_se3[2:])
            E_shell = np.column_stack([(groups == g).astype(np.float64) for g in range(1, n_groups)])
            G_surr = E_shell[1:-1] - 0.5 * (E_shell[:-2] + E_shell[2:])
            active_surr = np.any(np.abs(G_surr) > 0.1, axis=1)
            if int(np.sum(active_surr)) >= 4 * (n_groups - 1):
                Gs = G_surr[active_surr]
                us = u_se3[active_surr]
                w_s = np.ones(len(Gs), dtype=np.float64)
                B_shell = np.zeros((n_groups - 1, 6), dtype=np.float64)
                for _ in range(3):
                    B_shell, *_ = np.linalg.lstsq(Gs * np.sqrt(w_s)[:, None], us * np.sqrt(w_s)[:, None], rcond=None)
                    res_s = us - Gs @ B_shell
                    nrm_s = np.linalg.norm(res_s[:, :3], axis=1)
                    sc_s = max(1.4826 * float(np.median(np.abs(nrm_s - np.median(nrm_s)))), 1e-4)
                    r_s = nrm_s / (4.685 * sc_s)
                    w_s = np.where(r_s < 1.0, (1.0 - r_s ** 2) ** 2, 1e-4)
                Y_se3 = Y_se3 - E_shell @ B_shell
            d_com = Y_se3[:, :3]
            rv = Y_se3[:, 3:]
            # 2) Intrinsic PE-axis projection and non-PE motion-gated B0/eddy attribution
            t_par = d_com @ p_unit
            T_perp = d_com - np.outer(t_par, p_unit)
            t_norm = np.linspace(-1.0, 1.0, n_frames)
            X_pe = np.c_[np.ones(n_frames), t_norm, t_norm ** 2, t_norm ** 3, T_perp, rv]
            w_irls = np.ones(n_frames, dtype=np.float64)
            beta_pe = np.zeros(X_pe.shape[1], dtype=np.float64)
            for _ in range(3):
                Xw = X_pe * np.sqrt(w_irls)[:, None]
                yw = t_par * np.sqrt(w_irls)
                beta_pe, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
                res_pe = t_par - X_pe @ beta_pe
                s_pe = max(1.4826 * float(np.median(np.abs(res_pe - np.median(res_pe)))), 1e-4)
                u_pe = res_pe / (4.685 * s_pe)
                w_irls = np.where(np.abs(u_pe) < 1.0, (1.0 - u_pe ** 2) ** 2, 1e-4)
            t_par_hat = X_pe @ beta_pe
            dc_pe = float(np.median(t_par_hat))
            r_par = t_par - t_par_hat
            fd_rad = float(kwargs.get("fd_radius", 50.0))
            fd_perp_rot = np.r_[0.0, np.sqrt((np.diff(T_perp, axis=0) ** 2).sum(axis=1) + (fd_rad ** 2) * (np.diff(rv, axis=0) ** 2).sum(axis=1))]
            tau_perp = max(0.20, 3.0 * float(np.median(fd_perp_rot[1:])))
            w_b0 = np.exp(-0.5 * (np.maximum(0.0, fd_perp_rot - tau_perp) / tau_perp) ** 2)
            delta_pe_b0 = dc_pe + w_b0 * r_par
            shift_mm_all = shift_mm_all + delta_pe_b0
            d_com_clean = T_perp + np.outer(t_par - delta_pe_b0, p_unit)
            R_clean = _Rot.from_rotvec(rv).as_matrix()
            composed_rigid = []
            for i in range(n_frames):
                H_i = np.eye(4, dtype=np.float64)
                H_i[:3, :3] = R_clean[i]
                H_i[:3, 3] = d_com_clean[i] - (R_clean[i] - np.eye(3)) @ com
                composed_rigid.append(H_i)
            fd = jenkinson_framewise_displacement(composed_rigid, fd_rad, com)
        pe_affine = {"axis": [float(x) for x in p_unit], "coefficients_by_group": coef,
                     "a": a_all, "jacobian": jac_all, "shift_mm": shift_mm_all}
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
        composed_rigid = _compose(mats_motion)
        fd = jenkinson_framewise_displacement(composed_rigid, kwargs.get("fd_radius", 50.0), com)
        nuisance = {"axis": [float(a) for a in axis], "shift_mm": shift, "coefficients": coefficients, "r2": r2}

    # Extract composed 6-DOF rigid parameters in anchor space (centered at com), matching composed_rigid / fd
    translations_mm = np.zeros((n_frames, 3), dtype=np.float64)
    rotations_deg = np.zeros((n_frames, 3), dtype=np.float64)
    for i, H in enumerate(composed_rigid):
        R_i = H[:3, :3]
        translations_mm[i] = H[:3, 3] + (R_i - np.eye(3)) @ com
        sy = -R_i[2, 0]
        if np.abs(sy) < 0.999999:
            ry = np.arcsin(np.clip(sy, -1.0, 1.0))
            rx = np.arctan2(R_i[2, 1], R_i[2, 2])
            rz = np.arctan2(R_i[1, 0], R_i[0, 0])
        else:
            ry = (np.pi / 2.0) if sy > 0 else (-np.pi / 2.0)
            rx = np.arctan2(-R_i[1, 2], R_i[1, 1])
            rz = 0.0
        rotations_deg[i] = np.degrees([rx, ry, rz])

    out = GroupedMotionCorrectionResult()
    out["motion_corrected"] = motion_corrected
    out["fd"] = fd
    out["fd_raw"] = fd_raw
    out["translations_mm"] = translations_mm
    out["rotations_deg"] = rotations_deg
    out["nuisance"] = nuisance
    out["pe_affine"] = pe_affine
    out["reference_modes"] = reference_modes
    out["masked_metric"] = mask_pass
    out["bias"] = [r.get("bias") for r in results]
    out["encoding_pose_bias_mm_rad"] = [r.get("encoding_pose_bias_mm_rad") for r in results]
    # convergence of the per-frame registrations (every stage of every frame ended on the pose-step test) and of each cross-shell
    # registration (the winning path's stages); False entries are frames/steps that were CAPPED, never silently final.
    frame_converged = np.ones(int(sum(len(i) for i in idx)), dtype=bool)
    for g, r in enumerate(results):
        conv = r.get("convergence")
        if conv:
            for local_i, t in enumerate(idx[g]):
                frame_converged[t] = bool(conv["frame_converged"][local_i])
    out["frame_converged"] = frame_converged
    out["cross_converged"] = [bool(all(rec["converged"] for rec in c.get("convergence", []) if rec.get("path") == c.get("init_candidate") and rec.get("level", 1) == 1)) for c in cross]
    out["fwdtransforms"] = chains
    out["group_results"] = results
    out["group_indices"] = idx
    out["group_means"] = cmeans
    out["cross_registrations"] = cross
    out["preset"] = preset
    return out
