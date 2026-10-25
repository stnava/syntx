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
import warnings

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
    window_size: Optional[int] = None,
    sigma_enc: float = 0.5,
    exclude_self: bool = False,
) -> List[ants.ANTsImage]:
    """Build per-frame contrast-matched references with a kernel over encoding state (and optionally time):

    ``I_ref,t(x) = sum_tau K_time(t, tau) K_enc(e_t, e_tau) I_tau(x) / sum_tau K_time(t, tau) K_enc(e_t, e_tau)``

    ``K_enc(e_t, e_tau) = exp(-||e_t - e_tau||^2 / (2 K sigma_enc^2))`` weights frames of matching encoding state; for
    discrete states (slow-flow ``[b0, +X, +Y, +Z]``, ASL tag/control) the reference of frame t is the mean of all frames
    in t's state. ``window_size=None`` (default, what motion correction uses) sets ``K_time = 1``: every reference is a
    mean over the WHOLE series, so all references share one head position and the registered poses are in one common
    space. A finite ``window_size`` (``K_time = max(0, 1 - |t - tau| / w)``) gives local references; poses against
    them are relative to a moving average of the series, i.e. slow drift is NOT measured -- use it only to build
    templates, never as a motion-correction target.

    ``exclude_self``: leave-one-out, frame t gets no weight in its own reference. Off by default: for discrete states
    the inclusive state mean is ONE reference shared by every frame of the state, which is what puts their poses in a
    common space; leave-one-out gives every frame a slightly different reference (simulation, 3 seeds: slow-flow preset
    head error 0.031 -> 0.115 mm, ASL state-matched 0.017 -> 0.060 mm). With continuous encodings (one frame per DWI
    direction) the frame's own weight dominates the inclusive kernel; see ``encoding_reference_predictor``.
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
    if window_size is None:
        k_time = np.ones((T, T), dtype=np.float64)
    else:
        w = max(1.0, float(window_size))
        t_idx = np.arange(T, dtype=np.float64)
        dt = np.abs(t_idx[:, None] - t_idx[None, :])
        k_time = np.maximum(0.0, 1.0 - dt / w) + 1e-3 * np.exp(-0.5 * (dt / w) ** 2)

    if encoding_covariates is not None:
        enc_basis = build_encoding_covariates(
            encoding_covariates=encoding_covariates,
            num_frames=T,
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
    if exclude_self:
        np.fill_diagonal(W, 0.0)
    W = W / np.maximum(np.sum(W, axis=1, keepdims=True), 1e-12)
    W = W.astype(np.float32)
    ref_dtype = frame_list[0].dtype
    # one reference at a time: peak memory is the input stack plus the returned images, not a second 4-D stack
    return [ants.new_image_like(frame_list[0], np.tensordot(W[t], stack, axes=(0, 0)).astype(ref_dtype)) for t in range(T)]


def _group_offsets(H, groups, idx, com):
    """Per-group constant pose offset of composed rigid poses ``H`` (n, 4, 4) relative to the anchor group 0, from the
    surround Tukey projection on the group indicators. Returns ``(Y, offsets)`` with ``Y`` the (n, 6) poses
    ``[d_com, rotvec]`` and ``offsets`` {group: (6,)}, or ``None`` when the groups are not interleaved enough to separate an
    offset from head motion (fewer than 8 frames, or fewer than 4 active surround rows for some non-anchor group: a
    blocked acquisition's group offset is indistinguishable from the head motion between the blocks)."""
    from scipy.spatial.transform import Rotation as _Rot

    from .motion_batched import _project_encoding_surround_tukey_np, build_encoding_covariates

    n_frames, n_groups = len(H), len(idx)
    if n_frames < 8 or n_groups < 2:
        return None
    ind = np.column_stack([(groups == g).astype(np.float64) for g in range(1, n_groups)])
    surr = ind[1:-1] - 0.5 * (ind[:-2] + ind[2:])
    if int(np.min(np.sum(np.abs(surr) > 0.1, axis=0))) < 4:
        return None
    H = np.asarray(H)
    Y = np.c_[np.einsum("nij,j->ni", H[:, :3, :3], com) + H[:, :3, 3] - com, _Rot.from_matrix(H[:, :3, :3]).as_rotvec()]
    basis = build_encoding_covariates(state_labels=groups, num_frames=n_frames, window_size=n_frames)
    _, dY, _ = _project_encoding_surround_tukey_np(Y, basis["E"], basis["psi"])
    off = dY - np.median(dY[idx[0]], axis=0)          # relative to the anchor group (which stays put)
    return Y, {g: np.median(off[idx[g]], axis=0) for g in range(1, n_groups)}


def _pose_matrix(y, com):
    """4x4 rigid matrix of the pose ``y = [d_com, rotvec]`` about ``com``."""
    from scipy.spatial.transform import Rotation as _Rot

    R = _Rot.from_rotvec(y[3:]).as_matrix()
    h = np.eye(4)
    h[:3, :3] = R
    h[:3, 3] = com + y[:3] - R @ com
    return h


def _debias_cross_group_offsets(chains, groups, idx, results, cross, com, outprefix):
    """Per-frame rigid corrections removing the contrast-driven offset of each non-anchor group (see
    ``motion_correct_grouped(cross_group_debias=True)``). Returns ``None`` when the groups are not interleaved enough to
    separate an offset from head motion (fewer than 4 active surround rows per non-anchor group)."""
    import os
    import tempfile

    from .motion import _extract_rigid_parameters

    n_frames = len(chains)
    n_cross = [len(c["fwdtransforms"]) for c in cross]
    H = []
    for i in range(n_frames):
        g = int(groups[i])
        rigid_chain = chains[i][:sum(n_cross[:g])] + [chains[i][-1]]   # crosses + the frame's own rigid (no eddy)
        h = np.eye(4)
        for f in rigid_chain:
            h = _extract_rigid_parameters(ants.read_transform(f), 3)[2] @ h
        H.append(h)
    H = np.asarray(H)
    est = _group_offsets(H, groups, idx, com)
    if est is None:
        return None
    Y, per_group = est
    work = os.path.dirname(f"{outprefix}x") if outprefix else tempfile.mkdtemp(prefix="syntx_xgroup_")
    os.makedirs(work, exist_ok=True)
    paths = {}
    for i in range(n_frames):
        g = int(groups[i])
        if g == 0:
            continue
        Hc = _pose_matrix(Y[i] - per_group[g], com)
        corr = np.linalg.inv(H[i]) @ Hc                  # H_new = H @ corr: corr acts on the anchor-space point first
        tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
        tx.set_parameters(np.concatenate([corr[:3, :3].ravel(), corr[:3, 3]]))
        tx.set_fixed_parameters(np.zeros(3))
        path = os.path.join(work, f"{os.path.basename(outprefix) if outprefix else ''}xgroup_corr_{i:04d}.mat")
        ants.write_transform(tx, path)
        paths[i] = path
    return {
        "method": "surround_tukey_projection_on_group_indicators",
        "offset_mm_rad_by_group": {int(g): v.tolist() for g, v in per_group.items()},
        "offset_mm_by_group": {int(g): float(np.linalg.norm(v[:3])) for g, v in per_group.items()},
        "offset_deg_by_group": {int(g): float(np.degrees(np.linalg.norm(v[3:]))) for g, v in per_group.items()},
        "corrected_frames": sorted(paths),
        "correction_paths": paths,
    }


def encoding_reference_predictor(encoding_covariates: np.ndarray, sigma_enc: float = 0.5, exclude_self: bool = True):
    """``reference_predictor`` for ``motion_correct_grouped``: each frame's reference is the kernel-weighted mean of the
    group's OTHER corrected frames with a similar encoding (``build_bilateral_window_references``, whole series, no time
    window), i.e. a contrast-matched reference for continuous encodings. DWI: pass the even (l = 2) gradient terms --
    the image contrast is even in g. A group with no encoding variation (b0) keeps its mean (``None``).

    Measured on the simulated DWI series (b1000, 20 directions, 3 seeds, ``benchmarks/motion_encoding_bias_simulation.py``):
    leave-one-out gives no reliable gain over the shell mean (rotation error 0.56 -> 0.51 deg without eddy, 0.63 ->
    0.58 with). Including the frame (``exclude_self=False``) lowered it to 0.40 deg, but with 20 distinct directions the
    frame's own weight dominates that kernel (the frame is largely registered to its own previous resampling) and some
    frames stopped converging on antsxdwi's convergence test, so it is not used by default anywhere."""
    cov = np.asarray(encoding_covariates, dtype=np.float64)
    cov = cov[:, None] if cov.ndim == 1 else cov

    def predictor(frames, group_index=None, frame_indices=None):
        c = cov[np.asarray(frame_indices, dtype=int)]
        if not np.any(np.std(c, axis=0) > 1e-8):
            return None
        return build_bilateral_window_references(list(frames), encoding_covariates=c, sigma_enc=sigma_enc, exclude_self=exclude_self)

    return predictor


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
    cross_group_debias: bool = False,
    debias_pe_pseudomotion: bool = False,
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
    encoding_covariates : np.ndarray, shape (n_frames,) or (n_frames, k), optional
        Per-frame encoding covariates forwarded per group to ``motion_correction(encoding_covariates=...)`` (in-optimizer
        overlapping-window Tukey projection of encoding-locked registration bias out of the written poses). Use terms
        that describe the CONTRAST (DWI: even, l = 2, gradient terms), not the eddy displacement (odd in g, modelled by
        ``pe_affine_covariates`` and applied).
    debias_pe_pseudomotion : bool, default False
        Opt-in (needs ``pe_axis`` and >= 12 frames; otherwise a warning and ``{"method": "skipped"}``): shrink the
        high-frequency residual of the REPORTED rigid trajectory (``fd``, ``translations_mm``, ``rotations_deg``) along
        phase encode (and, gated on a negative step autocorrelation, the other degrees of freedom) as non-linear eddy /
        respiratory B0 pseudomotion. The transforms and the corrected series are unchanged; ``fd_before_pe_debias`` keeps
        the previous FD. Independent real jitter between volumes also alternates and is shrunk (audit: nodding jitter
        FD 0.94 -> 0.15 mm), so FD-based censoring after it under-counts real motion.
    cross_group_debias : bool, default False
        Opt-in: remove a contrast-driven bias of the cross-group (group mean -> group mean) registrations: when the groups
        are interleaved in time, the composed rigid trajectory is projected on the group indicators with the surround
        Tukey estimator (``motion_batched._project_encoding_surround_tukey_np``; smooth head drift and isolated jumps
        are in its nullspace / downweighted) and the per-frame offset of every non-anchor group relative to the anchor
        is removed by a correction transform at the START of its chain (applied before the single resampling).
        On the simulated DWI series (``benchmarks/motion_encoding_bias_simulation.py``, 3 seeds) the cross chain was
        already unbiased (head translation error 0.111 -> 0.115 mm with it), hence opt-in. Reported as
        ``cross_group_offset``; ``None`` when off or when the groups are not interleaved.
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

    # DVARS of every group is normalised against the anchor group's mean intensity (comparable across groups and subjects).
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
        if window_size is not None and "encoding_covariates" in kw:
            kw["window_size"] = int(window_size)
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

    chains: list = [None] * n_frames
    for g in range(n_groups):
        pre = [t for k in range(g) for t in cross[k]["fwdtransforms"]]  # [cross_1, ..., cross_g]
        for li, gi in enumerate(idx[g]):
            chains[gi] = pre + list(results[g].fwdtransforms[li])

    # Series-level Jenkinson FD in acquisition order: every frame's total rigid transform in group 0's space
    # (a point maps through cross_1 ... cross_g, then the frame's own transform: H = H_own @ H_cross_g @ ... @ H_cross_1).
    from .motion import jenkinson_framewise_displacement

    com = np.asarray(means[0].get_center_of_mass(), dtype=np.float64)

    def _compose(per_frame_mats):
        homog = []
        for ms in per_frame_mats:
            h = np.eye(4)
            for m in ms:
                h = m @ h
            homog.append(h)
        return homog

    cross_group_offset = None
    if cross_group_debias and n_groups > 1:
        cross_group_offset = _debias_cross_group_offsets(chains, groups, idx, results, cross, com, kwargs.get("outprefix"))
        if cross_group_offset is not None:
            for gi, corr_path in cross_group_offset.pop("correction_paths").items():
                chains[gi] = [corr_path] + chains[gi]   # first in the chain: acts on the anchor-space point first

    frames_out: list = [None] * n_frames
    for g in range(n_groups):
        for li, gi in enumerate(idx[g]):
            if g == 0 and len(chains[gi]) == len(results[0].fwdtransforms[li]):
                frames_out[gi] = ants.slice_image(results[0].motion_corrected, axis=3, idx=li)
            else:  # ONE resampling from the original through the composed chain: every interpolation blurs
                frames_out[gi] = _apply_transforms(fixed=means[0], moving=imgs[g][li], transformlist=chains[gi],
                                                   interpolator=interp)
                pe_g = results[g].get("pe_affine")
                if pe_g is not None and pe_g["jacobian"][li] != 1.0:
                    frames_out[gi] = frames_out[gi] * float(pe_g["jacobian"][li])
    motion_corrected = ants.list_to_ndimage(_make_4d_template_from_3d(frames_out[0], n_frames, tr=tr), frames_out)

    mats = [[_extract_rigid_parameters(ants.read_transform(f), 3)[2] for f in chain] for chain in chains]

    composed_all = _compose(mats)
    fd_raw = jenkinson_framewise_displacement(composed_all, kwargs.get("fd_radius", 50.0), com)
    nuisance = None
    pe_affine = None
    fd = fd_raw
    composed_rigid = composed_all
    if any(r.get("pe_affine") is not None for r in results):
        # the eddy PE-affine is in the chain (and in fd_raw); FD counts the rigid part only: crosses + the frame's rigid
        n_cross = [len(c["fwdtransforms"]) for c in cross]
        n_corr = [0 if cross_group_offset is None or gi not in cross_group_offset["corrected_frames"] else 1 for gi in range(n_frames)]
        mats_rigid = [ms[:n_corr[i] + sum(n_cross[:g])] + [ms[-1]]
                      for i, (ms, g) in enumerate(zip(mats, [int(groups[i]) for i in range(n_frames)]))]
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
        pe_affine = {"axis": [float(x) for x in p_unit], "coefficients_by_group": coef,
                     "a": a_all, "jacobian": jac_all, "shift_mm": a_all[:, 0].copy()}
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

    # Interleaved groups (multi-shell DWI switches shell almost every volume): a constant per-group pose offset left by the
    # cross-group registration is contrast-driven, not head motion, yet it would be a step at every group switch of the
    # reported motion and FD. Report motion and FD without it; the transforms are unchanged (cross_group_debias=True also
    # removes it from them). ``fd_with_group_offsets`` keeps the previous FD.
    group_offset_fd = None
    if cross_group_offset is None and n_groups > 1:
        est = _group_offsets(composed_rigid, groups, idx, com)
        if est is not None:
            Y_rigid, per_group = est
            composed_rigid = [h if int(groups[i]) == 0 else _pose_matrix(Y_rigid[i] - per_group[int(groups[i])], com)
                              for i, h in enumerate(composed_rigid)]
            group_offset_fd = {
                "method": "surround_tukey_projection_on_group_indicators",
                "offset_mm_by_group": {int(g): float(np.linalg.norm(v[:3])) for g, v in per_group.items()},
                "offset_deg_by_group": {int(g): float(np.degrees(np.linalg.norm(v[3:]))) for g, v in per_group.items()},
                "fd_with_group_offsets": np.asarray(fd, dtype=np.float64),
            }
            fd = jenkinson_framewise_displacement(composed_rigid, kwargs.get("fd_radius", 50.0), com)

    # Phase-encode B0 / non-linear eddy pseudomotion debiasing on the reported rigid trajectory (composed_rigid / fd):
    # After linear-in-g eddy removal and cross-group offset removal, high-b DWI volumes retain non-linear (g x g, long-time-constant)
    # eddy and respiratory Delta-B0 jitter strictly along the low-bandwidth phase-encode axis p_PE, whereas readout translation
    # and PE/slice rotations remain at the true rigid step floor (~0.05 mm). Shrink excess high-frequency PE residual toward
    # the smooth postural trajectory while gating to zero on frames where the B0-immune DOFs or PE steps indicate a real jump.
    # OPT-IN: an independent audit showed this shrinkage cannot tell independent real head jitter (step lag-1
    # autocorrelation ~ -0.5) from pseudomotion: real nodding jitter FD 0.94 -> 0.15 mm, PE jitter 0.53 -> 0.07 mm on
    # synthetic trajectories; only isolated jumps are protected by the gate. Requested but impossible: announced.
    pe_pseudomotion_debias = None
    if debias_pe_pseudomotion and (pe_axis is None or n_frames < 12):
        reason = "no pe_axis" if pe_axis is None else f"{n_frames} frames < 12"
        warnings.warn(f"motion_correct_grouped: debias_pe_pseudomotion requested but skipped ({reason})", RuntimeWarning, stacklevel=2)
        pe_pseudomotion_debias = {"method": "skipped", "reason": reason}
    if debias_pe_pseudomotion and pe_axis is not None and n_frames >= 12:
        from scipy.spatial.transform import Rotation as _Rot
        from .motion_batched import build_encoding_covariates

        fd_rad = float(kwargs.get("fd_radius", 50.0))
        D_mat = np.asarray(getattr(means[0], "direction", np.eye(3)), dtype=np.float64)[:3, :3].copy()
        for col in range(3):
            nrm = float(np.linalg.norm(D_mat[:, col]))
            if nrm > 1e-8:
                D_mat[:, col] /= nrm
        arr_pe = np.asarray(pe_axis, dtype=np.float64).reshape(-1)
        if arr_pe.size == 1 and float(arr_pe[0]) in (0.0, 1.0, 2.0):
            j_pe = int(arr_pe[0])
            p_pe = D_mat[:, j_pe].copy()
        else:
            p_pe = arr_pe / max(float(np.linalg.norm(arr_pe)), 1e-12)
            j_pe = int(np.argmax(np.abs(D_mat.T @ p_pe)))
        rem_axes = [a for a in range(3) if a != j_pe]
        k_sl = 2 if 2 in rem_axes else rem_axes[-1]
        p_sl_raw = D_mat[:, k_sl] - float(np.dot(D_mat[:, k_sl], p_pe)) * p_pe
        if float(np.linalg.norm(p_sl_raw)) < 1e-6:
            k_sl = rem_axes[0]
            p_sl_raw = D_mat[:, k_sl] - float(np.dot(D_mat[:, k_sl], p_pe)) * p_pe
        p_sl = p_sl_raw / max(float(np.linalg.norm(p_sl_raw)), 1e-12)
        p_ro = np.cross(p_pe, p_sl)
        p_ro = p_ro / max(float(np.linalg.norm(p_ro)), 1e-12)

        R_rig = np.array([H[:3, :3] for H in composed_rigid], dtype=np.float64)
        t_rig = np.array([H[:3, 3] + (H[:3, :3] - np.eye(3)) @ com for H in composed_rigid], dtype=np.float64)
        rv_rig = _Rot.from_matrix(R_rig).as_rotvec()
        rv_mm = fd_rad * rv_rig

        t_ro = t_rig @ p_ro
        t_pe = t_rig @ p_pe
        t_sl = t_rig @ p_sl
        r_ro = rv_mm @ p_ro
        r_pe = rv_mm @ p_pe
        r_sl = rv_mm @ p_sl

        def _fit_tukey(Y_col: np.ndarray, win_sz: int) -> np.ndarray:
            H_w = build_encoding_covariates(num_frames=n_frames, window_size=win_sz)["psi"]
            gram_w = H_w.T @ H_w + 1e-4 * np.eye(H_w.shape[1], dtype=np.float64)
            theta_w = np.linalg.solve(gram_w, H_w.T @ Y_col)
            for _ in range(3):
                res_w = Y_col - H_w @ theta_w
                med_w = np.median(res_w, axis=0)
                mad_w = np.maximum(1.4826 * np.median(np.abs(res_w - med_w), axis=0), 1e-6)
                u_w = (res_w - med_w) / (4.685 * mad_w)
                w_tk = np.where(np.abs(u_w) < 1.0, (1.0 - u_w ** 2) ** 2, 1e-4)
                for col_i in range(Y_col.shape[1]):
                    Hw_i = H_w * w_tk[:, col_i : col_i + 1]
                    theta_w[:, col_i] = np.linalg.solve(
                        Hw_i.T @ H_w + 1e-4 * np.eye(H_w.shape[1], dtype=np.float64),
                        Hw_i.T @ Y_col[:, col_i],
                    )
            return Y_col - H_w @ theta_w

        def _winsorized_ac1(series: np.ndarray) -> float:
            ds = np.diff(series)
            if ds.size < 4:
                return 0.0
            mu = float(np.mean(ds))
            q75 = max(float(np.percentile(np.abs(ds - mu), 75)), 1e-12)
            ds_c = np.clip(ds - mu, -3.0 * q75, 3.0 * q75)
            ds0 = ds_c[:-1] - float(np.mean(ds_c[:-1]))
            ds1 = ds_c[1:] - float(np.mean(ds_c[1:]))
            denom = float(np.sqrt(np.sum(ds0 ** 2) * np.sum(ds1 ** 2)))
            return float(np.sum(ds0 * ds1) / denom) if denom > 1e-12 else 0.0

        w_pe = max(8, min(16, n_frames // 5))
        w_nod = min(6, max(4, n_frames // 8))
        r_pe_res = _fit_tukey(t_pe[:, None], w_pe)[:, 0]
        raw_non_pe = np.column_stack([t_sl, r_ro, t_ro, r_pe, r_sl])
        r_non_pe = _fit_tukey(raw_non_pe, w_nod)
        r_epi = np.column_stack([r_pe_res, r_non_pe])

        raw_all6 = np.column_stack([t_pe, raw_non_pe])
        lag1_ac = [_winsorized_ac1(raw_all6[:, k]) for k in range(6)]
        sigs = [float(np.median(np.abs(np.diff(r_epi[:, k])))) for k in range(6)]

        etas_non_pe = [0.0] * 5
        for idx_np in range(5):
            k = idx_np + 1
            ac1_k = lag1_ac[k]
            sig_k_mm = 1.4826 * sigs[k]
            if ac1_k < -0.38 and sig_k_mm > 0.052:
                etas_non_pe[idx_np] = float(np.clip((-ac1_k - 0.20) / 0.45, 0.0, 0.85))

        imm_clean = []
        for k in (3, 4, 5):
            eta_k = etas_non_pe[k - 1]
            if eta_k > 0.0:
                dr_k = np.zeros(n_frames, dtype=np.float64)
                dr_k[1:] = np.abs(np.diff(r_epi[:, k]))
                dr_local = np.maximum(dr_k, np.r_[dr_k[1:], 0.0])
                scale_k = max(1.4826 * sigs[k], 0.02)
                w_step_k = np.exp(-(np.maximum(0.0, dr_local - 3.5 * scale_k) / (2.5 * scale_k)) ** 2)
                imm_clean.append(raw_all6[:, k] - eta_k * w_step_k * r_epi[:, k])
            else:
                imm_clean.append(raw_all6[:, k])

        sig_ref = float(np.median([np.median(np.abs(np.diff(c))) for c in imm_clean]))
        sag_min_sig = max(1.50 * sig_ref, 0.052 / 1.4826)
        for idx_sag in (0, 1):
            k = idx_sag + 1
            if lag1_ac[k] < -0.38 and sigs[k] > sag_min_sig:
                eta_wiener = max(0.0, 1.0 - (sig_ref / max(sigs[k], 1e-12)))
                etas_non_pe[idx_sag] = max(etas_non_pe[idx_sag], eta_wiener)

        sig_pe = sigs[0]
        eta_pe = max(0.0, 1.0 - (sig_ref / max(sig_pe, 1e-12))) if sig_pe > 1.20 * sig_ref else 0.0
        etas = [eta_pe] + etas_non_pe
        pe_pseudomotion_debias = {"method": "grouped_overlapping_window_wiener_tukey_epi", "applied": False,
                                  "wiener_shrinkage_eta_all6": [float(e) for e in etas],
                                  "lag1_step_autocorr_all6": [float(a) for a in lag1_ac], "removed_pe_rms_mm": 0.0}
        if max(etas) > 0.05:
            d_ref_step = np.zeros(n_frames, dtype=np.float64)
            d_ref_step[1:] = sum(np.abs(np.diff(c)) for c in imm_clean)
            d_ref_local = np.maximum(d_ref_step, np.r_[d_ref_step[1:], 0.0])
            tau_ref = max(0.25, 3.5 * float(np.median(d_ref_step[1:])))
            w_b0 = np.exp(-np.maximum(0.0, (d_ref_local - tau_ref) / max(0.15, tau_ref)) ** 2)

            deltas = np.zeros_like(r_epi)
            for k in range(6):
                if etas[k] <= 0.0:
                    continue
                dr_k = np.zeros(n_frames, dtype=np.float64)
                dr_k[1:] = np.abs(np.diff(r_epi[:, k]))
                dr_local = np.maximum(dr_k, np.r_[dr_k[1:], 0.0])
                scale_k = max(1.4826 * sigs[k], 0.02)
                u_step = np.maximum(0.0, dr_local - 3.5 * scale_k) / (2.5 * scale_k)
                w_step_k = np.exp(-(u_step ** 2))
                w_gate = w_b0 * w_step_k if k == 0 else w_step_k
                deltas[:, k] = etas[k] * w_gate * r_epi[:, k]

            t_clean = (
                t_rig
                - np.outer(deltas[:, 0], p_pe)
                - np.outer(deltas[:, 1], p_sl)
                - np.outer(deltas[:, 3], p_ro)
            )
            if max(etas[2], etas[4], etas[5]) > 0.0:
                rv_clean = (
                    rv_rig
                    - np.outer(deltas[:, 2] / fd_rad, p_ro)
                    - np.outer(deltas[:, 4] / fd_rad, p_pe)
                    - np.outer(deltas[:, 5] / fd_rad, p_sl)
                )
                R_clean = _Rot.from_rotvec(rv_clean).as_matrix()
            else:
                R_clean = R_rig
            composed_clean = []
            for i in range(n_frames):
                H_c = np.eye(4, dtype=np.float64)
                H_c[:3, :3] = R_clean[i]
                H_c[:3, 3] = com + t_clean[i] - R_clean[i] @ com
                composed_clean.append(H_c)
            fd_before_pe = np.asarray(fd, dtype=np.float64).copy()
            composed_rigid = composed_clean
            fd = jenkinson_framewise_displacement(composed_rigid, fd_rad, com)
            pe_pseudomotion_debias = {
                "method": "grouped_overlapping_window_wiener_tukey_epi",
                "pe_vector_lps": [float(x) for x in p_pe],
                "sigma_ref_mm": sig_ref,
                "sigma_pe_mm": sig_pe,
                "wiener_shrinkage_eta": eta_pe,
                "wiener_shrinkage_eta_all6": [float(e) for e in etas],
                "lag1_step_autocorr_all6": [float(a) for a in lag1_ac],
                "removed_pe_rms_mm": float(np.sqrt(np.mean(deltas[:, 0] ** 2))),
                "removed_non_pe_rms_mm": float(np.sqrt(np.mean(np.sum(deltas[:, 1:] ** 2, axis=1)))),
                "window_size": int(w_pe),
                "fd_before_pe_debias_mean_mm": float(np.mean(fd_before_pe[1:])) if n_frames > 1 else 0.0,
                "fd_before_pe_debias_max_mm": float(np.max(fd_before_pe[1:])) if n_frames > 1 else 0.0,
                "fd_before_pe_debias": fd_before_pe,
                "applied": True,
            }

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
    out["cross_group_offset"] = cross_group_offset
    out["group_offset_fd"] = group_offset_fd
    out["pe_pseudomotion_debias"] = pe_pseudomotion_debias
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
