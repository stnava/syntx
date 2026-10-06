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
    corrected = [
        ants.apply_transforms(fixed=selected_mean, moving=im, transformlist=f, interpolator="linear")
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


class GroupedMotionCorrectionResult(dict):
    """Dict subclass (attribute access) returned by ``motion_correct_grouped``.

    Keys: ``motion_corrected`` (4D, all frames resampled into the anchor group's own
    mean space), ``fd`` (per-frame, each computed against ITS OWN group's mean -- the
    real head-motion estimate, uncontaminated by the one-time cross-contrast offset),
    ``group_a_result`` / ``group_b_result`` (the two per-group ``MotionCorrectionResult``
    objects, in case the per-group detail is wanted), ``group_a_mean`` / ``group_b_mean``
    (the two same-contrast reference images actually used), ``cross_registration``
    (the ``robust_affine`` result aligning ``group_b_mean`` onto ``group_a_mean``).
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
    group_is_reference: np.ndarray,
    n_low_motion: int = 10,
    backend: str = "pytorch_batched_adaptive",
    verbose: bool = False,
    **kwargs,
) -> GroupedMotionCorrectionResult:
    """Motion-correct a contrast-heterogeneous series (e.g. DWI b0 + diffusion-weighted
    volumes) using the SAME-CONTRAST-only-registration design: (1) build a sharp mean
    for EACH group separately, (2) register the two means to each other ONCE, carefully
    (this is the only cross-contrast registration in the whole pipeline), (3) motion-
    correct each group's own frames to its OWN mean (same-contrast, small-motion --
    exactly the easy problem the fast adaptive backend is validated for), (4) resample
    the non-anchor group's corrected frames through the one mean-to-mean transform so
    everything ends up in a single shared (anchor-group) physical space.

    This replaced an earlier, WRONG design (registering every frame -- both groups --
    directly to one shared anchor-only reference) after the user caught that it doesn't
    match what was actually asked for, and verified empirically to perform worse: that
    design makes every non-anchor frame pay for its own cross-contrast registration
    (narrow-seed search, tuned for same-contrast motion, actively hurts there -- DWI b0
    FD rose to 7.28mm, worse than even the old blended-mean method's 4.66mm). This
    design does the hard cross-contrast alignment exactly ONCE (so it can afford to be
    careful/robust -- see ``robust_affine``) and keeps every per-frame registration
    same-contrast, where the fast adaptive backend is actually validated to work.

    Parameters
    ----------
    image : ants.ANTsImage
        4D time series (time is the last axis), e.g. the raw DWI 4D volume.
    group_is_reference : np.ndarray of bool, shape (n_frames,)
        True for "group A" (the anchor, e.g. ``bvals <= 50`` for DWI's b0 volumes) --
        the final shared space is group A's own mean's physical space. False entries
        are "group B" (e.g. the diffusion-weighted volumes).
    n_low_motion : int, default 10
        Forwarded to ``build_low_motion_reference`` for each group's own mean build.
    backend : str, default 'pytorch_batched_adaptive'
        Forwarded to ``syntx.motion_correction`` for each group's SAME-CONTRAST
        correction pass (steps 3 above) -- appropriate here since both passes really are
        same-contrast, small-motion problems.
    **kwargs
        Forwarded to ``syntx.motion_correction`` (e.g. ``fd_radius``).

    Returns
    -------
    GroupedMotionCorrectionResult
    """
    from .motion import motion_correction
    from .robust_affine import robust_affine

    group_is_reference = np.asarray(group_is_reference, dtype=bool)
    n_frames = image.shape[-1]
    if group_is_reference.shape[0] != n_frames:
        raise ValueError(
            f"group_is_reference must have one entry per frame ({n_frames}), "
            f"got shape {group_is_reference.shape}."
        )
    if not group_is_reference.any():
        raise ValueError("group_is_reference has no True entries -- no group-A frames.")
    if group_is_reference.all():
        raise ValueError("group_is_reference is all True -- no group-B frames to align to group A.")

    idx_a = np.where(group_is_reference)[0]
    idx_b = np.where(~group_is_reference)[0]

    # Build each group's own 4D container ONCE and re-slice every per-frame image FROM
    # it (rather than from independently-constructed lists) -- guarantees every frame
    # and every mean/reference derived from it shares EXACTLY the same grid. Floating-
    # point origin/direction drift across independently-built ants images (e.g. one
    # list_to_ndimage call per use) is real and silently rejected by the batched
    # passes' own grid-consistency check; this bit this exact function during testing.
    ref3d = ants.slice_image(image, axis=3, idx=0)
    tr = float(image.spacing[3]) if image.dimension == 4 else 1.0
    img4d_a = ants.list_to_ndimage(
        _make_4d_template_from_3d(ref3d, len(idx_a), tr=tr),
        [ants.slice_image(image, axis=3, idx=int(i)) for i in idx_a],
    )
    img4d_b = ants.list_to_ndimage(
        _make_4d_template_from_3d(ref3d, len(idx_b), tr=tr),
        [ants.slice_image(image, axis=3, idx=int(i)) for i in idx_b],
    )
    imgs_a = [ants.slice_image(img4d_a, axis=3, idx=i) for i in range(len(idx_a))]
    imgs_b = [ants.slice_image(img4d_b, axis=3, idx=i) for i in range(len(idx_b))]

    if verbose:
        print(f"[syntx.motion_reference] motion_correct_grouped: group A (anchor) "
              f"{len(imgs_a)}/{n_frames} frames, group B {len(imgs_b)}/{n_frames} frames.")

    # 1. Each group's own sharp, same-contrast mean.
    mean_a = build_low_motion_reference(imgs_a, n_low_motion=n_low_motion, verbose=verbose)
    mean_b = build_low_motion_reference(imgs_b, n_low_motion=n_low_motion, verbose=verbose)

    # 2. The ONE cross-contrast registration in this whole pipeline -- affordable to do
    # carefully (wide capture range) since it only runs once, not per frame.
    if verbose:
        print("[syntx.motion_reference] careful one-time cross-contrast registration "
              "(group B mean -> group A mean)...")
    cross_reg = robust_affine(fixed=mean_a, moving=mean_b, dof="rigid", mode="auto", verbose=verbose)

    # 3. Same-contrast, small-motion correction within each group, independently.
    result_a = motion_correction(img4d_a, reference=mean_a, backend=backend, verbose=verbose, **kwargs)
    result_b = motion_correction(img4d_b, reference=mean_b, backend=backend, verbose=verbose, **kwargs)

    # 4. Bring group B's corrected frames into group A's physical space via the one
    # mean-to-mean transform; group A's frames are already there.
    corrected_a = [ants.slice_image(result_a.motion_corrected, axis=3, idx=i) for i in range(len(imgs_a))]
    corrected_b_in_a_space = [
        ants.apply_transforms(fixed=mean_a, moving=ants.slice_image(result_b.motion_corrected, axis=3, idx=i),
                               transformlist=cross_reg["fwdtransforms"], interpolator="linear")
        for i in range(len(imgs_b))
    ]

    frames_out: list = [None] * n_frames
    fd_out = np.zeros(n_frames, dtype=np.float64)
    for local_i, global_i in enumerate(idx_a):
        frames_out[global_i] = corrected_a[local_i]
        fd_out[global_i] = result_a.fd[local_i]
    for local_i, global_i in enumerate(idx_b):
        frames_out[global_i] = corrected_b_in_a_space[local_i]
        fd_out[global_i] = result_b.fd[local_i]

    motion_corrected = ants.list_to_ndimage(
        _make_4d_template_from_3d(frames_out[0], n_frames, tr=tr), frames_out
    )

    out = GroupedMotionCorrectionResult()
    out["motion_corrected"] = motion_corrected
    out["fd"] = fd_out
    out["group_a_result"] = result_a
    out["group_b_result"] = result_b
    out["group_a_mean"] = mean_a
    out["group_b_mean"] = mean_b
    out["cross_registration"] = cross_reg
    return out
