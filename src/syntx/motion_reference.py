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
   nearly 3x higher than when b0 volumes are registered among themselves instead).
   ``build_grouped_reference``/``motion_correct_grouped`` below address this directly.

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


def motion_correct_grouped(
    image: ants.ANTsImage,
    group_is_reference: np.ndarray,
    n_low_motion: int = 10,
    backend: str = "pytorch_batched_adaptive",
    verbose: bool = False,
    **kwargs,
):
    """Motion-correct a contrast-heterogeneous series (e.g. DWI b0 + diffusion-weighted
    volumes) against a reference built ONLY from the "anchor" group, instead of a single
    blended whole-series mean.

    Real finding this addresses (2026-10-04, real SOCOM DWI data): registering every
    volume -- 7 b0 + 92 diffusion-weighted, 2 b-value shells -- to ``reference="mean"``
    (the blended mean of all 99) gave the b0 volumes (the numerical minority) the WORST
    apparent motion (mean FD 5.89mm) of any group, nearly 3x the FD a b0-only reference
    gives (1.43mm, registering the 7 b0 volumes only among themselves). The standard,
    literature-consistent fix (matching FSL eddy / MRtrix's own convention) is to build
    one consistent-contrast anchor (b0) reference and register EVERYTHING -- including
    the non-anchor group -- to it, rather than deriving the reference from a blend of
    both contrasts.

    Parameters
    ----------
    image : ants.ANTsImage
        4D time series (time is the last axis), e.g. the raw DWI 4D volume.
    group_is_reference : np.ndarray of bool, shape (n_frames,)
        True for frames belonging to the "anchor"/reference group (e.g. ``bvals <= 50``
        for DWI's b0 volumes). The reference is built ONLY from these frames (via
        ``build_low_motion_reference``); every frame (both groups) is then motion-
        corrected against that one reference.
    n_low_motion : int, default 10
        Forwarded to ``build_low_motion_reference`` for the anchor-group reference build.
    backend : str, default 'pytorch_batched_adaptive'
        Forwarded to ``syntx.motion_correction`` for the full-series correction pass.
    **kwargs
        Forwarded to ``syntx.motion_correction`` (e.g. ``fd_radius``).

    Returns
    -------
    MotionCorrectionResult
        Same contract as ``syntx.motion_correction``, computed against the anchor-group
        reference.
    """
    from .motion import motion_correction

    group_is_reference = np.asarray(group_is_reference, dtype=bool)
    n_frames = image.shape[-1]
    if group_is_reference.shape[0] != n_frames:
        raise ValueError(
            f"group_is_reference must have one entry per frame ({n_frames}), "
            f"got shape {group_is_reference.shape}."
        )
    if not group_is_reference.any():
        raise ValueError("group_is_reference has no True entries -- no anchor-group frames to build a reference from.")

    ref_group_idx = np.where(group_is_reference)[0]
    ref_group_imgs = [ants.slice_image(image, axis=3, idx=int(i)) for i in ref_group_idx]

    if verbose:
        print(f"[syntx.motion_reference] motion_correct_grouped: building reference from "
              f"{len(ref_group_imgs)}/{n_frames} anchor-group frames.")

    reference = build_low_motion_reference(ref_group_imgs, n_low_motion=n_low_motion, verbose=verbose)

    return motion_correction(image, reference=reference, backend=backend, verbose=verbose, **kwargs)
