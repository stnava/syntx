"""
dewarp.py — Distortion correction via anatomical reference alignment (canonical contract).
=============================================================================================

Systematic two-stage dewarping procedure originally developed in `antsxslowflow`:
1. Rigidly align the masked anatomical image (T1/T2) into the native reference space
   (DWI b0, functional EPI mean, ASL reference). This produces `anatomy_rigid` (`t1_rigid`),
   which has the exact voxel grid, FOV, and resolution of the distorted acquisition, but
   the undistorted anatomical geometry of the structural scan.
2. Estimate a non-linear susceptibility displacement field between the native reference
   and `anatomy_rigid` using SyNOnly (restricted along the phase-encoding axis).

Because stage 2 operates entirely on the native reference grid (e.g. 1.7 mm or 2 mm),
it avoids the large memory footprint, slow runtimes, and unnecessary interpolation artifacts
of upsampling distorted functional/diffusion data to a 1 mm anatomical grid during optimization.

When dual phase-encoded images are available (e.g., AP and PA), both are dewarped directly
into the common `anatomy_rigid` space, producing unbiased, undistorted, coregistered volumes
that can be combined or averaged.
"""

from __future__ import annotations

from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Sequence

import ants
import numpy as np


class DewarpRuntimeError(RuntimeError):
    """Exception raised for failures in the dewarping and canonical space contract."""
    pass


CanonicalContractRuntimeError = DewarpRuntimeError


class _DictWithAttributes(dict):
    """Dict subclass providing seamless dot-attribute access alongside dict key indexing."""

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class MotionReferenceResult(_DictWithAttributes):
    """Result of building a 3D isotropic motion reference from a 4D acquisition."""

    def __init__(
        self,
        *,
        motion_reference: ants.ANTsImage,
        motion_reference_isotropic: ants.ANTsImage,
        isotropic_spacing: tuple[float, float, float],
        source_spacing: tuple[float, float, float],
    ):
        super().__init__(
            motion_reference=motion_reference,
            motion_reference_isotropic=motion_reference_isotropic,
            isotropic_spacing=isotropic_spacing,
            source_spacing=source_spacing,
        )


class AnatomyRigidResult(_DictWithAttributes):
    """Result of rigid alignment of anatomical data to native reference grid."""

    def __init__(
        self,
        *,
        t1_rigid: ants.ANTsImage,
        anatomy_masked: ants.ANTsImage,
        forward_transformlist: list[str],
        inverse_transformlist: list[str],
        fixed_reference: ants.ANTsImage,
        moving_anatomy: ants.ANTsImage,
    ):
        super().__init__(
            t1_rigid=t1_rigid,
            anatomy_rigid=t1_rigid,
            warpedmovout=t1_rigid,
            t1_masked=anatomy_masked,
            anatomy_masked=anatomy_masked,
            forward_transformlist=forward_transformlist,
            fwdtransforms=forward_transformlist,
            inverse_transformlist=inverse_transformlist,
            invtransforms=inverse_transformlist,
            fixed_reference=fixed_reference,
            moving_anatomy=moving_anatomy,
        )


class CanonicalSynResult(_DictWithAttributes):
    """Result of restricted SyNOnly registration of reference to anatomy_rigid."""

    def __init__(
        self,
        *,
        dewarped: ants.ANTsImage,
        forward_transformlist: list[str],
        inverse_transformlist: list[str],
        fixed_reference: ants.ANTsImage,
        moving_reference: ants.ANTsImage,
    ):
        super().__init__(
            dewarped=dewarped,
            motion_reference_in_t1_rigid=dewarped,
            warpedmovout=dewarped,
            forward_transformlist=forward_transformlist,
            fwdtransforms=forward_transformlist,
            inverse_transformlist=inverse_transformlist,
            invtransforms=inverse_transformlist,
            fixed_reference=fixed_reference,
            moving_reference=moving_reference,
        )


class TransformApplication(_DictWithAttributes):
    """Result of applying a composed transform chain."""

    def __init__(
        self,
        *,
        warped_image: ants.ANTsImage,
        transformlist: list[str],
        whichtoinvert: list[bool] | None,
        source_space: str = "unknown",
        target_space: str = "unknown",
        interpolation: str = "linear",
    ):
        super().__init__(
            warped_image=warped_image,
            transformlist=transformlist,
            whichtoinvert=whichtoinvert,
            source_space=source_space,
            target_space=target_space,
            interpolation=interpolation,
        )


class DewarpResult(_DictWithAttributes):
    """Unified result containing dewarped images, transforms, and quality metrics."""

    def __init__(
        self,
        *,
        dewarped: ants.ANTsImage,
        dewarped_primary: ants.ANTsImage,
        dewarped_secondary: ants.ANTsImage | None,
        anatomy_rigid: ants.ANTsImage,
        rigid_result: AnatomyRigidResult,
        syn_primary: CanonicalSynResult,
        syn_secondary: CanonicalSynResult | None,
        forward_transformlist: list[str],
        inverse_transformlist: list[str],
        transforms_to_anatomy_native: list[str],
        transforms_to_anatomy_native_flags: list[bool],
        transforms_from_anatomy_native: list[str],
        transforms_from_anatomy_native_flags: list[bool],
        metrics: dict[str, float],
        forward_transformlist_secondary: list[str] | None = None,
        forward_transformlist_secondary_flags: list[bool] | None = None,
        transforms_secondary_to_anatomy_native: list[str] | None = None,
        transforms_secondary_to_anatomy_native_flags: list[bool] | None = None,
    ):
        super().__init__(
            dewarped=dewarped,
            warpedmovout=dewarped,
            dewarped_primary=dewarped_primary,
            dewarped_secondary=dewarped_secondary,
            anatomy_rigid=anatomy_rigid,
            t1_rigid=anatomy_rigid,
            rigid_result=rigid_result,
            syn_primary=syn_primary,
            syn_secondary=syn_secondary,
            forward_transformlist=forward_transformlist,
            fwdtransforms=forward_transformlist,
            forward_transformlist_secondary=forward_transformlist_secondary,
            forward_transformlist_secondary_flags=forward_transformlist_secondary_flags,
            inverse_transformlist=inverse_transformlist,
            invtransforms=inverse_transformlist,
            transforms_to_anatomy_native=transforms_to_anatomy_native,
            transforms_to_anatomy_native_flags=transforms_to_anatomy_native_flags,
            transforms_secondary_to_anatomy_native=transforms_secondary_to_anatomy_native,
            transforms_secondary_to_anatomy_native_flags=transforms_secondary_to_anatomy_native_flags,
            transforms_from_anatomy_native=transforms_from_anatomy_native,
            transforms_from_anatomy_native_flags=transforms_from_anatomy_native_flags,
            metrics=metrics,
        )


# ==============================================================================
# Image & Transform List Utilities
# ==============================================================================

def load_ants_image(data: Any, *, reference: Any = None) -> ants.ANTsImage:
    """Resolve an input into an ANTsImage (supports path, ANTsImage, or numpy array)."""
    if isinstance(data, (str, Path)):
        return ants.image_read(str(data))
    if hasattr(data, "numpy"):
        return data
    if isinstance(data, np.ndarray):
        if reference is not None:
            ref = load_ants_image(reference)
            return ref.new_image_like(data)
        return ants.from_numpy(data)
    raise TypeError(f"Cannot resolve {type(data)} to an ANTsImage")


def _compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """Normalized cross-correlation over positive brain voxels -- a thin wrapper over the
    canonical ``syntx.correlation`` (added 2026-10-02; this module was committed concurrently
    by a different session unaware of it, duplicating the same Pearson-correlation math --
    found and fixed in a 2026-10-03 synergy review). Only the positive-voxel masking and the
    degenerate-input 0.0 sentinel are specific to this call site."""
    from .image_compare import correlation

    a = np.asarray(img1, dtype=np.float32).ravel()
    b = np.asarray(img2, dtype=np.float32).ravel()
    mask = (a > 0) & (b > 0)
    if not np.any(mask):
        mask = np.ones_like(a, dtype=bool)
    a_m = a[mask]
    b_m = b[mask]
    if np.std(a_m) < 1e-7 or np.std(b_m) < 1e-7:
        return 0.0
    return correlation(a_m, b_m)


def _mk_outprefix(output_directory: str | Path | None, stem: str) -> str:
    """Generate output prefix string, creating directories if needed."""
    if output_directory is None:
        return str(Path(tempfile.mkdtemp(prefix=f"syntx_{stem}_")) / stem)
    outdir = Path(output_directory)
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir / stem)


def normalize_transformlist(x: Any) -> list[str]:
    """Clean and standardize a transform list, filtering out None and 'NA'."""
    if x is None or x == "NA":
        return []
    if isinstance(x, (list, tuple)) and len(x) == 1 and x[0] == "NA":
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        raw = [str(t) for t in x if t is not None and str(t) != "NA"]
        filtered = [t for t in raw if t.lower() != "identity"]
        if filtered:
            return filtered
        if any(t.lower() == "identity" for t in raw):
            return ["identity"]
        return []
    return [str(x)]


def compose_transformlists(*parts: Iterable[Any]) -> list[str]:
    """Concatenate transform lists in ANTs execution order."""
    out: list[str] = []
    for p in parts:
        out.extend(normalize_transformlist(p))
    return out


def _is_affine_transform(path: str) -> bool:
    """Check if transform file path represents an affine/linear transform."""
    p = str(path).lower()
    return p.endswith(".mat") or p.endswith(".txt") or p.endswith(".h5")


def infer_inverse_warp_path(forward_warp: str | Path) -> str:
    """Derive standard ANTs inverse warp filename from a forward warp path."""
    p = Path(str(forward_warp))
    name = p.name
    if "InverseWarp" in name:
        inv = p
    else:
        m = re.match(r"^(?P<prefix>\d+)?Warp\.nii\.gz$", name)
        if m:
            pref = m.group("prefix") or ""
            inv_name = f"{pref}InverseWarp.nii.gz"
        elif name.endswith("Warp.nii.gz"):
            inv_name = name.replace("Warp.nii.gz", "InverseWarp.nii.gz")
        else:
            inv_name = name.replace("Warp", "InverseWarp", 1) if "Warp" in name else name + ".InverseWarp.nii.gz"
        inv = p.with_name(inv_name)
    if not inv.exists():
        raise FileNotFoundError(f"Inverse warp not found for {p}: expected {inv}")
    return str(inv)


def invert_transformlist(transformlist: Iterable[Any]) -> tuple[list[str], list[bool]]:
    """Invert a list of ANTs transforms, reversing order and generating whichtoinvert flags."""
    fwd = normalize_transformlist(list(transformlist))
    inv_list: list[str] = []
    inv_flags: list[bool] = []
    for t in reversed(fwd):
        t_str = str(t)
        if t_str.lower() == "identity":
            inv_list.append("identity")
            inv_flags.append(False)
        elif _is_affine_transform(t_str):
            inv_list.append(t_str)
            inv_flags.append(True)
        else:
            inv_list.append(infer_inverse_warp_path(t_str))
            inv_flags.append(False)
    return inv_list, inv_flags


def apply_transform_chain(
    *,
    moving: Any,
    fixed: Any,
    transformlist: list[str] | tuple[str, ...],
    whichtoinvert: list[bool] | None = None,
    interpolator: str = "linear",
    source_space: str = "unknown",
    target_space: str = "unknown",
) -> TransformApplication:
    """Apply an ordered transform chain to resample moving image into fixed image space."""
    tl = normalize_transformlist(transformlist)
    if not tl:
        raise DewarpRuntimeError("apply_transform_chain requires a non-empty transformlist")
    flags = None if whichtoinvert is None else list(whichtoinvert)
    warped = ants.apply_transforms(
        fixed=load_ants_image(fixed),
        moving=load_ants_image(moving),
        transformlist=tl,
        whichtoinvert=flags,
        interpolator=interpolator,
    )
    return TransformApplication(
        warped_image=warped,
        transformlist=tl,
        whichtoinvert=flags,
        source_space=source_space,
        target_space=target_space,
        interpolation=interpolator,
    )


# ==============================================================================
# Motion Reference Construction
# ==============================================================================

def build_motion_reference(*, magnitude_4d: Any) -> MotionReferenceResult:
    """Compute 3D temporal mean from 4D series and resample to isotropic resolution."""
    mag = load_ants_image(magnitude_4d)
    if len(mag.shape) != 4:
        raise DewarpRuntimeError("build_motion_reference requires a 4D image")
    motion_reference = ants.get_average_of_timeseries(mag)
    spacing = tuple(float(v) for v in motion_reference.spacing[:3])
    if len(spacing) != 3:
        raise DewarpRuntimeError("Motion reference must have 3 spatial spacing values")
    iso = float(min(spacing))
    motion_reference_isotropic = ants.resample_image(
        motion_reference,
        resample_params=(iso, iso, iso),
        use_voxels=False,
        interp_type=0,
    )
    return MotionReferenceResult(
        motion_reference=motion_reference,
        motion_reference_isotropic=motion_reference_isotropic,
        isotropic_spacing=(iso, iso, iso),
        source_spacing=spacing,
    )


# ==============================================================================
# Stage 1: Rigid Alignment of Anatomy to Native Reference Grid
# ==============================================================================

def rigid_align_anatomy_to_reference(
    *,
    anatomy: Any = None,
    reference: Any = None,
    anatomy_mask: Any = None,
    output_directory: str | Path | None = None,
    initial_transform: str | None = None,
    verbose: bool = False,
    # Aliases for direct compatibility with antsxslowflow canonical_contract:
    anatomy_t1: Any = None,
    anatomy_brain_mask: Any = None,
    motion_reference_isotropic: Any = None,
    **kwargs: Any,
) -> AnatomyRigidResult:
    """Rigidly align masked anatomy into reference space.

    Resamples anatomy onto the reference voxel grid, orientation, and FOV, producing
    `anatomy_rigid` (`t1_rigid`).

    Parameters
    ----------
    anatomy : ANTsImage or Path
        High-resolution anatomical image (e.g. T1w or T2w).
    reference : ANTsImage or Path
        Distorted acquisition reference (e.g. DWI b0, mean DWI, EPI reference).
    anatomy_mask : ANTsImage or Path, optional
        Brain mask defined in anatomy space.
    output_directory : str or Path, optional
        Output folder for generated transform files.
    initial_transform : str, optional
        Initial transform ('identity', path to .mat, or None for automatic robust_affine).
    verbose : bool, default False
    """
    anat_in = anatomy if anatomy is not None else anatomy_t1
    ref_in = reference if reference is not None else motion_reference_isotropic
    mask_in = anatomy_mask if anatomy_mask is not None else anatomy_brain_mask

    if anat_in is None or ref_in is None:
        raise ValueError("Both anatomy and reference images must be provided")

    t1 = load_ants_image(anat_in)
    fixed = load_ants_image(ref_in)

    if mask_in is not None:
        mask = load_ants_image(mask_in, reference=t1)
        t1_masked = t1 * mask
    else:
        t1_masked = t1

    outprefix = _mk_outprefix(output_directory, "anatomy_rigid_")
    fwd: list[str] = []
    inv: list[str] = []

    # No silent fallback to plain ants.registration on a syntx.syn failure: this ecosystem's
    # own benchmarks show syntx.syn reliably EXCEEDS plain-ANTs accuracy (see
    # docs/antsx_implementation_standards.md), so a silent degrade-on-exception would
    # mask a real regression with no visible signal unless the caller happened to pass
    # verbose=True. Let a syntx.syn failure raise and surface directly instead.
    from .syn import registration as syn_reg
    reg = syn_reg(
        fixed=fixed,
        moving=t1_masked,
        type_of_transform="Rigid",
        initial_transform=initial_transform,
        outprefix=outprefix,
        verbose=verbose,
        **kwargs,
    )
    fwd = [str(p) for p in reg.get("fwdtransforms", [])]
    inv = [str(p) for p in reg.get("invtransforms", [])]

    if not fwd:
        raise DewarpRuntimeError("Rigid anatomy-to-reference registration did not produce transforms")

    t1_rigid = ants.apply_transforms(
        fixed=fixed,
        moving=t1_masked,
        transformlist=fwd,
        interpolator="linear",
    )

    return AnatomyRigidResult(
        t1_rigid=t1_rigid,
        anatomy_masked=t1_masked,
        forward_transformlist=fwd,
        inverse_transformlist=inv,
        fixed_reference=fixed,
        moving_anatomy=t1,
    )


# Alias for 100% antsxslowflow canonical_contract compatibility
rigid_align_masked_anatomy_to_motion_reference = rigid_align_anatomy_to_reference


# ==============================================================================
# Stage 2: SyNOnly Restricted Reference Dewarping to Anatomy Rigid
# ==============================================================================

def syn_only_reference_to_anatomy_rigid(
    *,
    reference: Any = None,
    anatomy_rigid: Any = None,
    output_directory: str | Path | None = None,
    restrict_transformation: tuple[int, ...] | str = (0, 1, 0),
    syn_metric: str = "cc2",
    reg_iterations: list[int] | None = None,
    verbose: bool = False,
    # Aliases for direct compatibility with antsxslowflow canonical_contract:
    motion_reference_isotropic: Any = None,
    t1_rigid: Any = None,
    **kwargs: Any,
) -> CanonicalSynResult:
    """Register moving reference to fixed anatomy_rigid using SyNOnly with PE restriction.

    Parameters
    ----------
    reference : ANTsImage or Path
        Moving distorted reference image on the native grid.
    anatomy_rigid : ANTsImage or Path
        Fixed undistorted anatomical image resampled onto the reference grid.
    output_directory : str or Path, optional
    restrict_transformation : tuple or str, default (0, 1, 0)
        Deformation restriction weights. Can be a 3-tuple (e.g. (0, 1, 0)), an anatomical
        axis ('AP', 'PA', 'LR', etc.), or a BIDS sidecar JSON path.
    syn_metric : str, default 'cc2'
    reg_iterations : list of int, optional
    verbose : bool, default False
    """
    ref_in = reference if reference is not None else motion_reference_isotropic
    anat_in = anatomy_rigid if anatomy_rigid is not None else t1_rigid

    if ref_in is None or anat_in is None:
        raise ValueError("Both reference and anatomy_rigid must be provided")

    moving = load_ants_image(ref_in)
    fixed = load_ants_image(anat_in)

    # Resolve restrict_transformation if given as string
    if isinstance(restrict_transformation, str):
        from .spatial import restriction_from_orientation
        if restrict_transformation.endswith(".json"):
            restrict_tuple = restriction_from_orientation(moving, json_sidecar=restrict_transformation)
        elif restrict_transformation.lower() in ("i", "j", "k", "i-", "j-", "k-"):
            restrict_tuple = restriction_from_orientation(moving, bids_phase_encoding_direction=restrict_transformation)
        else:
            restrict_tuple = restriction_from_orientation(moving, anatomical_axis=restrict_transformation)
    else:
        restrict_tuple = tuple(restrict_transformation)

    outprefix = _mk_outprefix(output_directory, "syn_dewarp_")
    fwd: list[str] = []
    inv: list[str] = []

    # No silent fallback to plain ants.registration on a syntx.syn failure -- same reasoning
    # as rigid_align_anatomy_to_reference above: let a real syntx.syn failure raise and
    # surface directly, never silently degrade to a different (validated-worse) registration
    # method with no visible signal.
    from .syn import registration as syn_reg
    syn_res = syn_reg(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyNOnly",
        initial_transform="identity",
        restrict_transformation=restrict_tuple,
        syn_metric=syn_metric,
        reg_iterations=reg_iterations or [100, 50, 20],
        outprefix=outprefix,
        verbose=verbose,
        **kwargs,
    )
    fwd = [str(p) for p in syn_res.get("fwdtransforms", [])]
    inv = [str(p) for p in syn_res.get("invtransforms", [])]

    if not fwd:
        raise DewarpRuntimeError("SyNOnly reference-to-anatomy-rigid registration did not produce transforms")

    dewarped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=fwd,
        interpolator="linear",
    )

    return CanonicalSynResult(
        dewarped=dewarped,
        forward_transformlist=fwd,
        inverse_transformlist=inv,
        fixed_reference=fixed,
        moving_reference=moving,
    )


# Alias for 100% antsxslowflow canonical_contract compatibility
syn_only_motion_reference_to_t1_rigid = syn_only_reference_to_anatomy_rigid


# ==============================================================================
# Transform Composition Helpers
# ==============================================================================

def compose_native_frame_to_anatomy_rigid_transform(
    *,
    syn_forward_transformlist: list[str] | tuple[str, ...],
    motion_transformlist: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[str], list[bool]]:
    """Compose native frame motion correction with SyN dewarping onto anatomy_rigid."""
    if motion_transformlist is not None:
        chain = compose_transformlists(syn_forward_transformlist, motion_transformlist)
    else:
        chain = normalize_transformlist(syn_forward_transformlist)
    return chain, [False] * len(chain)


compose_native_frame_to_t1_rigid_transform = compose_native_frame_to_anatomy_rigid_transform


def compose_anatomy_to_t1_rigid_transform(
    *,
    anatomy_rigid_transformlist: list[str] | tuple[str, ...],
) -> tuple[list[str], list[bool]]:
    """Anatomy native -> anatomy_rigid is simply the forward rigid transform."""
    chain = normalize_transformlist(anatomy_rigid_transformlist)
    return chain, [False] * len(chain)


def compose_anatomy_to_native_frame_transform(
    *,
    anatomy_rigid_transformlist: list[str] | tuple[str, ...],
    syn_inverse_transformlist: list[str] | tuple[str, ...],
    motion_transformlist: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[str], list[bool]]:
    """Compose native anatomy -> native frame (applying rigid, syn_inv, motion_inv)."""
    syn_inv = normalize_transformlist(syn_inverse_transformlist)
    rigid = normalize_transformlist(anatomy_rigid_transformlist)
    if motion_transformlist is not None:
        motion_inv, motion_inv_flags = invert_transformlist(motion_transformlist)
        chain = compose_transformlists(motion_inv, syn_inv, rigid)
        flags = list(motion_inv_flags) + ([False] * len(syn_inv)) + ([False] * len(rigid))
    else:
        chain = compose_transformlists(syn_inv, rigid)
        flags = ([False] * len(syn_inv)) + ([False] * len(rigid))
    return chain, flags


def compose_native_to_anatomy_transform(
    *,
    syn_forward_transformlist: list[str] | tuple[str, ...],
    anatomy_rigid_transformlist: list[str] | tuple[str, ...],
    motion_transformlist: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[str], list[bool]]:
    """Compose native reference/frame -> native anatomy (syn_fwd, then rigid_inv)."""
    rigid_inv, rigid_inv_flags = invert_transformlist(anatomy_rigid_transformlist)
    syn_fwd = normalize_transformlist(syn_forward_transformlist)
    if motion_transformlist is not None:
        motion_fwd = normalize_transformlist(motion_transformlist)
        chain = compose_transformlists(rigid_inv, syn_fwd, motion_fwd)
        flags = list(rigid_inv_flags) + ([False] * len(syn_fwd)) + ([False] * len(motion_fwd))
    else:
        chain = compose_transformlists(rigid_inv, syn_fwd)
        flags = list(rigid_inv_flags) + ([False] * len(syn_fwd))
    return chain, flags


# ==============================================================================
# High-Level Unified Entry Point: dewarp_to_anatomical
# ==============================================================================

def _create_negated_warp(warp_path: str | Path, outprefix: str | None = None) -> str:
    """Create negated displacement field representing reverse midpoint transform."""
    w = load_ants_image(warp_path)
    neg_w = w * -1.0
    if outprefix is not None:
        out_file = f"{outprefix}_inv_midpoint.nii.gz"
    else:
        out_file = tempfile.NamedTemporaryFile(suffix="_inv_midpoint.nii.gz", delete=False).name
    ants.image_write(neg_w, out_file)
    return out_file


def dewarp_to_anatomical(
    reference: Any,
    anatomy: Any,
    *,
    anatomy_mask: Any = None,
    restrict_transformation: tuple[int, ...] | str = (0, 1, 0),
    secondary_reference: Any = None,
    strategy: str = "hybrid",
    output_directory: str | Path | None = None,
    syn_metric: str = "cc2",
    reg_iterations: list[int] | None = None,
    initial_transform: str | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> DewarpResult:
    """Unified two-stage anatomical dewarping for single or dual phase-encoded acquisitions.

    Strategies when dual phase-encoded data is provided (`secondary_reference` is not None):
    1. ``strategy="hybrid"`` (default & recommended):
       - Step 1: Symmetrically registers primary and secondary PE images via contrast-matched
         reverse-PE SyN (antisymmetric=True), creating an unbiased, high-SNR midspace template.
       - Step 2: Rigidly aligns `anatomy` to this midspace template (`anatomy_rigid`).
       - Step 3: Deforms the midspace template into `anatomy_rigid` with restricted SyNOnly.
       - Step 4: Composes all transforms into unified end-to-end chains, guaranteeing that
         raw diffusion/EPI data is resampled into the final anatomically-anchored space in a
         **single interpolation step**.
    2. ``strategy="independent"``:
       - Rigidly aligns `anatomy` to primary reference.
       - Deforms primary reference to `anatomy_rigid` and secondary reference to `anatomy_rigid`
         independently, then averages.

    When `secondary_reference` is None, single-PE dewarping to `anatomy_rigid` is performed.

    Parameters
    ----------
    reference : ANTsImage or Path
        Primary distorted reference (e.g. AP b0).
    anatomy : ANTsImage or Path
        High-resolution anatomical image (e.g. T1w).
    anatomy_mask : ANTsImage or Path, optional
        Brain mask for anatomy in its native space.
    restrict_transformation : tuple or str, default (0, 1, 0)
        Phase-encoding axis restriction weights.
    secondary_reference : ANTsImage or Path, optional
        Opposite phase-encoding reference (e.g. PA b0).
    strategy : {'hybrid', 'independent'}, default 'hybrid'
        De-aliasing / dewarping strategy when secondary_reference is provided.
    output_directory : str or Path, optional
    syn_metric : str, default 'cc2'
    reg_iterations : list of int, optional
    initial_transform : str, optional
    verbose : bool, default False
    """
    ref_img = load_ants_image(reference)
    anat_img = load_ants_image(anatomy)

    # Resolve restrict_transformation if given as string
    if isinstance(restrict_transformation, str):
        from .spatial import restriction_from_orientation
        if restrict_transformation.endswith(".json"):
            restrict_tuple = restriction_from_orientation(ref_img, json_sidecar=restrict_transformation)
        elif restrict_transformation.lower() in ("i", "j", "k", "i-", "j-", "k-"):
            restrict_tuple = restriction_from_orientation(ref_img, bids_phase_encoding_direction=restrict_transformation)
        else:
            restrict_tuple = restriction_from_orientation(ref_img, anatomical_axis=restrict_transformation)
    else:
        restrict_tuple = tuple(restrict_transformation)

    # =========================================================================
    # Strategy A: Hybrid Dual-PE + Deformation to Rigid Anatomy
    # =========================================================================
    if secondary_reference is not None and str(strategy).lower() == "hybrid":
        sec_img = load_ants_image(secondary_reference)
        from .syn import registration as syn_reg

        # 1. Reverse-PE symmetric midpoint registration between contrast-matched b0s
        outprefix_mid = _mk_outprefix(output_directory, "reverse_pe_midspace_")
        reg_mid = syn_reg(
            fixed=ref_img,
            moving=sec_img,
            type_of_transform="SyN",
            dof="rigid",
            antisymmetric=True,
            restrict_transformation=restrict_tuple,
            syn_metric=syn_metric,
            reg_iterations=reg_iterations or [100, 50, 20],
            outprefix=outprefix_mid,
            verbose=verbose,
            **kwargs,
        )

        fwd_mid = reg_mid.get("fwd_midpoint_warp")
        inv_mid = reg_mid.get("inv_midpoint_warp")
        lin_fwd = [str(p) for p in reg_mid.get("fwdtransforms", []) if _is_affine_transform(p)]

        if fwd_mid and inv_mid:
            # Warp both into symmetric midpoint template
            ref_in_mid = ants.apply_transforms(fixed=ref_img, moving=ref_img, transformlist=[fwd_mid], interpolator="linear")
            sec_in_mid = ants.apply_transforms(fixed=ref_img, moving=sec_img, transformlist=[inv_mid] + lin_fwd, interpolator="linear")
            midspace_img = (ref_in_mid + sec_in_mid) * 0.5
        else:
            midspace_img = reg_mid.get("warpedmovout", ref_img)
            fwd_mid = reg_mid["fwdtransforms"][0]
            inv_mid = reg_mid["invtransforms"][0]

        # 2. Rigidly align anatomy to the distortion-free midspace template
        rigid_res = rigid_align_anatomy_to_reference(
            anatomy=anat_img,
            reference=midspace_img,
            anatomy_mask=anatomy_mask,
            output_directory=output_directory,
            initial_transform=initial_transform,
            verbose=verbose,
        )
        anat_rigid = rigid_res.anatomy_rigid

        # 3. Deform midspace template into anatomy_rigid (fine structural alignment)
        syn_mid_to_anat = syn_only_reference_to_anatomy_rigid(
            reference=midspace_img,
            anatomy_rigid=anat_rigid,
            output_directory=output_directory,
            restrict_transformation=restrict_tuple,
            syn_metric=syn_metric,
            reg_iterations=reg_iterations or [50, 30, 10],
            verbose=verbose,
            **kwargs,
        )
        warp_mid_to_anat = [str(p) for p in syn_mid_to_anat.forward_transformlist if str(p).endswith(".nii.gz")][0]
        warp_anat_to_mid = [str(p) for p in syn_mid_to_anat.inverse_transformlist if str(p).endswith(".nii.gz")][0]

        # 4. Compose unified single-interpolation chains
        chain_prim_to_rigid = [warp_mid_to_anat, str(fwd_mid)]
        flags_prim_to_rigid = [False, False]

        chain_sec_to_rigid = [warp_mid_to_anat, str(inv_mid)] + lin_fwd
        flags_sec_to_rigid = [False, False] + ([False] * len(lin_fwd))

        # Primary / Secondary to native anatomy
        chain_prim_to_anat = [rigid_res.forward_transformlist[0], warp_mid_to_anat, str(fwd_mid)]
        flags_prim_to_anat = [True, False, False]

        chain_sec_to_anat = [rigid_res.forward_transformlist[0], warp_mid_to_anat, str(inv_mid)] + lin_fwd
        flags_sec_to_anat = [True, False, False] + ([False] * len(lin_fwd))

        # Anatomy native to primary reference
        neg_fwd_mid = _create_negated_warp(fwd_mid, outprefix=outprefix_mid)
        chain_anat_to_prim = [neg_fwd_mid, warp_anat_to_mid, rigid_res.forward_transformlist[0]]
        flags_anat_to_prim = [False, False, False]

        # 5. Apply single-interpolation transform
        dewarped_primary = ants.apply_transforms(
            fixed=anat_rigid, moving=ref_img, transformlist=chain_prim_to_rigid, whichtoinvert=flags_prim_to_rigid
        )
        dewarped_secondary = ants.apply_transforms(
            fixed=anat_rigid, moving=sec_img, transformlist=chain_sec_to_rigid, whichtoinvert=flags_sec_to_rigid
        )
        dewarped = (dewarped_primary + dewarped_secondary) * 0.5

        metrics = {
            "raw_primary_secondary_ncc": _compute_ncc(ref_img.numpy(), sec_img.numpy()),
            "dewarped_primary_secondary_ncc": _compute_ncc(dewarped_primary.numpy(), dewarped_secondary.numpy()),
            "dewarped_to_anatomy_rigid_ncc": _compute_ncc(dewarped.numpy(), anat_rigid.numpy()),
        }

        return DewarpResult(
            dewarped=dewarped,
            dewarped_primary=dewarped_primary,
            dewarped_secondary=dewarped_secondary,
            anatomy_rigid=anat_rigid,
            rigid_result=rigid_res,
            syn_primary=syn_mid_to_anat,
            syn_secondary=None,
            forward_transformlist=chain_prim_to_rigid,
            forward_transformlist_secondary=chain_sec_to_rigid,
            forward_transformlist_secondary_flags=flags_sec_to_rigid,
            inverse_transformlist=[neg_fwd_mid, warp_anat_to_mid],
            transforms_to_anatomy_native=chain_prim_to_anat,
            transforms_to_anatomy_native_flags=flags_prim_to_anat,
            transforms_secondary_to_anatomy_native=chain_sec_to_anat,
            transforms_secondary_to_anatomy_native_flags=flags_sec_to_anat,
            transforms_from_anatomy_native=chain_anat_to_prim,
            transforms_from_anatomy_native_flags=flags_anat_to_prim,
            metrics=metrics,
        )

    # =========================================================================
    # Strategy B: Independent Dewarping to Anatomy Rigid
    # =========================================================================
    # 1. Rigid alignment: anatomy -> reference grid
    rigid_res = rigid_align_anatomy_to_reference(
        anatomy=anat_img,
        reference=ref_img,
        anatomy_mask=anatomy_mask,
        output_directory=output_directory,
        initial_transform=initial_transform,
        verbose=verbose,
    )
    anat_rigid = rigid_res.anatomy_rigid

    # 2. SyNOnly dewarping of primary reference to anatomy_rigid
    prim_syn = syn_only_reference_to_anatomy_rigid(
        reference=ref_img,
        anatomy_rigid=anat_rigid,
        output_directory=output_directory,
        restrict_transformation=restrict_tuple,
        syn_metric=syn_metric,
        reg_iterations=reg_iterations,
        verbose=verbose,
        **kwargs,
    )
    dewarped_primary = prim_syn.dewarped

    # 3. Handle secondary reference if provided
    dewarped_secondary = None
    sec_syn = None
    metrics: dict[str, float] = {}

    if secondary_reference is not None:
        sec_img = load_ants_image(secondary_reference)
        sec_outdir = str(Path(output_directory) / "secondary") if output_directory else None
        sec_syn = syn_only_reference_to_anatomy_rigid(
            reference=sec_img,
            anatomy_rigid=anat_rigid,
            output_directory=sec_outdir,
            restrict_transformation=restrict_tuple,
            syn_metric=syn_metric,
            reg_iterations=reg_iterations,
            verbose=verbose,
            **kwargs,
        )
        dewarped_secondary = sec_syn.dewarped
        dewarped = (dewarped_primary + dewarped_secondary) * 0.5
        metrics["raw_primary_secondary_ncc"] = _compute_ncc(ref_img.numpy(), sec_img.numpy())
        metrics["dewarped_primary_secondary_ncc"] = _compute_ncc(dewarped_primary.numpy(), dewarped_secondary.numpy())
    else:
        dewarped = dewarped_primary

    metrics["dewarped_to_anatomy_rigid_ncc"] = _compute_ncc(dewarped.numpy(), anat_rigid.numpy())

    # 4. Compose transform chains to and from native anatomy
    tl_to_anat, flags_to_anat = compose_native_to_anatomy_transform(
        syn_forward_transformlist=prim_syn.forward_transformlist,
        anatomy_rigid_transformlist=rigid_res.forward_transformlist,
    )
    tl_from_anat, flags_from_anat = compose_anatomy_to_native_frame_transform(
        anatomy_rigid_transformlist=rigid_res.forward_transformlist,
        syn_inverse_transformlist=prim_syn.inverse_transformlist,
    )

    return DewarpResult(
        dewarped=dewarped,
        dewarped_primary=dewarped_primary,
        dewarped_secondary=dewarped_secondary,
        anatomy_rigid=anat_rigid,
        rigid_result=rigid_res,
        syn_primary=prim_syn,
        syn_secondary=sec_syn,
        forward_transformlist=prim_syn.forward_transformlist,
        inverse_transformlist=prim_syn.inverse_transformlist,
        transforms_to_anatomy_native=tl_to_anat,
        transforms_to_anatomy_native_flags=flags_to_anat,
        transforms_from_anatomy_native=tl_from_anat,
        transforms_from_anatomy_native_flags=flags_from_anat,
        metrics=metrics,
    )


__all__ = [
    "DewarpRuntimeError",
    "CanonicalContractRuntimeError",
    "MotionReferenceResult",
    "AnatomyRigidResult",
    "CanonicalSynResult",
    "TransformApplication",
    "DewarpResult",
    "load_ants_image",
    "normalize_transformlist",
    "compose_transformlists",
    "invert_transformlist",
    "infer_inverse_warp_path",
    "apply_transform_chain",
    "build_motion_reference",
    "rigid_align_anatomy_to_reference",
    "rigid_align_masked_anatomy_to_motion_reference",
    "syn_only_reference_to_anatomy_rigid",
    "syn_only_motion_reference_to_t1_rigid",
    "compose_native_frame_to_anatomy_rigid_transform",
    "compose_native_frame_to_t1_rigid_transform",
    "compose_anatomy_to_t1_rigid_transform",
    "compose_anatomy_to_native_frame_transform",
    "compose_native_to_anatomy_transform",
    "dewarp_to_anatomical",
]
