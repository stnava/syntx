"""
syntx.motion — Motion Correction, Parameter Extraction, and Framewise Displacement
===================================================================================

High-accuracy, lightweight, multi-dimensional motion correction pipeline for
dynamic medical imaging (fMRI BOLD, dMRI DWI, DCE-MRI, dynamic PET/SPECT).

Features:
---------
- Supports 2D+t (3D) and 3D+t (4D) time-series images.
- Frame extraction along the temporal axis via `ants.slice_image(image, axis=image.dimension-1, idx=t)`.
- Flexible reference frame selection:
  * 'mean': temporal average volume (with optional iterative two-pass refinement).
  * int: specific frame index (e.g., 0).
  * ANTsImage: explicit user-provided reference volume.
- Fast rigid registration ('Rigid', 'QuickRigid', 'BOLDRigid', 'Affine', 'Translation').
- Extracts exact 6-DOF (or 3-DOF in 2D) physical motion parameters:
  * Translations: tx, ty, tz in physical millimetres (mm).
  * Rotations: rx, ry, rz in radians and degrees.
- Exact closed-form Framewise Displacement (FD):
  * Power FD (Power et al., NeuroImage 2012) converted on a 50mm head sphere.
  * Jenkinson FD (Jenkinson et al., NeuroImage 2002) RMS displacement.
- DVARS (Derivative of rms VARiance over voxelS) and temporal variance reduction metrics.
- Return full transformation paths (`fwdtransforms`, `invtransforms`).
- Seamless 4D / 3D time-series re-assembly preserving physical geometry and headers.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import ants
import numpy as np
import pandas as pd


class MotionParameters(np.ndarray):
    """
    Subclass of numpy.ndarray storing per-frame motion parameters.

    Supports dual access patterns:
    - Array slicing: `mp[:, 0]`, `mp[t]`
    - Column name indexing: `mp['tx']`, `mp['rx']`, `mp['rx_deg']`
    - Attribute access: `mp.tx`, `mp.ty`, `mp.tz`, `mp.rx`, `mp.ry`, `mp.rz`
    - Export to pandas DataFrame: `mp.to_dataframe()`
    - Export to dictionary: `mp.to_dict()`
    """

    def __new__(
        cls,
        input_array: np.ndarray,
        columns: Optional[List[str]] = None,
        rad_to_deg_cols: Optional[List[str]] = None,
        spatial_dim: int = 3,
    ) -> MotionParameters:
        obj = np.asarray(input_array, dtype=np.float64).view(cls)
        obj._columns = list(columns) if columns is not None else []
        obj._rad_cols = set(rad_to_deg_cols or [])
        obj._spatial_dim = spatial_dim
        return obj

    def __array_finalize__(self, obj: Optional[MotionParameters]) -> None:
        if obj is None:
            return
        self._columns = getattr(obj, "_columns", [])
        self._rad_cols = getattr(obj, "_rad_cols", set())
        self._spatial_dim = getattr(obj, "_spatial_dim", 3)

    @property
    def columns(self) -> List[str]:
        """Return list of column names."""
        return list(self._columns)

    @property
    def spatial_dimension(self) -> int:
        """Spatial dimension of each frame (2 or 3)."""
        return self._spatial_dim

    @property
    def translations(self) -> np.ndarray:
        """Physical translations in mm of shape (N, dim)."""
        dim = self._spatial_dim
        return np.asarray(self)[:, :dim]

    @property
    def rotations(self) -> np.ndarray:
        """Rotations in radians of shape (N, num_rot)."""
        dim = self._spatial_dim
        return np.asarray(self)[:, dim:]

    @property
    def rotations_deg(self) -> np.ndarray:
        """Rotations in degrees of shape (N, num_rot)."""
        return np.degrees(self.rotations)

    def keys(self) -> List[str]:
        """Return available dictionary keys."""
        deg_keys = [f"{c}_deg" for c in self._columns if c in self._rad_cols]
        return self._columns + deg_keys

    def values(self) -> List[np.ndarray]:
        """Return array values for each column."""
        return [self[k] for k in self.keys()]

    def items(self) -> List[Tuple[str, np.ndarray]]:
        """Return (column_name, array) pairs."""
        return [(k, self[k]) for k in self.keys()]

    def get(self, key: str, default: Any = None) -> Any:
        """Get column by name with optional default."""
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self._columns:
            return True
        if key.endswith("_deg") and key[:-4] in self._rad_cols:
            return True
        aliases = {"theta": "rz", "trans_x": "tx", "trans_y": "ty", "trans_z": "tz",
                   "rot_x": "rx", "rot_y": "ry", "rot_z": "rz"}
        if key in aliases and aliases[key] in self._columns:
            return True
        return False

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            aliases = {
                "trans_x": "tx", "trans_y": "ty", "trans_z": "tz",
                "rot_x": "rx", "rot_y": "ry", "rot_z": "rz",
                "theta": "rz" if "rz" in self._columns else "theta",
            }
            resolved = aliases.get(item, item)
            if resolved in self._columns:
                idx = self._columns.index(resolved)
                return np.asarray(self)[:, idx]
            if resolved.endswith("_deg") and resolved[:-4] in self._rad_cols:
                base_col = resolved[:-4]
                idx = self._columns.index(base_col)
                return np.degrees(np.asarray(self)[:, idx])
            raise KeyError(f"Column '{item}' not found in motion parameters {self.keys()}")
        return super().__getitem__(item)

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def to_dataframe(self, include_degrees: bool = True) -> pd.DataFrame:
        """Export motion parameters as a pandas DataFrame."""
        df = pd.DataFrame(np.asarray(self), columns=self._columns)
        if include_degrees:
            for col in self._rad_cols:
                if col in df.columns:
                    df[f"{col}_deg"] = np.degrees(df[col])
        return df

    def to_dict(self, include_degrees: bool = True) -> Dict[str, np.ndarray]:
        """Export motion parameters as a dictionary of 1D numpy arrays."""
        d = {col: np.asarray(self)[:, i] for i, col in enumerate(self._columns)}
        if include_degrees:
            for col in self._rad_cols:
                d[f"{col}_deg"] = np.degrees(d[col])
        return d


class TransformCollection(list):
    """
    Dual-access container for forward and inverse transformation files.

    Inherits from `list` where `transforms[t]` returns the list of forward transform
    file paths for volume `t`. Also supports dictionary-style access:
    - `transforms['fwdtransforms']` or `transforms['fwd']`: all forward transform paths.
    - `transforms['invtransforms']` or `transforms['inv']`: all inverse transform paths.
    - `transforms.fwdtransforms` / `transforms.invtransforms`.
    """

    def __init__(
        self,
        fwd_transforms: List[List[str]],
        inv_transforms: Optional[List[List[str]]] = None,
    ) -> None:
        super().__init__(fwd_transforms)
        self.fwdtransforms = fwd_transforms
        self.invtransforms = inv_transforms if inv_transforms is not None else []

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            lower = item.lower()
            if lower in ("fwd", "fwdtransforms", "forward", "forward_transforms"):
                return self.fwdtransforms
            if lower in ("inv", "invtransforms", "inverse", "inverse_transforms"):
                return self.invtransforms
            raise KeyError(f"Transform key '{item}' not recognized. Use 'fwdtransforms' or 'invtransforms'.")
        return super().__getitem__(item)

    def keys(self) -> List[str]:
        return ["fwdtransforms", "invtransforms"]

    def values(self) -> List[List[List[str]]]:
        return [self.fwdtransforms, self.invtransforms]

    def items(self) -> List[Tuple[str, List[List[str]]]]:
        return [("fwdtransforms", self.fwdtransforms), ("invtransforms", self.invtransforms)]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class MotionCorrectionResult(dict):
    """
    Dictionary subclass holding motion correction results with attribute access.

    Keys:
    - `motion_corrected`: 4D / 3D ANTsImage of corrected time-series.
    - `transforms`: `TransformCollection` of all forward and inverse transform paths.
    - `fwdtransforms`: List of forward transform paths per frame.
    - `invtransforms`: List of inverse transform paths per frame.
    - `fd`: 1D array of framewise displacement (mm).
    - `fd_power`: Power FD 1D array (mm).
    - `fd_jenkinson`: Jenkinson FD 1D array (mm).
    - `motion_parameters`: `MotionParameters` object (translations & rotations).
    - `dvars`: Post-correction DVARS 1D array.
    - `dvars_pre`: Pre-correction DVARS 1D array.
    - `dvars_post`: Post-correction DVARS 1D array.
    - `reference`: Reference ANTsImage used for alignment.
    - `summary`: Dictionary of motion summary metrics.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'MotionCorrectionResult' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __repr__(self) -> str:
        shape = getattr(self.get("motion_corrected"), "shape", None)
        fd_mean = self.get("summary", {}).get("fd_mean", None)
        var_red = self.get("summary", {}).get("temporal_variance_reduction_percent", None)
        return (
            f"<MotionCorrectionResult: shape={shape}, "
            f"mean_FD={fd_mean:.3f} mm, "
            f"variance_reduction={var_red:.1f}%>"
            if fd_mean is not None and var_red is not None
            else f"<MotionCorrectionResult: shape={shape}>"
        )


def _extract_rigid_parameters(
    tx: ants.ANTsTransform,
    dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts physical translations, rotation angles (radians), and homogeneous matrix
    from an ITK/ANTs rigid or affine transform.

    Parameters
    ----------
    tx : ants.ANTsTransform
        Read transform object.
    dim : int
        Spatial dimensionality (2 or 3).

    Returns
    -------
    translations : np.ndarray
        Translation parameters at center of rotation in mm, shape (dim,).
    rotations : np.ndarray
        Rotation angles in radians: [rx, ry, rz] for 3D or [theta] for 2D.
    T_homog : np.ndarray
        Full (dim+1, dim+1) homogeneous affine transformation matrix.
    """
    params = np.array(tx.parameters, dtype=np.float64)
    fixed = np.array(tx.fixed_parameters, dtype=np.float64)

    if dim == 2:
        A = params[:4].reshape((2, 2))
        t = params[4:6]
        c = fixed[:2] if len(fixed) >= 2 else np.zeros(2, dtype=np.float64)

        # Robust SO(2) projection via SVD
        U, _, Vt = np.linalg.svd(A)
        det = np.linalg.det(U @ Vt)
        R = U @ np.diag([1.0, det]) @ Vt

        theta = np.arctan2(R[1, 0], R[0, 0])
        rotations = np.array([theta], dtype=np.float64)
        translations = t.copy()

        # Homogeneous matrix: y = A(x - c) + c + t = A x + (t + c - A c)
        b = t + c - R @ c
        T_homog = np.eye(3, dtype=np.float64)
        T_homog[:2, :2] = R
        T_homog[:2, 2] = b

    elif dim == 3:
        A = params[:9].reshape((3, 3))
        t = params[9:12]
        c = fixed[:3] if len(fixed) >= 3 else np.zeros(3, dtype=np.float64)

        # Robust SO(3) projection via SVD
        U, _, Vt = np.linalg.svd(A)
        det = np.linalg.det(U @ Vt)
        R = U @ np.diag([1.0, 1.0, det]) @ Vt

        # Cardan / Euler angle decomposition (XYZ convention: R = Rz * Ry * Rx)
        sy = -R[2, 0]
        if np.abs(sy) < 0.999999:
            ry = np.arcsin(np.clip(sy, -1.0, 1.0))
            rx = np.arctan2(R[2, 1], R[2, 2])
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock singularity
            ry = (np.pi / 2.0) if sy > 0 else (-np.pi / 2.0)
            rx = np.arctan2(-R[1, 2], R[1, 1])
            rz = 0.0

        rotations = np.array([rx, ry, rz], dtype=np.float64)
        translations = t.copy()

        # Homogeneous matrix
        b = t + c - R @ c
        T_homog = np.eye(4, dtype=np.float64)
        T_homog[:3, :3] = R
        T_homog[:3, 3] = b

    else:
        raise ValueError(f"Unsupported spatial dimensionality {dim}. Expected 2 or 3.")

    return translations, rotations, T_homog


def calculate_framewise_displacement(
    motion_parameters: MotionParameters,
    homogeneous_matrices: List[np.ndarray],
    radius: float = 50.0,
    center_of_sphere: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes both Power FD and Jenkinson FD across time.

    Parameters
    ----------
    motion_parameters : MotionParameters
        Array of translations and rotations per frame.
    homogeneous_matrices : List[np.ndarray]
        List of (dim+1, dim+1) homogeneous affine matrices mapping reference space
        to frame space.
    radius : float, default=50.0
        Standard head sphere radius in mm (Power et al. 2012, Jenkinson et al. 2002).
    center_of_sphere : np.ndarray, optional
        Physical center coordinates for Jenkinson displacement evaluation.
        Defaults to zeros(dim).

    Returns
    -------
    fd_power : np.ndarray
        Power Framewise Displacement (1D array of length N).
    fd_jenkinson : np.ndarray
        Jenkinson Framewise Displacement (1D array of length N).
    """
    num_frames = len(motion_parameters)
    dim = motion_parameters.spatial_dimension

    fd_power = np.zeros(num_frames, dtype=np.float64)
    fd_jenkinson = np.zeros(num_frames, dtype=np.float64)

    if num_frames <= 1:
        return fd_power, fd_jenkinson

    if center_of_sphere is None:
        c_sphere = np.zeros(dim, dtype=np.float64)
    else:
        c_sphere = np.asarray(center_of_sphere, dtype=np.float64)[:dim]

    translations = motion_parameters.translations
    rotations = motion_parameters.rotations

    for t in range(1, num_frames):
        # 1. Power FD calculation
        dt = np.abs(translations[t] - translations[t - 1]).sum()

        # Wrap angular difference to [-pi, pi] to avoid 2*pi phase jumps
        drot_raw = rotations[t] - rotations[t - 1]
        drot_wrapped = (drot_raw + np.pi) % (2.0 * np.pi) - np.pi
        dr = radius * np.abs(drot_wrapped).sum()
        fd_power[t] = dt + dr

        # 2. Jenkinson FD calculation
        # Relative transform: M = T_t * T_{t-1}^{-1}
        T_t = homogeneous_matrices[t]
        T_prev = homogeneous_matrices[t - 1]
        try:
            T_prev_inv = np.linalg.inv(T_prev)
            M = T_t @ T_prev_inv
        except np.linalg.LinAlgError:
            M = np.eye(dim + 1, dtype=np.float64)

        A_rel = M[:dim, :dim]
        b_rel = M[:dim, dim]

        # Displacement of sphere center
        d_c = (A_rel - np.eye(dim)) @ c_sphere + b_rel

        # Mean square displacement on a sphere / disk of radius R
        diff_A = A_rel - np.eye(dim)
        tr_diff = np.trace(diff_A.T @ diff_A)

        if dim == 3:
            msd = 0.2 * (radius**2) * tr_diff + np.sum(d_c**2)
        else:  # dim == 2
            msd = 0.25 * (radius**2) * tr_diff + np.sum(d_c**2)

        fd_jenkinson[t] = np.sqrt(max(0.0, float(msd)))

    return fd_power, fd_jenkinson


def calculate_dvars(
    series: Union[ants.ANTsImage, np.ndarray],
    mask: Optional[Union[ants.ANTsImage, np.ndarray]] = None,
) -> np.ndarray:
    """
    Computes DVARS (Root-Mean-Square frame-to-frame intensity variance) across time.

    Parameters
    ----------
    series : ants.ANTsImage or np.ndarray
        Time series volume with time along the last dimension.
    mask : ants.ANTsImage or np.ndarray, optional
        Spatial mask defining foreground voxels. If None, voxels where the temporal
        mean exceeds 5% of the maximum intensity are used.

    Returns
    -------
    dvars : np.ndarray
        DVARS metric across frames (length N, with dvars[0] == 0.0).
    """
    if isinstance(series, ants.ANTsImage):
        arr = series.numpy()
    else:
        arr = np.asarray(series)

    num_frames = arr.shape[-1]
    dvars = np.zeros(num_frames, dtype=np.float64)

    if num_frames <= 1:
        return dvars

    if mask is not None:
        if isinstance(mask, ants.ANTsImage):
            mask_arr = mask.numpy() > 0
        else:
            mask_arr = np.asarray(mask) > 0
    else:
        mean_vol = np.mean(arr, axis=-1)
        thresh = 0.05 * np.max(mean_vol)
        mask_arr = mean_vol > thresh

    if not np.any(mask_arr):
        mask_arr = np.ones(arr.shape[:-1], dtype=bool)

    diff = np.diff(arr, axis=-1)  # shape: (..., num_frames - 1)
    diff_mask = diff[mask_arr]  # shape: (num_voxels, num_frames - 1)

    dvars[1:] = np.sqrt(np.mean(diff_mask**2, axis=0))
    return dvars


def _create_identity_transform(dim: int, filename: str) -> ants.ANTsTransform:
    """Create and save an exact identity AffineTransform to disk."""
    if dim not in (2, 3):
        raise ValueError(f"Unsupported dimension: {dim}. Expected 2 or 3.")

    tx = ants.new_ants_transform(precision="float", dimension=dim, transform_type="AffineTransform")
    if dim == 2:
        params = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    else:
        params = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    tx.set_parameters(params)
    tx.set_fixed_parameters(np.zeros(dim, dtype=np.float64))
    ants.write_transform(tx, filename)
    return tx


def motion_correction(
    image: Union[ants.ANTsImage, str, np.ndarray],
    reference: Union[str, int, ants.ANTsImage] = "mean",
    type_of_transform: str = "Rigid",
    aff_metric: str = "meansquares",
    fd_radius: float = 50.0,
    fd_method: str = "power",
    two_pass: bool = False,
    mask: Optional[Union[ants.ANTsImage, np.ndarray]] = None,
    interpolator: str = "linear",
    outprefix: Optional[str] = None,
    backend: str = "pytorch",
    verbose: bool = False,
    **kwargs: Any,
) -> MotionCorrectionResult:
    """
    Performs robust rigid motion correction on a 2D+t or 3D+t time-series image.

    Parameters
    ----------
    image : ants.ANTsImage, str, or np.ndarray
        Input time-series image. Dimension must be 3 (2D+t) or 4 (3D+t).
    reference : str, int, or ants.ANTsImage, default='mean'
        Reference frame for motion correction:
        - 'mean': temporal mean volume across all timepoints.
        - int: specific frame index (e.g., 0).
        - ants.ANTsImage: explicit reference image of spatial dimension (image.dimension - 1).
    type_of_transform : str, default='Rigid'
        Rigid transform type for registration ('Rigid', 'QuickRigid', 'BOLDRigid', 'Affine', 'Translation').
    aff_metric : str, default='meansquares'
        Metric for rigid intra-subject alignment. 'meansquares' is optimal and smooth
        for intra-subject mono-modal time series; 'mattes' or 'cc' can also be used.
    fd_radius : float, default=50.0
        Head sphere radius in millimetres for rotational displacement in FD calculation.
    fd_method : {'power', 'jenkinson'}, default='power'
        Primary framewise displacement algorithm to store in the `fd` key.
    two_pass : bool, default=False
        If True and `reference='mean'`, performs an iterative two-pass refinement:
        Pass 1 registers to raw mean, then Pass 2 registers to the sharper mean of Pass 1 corrected frames.
    mask : ants.ANTsImage or np.ndarray, optional
        Spatial mask for registration and metric evaluation.
    interpolator : str, default='linear'
        Interpolation method when resampling corrected frames ('linear', 'nearestNeighbor', 'bSpline').
    outprefix : str, optional
        File prefix for saving registration transform files. If None, a dedicated temp directory is used.
        Only honoured when `backend='ants'`; the pytorch backend writes its own transform files.
    backend : {'pytorch', 'ants'}, default='pytorch'
        Per-frame registration engine. 'pytorch' (default) uses `syntx.robust_affine`'s
        torch-native multi-start solver with `dof='rigid'` (translation + rotation only,
        `type_of_transform='Affine'` requests `dof='affine'` instead) -- a plain
        `dof='affine'` solve was found to drift into spurious scale contraction on
        small/low-texture volumes (near-perfect translation recovery but a corrupted
        warp, det(A) << 1), which the rigid constraint removes entirely since scale/shear
        are frozen at identity rather than merely regularized. 'ants' is the legacy
        `ants.registration` path, kept as an explicit named alternative for provenance
        comparisons. `mask` is not supported with `backend='pytorch'` (the solver has no
        masked-MI mode yet) -- pass `backend='ants'` if you need a spatial mask.
    verbose : bool, default=False
        If True, log progress per frame.
    **kwargs : Any
        Additional keyword arguments passed to `ants.registration` (backend='ants') or
        `syntx.robust_affine` (backend='pytorch', the default).

    Returns
    -------
    MotionCorrectionResult
        Dictionary containing:
        - `motion_corrected`: Re-assembled 4D / 3D ANTsImage.
        - `transforms`: `TransformCollection` of all forward and inverse transform paths.
        - `fwdtransforms`: List of forward transform file lists per frame.
        - `invtransforms`: List of inverse transform file lists per frame.
        - `fd`: Primary 1D array of framewise displacement (mm).
        - `fd_power`: Power FD 1D array (mm).
        - `fd_jenkinson`: Jenkinson FD 1D array (mm).
        - `motion_parameters`: `MotionParameters` object with 6-DOF (or 3-DOF) physical parameters.
        - `dvars`: Post-correction DVARS 1D array.
        - `dvars_pre`: Pre-correction DVARS 1D array.
        - `dvars_post`: Post-correction DVARS 1D array.
        - `reference`: The reference ANTsImage volume used.
        - `summary`: Detailed summary dictionary of motion statistics.
    """
    # 1. Input parsing and validation
    if backend not in ("pytorch", "ants"):
        raise ValueError(f"backend must be 'pytorch' or 'ants', got {backend!r}.")
    if mask is not None and backend != "ants":
        raise ValueError(
            "mask is only supported with backend='ants' -- syntx.robust_affine's pytorch "
            "solver has no masked-MI mode yet. Pass backend='ants' or omit mask."
        )

    if isinstance(image, str):
        image = ants.image_read(image)
    elif isinstance(image, np.ndarray):
        image = ants.from_numpy(image)

    dim = image.dimension
    if dim not in (3, 4):
        raise ValueError(
            f"motion_correction requires a time-series image of dimension 3 (2D+t) or 4 (3D+t), "
            f"got image with dimension {dim} and shape {image.shape}."
        )

    spatial_dim = dim - 1
    num_frames = image.shape[dim - 1]

    if num_frames == 0:
        raise ValueError("Input time-series contains 0 frames.")

    # 2. Extract frames along the last dimension using ants.slice_image
    frames: List[ants.ANTsImage] = [
        ants.slice_image(image, axis=dim - 1, idx=t) for t in range(num_frames)
    ]

    # Setup directory for transform output files
    if outprefix is None:
        temp_dir = tempfile.mkdtemp(prefix="syntx_moco_")
        run_prefix = os.path.join(temp_dir, "moco")
    else:
        run_prefix = outprefix

    # 3. Determine reference image
    ref_idx: Optional[int] = None
    if isinstance(reference, str):
        if reference.lower() == "mean":
            mean_data = np.mean([f.numpy() for f in frames], axis=0)
            ref_img = ants.new_image_like(frames[0], mean_data.astype(frames[0].dtype))
        else:
            raise ValueError(f"Unknown reference string '{reference}'. Expected 'mean'.")
    elif isinstance(reference, (int, np.integer)):
        ref_idx = int(reference)
        if not (0 <= ref_idx < num_frames):
            raise IndexError(
                f"Reference frame index {ref_idx} out of valid range [0, {num_frames - 1}]."
            )
        ref_img = frames[ref_idx]
    elif isinstance(reference, ants.ANTsImage):
        if reference.dimension != spatial_dim:
            raise ValueError(
                f"Reference image dimension {reference.dimension} does not match spatial dimension {spatial_dim}."
            )
        ref_img = reference
    else:
        raise TypeError(
            f"Unsupported reference type: {type(reference)}. Expected 'mean', int, or ants.ANTsImage."
        )

    # 4. Helper inner registration loop
    def _run_pass(
        current_ref: ants.ANTsImage,
        current_ref_idx: Optional[int],
        pass_tag: str = "",
    ) -> Tuple[
        List[ants.ANTsImage],
        List[List[str]],
        List[List[str]],
        np.ndarray,
        List[np.ndarray],
    ]:
        corrected_vols: List[ants.ANTsImage] = []
        fwd_tx_all: List[List[str]] = []
        inv_tx_all: List[List[str]] = []
        params_all = np.zeros(
            (num_frames, 6 if spatial_dim == 3 else 3), dtype=np.float64
        )
        homog_all: List[np.ndarray] = []

        for t in range(num_frames):
            frame_t = frames[t]
            frame_prefix = f"{run_prefix}_{pass_tag}vol{t:04d}_"

            # Check if this frame is identically the reference volume
            if current_ref_idx is not None and t == current_ref_idx:
                # Identity transform
                id_tx_path = f"{frame_prefix}IdentityAffine.mat"
                tx_obj = _create_identity_transform(spatial_dim, id_tx_path)
                fwd_tx = [id_tx_path]
                inv_tx = [id_tx_path]
                warped_t = frame_t.clone()
                trans, rot, T_h = _extract_rigid_parameters(tx_obj, spatial_dim)
            else:
                reg_args = dict(kwargs)

                if backend == "ants":
                    reg_args.setdefault("aff_metric", aff_metric)
                    reg_args.setdefault("verbose", verbose)
                    if mask is not None:
                        reg_args["mask"] = mask

                    reg = ants.registration(
                        fixed=current_ref,
                        moving=frame_t,
                        type_of_transform=type_of_transform,
                        outprefix=frame_prefix,
                        **reg_args,
                    )
                else:
                    from .robust_affine import robust_affine

                    affine_mode = "com_only" if type_of_transform == "Translation" else "auto"
                    solver_dof = "affine" if type_of_transform == "Affine" else "rigid"
                    reg_args.setdefault("dof", solver_dof)
                    reg = robust_affine(
                        fixed=current_ref,
                        moving=frame_t,
                        mode=affine_mode,
                        backend="pytorch",
                        verbose=verbose,
                        **reg_args,
                    )

                fwd_tx = reg["fwdtransforms"]
                inv_tx = reg["invtransforms"]

                # Resample frame with specified interpolator if needed
                if interpolator != "linear" or "warpedmovout" not in reg:
                    warped_t = ants.apply_transforms(
                        fixed=current_ref,
                        moving=frame_t,
                        transformlist=fwd_tx,
                        interpolator=interpolator,
                    )
                else:
                    warped_t = reg["warpedmovout"]

                # Extract motion parameters from the primary forward affine transform
                tx_obj = ants.read_transform(fwd_tx[0])
                trans, rot, T_h = _extract_rigid_parameters(tx_obj, spatial_dim)

            corrected_vols.append(warped_t)
            fwd_tx_all.append(fwd_tx)
            inv_tx_all.append(inv_tx)
            if spatial_dim == 3:
                params_all[t] = np.concatenate([trans, rot])
            else:
                params_all[t] = np.concatenate([trans, rot])
            homog_all.append(T_h)

            if verbose:
                print(f"[syntx.motion] Frame {t + 1}/{num_frames} aligned.")

        return corrected_vols, fwd_tx_all, inv_tx_all, params_all, homog_all

    # 5. Execute registration pass(es)
    pass1_tag = "p1_" if two_pass else ""
    corrected_frames, fwd_transforms, inv_transforms, raw_params, homog_matrices = _run_pass(
        ref_img, ref_idx, pass1_tag
    )

    if two_pass and isinstance(reference, str) and reference.lower() == "mean":
        # Pass 2: compute sharp mean from Pass 1 corrected images
        pass1_mean = np.mean([f.numpy() for f in corrected_frames], axis=0)
        ref_img = ants.new_image_like(frames[0], pass1_mean.astype(frames[0].dtype))
        corrected_frames, fwd_transforms, inv_transforms, raw_params, homog_matrices = _run_pass(
            ref_img, None, "p2_"
        )

    # 6. Re-assemble motion-corrected time-series image
    motion_corrected = ants.list_to_ndimage(image, corrected_frames)

    # 7. Package MotionParameters
    col_names = ["tx", "ty", "tz", "rx", "ry", "rz"] if spatial_dim == 3 else ["tx", "ty", "rz"]
    rad_cols = ["rx", "ry", "rz"] if spatial_dim == 3 else ["rz"]
    motion_params = MotionParameters(
        raw_params,
        columns=col_names,
        rad_to_deg_cols=rad_cols,
        spatial_dim=spatial_dim,
    )

    # 8. Calculate Framewise Displacement (Power and Jenkinson)
    center_of_sphere = ref_img.get_center_of_mass()
    fd_power, fd_jenkinson = calculate_framewise_displacement(
        motion_parameters=motion_params,
        homogeneous_matrices=homog_matrices,
        radius=fd_radius,
        center_of_sphere=center_of_sphere,
    )

    fd_primary = fd_jenkinson if fd_method.lower() == "jenkinson" else fd_power

    # 9. Calculate DVARS (pre and post)
    dvars_pre = calculate_dvars(image, mask=mask)
    dvars_post = calculate_dvars(motion_corrected, mask=mask)

    # 10. Temporal variance metrics
    img_arr = image.numpy()
    corr_arr = motion_corrected.numpy()

    if mask is not None:
        mask_arr = mask.numpy() > 0 if isinstance(mask, ants.ANTsImage) else np.asarray(mask) > 0
    else:
        mean_v = np.mean(img_arr, axis=-1)
        mask_arr = mean_v > (0.05 * np.max(mean_v))
        if not np.any(mask_arr):
            mask_arr = np.ones(img_arr.shape[:-1], dtype=bool)

    var_pre = float(np.mean(np.var(img_arr[mask_arr], axis=-1)))
    var_post = float(np.mean(np.var(corr_arr[mask_arr], axis=-1)))
    var_reduction = (1.0 - var_post / (var_pre + 1e-12)) * 100.0 if var_pre > 0 else 0.0

    mean_dvars_pre = float(np.mean(dvars_pre[1:])) if num_frames > 1 else 0.0
    mean_dvars_post = float(np.mean(dvars_post[1:])) if num_frames > 1 else 0.0
    dvars_reduction = (1.0 - mean_dvars_post / (mean_dvars_pre + 1e-12)) * 100.0 if mean_dvars_pre > 0 else 0.0

    summary_metrics = {
        "num_frames": num_frames,
        "spatial_dimension": spatial_dim,
        "reference_type": reference if isinstance(reference, (str, int)) else "ANTsImage",
        "fd_method": fd_method,
        "fd_radius": fd_radius,
        "fd_mean": float(np.mean(fd_primary[1:])) if num_frames > 1 else 0.0,
        "fd_median": float(np.median(fd_primary[1:])) if num_frames > 1 else 0.0,
        "fd_max": float(np.max(fd_primary)),
        "fd_power_mean": float(np.mean(fd_power[1:])) if num_frames > 1 else 0.0,
        "fd_jenkinson_mean": float(np.mean(fd_jenkinson[1:])) if num_frames > 1 else 0.0,
        "dvars_mean_pre": mean_dvars_pre,
        "dvars_mean_post": mean_dvars_post,
        "dvars_reduction_percent": dvars_reduction,
        "temporal_variance_pre": var_pre,
        "temporal_variance_post": var_post,
        "temporal_variance_reduction_percent": var_reduction,
    }

    transforms_collection = TransformCollection(fwd_transforms, inv_transforms)

    return MotionCorrectionResult({
        "motion_corrected": motion_corrected,
        "transforms": transforms_collection,
        "fwdtransforms": fwd_transforms,
        "invtransforms": inv_transforms,
        "fd": fd_primary,
        "fd_power": fd_power,
        "fd_jenkinson": fd_jenkinson,
        "motion_parameters": motion_params,
        "dvars": dvars_post,
        "dvars_pre": dvars_pre,
        "dvars_post": dvars_post,
        "reference": ref_img,
        "summary": summary_metrics,
    })
