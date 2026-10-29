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

from .sampling import DEFAULT_SAMPLING


class MotionParameters(np.ndarray):
    """
    Subclass of numpy.ndarray storing per-frame motion parameters.

    Supports dual access patterns:
    - Array slicing: `mp[:, 0]`, `mp[t]`
    - Column name indexing: `mp['tx']`, `mp['rx']`, `mp['rx_deg']`
    - Attribute access: `mp.tx`, `mp.ty`, `mp.tz`, `mp.rx`, `mp.ry`, `mp.rz`
    - Export to pandas DataFrame: `mp.to_dataframe()`
    - Export to dictionary: `mp.to_dict()`

    Columns: 3-D frames tx, ty, tz (mm), rx, ry, rz (rad); 2-D frames tx, ty (mm), rz (rad).
    ``'<rotation>_deg'`` gives degrees; aliases trans_x / trans_y / trans_z, rot_x / rot_y /
    rot_z, and theta (= rz).
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
        """Names usable with ``[]``: ``['fwdtransforms', 'invtransforms']``."""
        return ["fwdtransforms", "invtransforms"]

    def values(self) -> List[List[List[str]]]:
        """``[fwdtransforms, invtransforms]``."""
        return [self.fwdtransforms, self.invtransforms]

    def items(self) -> List[Tuple[str, List[List[str]]]]:
        """``[('fwdtransforms', ...), ('invtransforms', ...)]``."""
        return [("fwdtransforms", self.fwdtransforms), ("invtransforms", self.invtransforms)]

    def get(self, key: str, default: Any = None) -> Any:
        """``self[key]``, or ``default`` if the key is not recognised."""
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

    # Defensive guard: a degenerate per-frame registration (e.g. a near-blank or very
    # low-count frame in a dynamic PET series) can produce a non-finite affine matrix,
    # which makes np.linalg.svd raise LinAlgError("SVD did not converge") and crash the
    # whole batch -- confirmed directly on real ds002898 FDG-PET data (antsxfunctional's
    # FDG-template build, 2026-10-02). Falling back to the identity rotation (zero
    # rotation, zero translation) for just that one frame lets the caller's own
    # per-frame accept/fallback safeguard (e.g.
    # antsxfunctional.pet.static_frame.motion_corrected_static_pet_from_dynamic) reject
    # this frame's "correction" on its own merits (it will not correlate better than the
    # raw frame) instead of the whole series failing outright.
    if not np.all(np.isfinite(params)) or not np.all(np.isfinite(fixed)):
        rotations = np.zeros(1 if dim == 2 else 3, dtype=np.float64)
        translations = np.zeros(dim, dtype=np.float64)
        T_homog = np.eye(dim + 1, dtype=np.float64)
        return translations, rotations, T_homog

    if dim == 2:
        A = params[:4].reshape((2, 2))
        t = params[4:6]
        c = fixed[:2] if len(fixed) >= 2 else np.zeros(2, dtype=np.float64)

        # Robust SO(2) projection via SVD (falls back to identity on non-convergence --
        # see the finite-value guard above for why this can happen on real data).
        try:
            U, _, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            return np.zeros(2, dtype=np.float64), np.zeros(1, dtype=np.float64), np.eye(3, dtype=np.float64)
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

        # Robust SO(3) projection via SVD (falls back to identity on non-convergence --
        # see the finite-value guard above for why this can happen on real data).
        try:
            U, _, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), np.eye(4, dtype=np.float64)
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


def jenkinson_framewise_displacement(
    homogeneous_matrices: List[np.ndarray],
    radius: float = 50.0,
    center_of_sphere: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Jenkinson FD per frame: RMS displacement over a ball of ``radius`` mm between consecutive frames' rigid
    transforms (``H`` maps the reference to the frame). Frame 0 is 0. Needs no rotation-centre convention, so it
    is the one FD to compute for a series whose transforms come from several registrations (see
    ``motion_reference.motion_correct_grouped``). Shared by BOLD, ASL and DWI."""
    n = len(homogeneous_matrices)
    dim = homogeneous_matrices[0].shape[0] - 1 if n else 3
    c_sphere = np.zeros(dim) if center_of_sphere is None else np.asarray(center_of_sphere, dtype=np.float64)[:dim]
    fd = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        try:
            M = homogeneous_matrices[t] @ np.linalg.inv(homogeneous_matrices[t - 1])
        except np.linalg.LinAlgError:
            M = np.eye(dim + 1, dtype=np.float64)
        diff_A = M[:dim, :dim] - np.eye(dim)
        d_c = diff_A @ c_sphere + M[:dim, dim]
        tr_diff = np.trace(diff_A.T @ diff_A)
        msd = (0.2 if dim == 3 else 0.25) * (radius**2) * tr_diff + np.sum(d_c**2)
        fd[t] = np.sqrt(max(0.0, float(msd)))
    return fd


MIN_FRAMES_FOR_BAND = 32   # below this no band is sought: a 16-frame ASL series has eight spectral bins and no physiology to resolve


def band_limited_fd(
    translations_mm: np.ndarray,
    rotations_deg: np.ndarray,
    tr_s: float,
    band_hz: Optional[Tuple[float, float]] = None,
    radius: float = 50.0,
    min_hz: float = 0.1,
    min_prominence: float = 1.5,
    fd_method: str = "power",
    center_of_mass: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Framewise displacement with the near-Nyquist physiological alternation removed from the pose series.

    ``fd_method`` is the definition of the returned FD series: ``'power'`` (sum of absolute steps, rotations at
    ``radius``) or ``'jenkinson'`` (RMS displacement over the sphere, as ``motion_correct_grouped`` and the design engines
    report), so ``fd_bandlimited`` is comparable with the result's own ``fd``. Rotations are Euler xyz degrees as stored
    in ``rotations_deg``; ``center_of_mass`` is the point the translations are expressed about.

    A sub-voxel oscillation faster than the sampling (respiration aliased at a long TR, cardiac pulsation at a short one)
    is a real frame-to-frame displacement of the image content that the registration corrects well, but a per-step sum
    like FD counts it twice per cycle and inflates the mean three to four fold (breacher resting series: 0.42 and 0.62 mm
    raw, 0.14 and 0.19 mm after the band is removed; Power et al. 2019, Fair et al. 2020, Gratton et al. 2020). Censoring
    should use this series; the full poses stay the correction and the nuisance regressors.

    The band is ``band_hz`` when given (a recorded respiratory trace is the right source) or found automatically: the peak
    of the pose power spectrum (translations plus rotations at ``radius``, summed over the six degrees of freedom) above
    ``min_hz``, widened to the contiguous region where the smoothed spectrum stays above a quarter of the hump's height
    over the white floor (the median smoothed power above ``min_hz``) and to at least a quarter of the Nyquist frequency
    on either side, clipped to [``min_hz``, Nyquist]. The band is applied
    only when its power is at least ``min_prominence`` times what its width alone would hold (otherwise there is no
    oscillation and removing the band would merely understate white jitter); then ``band_source`` is ``'none'`` and the
    returned series is the plain Power FD. Every decision is in the returned dict; nothing is silent.

    Parameters
    ----------
    translations_mm, rotations_deg : (T, 3) arrays
        Rigid pose per frame about the reference centre of mass.
    tr_s : float
        Repetition time in seconds.
    band_hz : (lo, hi), optional
        Band to remove; None selects it from the pose spectrum.
    radius : float, default 50.0
        Sphere radius (mm) for the rotation lever, as for Power FD.
    min_hz : float, default 0.1
        Lowest frequency considered physiological (below it lies drift and real head motion).
    min_prominence : float, default 1.5
        Required ratio of the band's power fraction to its width fraction for the band to be applied.

    Returns
    -------
    dict with ``fd_bandlimited`` (T,), ``fd_bandlimited_mean``, ``fd_bandlimited_max``, ``fd_unfiltered`` (T,; the same FD
    definition as ``fd_bandlimited``, named by ``fd_method``, without the band removed),
    ``band_hz`` ((lo, hi) or None), ``band_source`` ('given' | 'auto' | 'none'), ``alternation_hz`` (spectral peak),
    ``alternation_amplitude_mm`` (RMS pose displacement inside the band at ``radius``), ``band_prominence``,
    ``nyquist_hz`` and ``tr_s``.
    """
    t = np.asarray(translations_mm, dtype=np.float64).reshape(-1, 3)
    rot = np.radians(np.asarray(rotations_deg, dtype=np.float64).reshape(-1, 3)) * float(radius)
    X = np.c_[t, rot]
    T = X.shape[0]
    tr_s = float(tr_s)
    if not np.isfinite(tr_s) or tr_s <= 0:
        raise ValueError(f"band_limited_fd needs a positive repetition time, got {tr_s!r}")
    nyq = 0.5 / tr_s

    if fd_method not in ("power", "jenkinson"):
        raise ValueError(f"fd_method must be 'power' or 'jenkinson', got {fd_method!r}")

    def _fd(Y):
        if fd_method == "power":
            return np.r_[0.0, np.abs(np.diff(Y, axis=0)).sum(axis=1)]
        from scipy.spatial.transform import Rotation as _Rot
        com = np.zeros(3) if center_of_mass is None else np.asarray(center_of_mass, dtype=np.float64)
        Hs = []
        for k in range(Y.shape[0]):
            R = _Rot.from_euler("xyz", Y[k, 3:] / float(radius), degrees=False).as_matrix()
            H = np.eye(4); H[:3, :3] = R; H[:3, 3] = Y[k, :3] - (R - np.eye(3)) @ com
            Hs.append(H)
        return np.asarray(jenkinson_framewise_displacement(Hs, float(radius), com), dtype=np.float64)

    fd_raw = _fd(X)
    out: Dict[str, Any] = {"fd_unfiltered": fd_raw, "fd_method": str(fd_method), "tr_s": tr_s, "nyquist_hz": float(nyq), "band_hz": None,
                           "band_source": "none", "alternation_hz": None, "alternation_amplitude_mm": 0.0,
                           "band_prominence": 0.0, "band_power_fraction": 0.0}
    if T < MIN_FRAMES_FOR_BAND or nyq <= float(min_hz):   # a band needs spectral resolution: 32 frames gives 16 bins
        out.update(fd_bandlimited=fd_raw, fd_bandlimited_mean=float(np.mean(fd_raw[1:])) if T > 1 else 0.0,
                   fd_bandlimited_max=float(fd_raw.max()))
        return out
    Xc = X - X.mean(axis=0)
    F = np.fft.rfft(Xc, axis=0)
    freqs = np.fft.rfftfreq(T, d=tr_s)
    power = (np.abs(F) ** 2).sum(axis=1)
    total = float(power[1:].sum())
    phys = (freqs >= float(min_hz)) & (freqs <= nyq) & (freqs > 0)
    if band_hz is not None:
        lo, hi = float(band_hz[0]), float(band_hz[1])
        source = "given"
    else:
        peak = int(np.flatnonzero(phys)[np.argmax(power[phys])]) if bool(phys.any()) else None
        if peak is None:
            out.update(fd_bandlimited=fd_raw, fd_bandlimited_mean=float(np.mean(fd_raw[1:])), fd_bandlimited_max=float(fd_raw.max()))
            return out
        f_peak = float(freqs[peak])
        # the band is the half-power width of the hump around the peak on the smoothed spectrum (a broad respiratory hump
        # at a long TR spans most of the upper half of the spectrum; a cardiac alias is narrow), never narrower than a
        # quarter of the Nyquist frequency on either side
        k = max(3, int(round(T / 64)) | 1)
        sm = np.convolve(power, np.ones(k) / k, mode="same")
        floor = float(np.median(sm[phys]))                      # the white jitter level above min_hz
        thresh = floor + 0.25 * (float(sm[peak]) - floor)       # quarter height of the hump above that floor
        lo_i = peak
        while lo_i - 1 > 0 and sm[lo_i - 1] >= thresh and freqs[lo_i - 1] >= float(min_hz):
            lo_i -= 1
        hi_i = peak
        while hi_i + 1 < len(freqs) and sm[hi_i + 1] >= thresh:
            hi_i += 1
        lo = max(float(min_hz), min(float(freqs[lo_i]), f_peak - 0.25 * nyq))
        hi = min(float(nyq), max(float(freqs[hi_i]), f_peak + 0.25 * nyq))
        source = "auto"
        out["alternation_hz"] = f_peak
    sel = (freqs >= lo) & (freqs <= hi) & (freqs > 0)
    band_power = float(power[sel].sum())
    width_frac = max((hi - lo) / nyq, 1e-9)
    prominence = (band_power / max(total, 1e-12)) / width_frac
    out["band_prominence"] = float(prominence)
    out["band_power_fraction"] = float(band_power / max(total, 1e-12))
    out["alternation_amplitude_mm"] = float(np.sqrt(2.0 * band_power / (T * T)))   # RMS amplitude of the band-limited pose
    if source == "auto" and prominence < float(min_prominence):
        out.update(fd_bandlimited=fd_raw, fd_bandlimited_mean=float(np.mean(fd_raw[1:])), fd_bandlimited_max=float(fd_raw.max()))
        return out
    Fn = F.copy()
    Fn[sel] = 0.0
    Xn = np.fft.irfft(Fn, n=T, axis=0) + X.mean(axis=0)
    fd_n = _fd(Xn)
    if source == "auto":
        out["alternation_hz"] = float(freqs[sel][np.argmax(power[sel])])
    out.update(fd_bandlimited=fd_n, fd_bandlimited_mean=float(np.mean(fd_n[1:])), fd_bandlimited_max=float(fd_n.max()),
               band_hz=(float(lo), float(hi)), band_source=source)
    return out


def attach_band_limited_fd(result: Dict[str, Any], image: Optional[ants.ANTsImage], translations_mm: np.ndarray,
                           rotations_deg: np.ndarray, tr_s: Optional[float] = None, radius: float = 50.0,
                           fd_method: str = "power", center_of_mass: Optional[np.ndarray] = None) -> None:
    """Write the ``band_limited_fd`` keys into a motion result in place (``fd_bandlimited``, ``fd_band`` dict and, when the
    result has a ``summary``, ``fd_bandlimited_mean`` / ``fd_bandlimited_max`` / ``fd_band_hz`` / ``fd_band_source`` /
    ``fd_alternation_hz`` / ``fd_alternation_amplitude_mm``). The repetition time is ``tr_s`` or the time spacing of a
    4-D ``image``; with neither, the keys record ``band_source='no_tr'`` and ``fd_bandlimited`` is None."""
    if tr_s is None and image is not None and image.dimension == 4 and len(image.spacing) == 4 and float(image.spacing[3]) > 0:
        tr_s = float(image.spacing[3])
    if tr_s is None or not np.isfinite(tr_s) or tr_s <= 0:
        result["fd_bandlimited"] = None
        result["fd_band"] = {"band_source": "no_tr", "band_hz": None, "tr_s": None}
        if isinstance(result.get("summary"), dict):
            result["summary"].update(fd_bandlimited_mean=None, fd_bandlimited_max=None, fd_band_hz=None, fd_band_source="no_tr",
                                     fd_alternation_hz=None, fd_alternation_amplitude_mm=None)
        return
    b = band_limited_fd(translations_mm, rotations_deg, tr_s, radius=radius, fd_method=fd_method, center_of_mass=center_of_mass)
    result["fd_bandlimited"] = b["fd_bandlimited"]
    result["fd_band"] = {k: b[k] for k in ("band_hz", "band_source", "alternation_hz", "alternation_amplitude_mm", "band_prominence", "band_power_fraction", "nyquist_hz", "tr_s")}
    if isinstance(result.get("summary"), dict):
        result["summary"].update(fd_bandlimited_mean=b["fd_bandlimited_mean"], fd_bandlimited_max=b["fd_bandlimited_max"],
                                 fd_band_hz=b["band_hz"], fd_band_source=b["band_source"], fd_alternation_hz=b["alternation_hz"],
                                 fd_alternation_amplitude_mm=b["alternation_amplitude_mm"])


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

    fd_jenkinson = jenkinson_framewise_displacement(homogeneous_matrices, radius, c_sphere)

    return fd_power, fd_jenkinson


def calculate_dvars(
    series: Union[ants.ANTsImage, np.ndarray],
    mask: Optional[Union[ants.ANTsImage, np.ndarray]] = None,
    reference_intensity: Optional[float] = None,
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
    reference_intensity : float, optional
        If given and > 0, the raw RMS frame-to-frame difference is divided by this value
        before being returned, expressing DVARS as a fraction of a reference signal level
        instead of the series' own raw intensity units. Centralizing the normalization
        here -- rather than each caller normalizing ad hoc, or not at all -- means every
        caller of ``calculate_dvars``, ``motion_correction`` and ``motion_correct_grouped``
        reports DVARS on a consistent, cross-subject/cross-acquisition-comparable scale.
        Confirmed real-world need: two DTI subjects processed through the identical code
        path reported ``dvars_mean`` differing by ~1000x (10,637 vs 11.6) purely because
        one acquisition's raw voxel intensities were stored on a ~1e6 scale and the
        other's on a ~1e1-1e3 scale -- not a difference in actual head motion (antsxmm
        cohort review, 2026-10-06). Pass the series' own reference-frame mean intensity
        (e.g. a DWI series' b0 mean) for the conventional normalization.

    Returns
    -------
    dvars : np.ndarray
        DVARS metric across frames (length N, with dvars[0] == 0.0). Dimensionless
        (fraction of ``reference_intensity``) when that argument is given; otherwise in
        the series' own raw intensity units.
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

    diff_mask = np.diff(arr[mask_arr], axis=-1)  # shape: (num_voxels, num_frames - 1)
    n_fg = max(1, int(diff_mask.shape[0]))
    dvars[1:] = np.sqrt(np.einsum("vt,vt->t", diff_mask, diff_mask, dtype=np.float64) / n_fg)
    if reference_intensity is not None and reference_intensity > 0:
        dvars = dvars / reference_intensity
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
    aff_metric: Optional[str] = None,
    fd_radius: float = 50.0,
    fd_method: str = "power",
    two_pass: bool = False,
    mask: Optional[Union[ants.ANTsImage, np.ndarray]] = None,
    interpolator: str = "linear",
    outprefix: Optional[str] = None,
    backend: str = "auto",
    dvars_reference_intensity: Optional[float] = None,
    verbose: bool = False,
    preset: Optional[str] = None,
    encoding_covariates: Optional[np.ndarray] = None,
    window_size: Optional[int] = None,
    debias_pe_pseudomotion: bool = False,
    debias_sagittal_nyquist: bool = False,
    slice_axis=None,
    sidecar=None,
    **kwargs: Any,
) -> MotionCorrectionResult:
    """
    Rigid motion correction of a 2D+t or 3D+t time series -- ``syntx.motion_correction``.

    Every frame is registered to a reference volume, resampled onto it, and the series is
    re-assembled; per-frame motion parameters, framewise displacement (FD) and DVARS are
    reported::

        mc = syntx.motion_correction(bold)               # rigid, reference = temporal mean
        mc.motion_corrected, mc.fd, mc.motion_parameters.to_dataframe()

    Parameters
    ----------
    image : ANTsImage, str (file) or np.ndarray
        Time series, dimension 3 (2D+t) or 4 (3D+t); time is the last axis.
    reference : 'mean', int or ANTsImage, default 'mean'
        Temporal mean, a frame index (that frame gets the identity), or an explicit volume of
        the spatial dimension.
    type_of_transform : str, default 'Rigid'
        'Rigid' (all backends), 'Affine' ('pytorch' / 'ants'), and with ``backend='ants'``
        only 'Translation' / 'QuickRigid' / 'BOLDRigid' (the PyTorch backends raise
        ValueError for them, also when 'auto' picks 'pytorch').
    aff_metric : str, optional
        Similarity for ``backend='ants'`` (None: 'meansquares'). The PyTorch backends use
        Mattes mutual information; giving ``aff_metric`` with them raises ValueError.
    fd_radius : float, default 50.0
        Head radius (mm) converting rotations to displacement in FD.
    fd_method : {'power', 'jenkinson'}, default 'power'
        Which FD goes in ``'fd'`` (both are returned).
    two_pass : bool, default False
        With ``reference='mean'``: register to the mean, then again to the mean of the
        corrected frames.
    mask : ANTsImage or array, optional
        Registration mask -- ``backend='ants'`` only (raises with the PyTorch backends).
    interpolator : str, default 'linear'
        For resampling the frames ('linear', 'nearestNeighbor', 'bSpline').
    outprefix : str, optional
        Prefix for the transform files (default: a temporary directory).
    dvars_reference_intensity : float, optional
        Forwarded to ``calculate_dvars`` for both the pre- and post-correction DVARS.
        Defaults to the resolved reference image's own mean foreground intensity, so
        DVARS is reported as a fraction of that baseline rather than in raw,
        acquisition-scale-dependent intensity units (see ``calculate_dvars``). Callers
        needing a specific shared baseline across multiple calls (e.g.
        ``motion_correct_grouped`` normalizing both its b0 and DWI groups by the b0
        group's own mean) should override this explicitly.
    device : str, optional
        Passed as a keyword argument: torch device for the PyTorch backends (None / 'auto' ->
        ``syntx.device.resolve_device``: ANTSXMM_REG_DEVICE / ANTSXMM_DEVICE, then cuda > mps > cpu;
        an unavailable explicit device raises). Recorded in ``summary['device']``. ``backend='ants'`` is
        CPU-only.
    backend : {'auto', 'pytorch_batched', 'pytorch_batched_temporal', 'pytorch', 'ants'}, default 'auto'
        'pytorch_batched': all frames registered rigidly in one batched GPU pass (3D+t,
        'Rigid', no mask); validated more accurate than ``ants.registration`` on 8 ground-truth
        caches (docs/SESSION_2026-09-25_PHASE_CORRELATION_AND_BATCHED_MOTION_CORRECTION.md,
        Sec 13 / 14). 'pytorch_batched_temporal': same batched GPU pass but tuned for
        TEMPORALLY COHERENT series (consecutive frames of the same subject/contrast, e.g. a
        dynamic PET series) rather than cross-subject/cross-modality registration -- trades
        capture range and precision for speed via a resolution cap, a narrow (small-motion)
        seed search, a shorter optimisation schedule, and a motion gate skipping
        already-aligned frames (see ``batched_rigid_register_pass_temporal``'s docstring for
        the 4 levers). NOT a drop-in accuracy-equivalent replacement for 'pytorch_batched' --
        opt in explicitly for series where real inter-frame motion is small (never selected by
        'auto'). Piloted on real 356-frame FDG-PET data: ~48x faster (100s vs ~80min via
        'pytorch_batched' once chunked) with alignment not hurt on any frame. 'pytorch':
        ``syntx.robust_affine`` per frame (``dof='rigid'``, or 'affine' for 'Affine'); also
        2D+t. 'ants': ``ants.registration`` per frame (the only backend with ``mask``).
        'auto': 'ants' if ``mask`` is given, else 'pytorch_batched' when eligible, else
        'pytorch'.
    verbose : bool, default False
    preset : str, optional
        A single-reference modality preset from ``syntx.motion_presets`` (``"bold"``: also for ASL; ``"slowflow"``:
        overlapping-window surround-coupled motion correction). It sets ``backend``
        and the backend options; passing a different ``backend`` or the same option again raises. The DWI preset is used through
        ``motion_correct_grouped``.
    encoding_covariates : np.ndarray, optional
        Per-frame encoding covariates (shape ``(T,)`` or ``(T, K)``) for in-optimizer overlapping-window
        joint estimation of encoding-induced pose bias and pure head motion.
    window_size : int, optional
        Temporal half-width (in frames) of the overlapping Hann windows for joint encoding-bias estimation.
    debias_pe_pseudomotion : bool, default False
        Opt-in (3-D, ``pe_axis`` required): after registration, separate respiratory B0 pseudomotion along the
        phase-encode axis from the REPORTED head-motion trajectory (``separate_bold_pe_pseudomotion``). Only the
        translation along ``pe_axis`` is touched; slice translation and pitch are real head motion and are never
        modified. The transforms and the resampled series are unchanged (the B0 shift is a real image displacement), so
        ``motion_parameters`` / ``fd`` then differ from ``fwdtransforms`` by ``pe_b0_shift_mm`` along ``pe_axis``;
        the unmodified values are kept as ``motion_parameters_raw`` / ``fd_raw``.
    debias_sagittal_nyquist : bool, default False
        Opt-in extension of ``debias_pe_pseudomotion``: also shrink a TR-to-TR alternation of slice translation and pitch
        (see ``separate_bold_pe_pseudomotion(sagittal_nyquist=True)``; independent real jitter alternates too).
    slice_axis : int or str, optional
        Acquired slice axis for the PE debias frame (voxel index or BIDS ``i/j/k``); default: measured from the series
        (``syntx.acquisition_axes.infer_slice_axis``: voxel thickness, slice interleave), raising when undecidable.
    sidecar : dict or Path or str, optional
        Optional BIDS sidecar dict or path for metadata-guided axis inference.
    **kwargs
        'ants': passed to ``ants.registration``; 'pytorch': passed to ``syntx.robust_affine``;
        'pytorch_batched': ``num_bins`` (default 32), ``seed_mode``, ``schedule_mode``, ``resolution_cap_mm``,
        ``metric`` ('mattes' default, or 'huber' for same-contrast series), ``huber_delta`` (default 0.1) and
        ``sample_fraction`` (fraction of the full-resolution foreground the metric is evaluated on; 1.0 = dense; the
        result's ``sampling`` key records what was used).
        'pytorch_batched_temporal': ``num_bins`` (default 32), ``resolution_cap_mm`` (default
        4.0mm, pass ``None`` to register at native resolution), ``motion_gate_threshold``
        (default 0.995, pass ``None`` to register every frame unconditionally).

    Returns
    -------
    MotionCorrectionResult (dict with attribute access)
        ``motion_corrected`` (re-assembled series), ``fwdtransforms`` / ``invtransforms``
        (per-frame lists), ``transforms`` (``TransformCollection`` of both),
        ``motion_parameters`` (``MotionParameters``: translations mm, rotations rad / deg),
        ``fd`` (``fd_method``), ``fd_power``, ``fd_jenkinson`` (mm), ``dvars_pre``,
        ``dvars_post`` (``dvars`` = post), ``reference``, ``summary`` (dict of summary
        statistics, e.g. ``fd_mean``, ``temporal_variance_reduction_percent``).
    """
    if preset is not None:
        from .motion_presets import get_motion_preset

        cfg = get_motion_preset(preset)
        if preset == "dwi":
            raise ValueError("the 'dwi' preset is used through motion_correct_grouped(..., preset='dwi')")
        if backend not in ("auto", cfg["backend"]):
            raise ValueError(f"preset={preset!r} sets backend={cfg['backend']!r}; do not also pass backend={backend!r}")
        tunable_keys = {"window_size", "window_stride", "surround_weight", "surround_delta_mm"}
        clash = sorted(set(kwargs) & (set(cfg) - {"backend"} - tunable_keys))
        if clash:
            raise ValueError(f"preset={preset!r} already sets {clash}; do not pass them as well")
        backend = cfg["backend"]
        kwargs = {**{k: v for k, v in cfg.items() if k != "backend"}, **kwargs}
    if encoding_covariates is not None:
        kwargs["encoding_covariates"] = encoding_covariates
        if preset == "slowflow" and isinstance(reference, str) and reference.lower() == "mean":
            reference = "bilateral"
    if window_size is not None:
        kwargs["window_size"] = window_size

    # 1. Input parsing and validation
    if debias_sagittal_nyquist and not debias_pe_pseudomotion:
        raise ValueError("debias_sagittal_nyquist extends debias_pe_pseudomotion; pass debias_pe_pseudomotion=True as well")
    if backend not in ("auto", "pytorch", "pytorch_batched", "pytorch_batched_temporal", "pytorch_batched_adaptive", "ants"):
        raise ValueError(
            f"backend must be 'auto', 'pytorch', 'pytorch_batched', 'pytorch_batched_temporal', "
            f"'pytorch_batched_adaptive', or 'ants', got {backend!r}."
        )

    # Compute device for the torch backends (None / 'auto' -> central resolver, cuda > mps > cpu).
    device = kwargs.pop("device", None)

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

    if backend == "auto":
        # Auto-select the validated-superior batched solver whenever it's eligible (3D,
        # Rigid, no mask -- its only current restrictions); fall back to the per-frame
        # 'pytorch' solver otherwise. This is a choice between two torch-native paths, not
        # a fallback to legacy `ants.registration` -- see
        # docs/antsx_implementation_standards.md rule 2. See `batched_rigid_register_pass`'s
        # docstring / docs/SESSION_2026-09-25_..._MOTION_CORRECTION.md for why
        # 'pytorch_batched' is now preferred by default: it's been validated to EXCEED
        # ants' own accuracy (translation AND rotation) across all 8 available
        # ground-truth caches, at roughly parity or better wall-clock time (measured
        # net of unrelated system load -- see session doc Sec 13/14).
        if mask is not None:
            backend = "ants"  # only backend supporting a spatial mask
        elif spatial_dim == 3 and type_of_transform == "Rigid":
            backend = "pytorch_batched"
        else:
            backend = "pytorch"

    if backend == "ants":
        if device is not None and str(device).strip().lower() not in ("auto", "cpu"):
            raise ValueError(f"backend='ants' runs on the CPU (ANTs/ITK); device={device!r} is not supported")
        motion_device, motion_device_source = "cpu", "ants-backend"
    else:
        from .device import resolve_device_with_source

        motion_device, motion_device_source = resolve_device_with_source(device, task="rigid_motion")
        if backend == "pytorch":
            kwargs["device"] = motion_device

    if backend == "pytorch" and type_of_transform in ("Translation", "QuickRigid", "BOLDRigid"):
        # robust_affine has no translation-only solver (this ran a centre-of-mass shift) and
        # QuickRigid / BOLDRigid are ANTs presets (this ran plain rigid)
        raise ValueError(f"type_of_transform={type_of_transform!r} is implemented by backend='ants' "
                         "only; use backend='ants', or 'Rigid' with the PyTorch backends")
    if mask is not None and backend != "ants":
        raise ValueError(
            f"mask is only supported with backend='ants' -- syntx's pytorch solvers have "
            f"no masked-MI mode yet. Pass backend='ants' or omit mask."
        )
    if aff_metric is not None and backend != "ants":
        raise ValueError(f"aff_metric is used by backend='ants' only (backend={backend!r} uses "
                         "Mattes mutual information)")
    _batched_allowed_kwargs = {"num_bins", "seed_mode", "schedule_mode", "resolution_cap_mm", "metric", "huber_delta",
                               "pe_affine_covariates", "pe_axis", "sample_mask", "bias_covariates", "sample_fraction",
                               "sampling", "resample_points", "sample_seed",
                               "convergence_tol_vox", "max_extend_factor", "encoding_covariates", "window_size",
                               "window_stride", "surround_weight", "surround_delta_mm", "slice_axis", "sidecar",
                               "frame_levels", "init_pose", "schedule", "design_matrix", "design_point_scale",
                               "design_levels", "design_model_live", "tr", "surround_scale", "surround_form", "design_gauge", "pose_shrinkage", "frame_gain", "pe_b0_split"}
    if backend == "pytorch_batched" and set(kwargs) - _batched_allowed_kwargs:
        raise TypeError(f"backend='pytorch_batched' accepts only {sorted(_batched_allowed_kwargs)}; got "
                        f"{sorted(set(kwargs) - _batched_allowed_kwargs)}")
    _temporal_allowed_kwargs = {"num_bins", "resolution_cap_mm", "motion_gate_threshold"}
    if backend == "pytorch_batched_temporal" and set(kwargs) - _temporal_allowed_kwargs:
        raise TypeError(
            f"backend='pytorch_batched_temporal' accepts only "
            f"{sorted(_temporal_allowed_kwargs)}; got {sorted(set(kwargs) - _temporal_allowed_kwargs)}"
        )
    _adaptive_allowed_kwargs = {
        "num_bins", "coarse_resolution_cap_mm", "fine_resolution_cap_mm", "gate_fd_threshold_mm",
    }
    if backend == "pytorch_batched_adaptive" and set(kwargs) - _adaptive_allowed_kwargs:
        raise TypeError(
            f"backend='pytorch_batched_adaptive' accepts only "
            f"{sorted(_adaptive_allowed_kwargs)}; got {sorted(set(kwargs) - _adaptive_allowed_kwargs)}"
        )
    if backend in ("pytorch_batched", "pytorch_batched_temporal", "pytorch_batched_adaptive") and type_of_transform != "Rigid":
        raise ValueError(
            f"backend={backend!r} only supports type_of_transform='Rigid' so far, "
            f"got {type_of_transform!r}. Use backend='pytorch' or backend='ants' for "
            f"'QuickRigid'/'BOLDRigid'/'Affine'/'Translation'."
        )

    if num_frames == 0:
        raise ValueError("Input time-series contains 0 frames.")
    if backend in ("pytorch_batched", "pytorch_batched_temporal", "pytorch_batched_adaptive") and spatial_dim != 3:
        raise NotImplementedError(
            f"backend={backend!r} only implements the 3D (3D+t) path, got a "
            f"{spatial_dim}D+t series. Use backend='pytorch' or backend='ants' for 2D+t."
        )

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
    per_frame_refs = None
    if isinstance(reference, str):
        if reference.lower() == "mean":
            mean_data = np.mean(image.numpy(), axis=-1)
            ref_img = ants.new_image_like(frames[0], mean_data.astype(frames[0].dtype))
        elif reference.lower() == "bilateral":
            from .motion_reference import build_bilateral_window_references

            # encoding-matched references averaged over the whole series (K_time = 1): one common head position
            per_frame_refs = build_bilateral_window_references(
                frames,
                encoding_covariates=kwargs.get("encoding_covariates"),
            )
            ref_img = ants.new_image_like(
                per_frame_refs[0],
                np.mean([r.numpy() for r in per_frame_refs], axis=0).astype(per_frame_refs[0].dtype),
            )
        else:
            raise ValueError(f"Unknown reference string '{reference}'. Expected 'mean' or 'bilateral'.")
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
    elif isinstance(reference, (list, tuple)):
        # one reference per frame, all on one grid (e.g. a reference predicted for each frame's own contrast); the
        # batched backends evaluate frame t against reference t; the series is resampled onto that shared grid and
        # DVARS / the ``reference`` key use the mean of the references. Not available for the per-frame backends.
        if len(reference) != num_frames:
            raise ValueError(f"{len(reference)} per-frame references for {num_frames} frames")
        if not all(isinstance(r, ants.ANTsImage) and r.dimension == spatial_dim for r in reference):
            raise TypeError(f"Unsupported reference type: a list of per-frame references must hold ants.ANTsImages of dimension {spatial_dim}")
        if backend not in ("pytorch_batched", "pytorch_batched_temporal", "pytorch_batched_adaptive"):
            raise ValueError("per-frame references need a pytorch_batched backend")
        per_frame_refs = list(reference)
        ref_img = ants.new_image_like(per_frame_refs[0], np.mean([r.numpy() for r in per_frame_refs], axis=0).astype(per_frame_refs[0].dtype))
    else:
        raise TypeError(
            f"Unsupported reference type: {type(reference)}. Expected 'mean', int, ants.ANTsImage or a list of per-frame images."
        )

    # 4. Helper inner registration loop
    pe_frames: dict = {"axis": None, "coefficients": [], "a": {}, "jacobian": {}, "bias": None,
                      "encoding_pose_bias_mm_rad": None, "sampling": None,
                      "frame_converged": {}, "stages": [], "pe_b0_debias": None}  # batched backends

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
        pe_frames["coefficients"].clear(); pe_frames["a"].clear(); pe_frames["jacobian"].clear()
        pe_frames["bias"] = None; pe_frames["encoding_pose_bias_mm_rad"] = None; pe_frames["pe_b0_debias"] = None
        corrected_vols: List[ants.ANTsImage] = []
        fwd_tx_all: List[List[str]] = []
        inv_tx_all: List[List[str]] = []
        params_all = np.zeros(
            (num_frames, 6 if spatial_dim == 3 else 3), dtype=np.float64
        )
        homog_all: List[np.ndarray] = []

        # For backend='pytorch_batched': register every non-reference-identity frame to
        # current_ref in ONE batched call up front, instead of one call per frame inside
        # the loop below -- that's the entire point of this backend (amortizing per-call
        # overhead across the whole series). The per-frame loop below still runs, for
        # resampling and parameter extraction, reusing the exact same code path as the
        # other backends; it just looks up the precomputed transform instead of calling
        # robust_affine/ants.registration per frame.
        batched_results: Dict[int, Tuple[List[str], List[str]]] = {}
        _cap_mm = kwargs.get("resolution_cap_mm") if backend == "pytorch_batched" else None

        def _cap(im: ants.ANTsImage) -> ants.ANTsImage:
            # Register on a copy resampled to at least ``resolution_cap_mm``; the rigid transform is a physical-space
            # transform, valid at native resolution (frames are still resampled from the native images below).
            if not _cap_mm:
                return im
            from .imaging_utils import cap_resolution_for_registration

            return cap_resolution_for_registration(im, _cap_mm)

        if backend in ("pytorch_batched", "pytorch_batched_temporal", "pytorch_batched_adaptive"):
            batch_indices = [t for t in range(num_frames) if not (current_ref_idx is not None and t == current_ref_idx)]
            if batch_indices:
                batch_prefix = f"{run_prefix}_{pass_tag}batch_" if outprefix is not None else None
                if backend == "pytorch_batched_temporal":
                    from .motion_batched import batched_rigid_register_pass_temporal

                    fwd_batch, inv_batch, batch_elapsed = batched_rigid_register_pass_temporal(
                        reference_img=current_ref,
                        moving_imgs=[frames[t] for t in batch_indices],
                        num_bins=kwargs.get("num_bins", 32),
                        verbose=verbose,
                        outprefix=batch_prefix,
                        device=motion_device,
                        resolution_cap_mm=kwargs.get("resolution_cap_mm", 4.0),
                        motion_gate_threshold=kwargs.get("motion_gate_threshold", 0.995),
                    )
                elif backend == "pytorch_batched_adaptive":
                    from .motion_batched import batched_rigid_register_pass_adaptive

                    fwd_batch, inv_batch, batch_elapsed = batched_rigid_register_pass_adaptive(
                        reference_img=current_ref,
                        moving_imgs=[frames[t] for t in batch_indices],
                        num_bins=kwargs.get("num_bins", 32),
                        verbose=verbose,
                        outprefix=batch_prefix,
                        device=motion_device,
                        coarse_resolution_cap_mm=kwargs.get("coarse_resolution_cap_mm", 8.0),
                        fine_resolution_cap_mm=kwargs.get("fine_resolution_cap_mm", 4.0),
                        gate_fd_threshold_mm=kwargs.get("gate_fd_threshold_mm", 0.4),
                        fd_radius=fd_radius,
                    )
                else:
                    from .motion_batched import batched_rigid_register_pass

                    cov_all = kwargs.get("pe_affine_covariates")
                    enc_all = kwargs.get("encoding_covariates")
                    slice_axis_vox = None
                    if (kwargs.get("pe_b0_split") or cov_all is not None) and kwargs.get("pe_axis") is not None and isinstance(image, ants.ANTsImage) and spatial_dim == 3:
                        from .acquisition_axes import AcquisitionAxisUndecidable, infer_slice_axis

                        try:
                            slice_axis_vox = int(
                                infer_slice_axis(image, slice_axis=slice_axis, sidecar=sidecar, levels=kwargs.get("frame_levels")).voxel_axis
                            )
                        except (AcquisitionAxisUndecidable, ValueError):
                            slice_axis_vox = None
                    fwd_batch, inv_batch, batch_elapsed, pe_batch = batched_rigid_register_pass(
                        reference_img=[_cap(per_frame_refs[t]) for t in batch_indices] if per_frame_refs is not None else _cap(current_ref),
                        moving_imgs=[_cap(frames[t]) for t in batch_indices],
                        num_bins=kwargs.get("num_bins", 32),
                        verbose=verbose,
                        outprefix=batch_prefix,
                        device=motion_device,
                        seed_mode=kwargs.get("seed_mode", "wide"),
                        schedule_mode=kwargs.get("schedule_mode", "default"),
                        schedule=kwargs.get("schedule"),
                        metric=kwargs.get("metric", "mattes"),
                        huber_delta=kwargs.get("huber_delta", 0.1),
                        pe_affine_covariates=None if cov_all is None else np.asarray(cov_all)[list(batch_indices)],
                        pe_axis=kwargs.get("pe_axis"),
                        return_pe_affine=True,
                        sample_mask=None if kwargs.get("sample_mask") is None else _cap(kwargs["sample_mask"]),
                        bias_covariates=None if kwargs.get("bias_covariates") is None else np.asarray(kwargs["bias_covariates"])[list(batch_indices)],
                        sample_fraction=kwargs.get("sample_fraction", 0.15),
                        sampling=kwargs.get("sampling", DEFAULT_SAMPLING),
                        resample_points=kwargs.get("resample_points", False), sample_seed=int(kwargs.get("sample_seed", 42)),
                        convergence_tol_vox=kwargs.get("convergence_tol_vox", 0.05),
                        max_extend_factor=kwargs.get("max_extend_factor", 4),
                        encoding_covariates=None if enc_all is None else np.asarray(enc_all)[list(batch_indices)],
                        window_size=kwargs.get("window_size"),
                        window_stride=kwargs.get("window_stride"),
                        surround_weight=kwargs.get("surround_weight"),
                        surround_delta_mm=kwargs.get("surround_delta_mm", 0.25),
                        frame_levels=None if kwargs.get("frame_levels") is None else np.asarray(kwargs["frame_levels"])[list(batch_indices)],
                        init_pose=None if kwargs.get("init_pose") is None else (np.asarray(kwargs["init_pose"][0])[list(batch_indices)],
                                                                                 np.asarray(kwargs["init_pose"][1])[list(batch_indices)]),
                        design_matrix=None if kwargs.get("design_matrix") is None else np.asarray(kwargs["design_matrix"])[list(batch_indices)],
                        design_levels=None if kwargs.get("design_levels") is None else np.asarray(kwargs["design_levels"])[list(batch_indices)],
                        design_point_scale=bool(kwargs.get("design_point_scale", True)),
                        design_model_live=bool(kwargs.get("design_model_live", False)),
                        surround_scale=str(kwargs.get("surround_scale", "raw")),
                        surround_form=str(kwargs.get("surround_form", "charbonnier")),
                        design_gauge=None if kwargs.get("design_gauge") is None else np.asarray(kwargs["design_gauge"])[list(batch_indices)],
                        pose_shrinkage=float(kwargs.get("pose_shrinkage", 0.0)),
                        frame_gain=bool(kwargs.get("frame_gain", False)),
                        pe_b0_split=bool(kwargs.get("pe_b0_split", False)),
                        slice_axis=slice_axis_vox,
                    )
                    if pe_batch is not None:
                        for k_pe in (
                            "sampling", "design_model", "prior",
                            "frame_gain", "bias", "pe_b0_debias",
                        ):
                            if k_pe in pe_batch:
                                pe_frames[k_pe] = pe_batch[k_pe]
                        if pe_batch.get("encoding_pose_bias_mm_rad") is not None:
                            enc_b = dict(pe_batch["encoding_pose_bias_mm_rad"])
                            dY_full = np.zeros((num_frames, 6), dtype=np.float64)   # the identity reference frame has no offset
                            dY_full[list(batch_indices)] = np.asarray(enc_b["delta_Y_mm_rad"], dtype=np.float64)
                            enc_b["delta_Y_mm_rad"] = dY_full
                            enc_b["delta_Y"] = dY_full.tolist()
                            pe_frames["encoding_pose_bias_mm_rad"] = enc_b
                        if "convergence" in pe_batch:
                            chunks = pe_batch["convergence"]
                            chunks = chunks if chunks and isinstance(chunks[0], list) else [chunks]   # one stage list per chunk
                            pos = 0
                            for stages in chunks:
                                n_chunk = len(stages[0]["converged"]) if stages else 0
                                for local_i in range(n_chunk):
                                    pe_frames["frame_converged"][batch_indices[pos + local_i]] = bool(all(st["converged"][local_i] for st in ([s for s in stages if s.get("level", 1) == 1][-1:] or stages[-1:])))
                                pe_frames["stages"].append(stages)
                                pos += n_chunk
                        if "a" in pe_batch and pe_batch["a"] is not None:
                            if pe_batch.get("coefficients") is not None:
                                pe_frames["coefficients"].append(pe_batch["coefficients"])
                            pe_frames["axis"] = pe_batch["axis"]
                            for local_i, t in enumerate(batch_indices):
                                pe_frames["a"][t] = pe_batch["a"][local_i]
                                pe_frames["jacobian"][t] = pe_batch["jacobian"][local_i]


                for local_i, t in enumerate(batch_indices):
                    batched_results[t] = (fwd_batch[local_i], inv_batch[local_i])
                if verbose:
                    print(f"[syntx.motion] {backend}: {len(batch_indices)} frames "
                          f"registered in one batch ({batch_elapsed:.2f}s, "
                          f"{batch_elapsed / len(batch_indices):.3f}s/frame amortized).")

        if backend in ("pytorch_batched", "pytorch_batched_temporal", "pytorch_batched_adaptive"):
            from .motion_batched import rigid_transform_of
            from .transforms import TransformChain, resample_series

            chains_pass: List[Optional[TransformChain]] = []
            vol_scales = np.ones(num_frames, dtype=np.float64)
            for t in range(num_frames):
                frame_prefix = f"{run_prefix}_{pass_tag}vol{t:04d}_"
                if current_ref_idx is not None and t == current_ref_idx:
                    id_tx_path = f"{frame_prefix}IdentityAffine.mat"
                    tx_obj = _create_identity_transform(spatial_dim, id_tx_path)
                    fwd_tx = [id_tx_path]
                    inv_tx = [id_tx_path]
                    chains_pass.append(None)
                    trans, rot, T_h = _extract_rigid_parameters(tx_obj, spatial_dim)
                else:
                    fwd_tx, inv_tx = batched_results[t]
                    chains_pass.append(TransformChain.from_paths(fwd_tx))
                    if t in pe_frames["jacobian"]:
                        vol_scales[t] = float(pe_frames["jacobian"][t])
                    # the rigid pose is the chain's rigid file, found by name (an eddy PE-affine, when present, follows it)
                    tx_obj = ants.read_transform(rigid_transform_of(fwd_tx))
                    trans, rot, T_h = _extract_rigid_parameters(tx_obj, spatial_dim)
                fwd_tx_all.append(fwd_tx)
                inv_tx_all.append(inv_tx)
                params_all[t] = np.concatenate([trans, rot])
                homog_all.append(T_h)
                if verbose:
                    print(f"[syntx.motion] Frame {t + 1}/{num_frames} aligned.")

            corrected_4d = resample_series(
                image,
                chains_pass,
                None,
                current_ref,
                jacobian=False,
                interpolator=interpolator,
                device=motion_device,
                volume_scales=vol_scales,
            )
            return corrected_4d, fwd_tx_all, inv_tx_all, params_all, homog_all

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
                    reg_args.setdefault("aff_metric", aff_metric or "meansquares")
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

                    affine_mode = "auto"
                    solver_dof = "affine" if type_of_transform == "Affine" else "rigid"
                    reg_args.setdefault("dof", solver_dof)
                    reg = robust_affine(
                        fixed=current_ref,
                        moving=frame_t,
                        mode=affine_mode,
                        verbose=verbose,
                        **reg_args,
                    )

                fwd_tx = reg["fwdtransforms"]
                inv_tx = reg["invtransforms"]

                # Resample frame with specified interpolator if needed
                if interpolator != "linear" or "warpedmovout" not in reg:
                    from .transforms import apply_transforms as _apply_transforms
                    warped_t = _apply_transforms(
                        fixed=current_ref,
                        moving=frame_t,
                        transformlist=fwd_tx,
                        interpolator=interpolator,
                        device=motion_device,
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
        if isinstance(corrected_frames, ants.ANTsImage):
            pass1_mean = np.mean(corrected_frames.numpy(), axis=-1)
        else:
            pass1_mean = np.mean([f.numpy() for f in corrected_frames], axis=0)
        ref_img = ants.new_image_like(frames[0], pass1_mean.astype(frames[0].dtype))
        corrected_frames, fwd_transforms, inv_transforms, raw_params, homog_matrices = _run_pass(
            ref_img, None, "p2_"
        )

    # 6. Re-assemble motion-corrected time-series image
    motion_corrected = (
        corrected_frames
        if isinstance(corrected_frames, ants.ANTsImage)
        else ants.list_to_ndimage(image, corrected_frames)
    )

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

    # 9. Calculate DVARS (pre and post), normalized by a reference intensity so DVARS is
    # comparable across subjects/acquisitions with very different raw intensity scales
    # (see calculate_dvars' reference_intensity docstring).
    if dvars_reference_intensity is None:
        ref_arr = ref_img.numpy()
        ref_fg = ref_arr[ref_arr > 0]
        dvars_reference_intensity = float(np.mean(ref_fg)) if ref_fg.size > 0 else None

    dvars_pre = calculate_dvars(image, mask=mask, reference_intensity=dvars_reference_intensity)
    dvars_post = calculate_dvars(motion_corrected, mask=mask, reference_intensity=dvars_reference_intensity)

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
        "reference_type": reference if isinstance(reference, (str, int)) else ("per_frame" if isinstance(reference, (list, tuple)) else "ANTsImage"),
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
        "device": motion_device,
        "device_source": motion_device_source,
    }

    transforms_collection = TransformCollection(fwd_transforms, inv_transforms)

    a_full = np.stack([pe_frames["a"].get(t, np.zeros(4)) for t in range(num_frames)]) if pe_frames["a"] else None
    result = MotionCorrectionResult({
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
        "preset": preset,
        "pe_affine": None if a_full is None else {
            "axis": pe_frames["axis"],
            "coefficients": pe_frames["coefficients"] if pe_frames["coefficients"] else None,
            "a": a_full,
            "jacobian": np.array([pe_frames["jacobian"].get(t, 1.0) for t in range(num_frames)]),
            "shift_mm": a_full[:, 0].copy(),
        },
        "bias": pe_frames["bias"],
        "encoding_pose_bias_mm_rad": pe_frames["encoding_pose_bias_mm_rad"],
        "sampling": pe_frames["sampling"],
        "design_model": pe_frames.get("design_model"),
        "prior": pe_frames.get("prior"),
        "frame_gain": pe_frames.get("frame_gain"),
        "convergence": None if not pe_frames["stages"] else {
            "tol_vox": float(kwargs.get("convergence_tol_vox", 0.05)), "max_extend_factor": int(kwargs.get("max_extend_factor", 4)),
            "frame_converged": [bool(pe_frames["frame_converged"].get(t, True)) for t in range(num_frames)],
            "stages": pe_frames["stages"],
        },
    })
    if a_full is not None and spatial_dim == 3:
        homog_all_composed = []
        raw_params_composed = np.zeros_like(raw_params)
        for t_i, chain in enumerate(fwd_transforms):
            h = np.eye(4)
            for p_tx in chain:
                h = _extract_rigid_parameters(ants.read_transform(p_tx), 3)[2] @ h
            homog_all_composed.append(h)
            tx_c = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
            c_ref = np.asarray(center_of_sphere, dtype=np.float64)
            tx_c.set_parameters(np.concatenate([h[:3, :3].flatten(), h[:3, 3] + (h[:3, :3] - np.eye(3)) @ c_ref]))
            tx_c.set_fixed_parameters(c_ref)
            tr_c, rot_c, _ = _extract_rigid_parameters(tx_c, 3)
            raw_params_composed[t_i] = np.concatenate([tr_c, rot_c])
        mp_raw = MotionParameters(raw_params_composed, columns=col_names, rad_to_deg_cols=rad_cols, spatial_dim=3)
        fd_p_raw, fd_j_raw = calculate_framewise_displacement(
            motion_parameters=mp_raw, homogeneous_matrices=homog_all_composed, radius=fd_radius, center_of_sphere=center_of_sphere
        )
        fd_raw_arr = fd_j_raw if fd_method.lower() == "jenkinson" else fd_p_raw
        result["fd_raw"] = fd_raw_arr
        result["motion_parameters_raw"] = mp_raw
        result["pe_b0_shift_mm"] = a_full[:, 0].copy()
        if isinstance(result.get("summary"), dict):
            result["summary"]["fd_raw_mean"] = float(np.mean(fd_raw_arr[1:])) if num_frames > 1 else 0.0
        if pe_frames.get("pe_b0_debias") is not None:
            result["pe_pseudomotion_debias"] = pe_frames["pe_b0_debias"]
    pe_b0_already_split = (
        bool(kwargs.get("pe_b0_split", False))
        and backend == "pytorch_batched"
        and kwargs.get("pe_axis") is not None
        and num_frames >= 12
    )
    if debias_pe_pseudomotion:
        if spatial_dim != 3:
            raise ValueError("debias_pe_pseudomotion needs a 3-D (+ time) series")
        if kwargs.get("pe_axis") is None:
            raise ValueError("debias_pe_pseudomotion needs pe_axis (physical unit vector, or voxel axis index, of phase encode)")
        if kwargs.get("encoding_covariates") is not None:
            raise ValueError("debias_pe_pseudomotion and encoding_covariates both re-attribute pose; pass one of them")
        if not (pe_b0_already_split and not debias_sagittal_nyquist):
            from .motion_batched import separate_bold_pe_pseudomotion

            result, _ = separate_bold_pe_pseudomotion(
                image,
                result,
                pe_axis=kwargs.get("pe_axis"),
                fd_radius=fd_radius,
                window_size=kwargs.get("window_size"),
                sagittal_nyquist=bool(debias_sagittal_nyquist),
                slice_axis=slice_axis,
                sidecar=sidecar,
            )
    final_mp = result["motion_parameters"]
    attach_band_limited_fd(result, image if isinstance(image, ants.ANTsImage) else None, final_mp.translations,
                           final_mp.rotations_deg, tr_s=kwargs.get("tr"), radius=fd_radius,
                           fd_method=("jenkinson" if fd_method.lower() == "jenkinson" else "power"), center_of_mass=center_of_sphere)
    return result


