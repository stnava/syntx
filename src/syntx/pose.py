"""Probe-displacement pose verification for rigid and affine transformations.

In multi-modality and slab registration (such as neuromelanin MRI to T1 structural
coregistration), intensity metrics like mutual information (MI) can fail to arbitrate
between competing poses because a truncated slab may achieve high local MI in an
incorrect anatomical location or incorrect tilt.

This module provides a physical-space probe-displacement metric:
    max_i || disp_A(q_i) - disp_B(q_i) ||  (in mm)
evaluated at a set of probe points q_i distributed in physical space around the
center of mass of the fixed image.

Transformation Convention
-------------------------
ANTs and ITK follow the fixed -> moving convention for forward coordinate mapping:
    p_moving = T(p_fixed)
where p_fixed is a coordinate in the physical space of the fixed image (in mm, LPS),
and p_moving is the corresponding coordinate in the moving image physical space.

The displacement of a probe point q located in fixed physical space under transform T is:
    disp(q) = T(q) - q
Given two candidate transforms T_A and T_B, the probe-displacement difference is:
    disp_A(q) - disp_B(q) = (T_A(q) - q) - (T_B(q) - q) = T_A(q) - T_B(q)
and the pose error is:
    pose_error(fixed, T_A, T_B) = max_i || T_A(q_i) - T_B(q_i) ||

Probe Geometry
--------------
For standard head volumes or anisotropic slabs, probe_points(fixed, kind='default')
computes the physical center of mass c = ants.get_center_of_mass(fixed * 0 + 1) and
constructs 9 probe points:
    - 8 corners at c + (+-15 mm, +-15 mm, +-5 mm)
    - 1 center point at c
For 'slab_thick', the z lever arm is expanded up to 1.5 slices from the outer slab
boundary (dz = max(5.0, (shape[2] / 2 - 1.5) * spacing[2])) to provide increased
sensitivity to out-of-plane rotation across thick or thin slabs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class PoseErrorReport(dict):
    """
    Per-probe displacement and pose error report.

    Inherits from dict for full dictionary compatibility (e.g. JSON serialization,
    key-based access) while also exposing properties for convenient attribute access.
    """

    def __init__(
        self,
        max_error_mm: float,
        mean_error_mm: float,
        z_mean_error_mm: float,
        probe_errors_mm: np.ndarray,
        probes: np.ndarray,
        displacements_a: np.ndarray,
        displacements_b: np.ndarray,
        worst_probe_index: int,
        worst_probe: np.ndarray,
    ):
        super().__init__(
            max_error_mm=float(max_error_mm),
            mean_error_mm=float(mean_error_mm),
            z_mean_error_mm=float(z_mean_error_mm),
            probe_errors_mm=probe_errors_mm,
            probes=probes,
            displacements_a=displacements_a,
            displacements_b=displacements_b,
            worst_probe_index=int(worst_probe_index),
            worst_probe=worst_probe,
        )

    @property
    def max_error_mm(self) -> float:
        """Maximum Euclidean displacement difference across all probes in mm."""
        return self["max_error_mm"]

    @property
    def mean_error_mm(self) -> float:
        """Mean Euclidean displacement difference across all probes in mm."""
        return self["mean_error_mm"]

    @property
    def z_mean_error_mm(self) -> float:
        """Mean absolute z-axis displacement difference in mm."""
        return self["z_mean_error_mm"]

    @property
    def probe_errors_mm(self) -> np.ndarray:
        """Array of shape (N,) containing per-probe displacement error magnitudes in mm."""
        return self["probe_errors_mm"]

    @property
    def probes(self) -> np.ndarray:
        """Array of shape (N, 3) containing physical probe points in mm."""
        return self["probes"]

    @property
    def displacements_a(self) -> np.ndarray:
        """Array of shape (N, 3) containing displacements under tx_a."""
        return self["displacements_a"]

    @property
    def displacements_b(self) -> np.ndarray:
        """Array of shape (N, 3) containing displacements under tx_b."""
        return self["displacements_b"]

    @property
    def worst_probe_index(self) -> int:
        """Index of the probe with the maximum error."""
        return self["worst_probe_index"]

    @property
    def worst_probe(self) -> np.ndarray:
        """Coordinates of the probe with the maximum error."""
        return self["worst_probe"]

    def summary(self) -> str:
        """Formatted human-readable summary string."""
        return (
            f"PoseErrorReport(max_error={self.max_error_mm:.3f} mm, "
            f"mean_error={self.mean_error_mm:.3f} mm, "
            f"z_mean_error={self.z_mean_error_mm:.3f} mm, "
            f"worst_probe_idx={self.worst_probe_index}, "
            f"worst_probe={self.worst_probe.tolist()})"
        )

    def __repr__(self) -> str:
        return self.summary()


def probe_points(fixed: Any, kind: str | float | Sequence[float] = "default") -> np.ndarray:
    """
    Generate physical-space probe points (in mm) centered on the fixed image.

    Parameters
    ----------
    fixed : ants.ANTsImage, str, Path, or array-like
        The fixed reference image, path to image, or a 3-element sequence specifying
        the physical center of mass [c_x, c_y, c_z] in mm.
    kind : str, float, or 3-tuple of floats, default 'default'
        - 'default': 8 corner points at c + (+-15, +-15, +-5) mm plus c (9 probes total).
        - 'slab_thick': expands z lever arm up to 1.5 slices from the outer boundary
          of the slab (dz = max(5.0, (shape[2] / 2 - 1.5) * spacing[2])), or 15 mm.
        - float: specifies a custom dz offset in mm with dx=15, dy=15.
        - (dx, dy, dz): specifies custom offsets along each axis in mm.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing physical coordinates (in mm, LPS).
    """
    if isinstance(fixed, (str, Path)):
        import ants

        fixed = ants.image_read(str(fixed))

    if hasattr(fixed, "spacing"):
        import ants

        c = np.asarray(ants.get_center_of_mass(fixed * 0 + 1), dtype=np.float64)
    elif isinstance(fixed, (list, tuple, np.ndarray)) and len(fixed) == 3 and np.isscalar(fixed[0]):
        c = np.asarray(fixed, dtype=np.float64)
    else:
        raise TypeError(
            f"fixed must be an ANTsImage, file path, or 3-element physical coordinate; got {type(fixed)}"
        )

    if kind == "default":
        offsets = [
            np.array([x, y, z], dtype=np.float64)
            for x in (-15.0, 15.0)
            for y in (-15.0, 15.0)
            for z in (-5.0, 5.0)
        ]
        return np.array([c + off for off in offsets] + [c], dtype=np.float64)

    elif kind == "slab_thick":
        dz = 5.0
        if hasattr(fixed, "shape") and hasattr(fixed, "spacing") and len(fixed.shape) >= 3:
            dz = max(5.0, (float(fixed.shape[2]) / 2.0 - 1.5) * float(fixed.spacing[2]))
        else:
            dz = 15.0
        offsets = [
            np.array([x, y, z], dtype=np.float64)
            for x in (-15.0, 15.0)
            for y in (-15.0, 15.0)
            for z in (-dz, dz)
        ]
        return np.array([c + off for off in offsets] + [c], dtype=np.float64)

    elif isinstance(kind, (int, float)):
        dz = float(kind)
        offsets = [
            np.array([x, y, z], dtype=np.float64)
            for x in (-15.0, 15.0)
            for y in (-15.0, 15.0)
            for z in (-dz, dz)
        ]
        return np.array([c + off for off in offsets] + [c], dtype=np.float64)

    elif isinstance(kind, (tuple, list)) and len(kind) == 3:
        dx, dy, dz = map(float, kind)
        offsets = [
            np.array([x, y, z], dtype=np.float64)
            for x in (-dx, dx)
            for y in (-dy, dy)
            for z in (-dz, dz)
        ]
        return np.array([c + off for off in offsets] + [c], dtype=np.float64)

    else:
        raise ValueError(
            f"Unknown probe kind: {kind!r}. Supported: 'default', 'slab_thick', float, or (dx, dy, dz) tuple."
        )


def _parse_single_transform(tx: Any) -> Any:
    """Normalize a single transform item into an ANTsTransform or 4x4 numpy matrix."""
    if tx is None:
        raise ValueError("Transform cannot be None")

    if isinstance(tx, (str, Path)):
        import ants

        path_str = str(tx)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Transform file not found: {path_str}")
        return ants.read_transform(path_str)

    if hasattr(tx, "apply_to_point"):
        return tx

    # Torch tensor or array-like
    if hasattr(tx, "detach"):
        tx = tx.detach().cpu().numpy()

    if isinstance(tx, np.ndarray) or (isinstance(tx, (list, tuple)) and tx and isinstance(tx[0], (list, tuple, np.ndarray))):
        mat = np.asarray(tx, dtype=np.float64)
        if mat.shape == (4, 4) or mat.shape == (3, 4) or mat.shape == (3, 3):
            return mat
        raise ValueError(f"Transform matrix must have shape (4, 4), (3, 4), or (3, 3), got {mat.shape}")

    raise TypeError(f"Unsupported transform type: {type(tx)}")


def _parse_transform(tx: Any) -> Any:
    """Normalize transform input, unwrapping single-item containers if present."""
    if isinstance(tx, (list, tuple)):
        if len(tx) == 0:
            raise ValueError("Transform list cannot be empty")
        if len(tx) == 1:
            return _parse_single_transform(tx[0])
        # Multi-transform list
        return [_parse_single_transform(t) for t in tx]
    return _parse_single_transform(tx)


def _apply_transform_to_points(tx: Any, points: np.ndarray) -> np.ndarray:
    """Apply a parsed transform (or list of transforms) to points of shape (N, 3)."""
    if isinstance(tx, list):
        # In ANTs convention, transformlist items are applied in reverse order
        curr = points
        for t in reversed(tx):
            curr = _apply_transform_to_points(t, curr)
        return curr

    if isinstance(tx, np.ndarray):
        r = tx[:3, :3]
        t = tx[:3, 3] if tx.shape[1] > 3 else np.zeros(3, dtype=np.float64)
        return points @ r.T + t

    if hasattr(tx, "apply_to_point"):
        res = [tx.apply_to_point(tuple(map(float, q))) for q in points]
        return np.asarray(res, dtype=np.float64)

    raise TypeError(f"Cannot apply transform of type {type(tx)}")


def _evaluate_displacements(tx: Any, probes: np.ndarray) -> np.ndarray:
    """Compute displacement vectors disp(q) = tx(q) - q for each probe point."""
    tx_parsed = _parse_transform(tx)
    transformed = _apply_transform_to_points(tx_parsed, probes)
    disp = transformed - probes
    if not np.all(np.isfinite(disp)):
        raise ValueError("Transform produced non-finite (NaN or Inf) probe displacements")
    return disp


def probe_displacements(
    fixed: Any,
    tx: Any,
    probes: Sequence[Any] | np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute physical displacement vectors disp(q) = tx(q) - q in mm at probe points.

    Parameters
    ----------
    fixed : ants.ANTsImage, str, Path, or array-like
        Fixed reference image used to compute default probe points if probes is None.
    tx : ANTsTransform, str, Path, or 4x4 array-like
        Transform to evaluate (ITK fixed -> moving convention).
    probes : array-like of shape (N, 3), optional
        Explicit physical probe points in mm. If None, probe_points(fixed, kind='default') is used.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing displacement vectors disp(q_i) = tx(q_i) - q_i in mm.
    """
    if probes is None:
        if fixed is None:
            raise ValueError("fixed image must be provided when probes is None")
        probe_pts = probe_points(fixed, kind="default")
    else:
        probe_pts = np.asarray(probes, dtype=np.float64)
        if probe_pts.ndim != 2 or probe_pts.shape[1] != 3:
            raise ValueError(f"probes must have shape (N, 3), got {probe_pts.shape}")

    return _evaluate_displacements(tx, probe_pts)


def pose_error_report(
    fixed: Any,
    tx_a: Any,
    tx_b: Any,
    probes: Sequence[Any] | np.ndarray | None = None,
) -> PoseErrorReport:
    """
    Compute a detailed per-probe displacement report comparing two transforms.

    Parameters
    ----------
    fixed : ants.ANTsImage, str, Path, or array-like
        Fixed reference image used to compute default probe points if probes is None,
        or a 3D coordinate representing the center of mass.
    tx_a : ANTsTransform, str, Path, or 4x4 array-like
        First transform (ITK fixed -> moving convention).
    tx_b : ANTsTransform, str, Path, or 4x4 array-like
        Second transform (ITK fixed -> moving convention).
    probes : array-like of shape (N, 3), optional
        Explicit physical probe points in mm. If None, probe_points(fixed, kind='default') is used.

    Returns
    -------
    PoseErrorReport (dict subclass)
        Per-probe displacement and error statistics.
    """
    if probes is None:
        if fixed is None:
            raise ValueError("fixed image must be provided when probes is None")
        probe_pts = probe_points(fixed, kind="default")
    else:
        probe_pts = np.asarray(probes, dtype=np.float64)
        if probe_pts.ndim != 2 or probe_pts.shape[1] != 3:
            raise ValueError(f"probes must have shape (N, 3), got {probe_pts.shape}")

    disp_a = _evaluate_displacements(tx_a, probe_pts)
    disp_b = _evaluate_displacements(tx_b, probe_pts)
    diff = disp_a - disp_b

    mag = np.linalg.norm(diff, axis=1)
    worst_idx = int(np.argmax(mag))

    return PoseErrorReport(
        max_error_mm=float(mag[worst_idx]),
        mean_error_mm=float(np.mean(mag)),
        z_mean_error_mm=float(np.mean(np.abs(diff[:, 2]))),
        probe_errors_mm=mag,
        probes=probe_pts,
        displacements_a=disp_a,
        displacements_b=disp_b,
        worst_probe_index=worst_idx,
        worst_probe=probe_pts[worst_idx],
    )


def pose_error(
    fixed: Any,
    tx_a: Any,
    tx_b: Any,
    probes: Sequence[Any] | np.ndarray | None = None,
) -> float:
    """
    Calculate the maximum probe displacement error in mm between two transforms:
        max_i || disp_A(q_i) - disp_B(q_i) ||

    Convention: ITK fixed -> moving transform mapping physical fixed space to moving space.

    Parameters
    ----------
    fixed : ants.ANTsImage, str, Path, or array-like
        Fixed reference image used to compute default probe points if probes is None,
        or a 3D coordinate representing the center of mass.
    tx_a : ANTsTransform, str, Path, or 4x4 array-like
        First transform.
    tx_b : ANTsTransform, str, Path, or 4x4 array-like
        Second transform.
    probes : array-like of shape (N, 3), optional
        Explicit physical probe points in mm. If None, probe_points(fixed, kind='default') is used.

    Returns
    -------
    float
        Maximum displacement difference across all probe points in mm.
    """
    rep = pose_error_report(fixed, tx_a, tx_b, probes=probes)
    return float(rep.max_error_mm)
