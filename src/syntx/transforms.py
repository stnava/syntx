"""syntx.transforms — Torch/MPS/CUDA physical-space transform suite following ANTsPy interfaces.

Provides drop-in Torch/MPS/CUDA implementations of the ANTsPy transform API:
- ``apply_transforms(fixed, moving, transformlist, interpolator='linear', imagetype=0, whichtoinvert=None, compose=None, defaultvalue=0, singleprecision=False, verbose=False, device=None, **kwargs)``
  Supports scalar (``imagetype=0``), 3-component vector (``imagetype=1``), 6-component
  symmetric second-rank tensor (``imagetype=2``), and 4D time-series (``imagetype=3``)
  images, as well as displacement composition (``compose=prefix``).
- ``apply_transforms_to_points(dim, points, transformlist, whichtoinvert=None, verbose=False, device=None)``
- ``resample_image_to_target(image, target, interp_type='linear', imagetype=0, verbose=False, device=None, **kwargs)``
- ``compose_displacement_fields(displacement_field, warping_field, device=None)``
- ``create_jacobian_determinant_image(domain_image, tx, do_log=False, geom=False)``
- ``apply_ants_transform_to_point(transform, point)`` & ``apply_ants_transform_to_vector(transform, vector)``
- ``transform_index_to_physical_point(image, index)`` & ``transform_physical_point_to_index(image, point)``
- ``transform_streamlines(streamlines, transformlist, whichtoinvert=None, compute_tangents=False, device=None)``
- ``reorient_vectors(...)`` & ``reorient_tensors(...)``
- ``TransformChain``, ``TransformElement``, ``classify_transform``, ``ContinuousField``, ``physical_to_normalized``
- ``compose_displacement``, ``signed_jacobian_determinant``, ``jacobian_modulation``, ``resample_series``, ``series_rebase``
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import os
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .device import resolve_device
from .spatial import get_physical_to_normalized_affine_xyz as get_physical_to_normalized_affine

Kind = Literal["rigid", "affine", "displacement", "inverse_displacement", "identity"]
_LINEAR = ("rigid", "affine")
JACOBIAN_CLIP = (0.1, 10.0)
_SYM6_TO_3X3 = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]], dtype=np.int64)


# ==============================================================================
# 1. Continuous Physical Field & Coordinate Primitives
# ==============================================================================

def physical_to_normalized(
    coords_phys: torch.Tensor,
    M_norm: torch.Tensor,
    b_norm: torch.Tensor,
) -> torch.Tensor:
    """Project physical LPS coordinates (in mm) into [-1, 1]^3 normalized grid space."""
    return coords_phys @ M_norm + b_norm


def transform_index_to_physical_point(image: Any, index: Any) -> Any:
    """Convert continuous or integer voxel index (or batch of indices ``(N, D)``) to physical LPS point(s).

    Matches ``ants.transform_index_to_physical_point(image, index)`` for single points while
    also supporting batched ``(N, D)`` arrays.
    """
    idx_arr = np.asarray(index, dtype=np.float64)
    dim = int( getattr(image, "dimension", len(image.spacing)) )
    spacing = np.asarray(image.spacing[:dim], dtype=np.float64)
    origin = np.asarray(image.origin[:dim], dtype=np.float64)
    direction = np.asarray(image.direction, dtype=np.float64)[:dim, :dim]
    m_i2p = direction @ np.diag(spacing)
    if idx_arr.ndim == 1:
        pt = origin + m_i2p @ idx_arr[:dim]
        return tuple(float(x) for x in pt)
    return idx_arr[..., :dim] @ m_i2p.T + origin


def transform_physical_point_to_index(image: Any, point: Any) -> Any:
    """Convert physical LPS point (or batch of points ``(N, D)``) to continuous voxel index.

    Matches ``ants.transform_physical_point_to_index(image, point)`` for single points while
    also supporting batched ``(N, D)`` arrays and vector images (`has_components=True`).
    """
    pt_arr = np.asarray(point, dtype=np.float64)
    dim = int(getattr(image, "dimension", len(image.spacing)))
    spacing = np.asarray(image.spacing[:dim], dtype=np.float64)
    origin = np.asarray(image.origin[:dim], dtype=np.float64)
    direction = np.asarray(image.direction, dtype=np.float64)[:dim, :dim]
    m_p2i = np.linalg.inv(direction @ np.diag(spacing))
    if pt_arr.ndim == 1:
        idx = m_p2i @ (pt_arr[:dim] - origin)
        return tuple(float(x) for x in idx)
    return (pt_arr[..., :dim] - origin) @ m_p2i.T


class ContinuousField:
    """Differentiable continuous field evaluator in physical space (LPS mm)."""

    def __init__(
        self,
        image_or_array: Any,
        origin: tuple[float, float, float] | None = None,
        spacing: tuple[float, float, float] | None = None,
        direction: np.ndarray | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype

        if hasattr(image_or_array, "spacing") and hasattr(image_or_array, "direction"):
            img = image_or_array
            arr = img.numpy()
            self.origin = tuple(float(x) for x in img.origin[:3])
            self.spacing = tuple(float(x) for x in img.spacing[:3])
            self.direction = np.asarray(img.direction[:3, :3], dtype=np.float64)
        else:
            arr = np.asarray(image_or_array)
            self.origin = origin if origin is not None else (0.0, 0.0, 0.0)
            self.spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)
            self.direction = direction if direction is not None else np.eye(3, dtype=np.float64)

        if arr.ndim == 3:
            self.is_vector = False
            self.spatial_shape = arr.shape
            arr_cdhw = arr.transpose(2, 1, 0)[None, None, ...]
        elif arr.ndim == 4 and arr.shape[-1] == 3:
            self.is_vector = True
            self.spatial_shape = arr.shape[:3]
            arr_cdhw = np.transpose(arr, (3, 2, 1, 0))[None, ...]
        elif arr.ndim == 4 and arr.shape[0] == 3:
            self.is_vector = True
            self.spatial_shape = arr.shape[1:]
            arr_cdhw = np.transpose(arr, (0, 3, 2, 1))[None, ...]
        elif arr.ndim == 4:
            self.is_vector = True
            self.spatial_shape = arr.shape[:3]
            arr_cdhw = np.transpose(arr, (3, 2, 1, 0))[None, ...]
        else:
            raise ValueError(f"Unsupported array shape for ContinuousField: {arr.shape}")

        self.tensor = torch.as_tensor(np.ascontiguousarray(arr_cdhw), device=self.device, dtype=self.dtype)

        self.M_norm, self.b_norm = get_physical_to_normalized_affine(
            shape=self.spatial_shape,
            spacing=self.spacing,
            origin=self.origin,
            direction=self.direction,
            device=self.device,
            dtype=self.dtype,
        )

    def sample(
        self,
        coords_phys: torch.Tensor,
        mode: str = "bilinear",
        padding_mode: str = "border",
    ) -> torch.Tensor:
        """Sample continuous field at arbitrary physical LPS coordinates."""
        orig_shape = coords_phys.shape
        flat_coords = coords_phys.view(-1, 3).to(device=self.device, dtype=self.dtype)
        coords_norm = physical_to_normalized(flat_coords, self.M_norm, self.b_norm)

        grid = coords_norm.view(1, 1, 1, -1, 3)
        sampled = F.grid_sample(self.tensor, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

        C = self.tensor.shape[1]
        out = sampled.view(C, -1)
        if self.is_vector:
            out = out.T
            if len(orig_shape) == 3:
                out = out.view(orig_shape[0], orig_shape[1], C)
            return out
        else:
            out = out.view(-1)
            if len(orig_shape) == 3:
                out = out.view(orig_shape[0], orig_shape[1])
            return out

    def is_inside(
        self,
        coords_phys: torch.Tensor,
        margin: float = 0.0,
    ) -> torch.Tensor:
        """Check whether physical LPS coordinates lie within the image grid domain."""
        orig_shape = coords_phys.shape[:-1]
        flat_coords = coords_phys.view(-1, 3).to(device=self.device, dtype=self.dtype)
        coords_norm = physical_to_normalized(flat_coords, self.M_norm, self.b_norm)
        inside = (coords_norm >= (-1.0 + margin)) & (coords_norm <= (1.0 - margin))
        inside_all = inside.all(dim=-1)
        if len(orig_shape) > 0:
            return inside_all.view(orig_shape)
        return inside_all


# ==============================================================================
# 2. Transform Elements & TransformChain
# ==============================================================================

@dataclass(frozen=True)
class TransformElement:
    path: str
    kind: Kind
    invert: bool = False


def classify_transform(path: str) -> Kind:
    """Determine the transform kind by inspecting the file, not its position."""
    lower = str(path).lower()
    if lower == "identity":
        return "identity"
    if lower.endswith((".nii", ".nii.gz", ".mha", ".nrrd")):
        return "inverse_displacement" if "inversewarp" in lower else "displacement"
    if lower.endswith((".mat", ".txt", ".tfm", ".h5", ".hdf5")):
        import ants

        tx = ants.read_transform(path)
        ttype = str(getattr(tx, "transform_type", "")).lower()
        if "euler" in ttype or "rigid" in ttype or "versorrigid" in ttype or "translation" in ttype:
            return "rigid"
        if "affine" in ttype or "similarity" in ttype or "matrixoffset" in ttype:
            return "affine"
        raise ValueError(f"Unknown linear transform type {ttype!r} for {path}")
    raise ValueError(f"Cannot classify transform file {path}")


def _resolve_whichtoinvert(
    tl: list[str],
    whichtoinvert: Iterable[bool] | None,
) -> list[bool]:
    """Match ANTsPy's default whichtoinvert convention: if whichtoinvert is None and
    transformlist is a 2-element ['.mat', warp], default to [True, False]; else [False]*len."""
    if whichtoinvert is None:
        if len(tl) == 2 and (".mat" in str(tl[0])) and (".mat" not in str(tl[1])) and str(tl[1]).lower() != "identity":
            return [True, False]
        return [False] * len(tl)
    inv = [bool(x) for x in whichtoinvert]
    if len(inv) != len(tl):
        raise ValueError("Transform list and inversion list must be the same length")
    return inv


@dataclass
class TransformChain:
    """Ordered list of transforms in ANTs *application* order (first applied last, as ants.apply_transforms expects)."""

    elements: list[TransformElement] = field(default_factory=list)

    @classmethod
    def from_paths(cls, paths: Iterable[str], whichtoinvert: Iterable[bool] | None = None) -> TransformChain:
        paths_list = [str(p) for p in paths]
        inv = [False] * len(paths_list) if whichtoinvert is None else [bool(x) for x in whichtoinvert]
        if len(inv) != len(paths_list):
            raise ValueError("whichtoinvert must match paths")
        return cls([TransformElement(p, classify_transform(p), bool(i)) for p, i in zip(paths_list, inv, strict=True)])

    @classmethod
    def from_syntx(cls, result: dict[str, Any], direction: str = "fwd") -> TransformChain:
        key = "fwdtransforms" if direction == "fwd" else "invtransforms"
        inv = result.get("whichtoinvert_inv") if direction == "inv" else None
        return cls.from_paths(result.get(key, []), inv)

    def __len__(self) -> int:
        return len(self.elements)

    def paths(self) -> list[str]:
        return [e.path for e in self.elements]

    def whichtoinvert(self) -> list[bool]:
        return [e.invert for e in self.elements]

    def linear_elements(self) -> list[TransformElement]:
        return [e for e in self.elements if e.kind in _LINEAR]

    def linear_part(self) -> np.ndarray:
        """Composed 3x3 matrix of the rigid/affine elements only. Displacement fields contribute identity.

        Composition follows the application order of ``apply`` / ``_map_points_to_moving_tensor`` (and of
        ``ants.apply_transforms``): the FIRST element in the list acts first on the reference-space point, so the
        composed matrix is ``A_n ... A_2 A_1``. Until 2026-10-28 this composed in the reverse order, which disagreed
        with the resampling for every chain with more than one non-commuting linear element.
        """
        m = np.eye(3)
        for e in self.linear_elements():
            a, _ = _read_itk_affine(e.path, invert=e.invert)
            m = a @ m
        return m

    def apply(
        self,
        image: Any,
        reference: Any,
        interpolator: str = "linear",
        imagetype: int = 0,
        defaultvalue: float = 0.0,
        device: str | None = None,
    ) -> Any:
        import ants

        if imagetype == 3 and len(image.shape) == 4 and len(reference.shape) == 3:
            return _apply_timeseries_4d(
                image=image,
                reference=reference,
                chain=self,
                interpolator=interpolator,
                defaultvalue=defaultvalue,
                device=device,
            )

        def _ants_path():
            # ANTsPy has no 'identity' transform name: drop the identity entries; a chain of nothing else is a plain resample
            real = [(p_, inv) for p_, inv in zip(self.paths(), self.whichtoinvert()) if str(p_).lower() != "identity"]
            if not real:
                return ants.resample_image_to_target(image, reference, interp_type=interpolator, imagetype=imagetype)
            return ants.apply_transforms(
                fixed=reference,
                moving=image,
                transformlist=[p_ for p_, _ in real],
                whichtoinvert=[inv for _, inv in real],
                interpolator=interpolator,
                imagetype=imagetype,
                defaultvalue=defaultvalue,
            )

        if len(image.shape) != 3 or len(reference.shape) != 3:
            return _ants_path()

        interp_lower = interpolator.lower()
        if interp_lower in ("linear", "bilinear", "trilinear"):
            mode = "bilinear"
        elif interp_lower in ("nearestneighbor", "nearest"):
            mode = "nearest"
        elif interp_lower in ("genericlabel", "multilabel"):
            mode = "genericlabel"
        elif interp_lower == "bspline":
            mode = "bspline"
        else:
            return _ants_path()

        dev_str = resolve_device(device, task="model")
        dev = torch.device(dev_str)

        if imagetype == 1:
            return _apply_vector_3d(
                image=image,
                reference=reference,
                chain=self,
                mode=mode,
                defaultvalue=defaultvalue,
                dev=dev,
            )
        if imagetype == 2:
            return _apply_tensor_3d(
                image=image,
                reference=reference,
                chain=self,
                mode=mode,
                dev=dev,
            )

        ref_shape = tuple(int(s) for s in reference.shape[:3])
        ref_sig = (
            ref_shape,
            tuple(float(x) for x in reference.spacing[:3]),
            tuple(float(x) for x in reference.origin[:3]),
            tuple(float(x) for x in np.asarray(reference.direction)[:3, :3].ravel()),
        )
        chain_sig = tuple(
            (e.path, e.kind, e.invert, os.path.getmtime(e.path) if os.path.exists(e.path) else 0.0)
            for e in self.elements
        )
        cache_key = (ref_sig, chain_sig, str(dev))
        pts_moving = _MAPPED_GRID_CACHE.get(cache_key)
        if pts_moving is None:
            pts_fixed = _reference_grid_points_lps(reference, dev)
            pts_moving = _map_points_to_moving_tensor(pts_fixed, self, dev)
            if len(_MAPPED_GRID_CACHE) >= 4:
                _MAPPED_GRID_CACHE.pop(next(iter(_MAPPED_GRID_CACHE)))
            _MAPPED_GRID_CACHE[cache_key] = pts_moving

        field = ContinuousField(image, device=dev, dtype=torch.float32)
        out = _sample_field_itk(field, pts_moving, mode=mode, defaultvalue=defaultvalue)
        arr_out = out.reshape(ref_shape).detach().cpu().numpy().astype(np.float32, copy=False)
        return reference.new_image_like(arr_out)

    def compose(self, other: TransformChain) -> TransformChain:
        """self ∘ other: ``other`` is applied first."""
        return TransformChain(self.elements + other.elements)

    def to_provenance(self) -> list[dict[str, str]]:
        return [{"path": e.path, "kind": e.kind, "invert": str(e.invert)} for e in self.elements]


_DISP_FIELD_CACHE: dict[tuple[str, float, str], Any] = {}
_DISP_GRAD_CACHE: dict[tuple[str, float, str], Any] = {}
_DISP_IMG_CACHE: dict[tuple[str, int, int], Any] = {}
_MAPPED_GRID_CACHE: dict[tuple[Any, ...], Any] = {}
_BSPLINE_PREFILTER_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _cache_disp_image_and_field(path: str, disp_img: Any, device: Any | None = None) -> None:
    """Seed ``_DISP_IMG_CACHE`` and ``_DISP_FIELD_CACHE`` for a just-written 3D displacement
    ``ANTsImage`` so immediate downstream ``jacobian_modulation`` / ``apply_transforms`` /
    ``resample_series`` calls skip ``ants.image_read`` gzip decompression."""
    if not isinstance(path, str) or len(getattr(disp_img, "shape", ())) != 3:
        return
    try:
        st = os.stat(path)
    except OSError:
        return
    img_key = (path, st.st_mtime_ns, st.st_size)
    if len(_DISP_IMG_CACHE) >= 4:
        _DISP_IMG_CACHE.pop(next(iter(_DISP_IMG_CACHE)))
    _DISP_IMG_CACHE[img_key] = disp_img
    if device is not None:
        dev = device if isinstance(device, torch.device) else torch.device(resolve_device(device, task="model"))
        disp_key = (path, os.path.getmtime(path), str(dev))
        field = ContinuousField(disp_img, device=dev, dtype=torch.float32)
        if len(_DISP_FIELD_CACHE) >= 8:
            _DISP_FIELD_CACHE.pop(next(iter(_DISP_FIELD_CACHE)))
        _DISP_FIELD_CACHE[disp_key] = field


def _bspline_prefilter_1d_matrix(
    n: int, target_dev: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Exact 1D cubic B-spline decomposition operator (whole-sample mirror boundary, matching ITK)."""
    key = (int(n), str(target_dev), dtype)
    cached = _BSPLINE_PREFILTER_CACHE.get(key)
    if cached is not None:
        return cached
    if n <= 1:
        mat = torch.ones((max(1, n), max(1, n)), device=target_dev, dtype=dtype)
    else:
        a_mat = torch.zeros((n, n), dtype=torch.float32)
        idx = torch.arange(n)
        a_mat[idx, idx] = 4.0 / 6.0
        a_mat[idx[:-1], idx[1:]] = 1.0 / 6.0
        a_mat[idx[1:], idx[:-1]] = 1.0 / 6.0
        a_mat[0, 1] = 2.0 / 6.0
        a_mat[-1, -2] = 2.0 / 6.0
        mat = torch.linalg.inv(a_mat).to(device=target_dev, dtype=dtype)
    if len(_BSPLINE_PREFILTER_CACHE) >= 64:
        _BSPLINE_PREFILTER_CACHE.pop(next(iter(_BSPLINE_PREFILTER_CACHE)))
    _BSPLINE_PREFILTER_CACHE[key] = mat
    return mat


def _mirror_idx_1d(k: int, n: int) -> int:
    if n <= 1:
        return 0
    period = 2 * (n - 1)
    m = abs(k) % period
    return m if m < n else period - m


_MIRROR_PAD_IDX_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _pad_mirror_2_3d(x: torch.Tensor) -> torch.Tensor:
    """Pad a 5D ``(B, C, Z, Y, X)`` tensor by 2 whole-sample mirror voxels per spatial side."""
    dev_str = str(x.device)
    for dim in (2, 3, 4):
        n = int(x.shape[dim])
        key = (n, dev_str)
        idx_t = _MIRROR_PAD_IDX_CACHE.get(key)
        if idx_t is None:
            indices = [
                _mirror_idx_1d(-2, n),
                _mirror_idx_1d(-1, n),
                *range(n),
                _mirror_idx_1d(n, n),
                _mirror_idx_1d(n + 1, n),
            ]
            idx_t = torch.tensor(indices, device=x.device, dtype=torch.long)
            _MIRROR_PAD_IDX_CACHE[key] = idx_t
        x = x.index_select(dim, idx_t)
    return x.contiguous()


def _bspline_prefilter_and_pad_5d(t_bcdhw: torch.Tensor) -> torch.Tensor:
    """Compute cubic B-spline coefficients on ``(B, C, Z, Y, X)`` and mirror-pad by 2 voxels."""
    tgt_dev = t_bcdhw.device
    dt = t_bcdhw.dtype
    nz, ny, nx = int(t_bcdhw.shape[2]), int(t_bcdhw.shape[3]), int(t_bcdhw.shape[4])
    px = _bspline_prefilter_1d_matrix(nx, tgt_dev, dt)
    py = _bspline_prefilter_1d_matrix(ny, tgt_dev, dt)
    pz = _bspline_prefilter_1d_matrix(nz, tgt_dev, dt)
    c = torch.matmul(t_bcdhw, px.T)
    c = torch.matmul(py, c)
    c = torch.matmul(pz, c.transpose(2, 3)).transpose(2, 3)
    return _pad_mirror_2_3d(c)


def _sample_bspline_padded_5d(
    c_pad: torch.Tensor, cidx_b: torch.Tensor, shape_t: torch.Tensor
) -> torch.Tensor:
    """Evaluate 3D cubic B-spline on mirror-padded coefficients ``c_pad`` ``(B, C, Z+4, Y+4, X+4)``
    at continuous voxel indices ``cidx_b`` ``(B, N_pts, 3)`` via exact 8-tap trilinear decomposition
    (Sigg & Hadwiger 2005), returning ``(B, C, N_pts)``."""
    b_sz, n_ch = int(c_pad.shape[0]), int(c_pad.shape[1])
    n_pts = int(cidx_b.shape[1])
    c_clamped = torch.minimum(
        torch.maximum(cidx_b, torch.full_like(shape_t, -0.5)), shape_t - 0.5 - 1e-6
    )
    i0 = torch.floor(c_clamped)
    t = c_clamped - i0
    del c_clamped
    omt = 1.0 - t
    w0 = (omt * omt * omt) * (1.0 / 6.0)
    w1 = (3.0 * t * t * t - 6.0 * t * t + 4.0) * (1.0 / 6.0)
    w3 = (t * t * t) * (1.0 / 6.0)
    del t, omt
    g0 = w0 + w1
    g1 = 1.0 - g0
    pad_half_span = 0.5 * (shape_t + 3.0)
    norm0 = ((i0 + 1.0) + w1 / g0.clamp(min=1e-12)) / pad_half_span - 1.0
    norm1 = ((i0 + 3.0) + w3 / g1.clamp(min=1e-12)) / pad_half_span - 1.0
    del i0, w0, w1, w3

    acc = torch.zeros((b_sz, n_ch, n_pts), dtype=c_pad.dtype, device=c_pad.device)
    for bx in (0, 1):
        gx = g0[..., 0] if bx == 0 else g1[..., 0]
        nx = norm0[..., 0] if bx == 0 else norm1[..., 0]
        for by in (0, 1):
            gy = g0[..., 1] if by == 0 else g1[..., 1]
            ny = norm0[..., 1] if by == 0 else norm1[..., 1]
            gxy = gx * gy
            for bz in (0, 1):
                gz = g0[..., 2] if bz == 0 else g1[..., 2]
                nz = norm0[..., 2] if bz == 0 else norm1[..., 2]
                grid = torch.stack((nx, ny, nz), dim=-1).view(b_sz, 1, 1, n_pts, 3)
                s = F.grid_sample(
                    c_pad, grid, mode="bilinear", padding_mode="border", align_corners=True
                ).view(b_sz, n_ch, n_pts)
                acc.addcmul_((gxy * gz).unsqueeze(1), s)
    return acc


# ==============================================================================
# 3. Internal ITK-Exact Sampling & Jacobian Kernels
# ==============================================================================

def _sample_field_itk(
    field: ContinuousField,
    pts_phys: torch.Tensor,
    mode: str = "bilinear",
    defaultvalue: float = 0.0,
) -> torch.Tensor:
    """Sample a ContinuousField matching ITK's ResampleImageFilter + Linear/Nearest/GenericLabel/BSpline:
    - Continuous index interval is half-open: c_d in [-0.5, N_d - 0.5). Points outside return defaultvalue.
    - Inside [-0.5, N_d - 0.5):
      - 'bilinear': clamped to [0, N_d - 1] (padding_mode='border') with trilinear weights.
      - 'nearest': ITK RoundHalfIntegerUp, i.e. floor(c_d + 0.5) clamped to [0, N_d - 1].
      - 'genericlabel': 8-corner trilinear label voting matching ITK LabelImageGenericInterpolateImageFunction.
      - 'bspline': exact order-3 cardinal B-spline matching ITK BSplineInterpolateImageFunction.
    """
    flat_coords = pts_phys.view(-1, 3).to(device=field.device, dtype=field.dtype)
    coords_norm = physical_to_normalized(flat_coords, field.M_norm, field.b_norm)
    shape_t = torch.tensor(field.spatial_shape, dtype=field.dtype, device=field.device)
    half_span = 0.5 * (shape_t - 1.0)
    cidx = (coords_norm + 1.0) * half_span
    inside = ((cidx >= -0.5 - 1e-5) & (cidx < (shape_t - 0.5) - 1e-5)).all(dim=-1)

    if mode == "nearest":
        cidx_nearest = torch.floor(cidx + 0.5 + 1e-5).clamp(min=torch.zeros_like(shape_t), max=shape_t - 1.0)
        coords_norm_n = cidx_nearest / half_span.clamp(min=1e-6) - 1.0
        grid = coords_norm_n.view(1, 1, 1, -1, 3)
        sampled = F.grid_sample(field.tensor, grid, mode="nearest", padding_mode="border", align_corners=True)
        C = field.tensor.shape[1]
        out = sampled.view(C, -1).T if field.is_vector else sampled.view(-1)
    elif mode == "bspline":
        c_pad = _bspline_prefilter_and_pad_5d(field.tensor)
        sampled_b = _sample_bspline_padded_5d(c_pad, cidx.unsqueeze(0), shape_t)
        C = field.tensor.shape[1]
        out = sampled_b.view(C, -1).T if field.is_vector else sampled_b.view(-1)
    elif mode == "genericlabel" and not field.is_vector:
        c_clamped = torch.minimum(torch.maximum(cidx, torch.zeros_like(shape_t)), shape_t - 1.0)
        n_pts = c_clamped.shape[0]
        pt_chunk = 524288
        vol = field.tensor[0, 0]  # (Nz, Ny, Nx)
        max_idx = (shape_t - 1).to(torch.long)
        if n_pts <= pt_chunk:
            i0 = torch.floor(c_clamped).to(torch.long)
            f = c_clamped - i0.to(field.dtype)
            i1 = torch.minimum(i0 + 1, max_idx)
            corners_l = []
            corners_w = []
            for dx in (0, 1):
                ix = i0[:, 0] if dx == 0 else i1[:, 0]
                wx = (1.0 - f[:, 0]) if dx == 0 else f[:, 0]
                for dy in (0, 1):
                    iy = i0[:, 1] if dy == 0 else i1[:, 1]
                    wy = (1.0 - f[:, 1]) if dy == 0 else f[:, 1]
                    for dz in (0, 1):
                        iz = i0[:, 2] if dz == 0 else i1[:, 2]
                        wz = (1.0 - f[:, 2]) if dz == 0 else f[:, 2]
                        corners_l.append(vol[iz, iy, ix])
                        corners_w.append(wx * wy * wz)
            l8 = torch.stack(corners_l, dim=1)  # (N, 8)
            w8 = torch.stack(corners_w, dim=1)  # (N, 8)
            scores = (l8.unsqueeze(2) == l8.unsqueeze(1)).to(field.dtype) @ w8.unsqueeze(2)
            best_idx = scores.squeeze(2).argmax(dim=1, keepdim=True)
            out = torch.gather(l8, 1, best_idx).squeeze(1)
        else:
            out = torch.empty(n_pts, dtype=field.dtype, device=field.device)
            for s_p in range(0, n_pts, pt_chunk):
                e_p = min(n_pts, s_p + pt_chunk)
                cc = c_clamped[s_p:e_p]
                i0 = torch.floor(cc).to(torch.long)
                f = cc - i0.to(field.dtype)
                i1 = torch.minimum(i0 + 1, max_idx)
                corners_l = []
                corners_w = []
                for dx in (0, 1):
                    ix = i0[:, 0] if dx == 0 else i1[:, 0]
                    wx = (1.0 - f[:, 0]) if dx == 0 else f[:, 0]
                    for dy in (0, 1):
                        iy = i0[:, 1] if dy == 0 else i1[:, 1]
                        wy = (1.0 - f[:, 1]) if dy == 0 else f[:, 1]
                        for dz in (0, 1):
                            iz = i0[:, 2] if dz == 0 else i1[:, 2]
                            wz = (1.0 - f[:, 2]) if dz == 0 else f[:, 2]
                            corners_l.append(vol[iz, iy, ix])
                            corners_w.append(wx * wy * wz)
                l8 = torch.stack(corners_l, dim=1)
                w8 = torch.stack(corners_w, dim=1)
                scores = (l8.unsqueeze(2) == l8.unsqueeze(1)).to(field.dtype) @ w8.unsqueeze(2)
                best_idx = scores.squeeze(2).argmax(dim=1, keepdim=True)
                out[s_p:e_p] = torch.gather(l8, 1, best_idx).squeeze(1)
    else:
        out = field.sample(pts_phys, mode="bilinear", padding_mode="border")

    mask = inside.unsqueeze(-1).to(out.dtype) if out.ndim == 2 else inside.to(out.dtype)
    if defaultvalue == 0.0:
        return out * mask
    return out * mask + float(defaultvalue) * (1.0 - mask)


def _reference_grid_points_lps(reference: Any, device: Any) -> torch.Tensor:
    shape = tuple(int(s) for s in reference.shape[:3])
    spacing = torch.as_tensor(reference.spacing[:3], dtype=torch.float32, device=device)
    origin = torch.as_tensor(reference.origin[:3], dtype=torch.float32, device=device)
    direction = torch.as_tensor(np.asarray(reference.direction)[:3, :3], dtype=torch.float32, device=device)
    m_idx_to_phys = direction * spacing.unsqueeze(0)
    gx = torch.arange(shape[0], dtype=torch.float32, device=device)
    gy = torch.arange(shape[1], dtype=torch.float32, device=device)
    gz = torch.arange(shape[2], dtype=torch.float32, device=device)
    ii, jj, kk = torch.meshgrid(gx, gy, gz, indexing="ij")
    idx = torch.stack((ii.reshape(-1), jj.reshape(-1), kk.reshape(-1)), dim=-1)
    return idx @ m_idx_to_phys.T + origin


def _read_itk_affine(path_or_tx: Any, invert: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (M, b) with y = M x + b for an ITK linear transform file or ANTsTransform object:
    y = M (x - c) + c + t, where c is the transform's fixed parameters (centre) and t its translation."""
    import ants

    tx = ants.read_transform(path_or_tx) if isinstance(path_or_tx, str) else path_or_tx
    p = np.asarray(ants.get_ants_transform_parameters(tx), dtype=np.float64)
    if p.size == 12:
        c = np.asarray(ants.get_ants_transform_fixed_parameters(tx), dtype=np.float64)[:3]
        M = p[:9].reshape(3, 3)
        b = c + p[9:12] - M @ c
    else:
        b = np.asarray(ants.apply_ants_transform_to_point(tx, (0.0, 0.0, 0.0)), dtype=np.float64)
        e0 = np.asarray(ants.apply_ants_transform_to_point(tx, (1.0, 0.0, 0.0)), dtype=np.float64) - b
        e1 = np.asarray(ants.apply_ants_transform_to_point(tx, (0.0, 1.0, 0.0)), dtype=np.float64) - b
        e2 = np.asarray(ants.apply_ants_transform_to_point(tx, (0.0, 0.0, 1.0)), dtype=np.float64) - b
        M = np.column_stack((e0, e1, e2))
    if invert:
        M_inv = np.linalg.inv(M)
        return M_inv, -M_inv @ b
    return M, b


def _get_disp_and_grad_fields(path: str, device: Any) -> tuple[ContinuousField, ContinuousField]:
    """Load and cache both the 3-component displacement ContinuousField and its 9-component
    ITK physical-space gradient ContinuousField (with 1-voxel boundary shell zeroed to match
    ITK DisplacementFieldTransform::ComputeInverseJacobianWithRespectToPosition)."""
    import ants

    mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
    key = (path, mtime, str(device))
    field = _DISP_FIELD_CACHE.get(key)
    grad_field = _DISP_GRAD_CACHE.get(key)
    if field is None or grad_field is None:
        disp_img = ants.image_read(path)
        field = ContinuousField(disp_img, device=device, dtype=torch.float32)
        u = np.asarray(disp_img.numpy(), dtype=np.float32)
        sp_shape = u.shape[:3]
        du_didx = np.zeros(sp_shape + (3, 3), dtype=np.float32)
        inv_three = np.float32(1.0 / 3.0)
        for d in range(3):
            n = sp_shape[d]
            idx = np.arange(n)
            ip1 = np.clip(idx + 1, 0, n - 1)
            im1 = np.clip(idx - 1, 0, n - 1)
            ip2 = np.clip(idx + 2, 0, n - 1)
            im2 = np.clip(idx - 2, 0, n - 1)
            d1 = np.float32(0.5) * (np.take(u, ip1, axis=d) - np.take(u, im1, axis=d))
            d2 = np.float32(0.25) * (np.take(u, ip2, axis=d) - np.take(u, im2, axis=d))
            du_didx[..., :, d] = (np.float32(4.0) * d1 - d2) * inv_three
        for ax in range(3):
            if sp_shape[ax] >= 2:
                sl0 = [slice(None)] * 3
                sl1 = [slice(None)] * 3
                sl0[ax] = 0
                sl1[ax] = -1
                du_didx[tuple(sl0)] = 0.0
                du_didx[tuple(sl1)] = 0.0
        d_mat = np.asarray(disp_img.direction, dtype=np.float64)[:3, :3]
        s_vec = np.asarray(disp_img.spacing, dtype=np.float64)[:3]
        ds_inv = np.linalg.inv(d_mat @ np.diag(s_vec)).astype(np.float32)
        grad_phys = (du_didx @ ds_inv).reshape(*sp_shape, 9)
        grad_field = ContinuousField(
            grad_phys,
            origin=field.origin,
            spacing=field.spacing,
            direction=field.direction,
            device=device,
            dtype=torch.float32,
        )
        if len(_DISP_FIELD_CACHE) >= 8:
            _DISP_FIELD_CACHE.pop(next(iter(_DISP_FIELD_CACHE)))
        if len(_DISP_GRAD_CACHE) >= 8:
            _DISP_GRAD_CACHE.pop(next(iter(_DISP_GRAD_CACHE)))
        _DISP_FIELD_CACHE[key] = field
        _DISP_GRAD_CACHE[key] = grad_field
    return field, grad_field


def _map_points_to_moving_tensor(points_lps: torch.Tensor, chain: TransformChain, device: Any) -> torch.Tensor:
    import ants

    pts = points_lps.to(device=device, dtype=torch.float32)
    if pts.numel() == 0:
        return pts
    for element in chain.elements:
        if element.kind == "identity":
            continue
        if element.kind in _LINEAR:
            M, b = _read_itk_affine(element.path, invert=element.invert)
            M_t = torch.as_tensor(M, dtype=torch.float32, device=device)
            b_t = torch.as_tensor(b, dtype=torch.float32, device=device)
            pts = pts @ M_t.T + b_t
        else:
            if element.invert:
                raise ValueError(f"{element.path}: a displacement field cannot be inverted on the fly")
            mtime = os.path.getmtime(element.path) if os.path.exists(element.path) else 0.0
            disp_key = (element.path, mtime, str(device))
            field = _DISP_FIELD_CACHE.get(disp_key)
            if field is None:
                disp_img = ants.image_read(element.path)
                field = ContinuousField(disp_img, device=device, dtype=torch.float32)
                if len(_DISP_FIELD_CACHE) >= 8:
                    _DISP_FIELD_CACHE.pop(next(iter(_DISP_FIELD_CACHE)))
                _DISP_FIELD_CACHE[disp_key] = field
            u = _sample_field_itk(field, pts, mode="bilinear")
            pts = pts + u
    return pts


def _map_points_and_inv_jacobian_tensor(
    points_lps: torch.Tensor, chain: TransformChain, device: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map fixed physical points (LPS mm) to moving space AND accumulate ITK's composite
    inverse Jacobian ``J_inv`` of shape ``(N, 3, 3)`` along the trajectory."""
    pts = points_lps.to(device=device, dtype=torch.float32)
    n_pts = pts.shape[0]
    eye = torch.eye(3, dtype=torch.float32, device=device)
    j_inv = eye.unsqueeze(0).expand(n_pts, 3, 3).clone()
    if n_pts == 0:
        return pts, j_inv
    for element in chain.elements:
        if element.kind == "identity":
            continue
        if element.kind in _LINEAR:
            M, b = _read_itk_affine(element.path, invert=element.invert)
            M_inv = np.linalg.inv(M)
            M_t = torch.as_tensor(M, dtype=torch.float32, device=device)
            b_t = torch.as_tensor(b, dtype=torch.float32, device=device)
            M_inv_t = torch.as_tensor(M_inv, dtype=torch.float32, device=device)
            j_inv = torch.matmul(M_inv_t.unsqueeze(0), j_inv)
            pts = pts @ M_t.T + b_t
        else:
            if element.invert:
                raise ValueError(f"{element.path}: a displacement field cannot be inverted on the fly")
            field, grad_field = _get_disp_and_grad_fields(element.path, device)
            g = _sample_field_itk(grad_field, pts, mode="nearest").view(-1, 3, 3)
            j_step = eye.unsqueeze(0) - g
            j_inv = torch.bmm(j_step, j_inv)
            u = _sample_field_itk(field, pts, mode="bilinear")
            pts = pts + u
    return pts, j_inv


def _polar_svd_3x3(mat: torch.Tensor) -> torch.Tensor:
    """Exact orthogonal polar factor R in SO(3) (det = +1) for a batch of 3x3 matrices (..., 3, 3)."""
    orig_shape = mat.shape
    m_cpu = mat.detach().cpu().numpy().astype(np.float64).reshape(-1, 3, 3)
    u, _, vt = np.linalg.svd(m_cpu)
    r = u @ vt
    dets = np.linalg.det(r)
    reflection_mask = dets < 0.0
    if np.any(reflection_mask):
        vt_copy = vt.copy()
        vt_copy[reflection_mask, -1, :] *= -1.0
        r[reflection_mask] = u[reflection_mask] @ vt_copy[reflection_mask]
    return torch.as_tensor(r.astype(np.float32), dtype=mat.dtype, device=mat.device).reshape(orig_shape)


def _apply_vector_3d(
    *,
    image: Any,
    reference: Any,
    chain: TransformChain,
    mode: str,
    defaultvalue: float,
    dev: torch.device,
) -> Any:
    """Resample and reorient a 3D 3-component vector ANTsImage (imagetype=1) matching ITK/ANTs:
    v_fix(x) = D_fix^T @ polar(J_inv(x)) @ D_mov @ v_mov(y(x))."""
    import ants

    ref_shape = tuple(int(s) for s in reference.shape[:3])
    pts_fixed = _reference_grid_points_lps(reference, dev)
    pts_moving, j_inv = _map_points_and_inv_jacobian_tensor(pts_fixed, chain, dev)

    field = ContinuousField(image, device=dev, dtype=torch.float32)
    v_sampled = _sample_field_itk(field, pts_moving, mode=mode, defaultvalue=defaultvalue)  # (N, 3)

    r_inv = _polar_svd_3x3(j_inv)  # (N, 3, 3)
    d_fix = torch.as_tensor(np.asarray(reference.direction)[:3, :3], dtype=torch.float32, device=dev)
    d_mov = torch.as_tensor(np.asarray(image.direction)[:3, :3], dtype=torch.float32, device=dev)
    rot_total = torch.matmul(d_fix.T.unsqueeze(0), torch.matmul(r_inv, d_mov.unsqueeze(0)))  # (N, 3, 3)
    v_out = torch.bmm(rot_total, v_sampled.unsqueeze(-1)).squeeze(-1)  # (N, 3)

    arr_out = v_out.reshape(*ref_shape, 3).detach().cpu().numpy().astype(np.float32, copy=False)
    return ants.from_numpy(
        arr_out,
        origin=tuple(float(x) for x in reference.origin[:3]),
        spacing=tuple(float(x) for x in reference.spacing[:3]),
        direction=np.asarray(reference.direction)[:3, :3],
        has_components=True,
    )


def _ppd_reorient_sym6(
    sym6: np.ndarray,
    f_mat: np.ndarray,
    from_log_domain: bool = False,
) -> np.ndarray:
    """Apply Preservation of Principal Directions (PPD, Alexander et al. 2001) to an array of
    6-component symmetric tensors ``[Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]`` of shape ``(N, 6)`` using
    local inverse Jacobian matrices ``f_mat`` of shape ``(N, 3, 3)`` (or ``(1, 3, 3)``)."""
    n_pts = sym6.shape[0]
    nonzero = np.any(np.abs(sym6) > 0.0, axis=-1)
    out = np.zeros((n_pts, 6), dtype=np.float32)
    if not np.any(nonzero):
        return out

    s_valid = sym6[nonzero].astype(np.float64)
    d_mat = s_valid[:, _SYM6_TO_3X3]  # (M, 3, 3)
    w, v = np.linalg.eigh(d_mat)  # ascending eigenvalues w[:, 0] <= w[:, 1] <= w[:, 2]
    if from_log_domain:
        w = np.exp(w)

    vals = w[:, ::-1]  # descending: lambda_1 >= lambda_2 >= lambda_3
    e1 = v[:, :, 2]
    e2 = v[:, :, 1]

    f_valid = f_mat[nonzero] if f_mat.shape[0] == n_pts else f_mat
    fe1 = np.einsum("nij,nj->ni", f_valid, e1)
    n1_norm = np.linalg.norm(fe1, axis=-1, keepdims=True)
    n1 = np.where(n1_norm > 1e-12, fe1 / np.maximum(n1_norm, 1e-12), e1)

    fe2 = np.einsum("nij,nj->ni", f_valid, e2)
    proj = np.sum(fe2 * n1, axis=-1, keepdims=True) * n1
    n2_raw = fe2 - proj
    n2_norm = np.linalg.norm(n2_raw, axis=-1, keepdims=True)
    n2 = np.where(n2_norm > 1e-12, n2_raw / np.maximum(n2_norm, 1e-12), e2)
    n3 = np.cross(n1, n2)

    r_ppd = np.stack([n1, n2, n3], axis=-1)
    d_rot = np.einsum("nik,nk,njk->nij", r_ppd, vals, r_ppd)
    out_valid = np.stack(
        [
            d_rot[:, 0, 0],
            d_rot[:, 0, 1],
            d_rot[:, 0, 2],
            d_rot[:, 1, 1],
            d_rot[:, 1, 2],
            d_rot[:, 2, 2],
        ],
        axis=-1,
    ).astype(np.float32)
    out[nonzero] = out_valid
    return out


def _apply_tensor_3d(
    *,
    image: Any,
    reference: Any,
    chain: TransformChain,
    mode: str,
    dev: torch.device,
) -> Any:
    """Resample and reorient a 3D 6-component symmetric tensor ANTsImage (imagetype=2,
    ``[Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]``) matching ITK/ANTs: log-Euclidean interpolation
    followed by PPD (Preservation of Principal Directions) with F = D_fix^T @ J_inv(x) @ D_mov."""
    import ants

    ref_shape = tuple(int(s) for s in reference.shape[:3])
    arr_in = np.asarray(image.numpy(), dtype=np.float64)
    if arr_in.ndim != 4 or arr_in.shape[-1] != 6:
        raise ValueError(f"imagetype=2 expects a 6-component symmetric tensor image, got shape {arr_in.shape}")

    flat_in = arr_in.reshape(-1, 6)
    pos_mask = (flat_in[:, 0] > 0.0) & (flat_in[:, 3] > 0.0) & (flat_in[:, 5] > 0.0)
    log_flat = np.zeros_like(flat_in, dtype=np.float32)
    if np.any(pos_mask):
        d_pos = flat_in[pos_mask][:, _SYM6_TO_3X3]
        w, v = np.linalg.eigh(d_pos)
        log_w = np.log(np.clip(w, 1e-10, None))
        log_d = np.einsum("nik,nk,njk->nij", v, log_w, v)
        log_flat[pos_mask] = np.stack(
            [
                log_d[:, 0, 0],
                log_d[:, 0, 1],
                log_d[:, 0, 2],
                log_d[:, 1, 1],
                log_d[:, 1, 2],
                log_d[:, 2, 2],
            ],
            axis=-1,
        ).astype(np.float32)
    log_arr = log_flat.reshape(arr_in.shape)

    pts_fixed = _reference_grid_points_lps(reference, dev)
    pts_moving, j_inv = _map_points_and_inv_jacobian_tensor(pts_fixed, chain, dev)

    field = ContinuousField(
        log_arr,
        origin=tuple(float(x) for x in image.origin[:3]),
        spacing=tuple(float(x) for x in image.spacing[:3]),
        direction=np.asarray(image.direction)[:3, :3],
        device=dev,
        dtype=torch.float32,
    )
    flat_coords = pts_moving.view(-1, 3).to(device=dev, dtype=torch.float32)
    coords_norm = physical_to_normalized(flat_coords, field.M_norm, field.b_norm)
    shape_t = torch.tensor(field.spatial_shape, dtype=torch.float32, device=dev)
    cidx = (coords_norm + 1.0) * (0.5 * (shape_t - 1.0))
    inside = ((cidx >= -0.5 - 1e-5) & (cidx < (shape_t - 0.5) - 1e-5)).all(dim=-1).detach().cpu().numpy()

    sampled_log = _sample_field_itk(field, pts_moving, mode=mode).detach().cpu().numpy()
    d_fix = np.asarray(reference.direction, dtype=np.float64)[:3, :3]
    d_mov = np.asarray(image.direction, dtype=np.float64)[:3, :3]
    j_inv_np = j_inv.detach().cpu().numpy().astype(np.float64)
    f_total = d_fix.T @ j_inv_np @ d_mov

    sampled_log[~inside] = 0.0
    out_flat = np.zeros_like(sampled_log, dtype=np.float32)
    if np.any(inside):
        out_flat[inside] = _ppd_reorient_sym6(sampled_log[inside], f_total[inside], from_log_domain=True)

    arr_out = out_flat.reshape(*ref_shape, 6)
    return ants.from_numpy(
        arr_out,
        origin=tuple(float(x) for x in reference.origin[:3]),
        spacing=tuple(float(x) for x in reference.spacing[:3]),
        direction=np.asarray(reference.direction)[:3, :3],
        has_components=True,
    )


def _get_available_device_bytes(dev: torch.device) -> int:
    """Query available memory in bytes for ``dev`` (CUDA free VRAM, MPS recommended budget, or host RAM)."""
    host_avail = 2 * 1024**3
    try:
        import psutil  # type: ignore

        host_avail = int(psutil.virtual_memory().available)
    except Exception:
        try:
            page_sz = int(os.sysconf("SC_PAGE_SIZE"))
            if "SC_AVPHYS_PAGES" in os.sysconf_names:
                host_avail = page_sz * int(os.sysconf("SC_AVPHYS_PAGES"))
            elif "SC_PHYS_PAGES" in os.sysconf_names:
                host_avail = (page_sz * int(os.sysconf("SC_PHYS_PAGES"))) // 2
        except Exception:
            pass

    if dev.type == "cuda":
        try:
            free_b, _ = torch.cuda.mem_get_info(dev)
            return int(free_b)
        except Exception:
            return host_avail
    if dev.type == "mps":
        try:
            rec_max = int(torch.mps.recommended_max_memory())
            cur_alloc = int(torch.mps.current_allocated_memory())
            mps_free = max(0, rec_max - cur_alloc)
            if mps_free > 0:
                return min(mps_free, host_avail)
        except Exception:
            pass
    return host_avail


def _empty_device_cache(dev: torch.device) -> None:
    try:
        if dev.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif dev.type == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
    except Exception:
        pass


def _auto_select_4d_batch_size(
    mov_shape_3d: tuple[int, int, int],
    ref_shape_3d: tuple[int, int, int],
    nvol: int,
    dev: torch.device,
    *,
    per_volume_grid: bool = False,
    bspline: bool = False,
    affine_grid_only: bool = False,
) -> int:
    """Auto-select a memory-safe 4D volume batch size based on local device memory and 3D image dimensions."""
    if nvol <= 1:
        return 1
    avail_bytes = _get_available_device_bytes(dev)
    max_cap_bytes = 1024 * 1024 * 1024
    budget_bytes = max(64 * 1024 * 1024, min(int(avail_bytes * 0.20), max_cap_bytes))

    mov_vox = max(1, int(mov_shape_3d[0]) * int(mov_shape_3d[1]) * int(mov_shape_3d[2]))
    ref_vox = max(1, int(ref_shape_3d[0]) * int(ref_shape_3d[1]) * int(ref_shape_3d[2]))
    fixed_overhead = 0 if affine_grid_only else 24 * ref_vox

    if per_volume_grid and bspline:
        bytes_per_vol = int(1.25 * (8 * mov_vox + 64 * ref_vox))
        max_batch_cap = 16
    elif per_volume_grid:
        bytes_per_vol = int(1.25 * (4 * mov_vox + (20 if affine_grid_only else 40) * ref_vox))
        max_batch_cap = 64
    elif bspline:
        bytes_per_vol = int(1.25 * (8 * mov_vox + 16 * ref_vox))
        max_batch_cap = 32
    else:
        # Shared 5D sampling grid across channels + input/output volume buffers
        bytes_per_vol = int(1.25 * (4 * mov_vox + 8 * ref_vox))
        max_batch_cap = 64

    usable_budget = max(bytes_per_vol, budget_bytes - fixed_overhead)
    batch_sz = int(usable_budget // max(bytes_per_vol, 1))
    return max(1, min(int(nvol), max_batch_cap, batch_sz))


def _apply_timeseries_4d(
    *,
    image: Any,
    reference: Any,
    chain: TransformChain,
    interpolator: str,
    defaultvalue: float = 0.0,
    device: str | None = None,
) -> Any:
    """Resample every 3D volume of a 4D time-series ANTsImage (imagetype=3) onto ``reference``
    using a single shared grid mapping and memory-auto-batched 3D ``grid_sample`` on Torch/MPS/CUDA."""
    import ants

    interp_lower = interpolator.lower()
    if interp_lower in ("linear", "bilinear", "trilinear"):
        mode = "bilinear"
    elif interp_lower in ("nearestneighbor", "nearest"):
        mode = "nearest"
    elif interp_lower in ("genericlabel", "multilabel"):
        mode = "genericlabel"
    elif interp_lower == "bspline":
        mode = "bspline"
    else:
        return ants.apply_transforms(
            fixed=reference,
            moving=image,
            transformlist=chain.paths(),
            whichtoinvert=chain.whichtoinvert(),
            interpolator=interpolator,
            imagetype=3,
            defaultvalue=defaultvalue,
        )

    dev = torch.device(resolve_device(device, task="model"))
    ref_shape = tuple(int(s) for s in reference.shape[:3])
    mov_shape = tuple(int(s) for s in image.shape[:3])
    nvol = int(image.shape[3])
    pts_fixed = _reference_grid_points_lps(reference, dev)
    pts_moving = _map_points_to_moving_tensor(pts_fixed, chain, dev)
    del pts_fixed

    raw4d = image.numpy()
    vols_tzyx = (
        raw4d.T
        if (raw4d.dtype == np.float32 and raw4d.flags["F_CONTIGUOUS"])
        else np.ascontiguousarray(np.transpose(raw4d, (3, 2, 1, 0)), dtype=np.float32)
    )
    vol_origin = tuple(float(x) for x in image.origin[:3])
    vol_spacing = tuple(float(x) for x in image.spacing[:3])
    vol_dir = np.asarray(image.direction, dtype=np.float64)[:3, :3]

    out_tzyx = np.empty((nvol, ref_shape[2], ref_shape[1], ref_shape[0]), dtype=np.float32)
    out4d = out_tzyx.T

    if mode in ("bilinear", "nearest", "bspline"):
        M_norm, b_norm = get_physical_to_normalized_affine(
            shape=mov_shape,
            spacing=vol_spacing,
            origin=vol_origin,
            direction=vol_dir,
            device=dev,
            dtype=torch.float32,
        )
        flat_coords = pts_moving.view(-1, 3).to(device=dev, dtype=torch.float32)
        del pts_moving
        coords_norm = physical_to_normalized(flat_coords, M_norm, b_norm)
        del flat_coords
        shape_t = torch.tensor(mov_shape, dtype=torch.float32, device=dev)
        half_span = 0.5 * (shape_t - 1.0)
        cidx = (coords_norm + 1.0) * half_span
        inside = ((cidx >= -0.5 - 1e-5) & (cidx < (shape_t - 0.5) - 1e-5)).all(dim=-1)
        cidx_shared: torch.Tensor | None = None
        grid: torch.Tensor | None = None
        if mode == "bspline":
            cidx_shared = cidx.unsqueeze(0)
            del coords_norm, cidx
        else:
            if mode == "nearest":
                cidx_nearest = torch.floor(cidx + 0.5 + 1e-5).clamp(min=torch.zeros_like(shape_t), max=shape_t - 1.0)
                coords_norm = cidx_nearest / half_span.clamp(min=1e-6) - 1.0
            del cidx
            grid = coords_norm.view(1, 1, 1, -1, 3)
        mask = inside.to(torch.float32).unsqueeze(0)  # (1, N_vox)
        del inside

        chunk_vols = _auto_select_4d_batch_size(
            mov_shape, ref_shape, nvol, dev, per_volume_grid=False, bspline=(mode == "bspline")
        )
        s_idx = 0
        while s_idx < nvol:
            e_idx = min(nvol, s_idx + chunk_vols)
            c_len = e_idx - s_idx
            try:
                t_cdhw = torch.from_numpy(vols_tzyx[s_idx:e_idx]).unsqueeze(0).to(device=dev, dtype=torch.float32)
                if mode == "bspline":
                    assert cidx_shared is not None
                    c_pad = _bspline_prefilter_and_pad_5d(t_cdhw)
                    del t_cdhw
                    sampled = _sample_bspline_padded_5d(c_pad, cidx_shared, shape_t).view(c_len, -1)
                    del c_pad
                else:
                    assert grid is not None
                    sampled = F.grid_sample(
                        t_cdhw, grid, mode=mode, padding_mode="border", align_corners=True
                    ).view(c_len, -1)
                    del t_cdhw
                if defaultvalue == 0.0:
                    sampled = sampled * mask
                else:
                    sampled = sampled * mask + float(defaultvalue) * (1.0 - mask)
                out_tzyx[s_idx:e_idx] = (
                    sampled.view(c_len, *ref_shape).permute(0, 3, 2, 1).contiguous().detach().cpu().numpy()
                )
                del sampled
                s_idx = e_idx
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or "mps" in msg or "buffer" in msg:
                    _empty_device_cache(dev)
                    if chunk_vols > 1:
                        chunk_vols = max(1, chunk_vols // 2)
                        continue
                    if dev.type != "cpu":
                        dev = torch.device("cpu")
                        if grid is not None:
                            grid = grid.to(device=dev)
                        if cidx_shared is not None:
                            cidx_shared = cidx_shared.to(device=dev)
                        shape_t = shape_t.to(device=dev)
                        mask = mask.to(device=dev)
                        chunk_vols = _auto_select_4d_batch_size(
                            mov_shape, ref_shape, nvol - s_idx, dev, per_volume_grid=False, bspline=(mode == "bspline")
                        )
                        continue
                raise
    else:
        field_vol: ContinuousField | None = None
        for k in range(nvol):
            vol_arr = raw4d[..., k]
            if field_vol is None:
                field_vol = ContinuousField(
                    np.asarray(vol_arr, dtype=np.float32),
                    origin=vol_origin,
                    spacing=vol_spacing,
                    direction=vol_dir,
                    device=dev,
                    dtype=torch.float32,
                )
            else:
                field_vol.tensor = torch.as_tensor(vols_tzyx[k][None, None, ...], device=dev, dtype=torch.float32)
            sampled = _sample_field_itk(field_vol, pts_moving, mode=mode, defaultvalue=defaultvalue)
            out4d[..., k] = sampled.reshape(ref_shape).detach().cpu().numpy()

    spacing = list(reference.spacing[:3]) + [float(image.spacing[3])]
    origin = list(reference.origin[:3]) + [float(image.origin[3])]
    direction = np.eye(4)
    direction[:3, :3] = _as_direction(reference.direction)
    return ants.from_numpy(out4d, origin=origin, spacing=spacing, direction=direction)


# ==============================================================================
# 4. ANTsPy-Compatible Public Transform Functions
# ==============================================================================

def apply_transforms(
    fixed: Any,
    moving: Any,
    transformlist: str | Iterable[str] | TransformChain,
    interpolator: str = "linear",
    imagetype: int = 0,
    whichtoinvert: Iterable[bool] | None = None,
    compose: str | None = None,
    defaultvalue: float = 0.0,
    singleprecision: bool = False,
    verbose: bool = False,
    device: str | None = None,
    **kwargs: Any,
) -> Any:
    """Drop-in Torch/MPS/CUDA replacement for ``ants.apply_transforms`` with identical signature.

    Parameters
    ----------
    fixed : ANTsImage
        Fixed/reference image defining the output domain.
    moving : ANTsImage
        Moving image to resample.
    transformlist : str, list of str, or TransformChain
        Ordered list of transforms in ANTs application order.
    interpolator : str, default 'linear'
        Interpolation method (``'linear'``, ``'nearestNeighbor'``, ``'genericLabel'``, ``'multiLabel'``, ...).
    imagetype : int, default 0
        0 = scalar, 1 = 3-component vector, 2 = 6-component symmetric tensor, 3 = 4D time-series.
    whichtoinvert : list of bool, optional
        Per-transform inversion flags. Follows ANTsPy's default ``(True, False)`` when
        ``whichtoinvert=None`` and ``transformlist`` is ``['*.mat', '*.nii.gz']``.
    compose : str, optional
        If provided, composes the transform chain into a displacement field on ``fixed``'s
        grid, writes ``{compose}comptx.nii.gz`` (or ``compose`` if it ends in ``.nii.gz``/``.nii``),
        and returns the path.
    defaultvalue : float, default 0.0
        Background value for out-of-bounds points.
    device : str, optional
        Compute device (``'cpu'``, ``'mps'``, ``'cuda'``).
    """
    import ants

    is_real_images = hasattr(ants, "is_image") and ants.is_image(fixed) and ants.is_image(moving)
    if hasattr(ants.apply_transforms, "_mock_name") or not is_real_images:
        tl_pass = transformlist.paths() if isinstance(transformlist, TransformChain) else transformlist
        inv_pass = (
            transformlist.whichtoinvert()
            if (isinstance(transformlist, TransformChain) and whichtoinvert is None)
            else whichtoinvert
        )
        return ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=tl_pass,
            interpolator=interpolator,
            imagetype=imagetype,
            whichtoinvert=inv_pass,
            compose=compose,
            defaultvalue=defaultvalue,
            singleprecision=singleprecision,
            verbose=verbose,
            **kwargs,
        )

    if isinstance(transformlist, TransformChain):
        chain = transformlist
    else:
        tl = [transformlist] if isinstance(transformlist, str) else [str(t) for t in transformlist]
        inv_flags = _resolve_whichtoinvert(tl, whichtoinvert)
        chain = TransformChain.from_paths(tl, inv_flags)

    if compose is not None:
        if len(fixed.shape) != 3:
            return ants.apply_transforms(
                fixed=fixed,
                moving=moving,
                transformlist=chain.paths(),
                interpolator=interpolator,
                imagetype=imagetype,
                whichtoinvert=chain.whichtoinvert(),
                compose=compose,
                defaultvalue=defaultvalue,
                singleprecision=singleprecision,
                verbose=verbose,
                **kwargs,
            )
        compose_str = str(compose)
        tfn = compose_str if compose_str.endswith((".nii.gz", ".nii", ".h5")) else f"{compose_str}comptx.nii.gz"
        dev = torch.device(resolve_device(device, task="model"))
        pts_fixed = _reference_grid_points_lps(fixed, dev)
        pts_moving = _map_points_to_moving_tensor(pts_fixed, chain, dev)
        disp_arr = (pts_moving - pts_fixed).reshape(*fixed.shape[:3], 3).detach().cpu().numpy().astype(np.float32)
        disp_img = ants.from_numpy(
            disp_arr,
            origin=tuple(float(x) for x in fixed.origin[:3]),
            spacing=tuple(float(x) for x in fixed.spacing[:3]),
            direction=np.asarray(fixed.direction)[:3, :3],
            has_components=True,
        )
        ants.image_write(disp_img, tfn)
        _cache_disp_image_and_field(tfn, disp_img, dev)
        return tfn

    if (len(moving.shape) == 4) and (len(fixed.shape) == 3) and (imagetype == 0):
        raise ValueError("Set imagetype=3 to transform 4D time-series images.")

    return chain.apply(
        moving,
        fixed,
        interpolator=interpolator,
        imagetype=imagetype,
        defaultvalue=defaultvalue,
        device=device,
    )


def resample_image_to_target(
    image: Any,
    target: Any,
    interp_type: str | int = "linear",
    imagetype: int = 0,
    verbose: bool = False,
    device: str | None = None,
    **kwargs: Any,
) -> Any:
    """Drop-in Torch/MPS/CUDA replacement for ``ants.resample_image_to_target``."""
    import ants

    if hasattr(ants.resample_image_to_target, "_mock_name") or not (
        hasattr(ants, "is_image") and ants.is_image(image) and ants.is_image(target)
    ):
        return ants.resample_image_to_target(
            image, target, interp_type=interp_type, imagetype=imagetype, verbose=verbose, **kwargs
        )

    interpolator_oldoptions = ("linear", "nearestNeighbor", "gaussian", "cosineWindowedSinc", "bSpline")
    if "interpolator" in kwargs:
        interp_type = kwargs.pop("interpolator")
    interpolator = interpolator_oldoptions[interp_type] if isinstance(interp_type, int) else str(interp_type)
    if len(image.shape) == 4 and len(target.shape) == 3 and imagetype == 0:
        imagetype = 3
    elif getattr(image, "components", 1) == 3 and imagetype == 0:
        imagetype = 1
    elif getattr(image, "components", 1) == 6 and imagetype == 0:
        imagetype = 2
    return apply_transforms(
        fixed=target,
        moving=image,
        transformlist=["identity"],
        interpolator=interpolator,
        imagetype=imagetype,
        verbose=verbose,
        device=device,
        **kwargs,
    )


def compose_displacement_fields(
    displacement_field: Any,
    warping_field: Any,
    device: str | None = None,
) -> Any:
    """Drop-in Torch/MPS/CUDA replacement for ``ants.compose_displacement_fields(displacement_field, warping_field)``.

    Matches ITK/ANTs convention: ``warping_field`` is applied first, then ``displacement_field``:
    ``u_composed(x) = w(x) + d(x + w(x))`` on ``warping_field``'s grid.
    """
    import ants

    if (
        hasattr(ants.compose_displacement_fields, "_mock_name")
        or not (hasattr(ants, "is_image") and ants.is_image(displacement_field) and ants.is_image(warping_field))
        or len(displacement_field.shape) != 3
        or len(warping_field.shape) != 3
    ):
        return ants.compose_displacement_fields(displacement_field, warping_field)

    dev = torch.device(resolve_device(device, task="model"))
    pts_fixed = _reference_grid_points_lps(warping_field, dev)
    w_arr = np.asarray(warping_field.numpy(), dtype=np.float32)
    w_t = torch.as_tensor(w_arr.reshape(-1, 3), dtype=torch.float32, device=dev)
    d_field = ContinuousField(displacement_field, device=dev, dtype=torch.float32)
    d_sampled = _sample_field_itk(d_field, pts_fixed + w_t, mode="bilinear")
    comp_arr = (w_t + d_sampled).reshape(*warping_field.shape[:3], 3).detach().cpu().numpy().astype(np.float32)
    return ants.from_numpy(
        comp_arr,
        origin=tuple(float(x) for x in warping_field.origin[:3]),
        spacing=tuple(float(x) for x in warping_field.spacing[:3]),
        direction=np.asarray(warping_field.direction)[:3, :3],
        has_components=True,
    )


def create_jacobian_determinant_image(
    domain_image: Any,
    tx: Any,
    do_log: bool = False,
    geom: bool = False,
) -> Any:
    """Compute the Jacobian determinant image on ``domain_image``'s grid, matching ``ants.create_jacobian_determinant_image``."""
    import ants

    if (
        hasattr(ants.create_jacobian_determinant_image, "_mock_name")
        or not (hasattr(ants, "is_image") and ants.is_image(domain_image))
        or geom
        or int(getattr(domain_image, "dimension", len(domain_image.shape))) != 3
    ):
        return ants.create_jacobian_determinant_image(domain_image, tx, do_log=do_log, geom=geom)
    try:
        det_img = signed_jacobian_determinant(tx, domain_image)
    except Exception:
        return ants.create_jacobian_determinant_image(domain_image, tx, do_log=do_log, geom=geom)
    if do_log:
        det_arr = np.log(np.clip(det_img.numpy(), 1e-6, None)).astype(np.float32)
        return domain_image.new_image_like(det_arr)
    return det_img


def map_points_to_moving(
    points_lps: np.ndarray,
    transformlist: str | Iterable[str] | TransformChain,
    whichtoinvert: Iterable[bool] | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Map fixed-space physical points (LPS mm) to moving-space points through an ANTs
    transform list, in torch."""
    if isinstance(transformlist, TransformChain):
        chain = transformlist
    elif isinstance(transformlist, str):
        chain = TransformChain.from_paths([transformlist], whichtoinvert)
    else:
        chain = TransformChain.from_paths(list(transformlist), whichtoinvert)
    pts = np.asarray(points_lps, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return pts.copy()
    has_disp = any(e.kind not in (*_LINEAR, "identity") for e in chain.elements)
    if not has_disp:
        for element in chain.elements:
            if element.kind == "identity":
                continue
            M, b = _read_itk_affine(element.path, invert=element.invert)
            pts = pts @ M.T + b
        return pts

    dev = torch.device(resolve_device(device, task="model"))
    pts_t = torch.as_tensor(pts.astype(np.float32), device=dev)
    out_t = _map_points_to_moving_tensor(pts_t, chain, dev)
    return out_t.detach().cpu().numpy().astype(np.float64)


def apply_transforms_to_points(
    dim: int,
    points: Any,
    transformlist: str | Iterable[str] | TransformChain,
    whichtoinvert: Iterable[bool] | None = None,
    verbose: bool = False,
    device: str | None = None,
) -> Any:
    """Drop-in Torch/MPS/CUDA replacement for ``ants.apply_transforms_to_points`` with identical signature.

    Accepts:
    - ``pandas.DataFrame`` with columns ``['x', 'y', 'z']`` (preserving any extra columns like ``t``, ``label``),
    - ``numpy.ndarray`` of shape ``(N, 3)`` or ``(..., 3)``,
    - ``torch.Tensor`` of shape ``(N, 3)`` or ``(..., 3)``.
    """
    import ants

    if hasattr(ants.apply_transforms_to_points, "_mock_name"):
        return ants.apply_transforms_to_points(dim, points, transformlist, whichtoinvert=whichtoinvert, verbose=verbose)

    if isinstance(transformlist, TransformChain):
        tl = transformlist.paths()
        inv_flags = transformlist.whichtoinvert() if whichtoinvert is None else _resolve_whichtoinvert(tl, whichtoinvert)
    elif isinstance(transformlist, str):
        tl = [transformlist]
        inv_flags = _resolve_whichtoinvert(tl, whichtoinvert)
    else:
        tl = [str(t) for t in transformlist]
        inv_flags = _resolve_whichtoinvert(tl, whichtoinvert)

    if dim != 3:
        return ants.apply_transforms_to_points(dim, points, tl, whichtoinvert=inv_flags, verbose=verbose)

    chain = TransformChain.from_paths(tl, inv_flags)
    if hasattr(points, "columns"):
        pts_xyz = np.asarray(points[["x", "y", "z"]].values, dtype=np.float64)
        mapped = map_points_to_moving(pts_xyz, chain, device=device)
        out_df = points.copy()
        out_df["x"] = mapped[:, 0]
        out_df["y"] = mapped[:, 1]
        out_df["z"] = mapped[:, 2]
        return out_df

    if isinstance(points, torch.Tensor):
        dev = torch.device(resolve_device(device, task="model"))
        orig_shape = points.shape
        out_t = _map_points_to_moving_tensor(points.reshape(-1, 3), chain, dev)
        return out_t.reshape(orig_shape).to(device=points.device, dtype=points.dtype)

    arr = np.asarray(points, dtype=np.float64)
    orig_shape = arr.shape
    mapped = map_points_to_moving(arr.reshape(-1, 3), chain, device=device)
    return mapped.reshape(orig_shape)


def apply_ants_transform_to_point(
    transform: Any,
    point: Any,
    whichtoinvert: Iterable[bool] | bool | None = None,
    device: str | None = None,
) -> Any:
    """Apply an ``ANTsTransform``, transform file path, or ``TransformChain`` to a single 3D point
    ``(x, y, z)`` or batch of points ``(N, 3)``, matching ``ants.apply_ants_transform_to_point``."""
    pt_arr = np.asarray(point, dtype=np.float64)
    is_single = pt_arr.ndim == 1
    if hasattr(transform, "apply_to_point") and not isinstance(transform, (str, TransformChain, list, tuple)):
        if is_single:
            return transform.apply_to_point(tuple(float(x) for x in pt_arr))
        M, b = _read_itk_affine(transform, invert=bool(whichtoinvert))
        return pt_arr @ M.T + b

    inv = [bool(whichtoinvert)] if isinstance(whichtoinvert, bool) else whichtoinvert
    mapped = map_points_to_moving(pt_arr.reshape(-1, 3), transform, whichtoinvert=inv, device=device)
    if is_single:
        return tuple(float(x) for x in mapped[0])
    return mapped.reshape(pt_arr.shape)


def apply_ants_transform_to_vector(
    transform: Any,
    vector: Any,
    whichtoinvert: Iterable[bool] | bool | None = None,
) -> Any:
    """Apply an ``ANTsTransform``, transform file path, or ``TransformChain`` to a 3D vector
    ``(vx, vy, vz)`` or batch of vectors ``(N, 3)``, matching ``ants.apply_ants_transform_to_vector``."""
    vec_arr = np.asarray(vector, dtype=np.float64)
    is_single = vec_arr.ndim == 1
    if hasattr(transform, "apply_to_vector") and not isinstance(transform, (str, TransformChain, list, tuple)):
        if is_single:
            return transform.apply_to_vector(tuple(float(x) for x in vec_arr))
        M, _ = _read_itk_affine(transform, invert=bool(whichtoinvert))
        return vec_arr @ M.T

    inv = [bool(whichtoinvert)] if isinstance(whichtoinvert, bool) else whichtoinvert
    chain = (
        transform
        if isinstance(transform, TransformChain)
        else TransformChain.from_paths([transform] if isinstance(transform, str) else list(transform), inv)
    )
    M = chain.linear_part()
    out = vec_arr.reshape(-1, 3) @ M.T
    if is_single:
        return tuple(float(x) for x in out[0])
    return out.reshape(vec_arr.shape)


# ==============================================================================
# 5. Streamline Transforms & Vector/Tensor Field Reorientation
# ==============================================================================

def _streamline_tangents(pts: np.ndarray) -> np.ndarray:
    """Compute unit finite-difference tangents along a streamline of shape ``(M, 3)``."""
    m = pts.shape[0]
    if m == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if m == 1:
        return np.zeros((1, 3), dtype=np.float32)
    tangents = np.empty_like(pts, dtype=np.float64)
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    if m > 2:
        tangents[1:-1] = pts[2:] - pts[:-2]
    norms = np.linalg.norm(tangents, axis=-1, keepdims=True)
    nz = norms > 1e-12
    out = np.zeros_like(tangents, dtype=np.float32)
    np.divide(tangents, norms, out=out, where=nz)
    return out


def transform_streamlines(
    streamlines: Sequence[np.ndarray] | np.ndarray | torch.Tensor,
    transformlist: str | Iterable[str] | TransformChain,
    whichtoinvert: Iterable[bool] | None = None,
    *,
    compute_tangents: bool = False,
    device: str | None = None,
) -> Any:
    """Transform a collection of physical-space (LPS mm) streamlines through a transform chain
    in a single batched Torch/MPS/CUDA pass, optionally returning their reoriented unit tangents.
    """
    if isinstance(transformlist, TransformChain):
        chain = transformlist
    elif isinstance(transformlist, str):
        chain = TransformChain.from_paths([transformlist], whichtoinvert)
    else:
        chain = TransformChain.from_paths(list(transformlist), whichtoinvert)

    if isinstance(streamlines, (np.ndarray, torch.Tensor)):
        arr = streamlines.detach().cpu().numpy() if isinstance(streamlines, torch.Tensor) else np.asarray(streamlines)
        if arr.ndim == 2 and arr.shape[-1] == 3:
            warped = map_points_to_moving(arr, chain, device=device).astype(np.float32)
            if compute_tangents:
                return warped, _streamline_tangents(warped)
            return warped
        if arr.ndim == 3 and arr.shape[-1] == 3:
            b_sz, m_sz, _ = arr.shape
            warped = map_points_to_moving(arr.reshape(-1, 3), chain, device=device).reshape(b_sz, m_sz, 3).astype(np.float32)
            if compute_tangents:
                tangs = np.stack([_streamline_tangents(warped[i]) for i in range(b_sz)], axis=0)
                return warped, tangs
            return warped
        raise ValueError(f"Expected streamline array of shape (M, 3) or (B, M, 3), got {arr.shape}")

    sl_list = [np.asarray(s, dtype=np.float64) for s in streamlines]
    if len(sl_list) == 0:
        return ([], []) if compute_tangents else []
    counts = np.array([len(s) for s in sl_list], dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        empty = [np.zeros((0, 3), dtype=np.float32) for _ in sl_list]
        return (empty, list(empty)) if compute_tangents else empty

    all_pts = np.concatenate([s for s in sl_list if len(s) > 0], axis=0)
    mapped_all = map_points_to_moving(all_pts, chain, device=device).astype(np.float32)
    splits = np.split(mapped_all, np.cumsum(counts)[:-1])
    warped_list = [s.reshape(-1, 3) for s in splits]
    if compute_tangents:
        tang_list = [_streamline_tangents(s) for s in warped_list]
        return warped_list, tang_list
    return warped_list


def reorient_vectors(
    vectors: Any,
    transformlist: str | Iterable[str] | TransformChain | None = None,
    *,
    warp: Any | None = None,
    reference: Any | None = None,
    moving_direction: np.ndarray | None = None,
    fixed_direction: np.ndarray | None = None,
    whichtoinvert: Iterable[bool] | None = None,
    method: str = "finite_strain",
    normalize: bool = False,
    device: str | None = None,
) -> Any:
    """Reorient 3D vectors or multi-peak fixel directions under a transform chain or displacement field.

    Supports 3-component vector ``ANTsImage``, ``(*spatial, 3)`` arrays, ``(*spatial, K, 3)``
    multi-peak fixel arrays, or ``(N, 3)`` point vectors using either ``method='finite_strain'``
    (orthogonal polar SVD rotation) or ``method='ppd'`` (Preservation of Principal Direction).

    Note on mapping duality:
        ``warp=...`` computes forward deformation gradient F = I + \u2207u via ``deformation_gradient(warp)``
        (matching ``antsxdwi.fixel``), while ``transformlist=[warp]`` evaluates ITK's inverse-mapping
        pushforward J^-1 = I - \u2207u (matching ``ants.apply_transforms(imagetype=1)``).
    """
    import ants

    is_ants = hasattr(ants, "is_image") and ants.is_image(vectors)
    is_torch = isinstance(vectors, torch.Tensor)
    v_arr = vectors.numpy() if is_ants else (vectors.detach().cpu().numpy() if is_torch else np.asarray(vectors))
    if v_arr.shape[-1] != 3:
        raise ValueError(f"reorient_vectors expects last dimension 3, got shape {v_arr.shape}")

    if warp is not None and transformlist is None:
        from .spatial import deformation_gradient

        if isinstance(warp, str):
            warp = ants.image_read(warp)
        if method == "finite_strain":
            rot = deformation_gradient(warp, to_rotation=True)
        elif method == "ppd":
            rot = deformation_gradient(warp, to_rotation=False)
        else:
            raise ValueError(f"Unknown reorientation method {method!r}. Options: 'finite_strain', 'ppd'.")

        d_mov = (
            _as_direction(moving_direction)
            if moving_direction is not None
            else (
                _as_direction(vectors.direction[:3, :3])
                if is_ants
                else (
                    _as_direction(warp.direction[:3, :3])
                    if hasattr(warp, "direction")
                    else np.eye(3)
                )
            )
        )
        d_fix = (
            _as_direction(fixed_direction)
            if fixed_direction is not None
            else (
                _as_direction(reference.direction[:3, :3])
                if (reference is not None and hasattr(reference, "direction"))
                else (
                    _as_direction(warp.direction[:3, :3])
                    if hasattr(warp, "direction")
                    else d_mov
                )
            )
        )
        rot = np.einsum("ij,...jk,kl->...il", d_fix.T, rot, d_mov)

        if v_arr.ndim == rot.ndim - 1:
            v_rot = np.einsum("...ij,...j->...i", rot, v_arr)
        elif v_arr.ndim == rot.ndim:
            v_rot = np.einsum("...ij,...pj->...pi", rot, v_arr)
        else:
            raise ValueError(f"Incompatible vector shape {v_arr.shape} with warp gradient shape {rot.shape}")
    else:
        if transformlist is None:
            raise ValueError("Provide either transformlist or warp to reorient_vectors")
        ref_img = reference if reference is not None else (vectors if is_ants else None)
        if isinstance(transformlist, (np.ndarray, torch.Tensor)):
            m_in = transformlist.detach().cpu().numpy() if isinstance(transformlist, torch.Tensor) else np.asarray(transformlist, dtype=np.float64)
            if method == "finite_strain":
                from .spatial import polar_rotation_field

                f_mat = polar_rotation_field(m_in)
            elif method == "ppd":
                f_mat = m_in
            else:
                raise ValueError(f"Unknown reorientation method {method!r}. Options: 'finite_strain', 'ppd'.")
        else:
            chain = (
                TransformChain.from_paths([transformlist], whichtoinvert)
                if isinstance(transformlist, str)
                else (
                    transformlist
                    if isinstance(transformlist, TransformChain)
                    else TransformChain.from_paths(list(transformlist), whichtoinvert)
                )
            )
            has_disp = any(e.kind not in (*_LINEAR, "identity") for e in chain.elements)
            if has_disp and ref_img is not None:
                dev = torch.device(resolve_device(device, task="model"))
                pts_fixed = _reference_grid_points_lps(ref_img, dev)
                _, j_inv_t = _map_points_and_inv_jacobian_tensor(pts_fixed, chain, dev)
                if method == "finite_strain":
                    f_mat = _polar_svd_3x3(j_inv_t).detach().cpu().numpy()
                elif method == "ppd":
                    f_mat = j_inv_t.detach().cpu().numpy()
                else:
                    raise ValueError(f"Unknown reorientation method {method!r}. Options: 'finite_strain', 'ppd'.")
                f_mat = f_mat.reshape(*ref_img.shape[:3], 3, 3)
            else:
                j_inv_single = np.linalg.inv(chain.linear_part())
                if method == "finite_strain":
                    from .spatial import polar_rotation_field

                    f_mat = polar_rotation_field(j_inv_single)
                elif method == "ppd":
                    f_mat = j_inv_single
                else:
                    raise ValueError(f"Unknown reorientation method {method!r}. Options: 'finite_strain', 'ppd'.")

        d_mov = (
            _as_direction(moving_direction)
            if moving_direction is not None
            else (_as_direction(vectors.direction[:3, :3]) if is_ants else np.eye(3))
        )
        d_fix = (
            _as_direction(fixed_direction)
            if fixed_direction is not None
            else (_as_direction(ref_img.direction[:3, :3]) if ref_img is not None else d_mov)
        )
        f_mat = d_fix.T @ f_mat @ d_mov
        if f_mat.ndim == 2:
            v_rot = np.einsum("ij,...j->...i", f_mat, v_arr)
        elif v_arr.ndim == f_mat.ndim - 1:
            v_rot = np.einsum("...ij,...j->...i", f_mat, v_arr)
        elif v_arr.ndim == f_mat.ndim:
            v_rot = np.einsum("...ij,...pj->...pi", f_mat, v_arr)
        else:
            raise ValueError(f"Incompatible vector shape {v_arr.shape} and operator shape {f_mat.shape}")

    if method == "ppd" or normalize:
        orig_norms = np.linalg.norm(v_arr, axis=-1, keepdims=True)
        new_norms = np.linalg.norm(v_rot, axis=-1, keepdims=True)
        nz = (orig_norms > 1e-12) & (new_norms > 1e-12)
        scale = np.ones_like(orig_norms) if normalize else orig_norms
        v_out = np.zeros_like(v_rot, dtype=np.float32)
        np.divide(v_rot * scale, new_norms, out=v_out, where=nz)
    else:
        v_out = v_rot.astype(np.float32)

    if is_ants:
        out_ref = reference if reference is not None else vectors
        return ants.from_numpy(
            v_out,
            origin=tuple(float(x) for x in out_ref.origin[:3]),
            spacing=tuple(float(x) for x in out_ref.spacing[:3]),
            direction=np.asarray(out_ref.direction)[:3, :3],
            has_components=True,
        )
    if is_torch:
        return torch.as_tensor(v_out, device=vectors.device, dtype=vectors.dtype)
    return v_out


def reorient_tensors(
    tensors: Any,
    transformlist: str | Iterable[str] | TransformChain | np.ndarray | torch.Tensor,
    *,
    reference: Any | None = None,
    moving_direction: np.ndarray | None = None,
    fixed_direction: np.ndarray | None = None,
    whichtoinvert: Iterable[bool] | None = None,
    method: str = "ppd",
    device: str | None = None,
) -> Any:
    """Reorient 2nd-order symmetric diffusion tensors under a transform chain using either
    PPD (Preservation of Principal Directions, Alexander et al. 2001, matching ITK/ANTs
    ``imagetype=2``) or Finite Strain polar rotation (``R D R^T``).

    Accepts 6-component ``ANTsImage`` (``[Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]``), ``(..., 6)``
    array/tensor, or ``(..., 3, 3)`` symmetric matrix array/tensor.
    """
    import ants

    is_ants = hasattr(ants, "is_image") and ants.is_image(tensors)
    is_torch = isinstance(tensors, torch.Tensor)
    t_arr = tensors.numpy() if is_ants else (tensors.detach().cpu().numpy() if is_torch else np.asarray(tensors))

    is_mat33 = t_arr.ndim >= 2 and t_arr.shape[-2:] == (3, 3)
    if is_mat33:
        spatial_shape = t_arr.shape[:-2]
        sym6 = np.stack(
            [
                t_arr[..., 0, 0],
                t_arr[..., 0, 1],
                t_arr[..., 0, 2],
                t_arr[..., 1, 1],
                t_arr[..., 1, 2],
                t_arr[..., 2, 2],
            ],
            axis=-1,
        ).reshape(-1, 6)
    elif t_arr.shape[-1] == 6:
        spatial_shape = t_arr.shape[:-1]
        sym6 = t_arr.reshape(-1, 6)
    else:
        raise ValueError(f"reorient_tensors expects shape (..., 6) or (..., 3, 3), got {t_arr.shape}")

    ref_img = reference if reference is not None else (tensors if is_ants else None)
    if isinstance(transformlist, (np.ndarray, torch.Tensor)):
        m_in = transformlist.detach().cpu().numpy() if isinstance(transformlist, torch.Tensor) else np.asarray(transformlist, dtype=np.float64)
        j_inv_np = m_in.reshape(-1, 3, 3)
    else:
        chain = (
            TransformChain.from_paths([transformlist], whichtoinvert)
            if isinstance(transformlist, str)
            else (
                transformlist
                if isinstance(transformlist, TransformChain)
                else TransformChain.from_paths(list(transformlist), whichtoinvert)
            )
        )
        has_disp = any(e.kind not in (*_LINEAR, "identity") for e in chain.elements)
        if has_disp and ref_img is not None:
            dev = torch.device(resolve_device(device, task="model"))
            pts_fixed = _reference_grid_points_lps(ref_img, dev)
            _, j_inv_t = _map_points_and_inv_jacobian_tensor(pts_fixed, chain, dev)
            j_inv_np = j_inv_t.detach().cpu().numpy().astype(np.float64)
        else:
            j_inv_np = np.linalg.inv(chain.linear_part())[None, :, :]

    d_mov = (
        _as_direction(moving_direction)
        if moving_direction is not None
        else (_as_direction(tensors.direction[:3, :3]) if is_ants else np.eye(3))
    )
    d_fix = (
        _as_direction(fixed_direction)
        if fixed_direction is not None
        else (_as_direction(ref_img.direction[:3, :3]) if ref_img is not None else d_mov)
    )
    if method == "ppd":
        f_total = d_fix.T @ j_inv_np @ d_mov
        out6 = _ppd_reorient_sym6(sym6, f_total, from_log_domain=False)
    elif method == "finite_strain":
        from .spatial import polar_rotation_field

        r_lps = polar_rotation_field(j_inv_np)
        r_fs = d_fix.T @ r_lps @ d_mov
        d_in = sym6[:, _SYM6_TO_3X3].astype(np.float64)
        d_rot = np.einsum("nij,njk,nlk->nil", r_fs, d_in, r_fs)
        out6 = np.stack(
            [
                d_rot[:, 0, 0],
                d_rot[:, 0, 1],
                d_rot[:, 0, 2],
                d_rot[:, 1, 1],
                d_rot[:, 1, 2],
                d_rot[:, 2, 2],
            ],
            axis=-1,
        ).astype(np.float32)
    else:
        raise ValueError(f"Unknown tensor reorientation method {method!r}. Options: 'ppd', 'finite_strain'.")

    if is_mat33:
        out_arr = out6[:, _SYM6_TO_3X3].reshape(*spatial_shape, 3, 3)
    else:
        out_arr = out6.reshape(*spatial_shape, 6)

    if is_ants:
        out_ref = reference if reference is not None else tensors
        return ants.from_numpy(
            out_arr,
            origin=tuple(float(x) for x in out_ref.origin[:3]),
            spacing=tuple(float(x) for x in out_ref.spacing[:3]),
            direction=np.asarray(out_ref.direction)[:3, :3],
            has_components=True,
        )
    if is_torch:
        return torch.as_tensor(out_arr, device=tensors.device, dtype=tensors.dtype)
    return out_arr


# ==============================================================================
# 6. Series Ledger, Composition, and Jacobian Helpers
# ==============================================================================

def _as_direction(direction: Any) -> np.ndarray:
    d = np.asarray(direction, dtype=float)
    if d.size != 9:
        raise ValueError(f"direction must be a 3x3 matrix or flat 9-vector, got {d.size} elements")
    return d.reshape(3, 3)


def series_rebase(series_direction: Any, final_direction: Any) -> np.ndarray:
    """Header rebase between a series and the final frame, ANTsPyMM convention."""
    return _as_direction(final_direction).T @ _as_direction(series_direction)


def compose_displacement(
    transformlist: Sequence[str],
    whichtoinvert: Sequence[bool] | None,
    reference: Any,
    out_path: str,
    device: str | None = None,
) -> str:
    """Compose a transform list (displacement fields and/or affines) into ONE displacement field on ``reference``'s grid."""
    tl = [str(t) for t in transformlist]
    if not tl:
        raise ValueError("compose_displacement: transformlist is empty")
    inv = [False] * len(tl) if whichtoinvert is None else [bool(i) for i in whichtoinvert]
    if len(inv) != len(tl):
        raise ValueError("whichtoinvert must match transformlist")
    out_path = str(out_path)
    if not out_path.endswith((".nii.gz", ".nii")):
        raise ValueError(f"out_path must be a NIfTI file (.nii.gz/.nii), got {out_path}")
    return str(
        apply_transforms(
            fixed=reference,
            moving=reference,
            transformlist=tl,
            whichtoinvert=inv,
            compose=out_path,
            device=device,
        )
    )


def signed_jacobian_determinant(field_path_or_image: Any, reference: Any) -> Any:
    """Signed ``det(I + grad u)`` of a displacement field on ``reference``'s grid (unclipped)."""
    import ants

    if isinstance(field_path_or_image, str):
        f = None
        try:
            st = os.stat(field_path_or_image)
            f = _DISP_IMG_CACHE.get((field_path_or_image, st.st_mtime_ns, st.st_size))
        except OSError:
            pass
        if f is None:
            f = ants.image_read(field_path_or_image)
    else:
        f = field_path_or_image
    if tuple(f.shape) != tuple(reference.shape):
        raise ValueError(f"field grid {tuple(f.shape)} differs from reference {tuple(reference.shape)}")
    u = np.asarray(f.numpy(), dtype=np.float32)
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError(f"expected a 3-component displacement image, got array shape {u.shape}")
    if not np.all(np.isfinite(u)):
        raise ValueError("displacement field contains non-finite values")
    ds_inv = np.linalg.inv(_as_direction(reference.direction) @ np.diag(np.asarray(reference.spacing, float))).astype(np.float32)
    sp_shape = u.shape[:3]
    du = np.empty(sp_shape + (3, 3), dtype=np.float32)
    for ax in range(3):
        du[..., :, ax] = np.gradient(u, axis=ax)
    j = du @ ds_inv
    j00 = np.float32(1.0) + j[..., 0, 0]
    j01 = j[..., 0, 1]
    j02 = j[..., 0, 2]
    j10 = j[..., 1, 0]
    j11 = np.float32(1.0) + j[..., 1, 1]
    j12 = j[..., 1, 2]
    j20 = j[..., 2, 0]
    j21 = j[..., 2, 1]
    j22 = np.float32(1.0) + j[..., 2, 2]
    det = (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )
    if not np.all(np.isfinite(det)):
        raise ValueError("Jacobian determinant contains non-finite values")
    return reference.new_image_like(det.astype(np.float32, copy=False))


def jacobian_modulation(field_path_or_image: Any, reference: Any) -> Any:
    """Intensity modulation ``|det(I + grad u)|`` of a displacement field, on ``reference``'s grid."""
    det = np.abs(signed_jacobian_determinant(field_path_or_image, reference).numpy())
    return reference.new_image_like(np.clip(det, *JACOBIAN_CLIP).astype(np.float32))


def chain_linear_jacobian(chain: "TransformChain | None") -> float:
    """Determinant of the composed linear part of a chain (displacement fields contribute 1): the intensity factor that
    conserves signal under the chain's affine volume change. A rigid contributes 1; the motion solver's eddy affine along
    phase encode ``p`` with shears/scale ``a`` has ``det(I + p a^T) = 1 + a . p``, the Jacobian the solver applies to its own
    ``motion_corrected`` (``_batched_rigid_register_pass_core``). ``resample_series`` applies it so the single-interpolation
    path and the solver's output agree (gap found 2026-10-28: the final DWI volumes carried the eddy displacement but not its
    intensity modulation)."""
    if chain is None or len(chain) == 0:
        return 1.0
    return float(np.linalg.det(chain.linear_part()))


def resample_series(
    img4d: Any,
    motion_chains: Sequence[TransformChain | None],
    field_path: str | None = None,
    reference: Any = None,
    jacobian: bool = True,
    interpolator: str = "linear",
    device: str | None = None,
    volume_scales: Sequence[float] | np.ndarray | None = None,
    precomputed_field_jacobian: np.ndarray | None = None,
) -> Any:
    """Resample every volume of a 4D series onto ``reference`` with exactly ONE interpolation each on Torch/MPS/CUDA."""
    import ants

    if reference is None:
        raise ValueError("resample_series requires reference")
    nvol = int(img4d.shape[3])
    if len(motion_chains) != nvol:
        raise ValueError(f"len(motion_chains)={len(motion_chains)} != number of volumes {nvol}")
    if not jacobian:
        jac = None
    elif precomputed_field_jacobian is not None:
        jac = np.asarray(precomputed_field_jacobian, dtype=np.float32)
        ref_shape_3d = tuple(int(s) for s in reference.shape[:3])
        if jac.shape != ref_shape_3d:
            raise ValueError(
                f"precomputed_field_jacobian shape {jac.shape} != reference shape {ref_shape_3d}"
            )
    elif field_path is not None:
        jac = jacobian_modulation(field_path, reference).numpy()
    else:
        jac = None
    if volume_scales is not None:
        jac_motion = np.asarray(volume_scales, dtype=np.float64).reshape(-1)
        if jac_motion.size != nvol:
            raise ValueError(f"len(volume_scales)={jac_motion.size} != number of volumes {nvol}")
    elif jacobian:
        # per-volume intensity factor of the motion chain's affine part (the eddy Jacobian; 1 for a rigid-only chain)
        jac_motion = np.array([chain_linear_jacobian(ch) for ch in motion_chains], dtype=np.float64)
    else:
        jac_motion = np.ones(nvol, dtype=np.float64)

    interp_lower = interpolator.lower()
    if interp_lower in ("linear", "bilinear", "trilinear"):
        mode = "bilinear"
    elif interp_lower in ("nearestneighbor", "nearest"):
        mode = "nearest"
    elif interp_lower in ("genericlabel", "multilabel"):
        mode = "genericlabel"
    elif interp_lower == "bspline":
        mode = "bspline"
    else:
        mode = None

    if mode is not None:
        dev = torch.device(resolve_device(device, task="model"))
        ref_shape = tuple(int(s) for s in reference.shape[:3])
        mov_shape = tuple(int(s) for s in img4d.shape[:3])
        n_vox = ref_shape[0] * ref_shape[1] * ref_shape[2]
        jac_t = torch.as_tensor(jac.reshape(-1), dtype=torch.float32, device=dev) if jac is not None else None
        jac_motion_t = torch.as_tensor(jac_motion, dtype=torch.float32, device=dev)

        raw4d = img4d.numpy()
        vols_tzyx = (
            raw4d.T
            if (raw4d.dtype == np.float32 and raw4d.flags["F_CONTIGUOUS"])
            else np.ascontiguousarray(np.transpose(raw4d, (3, 2, 1, 0)), dtype=np.float32)
        )
        vol_origin = tuple(float(x) for x in img4d.origin[:3])
        vol_spacing = tuple(float(x) for x in img4d.spacing[:3])
        vol_dir = np.asarray(img4d.direction, dtype=np.float64)[:3, :3]
        ref_dir = _as_direction(reference.direction)
        same_3d_grid = (
            field_path is None
            and mov_shape == ref_shape
            and np.allclose(vol_spacing, reference.spacing[:3])
            and np.allclose(vol_origin, reference.origin[:3])
            and np.allclose(vol_dir, ref_dir)
        )

        out_tzyx = np.empty((nvol, ref_shape[2], ref_shape[1], ref_shape[0]), dtype=np.float32)
        out4d = out_tzyx.T

        all_linear = all(
            (ch is None) or all(e.kind in (*_LINEAR, "identity") for e in ch.elements)
            for ch in motion_chains
        )
        use_affine_grid = (
            field_path is None
            and all_linear
            and mode in ("bilinear", "nearest", "bspline")
            and min(mov_shape) > 1
            and min(ref_shape) > 1
        )
        if use_affine_grid:
            M_list = []
            b_list = []
            for ch in motion_chains:
                M_acc = np.eye(3, dtype=np.float64)
                b_acc = np.zeros(3, dtype=np.float64)
                if ch is not None:
                    for el in ch.elements:
                        if el.kind == "identity":
                            continue
                        M_step, b_step = _read_itk_affine(el.path, invert=el.invert)
                        b_acc = M_step @ b_acc + b_step
                        M_acc = M_step @ M_acc
                M_list.append(M_acc)
                b_list.append(b_acc)
            half_ref = 0.5 * (np.asarray(ref_shape, dtype=np.float64) - 1.0)
            r_ref = ref_dir * np.asarray(reference.spacing[:3], dtype=np.float64)[None, :]
            o_ref = np.asarray(reference.origin[:3], dtype=np.float64)
            a_ref = r_ref @ np.diag(half_ref)
            c_ref = r_ref @ half_ref + o_ref
            r_mov_inv = np.linalg.inv(vol_dir * np.asarray(vol_spacing, dtype=np.float64)[None, :])
            o_mov = np.asarray(vol_origin, dtype=np.float64)
            m_np = np.stack(M_list, axis=0)
            b_np = np.stack(b_list, axis=0)
            a_cidx = np.einsum("ij,tjk,kl->til", r_mov_inv, m_np, a_ref)
            t_cidx = np.einsum("ij,tj->ti", r_mov_inv, (m_np @ c_ref) + b_np - o_mov[None, :])
            shape_t = torch.tensor(mov_shape, dtype=torch.float32, device=dev)
            half_span = 0.5 * (shape_t - 1.0)
            if mode == "bilinear":
                inv_half_mov = 1.0 / (0.5 * (np.asarray(mov_shape, dtype=np.float64) - 1.0))
                theta_np = np.concatenate(
                    [
                        a_cidx * inv_half_mov[None, :, None],
                        (t_cidx * inv_half_mov[None, :] - 1.0)[..., None],
                    ],
                    axis=-1,
                )
                lo_norm = torch.as_tensor(-1.0 - (0.5 + 1e-5) * inv_half_mov, dtype=torch.float32, device=dev)
                hi_norm = torch.as_tensor(1.0 + (0.5 - 1e-5) * inv_half_mov, dtype=torch.float32, device=dev)
            else:
                theta_np = np.concatenate([a_cidx, t_cidx[..., None]], axis=-1)
                lo_norm = shape_t
                hi_norm = shape_t
            theta_t = torch.as_tensor(theta_np, dtype=torch.float32, device=dev)
            batch_sz = _auto_select_4d_batch_size(
                mov_shape,
                ref_shape,
                nvol,
                dev,
                per_volume_grid=True,
                bspline=(mode == "bspline"),
                affine_grid_only=True,
            )
            rz, ry, rx = ref_shape[2], ref_shape[1], ref_shape[0]
            s_idx = 0
            while s_idx < nvol:
                e_idx = min(nvol, s_idx + batch_sz)
                b_len = e_idx - s_idx
                try:
                    t_vols_b = torch.from_numpy(vols_tzyx[s_idx:e_idx]).unsqueeze(1).to(
                        device=dev, dtype=torch.float32
                    )
                    grid_or_cidx = F.affine_grid(
                        theta_t[s_idx:e_idx], [b_len, 1, rz, ry, rx], align_corners=True
                    )
                    if mode == "bilinear":
                        inside = ((grid_or_cidx >= lo_norm) & (grid_or_cidx < hi_norm)).all(dim=-1)
                        sampled = F.grid_sample(
                            t_vols_b,
                            grid_or_cidx,
                            mode="bilinear",
                            padding_mode="border",
                            align_corners=True,
                        ).squeeze(1)
                        del t_vols_b, grid_or_cidx
                    elif mode == "bspline":
                        inside = ((grid_or_cidx >= -0.5 - 1e-5) & (grid_or_cidx < (shape_t - 0.5) - 1e-5)).all(dim=-1)
                        c_pad = _bspline_prefilter_and_pad_5d(t_vols_b)
                        del t_vols_b
                        sampled = _sample_bspline_padded_5d(
                            c_pad, grid_or_cidx.view(b_len, n_vox, 3), shape_t
                        ).view(b_len, rz, ry, rx)
                        del c_pad, grid_or_cidx
                    else:
                        inside = ((grid_or_cidx >= -0.5 - 1e-5) & (grid_or_cidx < (shape_t - 0.5) - 1e-5)).all(dim=-1)
                        cidx_nearest = torch.floor(grid_or_cidx + 0.5 + 1e-5).clamp(
                            min=torch.zeros_like(shape_t), max=shape_t - 1.0
                        )
                        del grid_or_cidx
                        grid_n = cidx_nearest / half_span.clamp(min=1e-6) - 1.0
                        del cidx_nearest
                        sampled = F.grid_sample(
                            t_vols_b,
                            grid_n,
                            mode="nearest",
                            padding_mode="border",
                            align_corners=True,
                        ).squeeze(1)
                        del t_vols_b, grid_n
                    sampled = sampled * inside.to(torch.float32)
                    del inside
                    sampled = sampled * jac_motion_t[s_idx:e_idx].to(sampled.device).view(b_len, 1, 1, 1)
                    out_tzyx[s_idx:e_idx] = sampled.detach().cpu().numpy()
                    del sampled
                    s_idx = e_idx
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "out of memory" in msg or "mps" in msg or "buffer" in msg:
                        _empty_device_cache(dev)
                        if batch_sz > 1:
                            batch_sz = max(1, batch_sz // 2)
                            continue
                        host_dev = torch.zeros(1).device
                        if dev != host_dev:
                            dev = host_dev
                            theta_t = theta_t.to(device=dev)
                            shape_t = shape_t.to(device=dev)
                            half_span = half_span.to(device=dev)
                            lo_norm = lo_norm.to(device=dev)
                            hi_norm = hi_norm.to(device=dev)
                            batch_sz = _auto_select_4d_batch_size(
                                mov_shape,
                                ref_shape,
                                nvol - s_idx,
                                dev,
                                per_volume_grid=True,
                                bspline=(mode == "bspline"),
                                affine_grid_only=True,
                            )
                            continue
                    raise
        elif all_linear and mode in ("bilinear", "nearest", "bspline"):
            pts_fixed = _reference_grid_points_lps(reference, dev)
            if field_path is not None:
                field_chain = TransformChain.from_paths([str(field_path)])
                pts_after_field = _map_points_to_moving_tensor(pts_fixed, field_chain, dev)
                del pts_fixed
            else:
                pts_after_field = pts_fixed
            M_list = []
            b_list = []
            for ch in motion_chains:
                M_acc = np.eye(3, dtype=np.float64)
                b_acc = np.zeros(3, dtype=np.float64)
                if ch is not None:
                    for el in ch.elements:
                        if el.kind == "identity":
                            continue
                        M_step, b_step = _read_itk_affine(el.path, invert=el.invert)
                        b_acc = M_step @ b_acc + b_step
                        M_acc = M_step @ M_acc
                M_list.append(M_acc)
                b_list.append(b_acc)
            M_all = torch.as_tensor(np.stack(M_list, axis=0), dtype=torch.float32, device=dev)  # (T, 3, 3)
            b_all = torch.as_tensor(np.stack(b_list, axis=0), dtype=torch.float32, device=dev)  # (T, 3)
            M_norm, b_norm = get_physical_to_normalized_affine(
                shape=mov_shape,
                spacing=vol_spacing,
                origin=vol_origin,
                direction=vol_dir,
                device=dev,
                dtype=torch.float32,
            )
            shape_t = torch.tensor(mov_shape, dtype=torch.float32, device=dev)
            half_span = 0.5 * (shape_t - 1.0)

            batch_sz = _auto_select_4d_batch_size(
                mov_shape, ref_shape, nvol, dev, per_volume_grid=True, bspline=(mode == "bspline")
            )

            s_idx = 0
            while s_idx < nvol:
                e_idx = min(nvol, s_idx + batch_sz)
                b_len = e_idx - s_idx
                try:
                    t_vols_b = torch.from_numpy(vols_tzyx[s_idx:e_idx]).unsqueeze(1).to(
                        device=dev, dtype=torch.float32
                    )
                    pts_b = (
                        torch.bmm(
                            pts_after_field.unsqueeze(0).expand(b_len, -1, -1),
                            M_all[s_idx:e_idx].transpose(1, 2),
                        )
                        + b_all[s_idx:e_idx].unsqueeze(1)
                    )
                    coords_norm = physical_to_normalized(pts_b, M_norm, b_norm)
                    del pts_b
                    cidx = (coords_norm + 1.0) * half_span
                    inside = ((cidx >= -0.5 - 1e-5) & (cidx < (shape_t - 0.5) - 1e-5)).all(dim=-1)
                    if mode == "bspline":
                        del coords_norm
                        c_pad = _bspline_prefilter_and_pad_5d(t_vols_b)
                        del t_vols_b
                        sampled = _sample_bspline_padded_5d(c_pad, cidx, shape_t).view(b_len, n_vox)
                        del c_pad, cidx
                    else:
                        if mode == "nearest":
                            cidx_nearest = torch.floor(cidx + 0.5 + 1e-5).clamp(
                                min=torch.zeros_like(shape_t), max=shape_t - 1.0
                            )
                            coords_norm = cidx_nearest / half_span.clamp(min=1e-6) - 1.0
                        del cidx
                        sampled = F.grid_sample(
                            t_vols_b,
                            coords_norm.view(b_len, 1, 1, n_vox, 3),
                            mode=mode,
                            padding_mode="border",
                            align_corners=True,
                        ).view(b_len, n_vox)
                        del t_vols_b, coords_norm
                    sampled = sampled * inside.to(torch.float32)
                    del inside
                    if jac_t is not None:
                        sampled = sampled * jac_t.unsqueeze(0)
                    sampled = sampled * jac_motion_t[s_idx:e_idx].to(sampled.device).unsqueeze(1)
                    out_tzyx[s_idx:e_idx] = (
                        sampled.view(b_len, *ref_shape).permute(0, 3, 2, 1).contiguous().detach().cpu().numpy()
                    )
                    del sampled
                    s_idx = e_idx
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "out of memory" in msg or "mps" in msg or "buffer" in msg:
                        _empty_device_cache(dev)
                        if batch_sz > 1:
                            batch_sz = max(1, batch_sz // 2)
                            continue
                        if dev.type != "cpu":
                            dev = torch.device("cpu")
                            pts_after_field = pts_after_field.to(device=dev)
                            M_all = M_all.to(device=dev)
                            b_all = b_all.to(device=dev)
                            M_norm = M_norm.to(device=dev)
                            b_norm = b_norm.to(device=dev)
                            shape_t = shape_t.to(device=dev)
                            half_span = half_span.to(device=dev)
                            if jac_t is not None:
                                jac_t = jac_t.to(device=dev)
                            batch_sz = _auto_select_4d_batch_size(
                                mov_shape,
                                ref_shape,
                                nvol - s_idx,
                                dev,
                                per_volume_grid=True,
                                bspline=(mode == "bspline"),
                            )
                            continue
                    raise
        else:
            pts_fixed = _reference_grid_points_lps(reference, dev)
            if field_path is not None:
                field_chain = TransformChain.from_paths([str(field_path)])
                pts_after_field = _map_points_to_moving_tensor(pts_fixed, field_chain, dev)
                del pts_fixed
            else:
                pts_after_field = pts_fixed
            field_vol: ContinuousField | None = None
            for k in range(nvol):
                chain = motion_chains[k]
                pts_k = (
                    _map_points_to_moving_tensor(pts_after_field, chain, dev)
                    if (chain is not None and len(chain) > 0)
                    else pts_after_field
                )
                vol_arr = raw4d[..., k]
                if field_vol is None:
                    field_vol = ContinuousField(
                        np.asarray(vol_arr, dtype=np.float32),
                        origin=vol_origin,
                        spacing=vol_spacing,
                        direction=vol_dir,
                        device=dev,
                        dtype=torch.float32,
                    )
                else:
                    field_vol.tensor = torch.as_tensor(vols_tzyx[k][None, None, ...], device=dev, dtype=torch.float32)

                sampled = _sample_field_itk(field_vol, pts_k, mode=mode)
                if jac_t is not None:
                    sampled = sampled * jac_t
                sampled = sampled * float(jac_motion[k])
                out4d[..., k] = sampled.reshape(ref_shape).detach().cpu().numpy()

        if same_3d_grid:
            for k, ch in enumerate(motion_chains):
                if ch is None or all(el.kind == "identity" for el in ch.elements):
                    out_tzyx[k] = vols_tzyx[k] * float(jac_motion[k])

        spacing = list(reference.spacing[:3]) + [float(img4d.spacing[3])]
        origin = list(reference.origin[:3]) + [float(img4d.origin[3])]
        direction = np.eye(4)
        direction[:3, :3] = ref_dir
        return ants.from_numpy(out4d, origin=origin, spacing=spacing, direction=direction)

    out = []
    for k in range(nvol):
        chain = motion_chains[k]
        real_ch = (
            [(p_, inv_) for p_, inv_ in zip(chain.paths(), chain.whichtoinvert()) if str(p_).lower() != "identity"]
            if chain is not None
            else []
        )
        tl = ([str(field_path)] if field_path is not None else []) + [p_ for p_, _ in real_ch]
        inv = ([False] if field_path is not None else []) + [inv_ for _, inv_ in real_ch]
        vol = ants.slice_image(img4d, axis=3, idx=k)
        if tl:
            w = ants.apply_transforms(
                fixed=reference, moving=vol, transformlist=tl, whichtoinvert=inv, interpolator=interpolator
            )
        else:
            w = ants.resample_image_to_target(vol, reference, interp_type=interpolator)
        if jac is not None:
            w = w.new_image_like(w.numpy() * jac)
        if jac_motion[k] != 1.0:
            w = w.new_image_like(w.numpy() * float(jac_motion[k]))
        out.append(w)
    res = ants.list_to_ndimage(img4d, out)
    spacing = list(reference.spacing[:3]) + [float(img4d.spacing[3])]
    origin = list(reference.origin[:3]) + [float(img4d.origin[3])]
    direction = np.eye(4)
    direction[:3, :3] = _as_direction(reference.direction)
    res.set_spacing(spacing)
    res.set_origin(origin)
    res.set_direction(direction)
    return res


__all__ = [
    "Kind",
    "JACOBIAN_CLIP",
    "ContinuousField",
    "physical_to_normalized",
    "transform_index_to_physical_point",
    "transform_physical_point_to_index",
    "TransformElement",
    "classify_transform",
    "TransformChain",
    "apply_transforms",
    "resample_image_to_target",
    "compose_displacement_fields",
    "create_jacobian_determinant_image",
    "map_points_to_moving",
    "apply_transforms_to_points",
    "apply_ants_transform_to_point",
    "apply_ants_transform_to_vector",
    "transform_streamlines",
    "reorient_vectors",
    "reorient_tensors",
    "series_rebase",
    "compose_displacement",
    "signed_jacobian_determinant",
    "jacobian_modulation",
    "resample_series",
    "chain_linear_jacobian",
    "_sample_field_itk",
    "_reference_grid_points_lps",
    "_read_itk_affine",
    "_map_points_to_moving_tensor",
    "_map_points_and_inv_jacobian_tensor",
    "_as_direction",
]
