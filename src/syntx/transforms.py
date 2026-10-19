"""syntx.transforms — Core Torch/MPS/CUDA physical-space transform application & series resampling.

Provides:
- ``ContinuousField`` & ``physical_to_normalized``: Differentiable O(N) physical-space (ITK LPS mm)
  scalar, vector, and multi-channel field evaluator on Torch/MPS/CUDA.
- ``TransformElement``, ``classify_transform``, ``TransformChain``: Typed, ordered, provenance-carrying
  wrapper over ANTs/syntx transform lists with Torch/MPS/CUDA ``TransformChain.apply`` matching ITK's
  ``ResampleImageFilter`` boundary semantics.
- ``map_points_to_moving``: Point-space counterpart of ``ants.apply_transforms`` in physical LPS mm.
- ``apply_transforms``: Drop-in Torch/MPS/CUDA replacement for 3D ``ants.apply_transforms``.
- ``compose_displacement``, ``signed_jacobian_determinant``, ``jacobian_modulation``, ``resample_series``:
  Single-interpolation 4D series resampling with optional Jacobian intensity modulation on Torch/MPS/CUDA.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import os
import shutil
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .device import resolve_device
from .spatial import get_physical_to_normalized_affine_xyz as get_physical_to_normalized_affine

Kind = Literal["rigid", "affine", "displacement", "inverse_displacement", "identity"]
_LINEAR = ("rigid", "affine")
JACOBIAN_CLIP = (0.1, 10.0)


def physical_to_normalized(
    coords_phys: torch.Tensor,
    M_norm: torch.Tensor,
    b_norm: torch.Tensor,
) -> torch.Tensor:
    """Project physical LPS coordinates (in mm) into [-1, 1]^3 normalized grid space."""
    return coords_phys @ M_norm + b_norm


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


@dataclass
class TransformChain:
    """Ordered list of transforms in ANTs *application* order (first applied last, as ants.apply_transforms expects)."""

    elements: list[TransformElement] = field(default_factory=list)

    @classmethod
    def from_paths(cls, paths: Iterable[str], whichtoinvert: Iterable[bool] | None = None) -> TransformChain:
        paths = list(paths)
        inv = list(whichtoinvert) if whichtoinvert is not None else [False] * len(paths)
        if len(inv) != len(paths):
            raise ValueError("whichtoinvert must match paths")
        return cls([TransformElement(p, classify_transform(p), bool(i)) for p, i in zip(paths, inv, strict=True)])

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

        Composition follows application order: the last element in the list is applied first.
        """
        m = np.eye(3)
        for e in reversed(self.linear_elements()):
            a, _ = _read_itk_affine(e.path, invert=e.invert)
            m = a @ m
        return m

    def apply(
        self,
        image: Any,
        reference: Any,
        interpolator: str = "linear",
        device: str | None = None,
    ) -> Any:
        import ants

        if len(image.shape) != 3 or len(reference.shape) != 3:
            return ants.apply_transforms(
                fixed=reference,
                moving=image,
                transformlist=self.paths(),
                whichtoinvert=self.whichtoinvert(),
                interpolator=interpolator,
            )

        interp_lower = interpolator.lower()
        if interp_lower in ("linear", "bilinear", "trilinear"):
            mode = "bilinear"
        elif interp_lower in ("nearestneighbor", "nearest"):
            mode = "nearest"
        elif interp_lower in ("genericlabel", "multilabel"):
            mode = "genericlabel"
        else:
            return ants.apply_transforms(
                fixed=reference,
                moving=image,
                transformlist=self.paths(),
                whichtoinvert=self.whichtoinvert(),
                interpolator=interpolator,
            )

        dev_str = resolve_device(device, task="model")
        dev = torch.device(dev_str)

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
        out = _sample_field_itk(field, pts_moving, mode=mode)
        arr_out = out.reshape(ref_shape).detach().cpu().numpy().astype(np.float32, copy=False)
        return reference.new_image_like(arr_out)

    def compose(self, other: TransformChain) -> TransformChain:
        """self ∘ other: ``other`` is applied first."""
        return TransformChain(self.elements + other.elements)

    def to_provenance(self) -> list[dict[str, str]]:
        return [{"path": e.path, "kind": e.kind, "invert": str(e.invert)} for e in self.elements]


_DISP_FIELD_CACHE: dict[tuple[str, float, str], Any] = {}
_MAPPED_GRID_CACHE: dict[tuple[Any, ...], Any] = {}


def _sample_field_itk(field: ContinuousField, pts_phys: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Sample a ContinuousField matching ITK's ResampleImageFilter + Linear/Nearest/GenericLabel:
    - Continuous index interval is half-open: c_d in [-0.5, N_d - 0.5). Points outside return 0.
    - Inside [-0.5, N_d - 0.5):
      - 'bilinear': clamped to [0, N_d - 1] (padding_mode='border') with trilinear weights.
      - 'nearest': ITK RoundHalfIntegerUp, i.e. floor(c_d + 0.5) clamped to [0, N_d - 1].
      - 'genericlabel': 8-corner trilinear label voting matching ITK LabelImageGenericInterpolateImageFunction.
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
    elif mode == "genericlabel" and not field.is_vector:
        c_clamped = torch.minimum(torch.maximum(cidx, torch.zeros_like(shape_t)), shape_t - 1.0)
        i0 = torch.floor(c_clamped).to(torch.long)
        f = c_clamped - i0.to(field.dtype)
        i1 = torch.minimum(i0 + 1, (shape_t - 1).to(torch.long))
        vol = field.tensor[0, 0]  # (Nz, Ny, Nx)
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
        out = field.sample(pts_phys, mode="bilinear", padding_mode="border")

    return out * (inside.unsqueeze(-1).to(out.dtype) if out.ndim == 2 else inside.to(out.dtype))


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


def _read_itk_affine(path: str, invert: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (M, b) with y = M x + b for an ITK linear transform file: y = M (x - c) + c + t, where
    c is the transform's fixed parameters (centre) and t its translation."""
    import ants

    tx = ants.read_transform(path)
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


def map_points_to_moving(
    points_lps: np.ndarray,
    transformlist: Iterable[str] | TransformChain,
    whichtoinvert: Iterable[bool] | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Map fixed-space physical points (LPS mm) to moving-space points through an ANTs
    transform list, in torch."""
    if isinstance(transformlist, TransformChain):
        chain = transformlist
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


def apply_transforms(
    fixed: Any,
    moving: Any,
    transformlist: Iterable[str] | TransformChain,
    whichtoinvert: Iterable[bool] | None = None,
    interpolator: str = "linear",
    device: str | None = None,
) -> Any:
    """Drop-in Torch/MPS/CUDA replacement for 3D ``ants.apply_transforms``."""
    chain = (
        transformlist
        if isinstance(transformlist, TransformChain)
        else TransformChain.from_paths(list(transformlist), whichtoinvert)
    )
    return chain.apply(moving, fixed, interpolator=interpolator, device=device)


def _as_direction(direction: Any) -> np.ndarray:
    d = np.asarray(direction, dtype=float)
    if d.size != 9:
        raise ValueError(f"direction must be a 3x3 matrix or flat 9-vector, got {d.size} elements")
    return d.reshape(3, 3)


def series_rebase(series_direction: Any, final_direction: Any) -> np.ndarray:
    """Header rebase between a series and the final frame, ANTsPyMM convention."""
    return _as_direction(final_direction).T @ _as_direction(series_direction)


def compose_displacement(
    transformlist: Sequence[str], whichtoinvert: Sequence[bool] | None, reference: Any, out_path: str
) -> str:
    """Compose a transform list (displacement fields and/or affines) into ONE displacement field."""
    import ants

    tl = [str(t) for t in transformlist]
    if not tl:
        raise ValueError("compose_displacement: transformlist is empty")
    inv = [False] * len(tl) if whichtoinvert is None else [bool(i) for i in whichtoinvert]
    if len(inv) != len(tl):
        raise ValueError("whichtoinvert must match transformlist")
    out_path = str(out_path)
    if not out_path.endswith((".nii.gz", ".nii")):
        raise ValueError(f"out_path must be a NIfTI file (.nii.gz/.nii), got {out_path}")
    prefix = out_path[: -len(".nii.gz")] if out_path.endswith(".nii.gz") else out_path[: -len(".nii")]
    comp = ants.apply_transforms(fixed=reference, moving=reference, transformlist=tl, whichtoinvert=inv, compose=prefix)
    if not isinstance(comp, str):
        raise RuntimeError("ants.apply_transforms(compose=...) did not return a file path")
    if str(comp) != out_path:
        shutil.move(str(comp), out_path)
    return out_path


def signed_jacobian_determinant(field_path_or_image: Any, reference: Any) -> Any:
    """Signed ``det(I + grad u)`` of a displacement field on ``reference``'s grid (unclipped)."""
    import ants

    f = ants.image_read(str(field_path_or_image)) if isinstance(field_path_or_image, str) else field_path_or_image
    if tuple(f.shape) != tuple(reference.shape):
        raise ValueError(f"field grid {tuple(f.shape)} differs from reference {tuple(reference.shape)}")
    u = np.asarray(f.numpy(), dtype=np.float64)
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError(f"expected a 3-component displacement image, got array shape {u.shape}")
    if not np.all(np.isfinite(u)):
        raise ValueError("displacement field contains non-finite values")
    ds_inv = np.linalg.inv(_as_direction(reference.direction) @ np.diag(np.asarray(reference.spacing, float)))
    du = np.stack([np.stack(np.gradient(u[..., i]), axis=-1) for i in range(3)], axis=-2)
    det = np.linalg.det(np.eye(3) + du @ ds_inv)
    if not np.all(np.isfinite(det)):
        raise ValueError("Jacobian determinant contains non-finite values")
    return reference.new_image_like(det.astype(np.float32))


def jacobian_modulation(field_path_or_image: Any, reference: Any) -> Any:
    """Intensity modulation ``|det(I + grad u)|`` of a displacement field, on ``reference``'s grid."""
    det = np.abs(signed_jacobian_determinant(field_path_or_image, reference).numpy())
    return reference.new_image_like(np.clip(det, *JACOBIAN_CLIP).astype(np.float32))


def resample_series(
    img4d: Any,
    motion_chains: Sequence[TransformChain | None],
    field_path: str,
    reference: Any,
    jacobian: bool = True,
    interpolator: str = "linear",
    device: str | None = None,
) -> Any:
    """Resample every volume of a 4D series onto ``reference`` with exactly ONE interpolation each on Torch/MPS/CUDA."""
    import ants

    nvol = int(img4d.shape[3])
    if len(motion_chains) != nvol:
        raise ValueError(f"len(motion_chains)={len(motion_chains)} != number of volumes {nvol}")
    jac = jacobian_modulation(field_path, reference).numpy() if jacobian else None

    interp_lower = interpolator.lower()
    if interp_lower in ("linear", "bilinear", "trilinear"):
        mode = "bilinear"
    elif interp_lower in ("nearestneighbor", "nearest"):
        mode = "nearest"
    elif interp_lower in ("genericlabel", "multilabel"):
        mode = "genericlabel"
    else:
        mode = None

    if mode is not None:
        dev = torch.device(resolve_device(device, task="model"))
        ref_shape = tuple(int(s) for s in reference.shape[:3])
        pts_fixed = _reference_grid_points_lps(reference, dev)
        field_chain = TransformChain.from_paths([str(field_path)])
        pts_after_field = _map_points_to_moving_tensor(pts_fixed, field_chain, dev)
        jac_t = torch.as_tensor(jac.reshape(-1), dtype=torch.float32, device=dev) if jac is not None else None

        arr4d = np.asarray(img4d.numpy(), dtype=np.float32)
        vol_origin = tuple(float(x) for x in img4d.origin[:3])
        vol_spacing = tuple(float(x) for x in img4d.spacing[:3])
        vol_dir = np.asarray(img4d.direction)[:3, :3]
        out4d = np.empty(ref_shape + (nvol,), dtype=np.float32)

        field_vol: ContinuousField | None = None
        for k in range(nvol):
            chain = motion_chains[k]
            pts_k = (
                _map_points_to_moving_tensor(pts_after_field, chain, dev)
                if (chain is not None and len(chain) > 0)
                else pts_after_field
            )
            vol_arr = arr4d[..., k]
            if field_vol is None:
                field_vol = ContinuousField(
                    vol_arr,
                    origin=vol_origin,
                    spacing=vol_spacing,
                    direction=vol_dir,
                    device=dev,
                    dtype=torch.float32,
                )
            else:
                arr_dhw = np.transpose(vol_arr, (2, 1, 0))[None, None, ...]
                field_vol.tensor = torch.as_tensor(np.ascontiguousarray(arr_dhw), device=dev, dtype=torch.float32)

            sampled = _sample_field_itk(field_vol, pts_k, mode=mode)
            if jac_t is not None:
                sampled = sampled * jac_t
            out4d[..., k] = sampled.reshape(ref_shape).detach().cpu().numpy()

        spacing = list(reference.spacing) + [float(img4d.spacing[3])]
        origin = list(reference.origin) + [float(img4d.origin[3])]
        direction = np.eye(4)
        direction[:3, :3] = _as_direction(reference.direction)
        return ants.from_numpy(out4d, origin=origin, spacing=spacing, direction=direction)

    out = []
    for k in range(nvol):
        chain = motion_chains[k]
        tl = [str(field_path)] + (chain.paths() if chain is not None else [])
        inv = [False] + (chain.whichtoinvert() if chain is not None else [])
        vol = ants.slice_image(img4d, axis=3, idx=k)
        w = ants.apply_transforms(
            fixed=reference, moving=vol, transformlist=tl, whichtoinvert=inv, interpolator=interpolator
        )
        if jac is not None:
            w = w.new_image_like(w.numpy() * jac)
        out.append(w)
    res = ants.list_to_ndimage(img4d, out)
    spacing = list(reference.spacing) + [float(img4d.spacing[3])]
    origin = list(reference.origin) + [float(img4d.origin[3])]
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
    "TransformElement",
    "classify_transform",
    "TransformChain",
    "map_points_to_moving",
    "apply_transforms",
    "series_rebase",
    "compose_displacement",
    "signed_jacobian_determinant",
    "jacobian_modulation",
    "resample_series",
    "_sample_field_itk",
    "_reference_grid_points_lps",
    "_read_itk_affine",
    "_map_points_to_moving_tensor",
    "_as_direction",
]
