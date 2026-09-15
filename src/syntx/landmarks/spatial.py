"""
syntx.landmarks.spatial — Canonical Physical-Space Coordinate and Display Framework
==================================================================================

Single source of truth for every voxel <-> physical conversion, tensor layout
convention, orthographic slice extraction and landmark overlay projection used
in ``syntx.landmarks``.

Conventions (ITK / ANTsPy)
--------------------------
* Physical coordinates are millimetres in **LPS** scanner space:
  +x = patient Left, +y = patient Posterior, +z = patient Superior.
* ``ants.ANTsImage.numpy()`` returns an array of shape ``(nx, ny, nz)`` indexed
  ``arr[ix, iy, iz]``.  ``vox_to_physical`` takes those XYZ indices.
* ``image_to_tensor`` keeps that layout: tensor ``[1, 1, nx, ny, nz]`` so that
  ``torch`` dims (2, 3, 4) are array axes (0, 1, 2) and ``np.argwhere`` on a
  squeezed tensor already yields ``(ix, iy, iz)``.  Nothing is transposed.
* ``vox_zyx_to_physical`` exists only for tensors that were *explicitly*
  transposed to ``(nz, ny, nx)`` (e.g. ``arr.transpose(2, 1, 0)``); do not use
  it on tensors built with ``image_to_tensor``.
* Mapping:  ``physical = origin + direction @ (index * spacing)``.

Display standards (no array reorientation, ever)
------------------------------------------------
* Axial   : Anterior UP, Posterior DOWN.
* Coronal : Superior UP, Inferior DOWN.
* Sagittal: Superior UP, Inferior DOWN, Anterior on the viewer's RIGHT.
* Left/Right on axial and coronal views follows ``convention``:
  ``'radiological'`` (default): patient Left on the viewer's RIGHT.
  ``'neurological'``           : patient Left on the viewer's LEFT.

All of this is derived from the direction cosine matrix, so two images stored
in different native frames (e.g. LAS vs RPS arrays, permuted axes, anisotropic
spacing) are displayed identically and their landmarks project consistently.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Anatomical letter for the negative / positive end of each LPS physical axis.
_LPS_LABELS = (("R", "L"), ("A", "P"), ("I", "S"))

# View definitions: which physical axis is the slice normal, which physical axes
# span the display horizontal (u) and vertical (v), and the *desired* direction
# of increase for v (u for L/R views depends on the convention).
#   axial   : normal z ; u = x (L/R) ; v = y with Anterior UP  -> v increases toward -y
#   coronal : normal y ; u = x (L/R) ; v = z with Superior UP  -> v increases toward +z
#   sagittal: normal x ; u = y with Anterior RIGHT (-y) ; v = z Superior UP (+z)
_VIEW_TABLE = {
    "axial":    dict(normal=2, u=0, v=1, u_sign=None, v_sign=-1),
    "coronal":  dict(normal=1, u=0, v=2, u_sign=None, v_sign=+1),
    "sagittal": dict(normal=0, u=1, v=2, u_sign=-1,   v_sign=+1),
}
_VIEW_ALIASES = {"ax": "axial", "cor": "coronal", "sag": "sagittal"}


# ---------------------------------------------------------------------------
# Affine and coordinate mapping
# ---------------------------------------------------------------------------

def get_image_affine(image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract ``(origin, spacing, direction)`` as float64 arrays of shape
    ``[3]``, ``[3]``, ``[3, 3]`` from an ``ants.ANTsImage`` or an
    ``(origin, spacing, direction)`` tuple.  2-D images are padded to 3-D.
    """
    if isinstance(image, tuple) and len(image) == 3:
        org, sp, D = image
        org = np.asarray(org, dtype=np.float64).ravel()
        sp = np.asarray(sp, dtype=np.float64).ravel()
        D = np.asarray(D, dtype=np.float64)
        if sp.size == 2:
            D2 = D.reshape(2, 2)
            D = np.eye(3)
            D[:2, :2] = D2
            sp = np.append(sp, 1.0)
            org = np.append(org, 0.0)
        elif sp.size >= 4:
            D_full = D.reshape(sp.size, sp.size)
            return org[:3], sp[:3], D_full[:3, :3]
        return org[:3], sp[:3], D.reshape(3, 3)
    if hasattr(image, "spacing"):
        sp = np.array(image.spacing, dtype=np.float64)
        org = np.array(image.origin, dtype=np.float64)
        D = np.array(image.direction, dtype=np.float64)
        if sp.size == 2:
            D2 = D.reshape(2, 2)
            D = np.eye(3)
            D[:2, :2] = D2
            sp = np.append(sp, 1.0)
            org = np.append(org, 0.0)
        elif sp.size >= 4:
            D_full = D.reshape(sp.size, sp.size)
            return org[:3], sp[:3], D_full[:3, :3]
        return org[:3], sp[:3], D.reshape(3, 3)
    return np.zeros(3), np.ones(3), np.eye(3)


def _as_points(x, n_cols: int = 3) -> np.ndarray:
    pts = np.asarray(x, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[np.newaxis]
    if pts.ndim != 2 or pts.shape[1] < n_cols:
        raise ValueError(f"expected [N, {n_cols}] array, got shape {pts.shape}")
    return pts[:, :n_cols]


def vox_to_physical(image, indices_xyz) -> np.ndarray:
    """
    ANTs array indices ``(ix, iy, iz)`` -> physical mm ``(x, y, z)``.

        physical = origin + direction @ (index * spacing)

    Matches ``ants.transform_index_to_physical_point`` (continuous indices allowed).
    Returns ``[N, 3]`` float32.
    """
    org, sp, D = get_image_affine(image)
    idx = _as_points(indices_xyz)
    if idx.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return (org + (idx * sp) @ D.T).astype(np.float32)


def vox_zyx_to_physical(image, indices_zyx) -> np.ndarray:
    """
    Indices of an explicitly transposed ``(nz, ny, nx)`` tensor -> physical mm.
    Only for tensors built as ``arr.transpose(2, 1, 0)``; tensors from
    ``image_to_tensor`` keep XYZ layout and must use ``vox_to_physical``.
    """
    idx = _as_points(indices_zyx)
    return vox_to_physical(image, idx[:, ::-1])


def physical_to_vox(image, points_mm) -> np.ndarray:
    """
    Physical mm ``(x, y, z)`` -> continuous ANTs array indices ``(ix, iy, iz)``.

        index = (direction^-1 @ (physical - origin)) / spacing

    Matches ``ants.transform_physical_point_to_index``.  Returns ``[N, 3]`` float64.
    """
    org, sp, D = get_image_affine(image)
    pts = _as_points(points_mm)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return ((pts - org) @ np.linalg.inv(D).T) / sp


def physical_offset_to_voxel(image, offsets_mm) -> np.ndarray:
    """
    Physical displacement vectors (mm, LPS axes) -> voxel displacement
    ``(dix, diy, diz)`` (float).  Ignores the origin.  Used to express
    neighbourhood offsets (MIND) and descriptor windows in scanner space so
    that images stored in different native frames produce comparable features.
    """
    _, sp, D = get_image_affine(image)
    off = _as_points(offsets_mm)
    return (off @ np.linalg.inv(D).T) / sp


def voxel_gradient_to_physical(image, grad_axes: np.ndarray) -> np.ndarray:
    """
    Convert per-array-axis finite differences (in intensity per *index*) to
    the physical gradient in intensity per mm along LPS axes.

        g_phys = direction @ (g_index / spacing)

    ``grad_axes`` : ``[..., 3]`` with last dim ordered (d/dix, d/diy, d/diz).
    """
    _, sp, D = get_image_affine(image)
    g = np.asarray(grad_axes, dtype=np.float64) / sp
    return g @ D.T


# ---------------------------------------------------------------------------
# Torch layout helpers
# ---------------------------------------------------------------------------

def image_to_tensor(image, device=None):
    """
    ``ants.ANTsImage`` / ndarray / tensor -> float32 tensor ``[1, 1, nx, ny, nz]``.

    The array layout is preserved (dims 2, 3, 4 == array axes 0, 1, 2), so
    ``np.argwhere(t.squeeze().cpu().numpy())`` returns ``(ix, iy, iz)`` rows
    that feed ``vox_to_physical`` directly.
    """
    import torch
    if hasattr(image, "numpy"):
        arr = image.numpy()
    elif isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[..., 0]
    t = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))
    while t.ndim < 5:
        t = t.unsqueeze(0)
    return t.to(device) if device is not None else t


def sample_tensor_at_physical(vol, image, points_mm, mode: str = "bilinear",
                              padding_mode: str = "border"):
    """
    Trilinearly sample a ``[1, C, nx, ny, nz]`` tensor (XYZ layout from
    ``image_to_tensor``) at physical mm coordinates.  Returns ``[N, C]``.

    ``torch.nn.functional.grid_sample`` expects grid coordinates ordered
    ``(W, H, D)`` = dims ``(4, 3, 2)`` = ``(iz, iy, ix)`` here, hence the
    reversed stacking below.  ``align_corners=True`` makes normalised
    coordinate -1 the centre of index 0 and +1 the centre of index n-1.
    """
    import torch
    import torch.nn.functional as F
    pts = _as_points(points_mm)
    C = vol.shape[1]
    if pts.shape[0] == 0:
        return torch.zeros((0, C), dtype=vol.dtype, device=vol.device)
    idx = physical_to_vox(image, pts)                          # [N, 3] (ix, iy, iz)
    n = np.array(vol.shape[2:5], dtype=np.float64)             # (nx, ny, nz)
    denom = np.maximum(n - 1.0, 1.0)
    norm = idx / denom * 2.0 - 1.0                             # [N, 3]
    grid = torch.from_numpy(norm[:, ::-1].copy().astype(np.float32)).to(vol.device)
    grid = grid.view(1, 1, 1, -1, 3)                           # (iz_n, iy_n, ix_n)
    out = F.grid_sample(vol, grid, mode=mode, padding_mode=padding_mode,
                        align_corners=True)                    # [1, C, 1, 1, N]
    return out.reshape(C, -1).permute(1, 0)


# ---------------------------------------------------------------------------
# Orientation analysis
# ---------------------------------------------------------------------------

def dominant_axes(direction) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each physical LPS axis ``p`` return the array axis ``a[p]`` whose
    index increment moves mostly along ``p``, and the sign ``s[p]`` of that
    motion.  Exact for axis-aligned direction matrices (any permutation and
    flips); oblique matrices are approximated and a warning is logged.
    """
    D = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    a = np.argmax(np.abs(D), axis=1)
    s = np.sign(D[np.arange(3), a]).astype(int)
    s[s == 0] = 1
    if len(set(a.tolist())) != 3:
        logger.warning("direction matrix is not a permutation of axes; falling back to identity mapping")
        a = np.arange(3)
        s = np.sign(np.diag(D)).astype(int)
        s[s == 0] = 1
    if np.max(np.abs(np.abs(D).sum(axis=1) - 1.0)) > 1e-3:
        logger.warning("oblique direction matrix; anatomical display axes are approximate")
    return a, s


def axis_orientation_code(image_or_direction) -> str:
    """
    Three-letter code naming the anatomical direction each *array* axis points
    to, e.g. ``'LAS'`` for direction ``diag(1, -1, 1)`` (ix -> Left,
    iy -> Anterior, iz -> Superior).
    """
    D = image_or_direction if isinstance(image_or_direction, np.ndarray) and np.asarray(image_or_direction).size == 9 \
        else get_image_affine(image_or_direction)[2]
    D = np.asarray(D, dtype=np.float64).reshape(3, 3)
    code = ""
    for a in range(3):
        p = int(np.argmax(np.abs(D[:, a])))
        code += _LPS_LABELS[p][1] if D[p, a] > 0 else _LPS_LABELS[p][0]
    return code


def anatomical_axis_labels(image_or_direction) -> list:
    """Per array axis: ``(label_at_index_0_end, label_at_index_max_end)``."""
    code = axis_orientation_code(image_or_direction)
    opposite = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}
    return [(opposite[c], c) for c in code]


def format_axis_xlabel(image_or_direction, array_axis: int, unit: str = "vox") -> str:
    """Matplotlib-ready label such as ``'ix  (R -> L) [vox]'`` for a given array axis."""
    lo, hi = anatomical_axis_labels(image_or_direction)[array_axis]
    return f"i{'xyz'[array_axis]}  ({lo} -> {hi}) [{unit}]"


# ---------------------------------------------------------------------------
# Orthographic views
# ---------------------------------------------------------------------------

def _resolve_convention(convention: str) -> int:
    """Return desired sign of physical x for increasing display u on L/R views."""
    c = (convention or "radiological").lower()
    if c.startswith("rad"):
        return +1     # u increases toward +x = patient Left on the viewer's right
    if c.startswith("neuro"):
        return -1     # u increases toward -x = patient Right on the viewer's right
    raise ValueError(f"unknown convention '{convention}' (use 'radiological' or 'neurological')")


def ortho_view_spec(image, view: str, center_mm=None, convention: str = "radiological") -> Dict[str, Any]:
    """
    Describe how to cut and arrange a native array so that the 2-D result obeys
    the display standard for ``view`` in {'axial', 'coronal', 'sagittal'}.

    Returns a dict with:
      name, convention
      slice_axis   : array axis that is held fixed
      slice_index  : integer index along slice_axis
      slice_pos_mm : physical coordinate of that plane along the normal axis
      normal_axis  : physical axis (0=x, 1=y, 2=z) normal to the plane
      normal_dir   : [3] unit vector (physical) of the array slice axis
      u_axis, v_axis : array axes displayed horizontally / vertically
      flip_u, flip_v : whether the display index runs opposite to the array index
      n_u, n_v     : display sizes
      aspect       : spacing[v_axis] / spacing[u_axis]
      labels       : dict(left, right, bottom, top) anatomical letters
    """
    view = _VIEW_ALIASES.get(view, view)
    if view not in _VIEW_TABLE:
        raise ValueError(f"unknown view '{view}'")
    tbl = _VIEW_TABLE[view]
    org, sp, D = get_image_affine(image)
    shape = tuple(int(s) for s in image.shape) if hasattr(image, "shape") else None
    a, s = dominant_axes(D)

    u_sign = tbl["u_sign"] if tbl["u_sign"] is not None else _resolve_convention(convention)
    v_sign = tbl["v_sign"]
    pu, pv, pn = tbl["u"], tbl["v"], tbl["normal"]
    u_axis, v_axis, slice_axis = int(a[pu]), int(a[pv]), int(a[pn])
    flip_u = bool(s[pu] != u_sign)
    flip_v = bool(s[pv] != v_sign)

    if center_mm is None:
        if hasattr(image, "numpy"):
            import ants
            center_mm = np.array(ants.get_center_of_mass(image), dtype=np.float64)
        else:
            center_mm = vox_to_physical(image, (np.array(shape) - 1) / 2.0)[0]
    ctr_vox = physical_to_vox(image, center_mm)[0]
    n_slice = shape[slice_axis] if shape is not None else None
    slice_index = int(round(ctr_vox[slice_axis]))
    if n_slice is not None:
        slice_index = int(np.clip(slice_index, 0, n_slice - 1))
    plane_vox = ctr_vox.copy()
    plane_vox[slice_axis] = slice_index
    plane_mm = vox_to_physical(image, plane_vox)[0]
    normal_dir = D[:, slice_axis] / (np.linalg.norm(D[:, slice_axis]) + 1e-12)

    lab_lo_u, lab_hi_u = _LPS_LABELS[pu]
    lab_lo_v, lab_hi_v = _LPS_LABELS[pv]
    labels = dict(
        left=lab_hi_u if u_sign < 0 else lab_lo_u,
        right=lab_hi_u if u_sign > 0 else lab_lo_u,
        bottom=lab_hi_v if v_sign < 0 else lab_lo_v,
        top=lab_hi_v if v_sign > 0 else lab_lo_v,
    )
    return dict(
        name=view, convention=convention,
        slice_axis=slice_axis, slice_index=slice_index,
        slice_pos_mm=float(plane_mm[pn]), plane_point_mm=plane_mm.astype(np.float32),
        normal_axis=pn, normal_dir=normal_dir,
        u_axis=u_axis, v_axis=v_axis, flip_u=flip_u, flip_v=flip_v,
        n_u=int(shape[u_axis]) if shape is not None else None,
        n_v=int(shape[v_axis]) if shape is not None else None,
        aspect=float(sp[v_axis] / (sp[u_axis] + 1e-12)),
        spacing_u=float(sp[u_axis]), spacing_v=float(sp[v_axis]),
        labels=labels,
    )


def slice_from_spec(arr: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    """
    Cut ``arr[ix, iy, iz]`` per ``spec`` and return a 2-D array ``[n_v, n_u]``
    to be shown with ``imshow(..., origin='lower')`` so that row index grows
    upward on screen.  Only slicing, transposition and reversal are used.
    """
    sl = np.take(arr, spec["slice_index"], axis=spec["slice_axis"])
    # remaining axes appear in increasing array-axis order
    remaining = [ax for ax in range(3) if ax != spec["slice_axis"]]
    if remaining.index(spec["v_axis"]) == 1:      # sl is [u, v] -> want [v, u]
        sl = sl.T
    if spec["flip_u"]:
        sl = sl[:, ::-1]
    if spec["flip_v"]:
        sl = sl[::-1, :]
    return np.ascontiguousarray(sl)


def extract_ortho_slices(image, center_mm=None, convention: str = "radiological") -> Dict[str, Any]:
    """
    Axial, coronal and sagittal display slices through ``center_mm`` (default:
    centre of mass) following the module's display standards.

    Returns dict with 2-D arrays ``'ax'``, ``'cor'``, ``'sag'`` (show with
    ``origin='lower'``), their ``aspect_*`` ratios, ``center_vox``,
    ``center_mm``, per-view ``flip_*`` tuples, and ``'views'`` holding the full
    ``ortho_view_spec`` dicts to pass to ``project_to_slice``.
    """
    arr = image.numpy()
    if center_mm is None:
        import ants
        center_mm = np.array(ants.get_center_of_mass(image), dtype=np.float64)
    center_mm = np.asarray(center_mm, dtype=np.float64)
    views = {k: ortho_view_spec(image, k, center_mm, convention) for k in ("ax", "cor", "sag")}
    out: Dict[str, Any] = {}
    for k, spec in views.items():
        out[k] = slice_from_spec(arr, spec)
        out[f"aspect_{k}"] = spec["aspect"]
        out[f"flip_{k}"] = (spec["flip_u"], spec["flip_v"])
        out[f"labels_{k}"] = spec["labels"]
    ctr_vox = np.rint(physical_to_vox(image, center_mm)[0]).astype(int)
    ctr_vox = np.clip(ctr_vox, 0, np.array(arr.shape) - 1)
    org, sp, D = get_image_affine(image)
    out.update(
        center_vox=ctr_vox,
        center_mm=center_mm.astype(np.float32),
        spacing=sp.astype(np.float32), origin=org.astype(np.float32), direction=D.astype(np.float32),
        orientation_code=axis_orientation_code(D),
        convention=convention,
        views=views,
    )
    return out


# ---------------------------------------------------------------------------
# Keypoint projection onto display slices
# ---------------------------------------------------------------------------

def project_to_slice(
    image,
    points_mm,
    view=None,
    slab_half_mm: Optional[float] = None,
    slice_axis: Optional[int] = None,
    slice_pos_mm: Optional[float] = None,
    flip_u: Optional[bool] = None,
    flip_v: Optional[bool] = None,
    convention: str = "radiological",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project physical points onto a display slice produced by
    ``extract_ortho_slices`` / ``slice_from_spec``.

    Preferred call: ``project_to_slice(image, pts, view=slices['views']['ax'])``.
    Legacy call: ``slice_axis`` (0=sagittal, 1=coronal, 2=axial, i.e. the
    *physical* normal axis) with ``slice_pos_mm``; the view spec is rebuilt from
    the image so the flips always agree with the extractor (explicit ``flip_*``
    arguments are ignored with a warning if they disagree).

    Returns ``(u, v, mask)``: display column / row coordinates (float32, in
    display pixel units matching ``imshow(origin='lower')``) and a boolean
    slab-membership mask (``|signed distance to plane| <= slab_half_mm``,
    default 4 * max spacing).
    """
    pts = _as_points(points_mm)
    org, sp, D = get_image_affine(image)
    if pts.shape[0] == 0:
        z = np.zeros(0, dtype=np.float32)
        return z, z.copy(), np.zeros(0, dtype=bool)

    if view is None:
        if slice_axis is None or slice_pos_mm is None:
            raise ValueError("pass view=spec or (slice_axis, slice_pos_mm)")
        name = {0: "sagittal", 1: "coronal", 2: "axial"}[int(slice_axis)]
        import ants
        c = np.array(ants.get_center_of_mass(image), dtype=np.float64) if hasattr(image, "numpy") else np.zeros(3)
        c[int(slice_axis)] = float(slice_pos_mm)
        view = ortho_view_spec(image, name, c, convention)
        if (flip_u is not None and bool(flip_u) != view["flip_u"]) or \
           (flip_v is not None and bool(flip_v) != view["flip_v"]):
            logger.warning("project_to_slice: explicit flip flags disagree with the direction matrix; using the matrix")

    if slab_half_mm is None:
        slab_half_mm = float(np.max(sp)) * 4.0

    dist = (pts - view["plane_point_mm"].astype(np.float64)) @ view["normal_dir"]
    mask = np.abs(dist) <= slab_half_mm

    vox = physical_to_vox(image, pts)
    u = vox[:, view["u_axis"]]
    v = vox[:, view["v_axis"]]
    if view["flip_u"]:
        u = (view["n_u"] - 1) - u
    if view["flip_v"]:
        v = (view["n_v"] - 1) - v
    return u.astype(np.float32), v.astype(np.float32), mask


# ---------------------------------------------------------------------------
# Transform list guard
# ---------------------------------------------------------------------------

def safe_whichtoinvert(transformlist: list, invert_flags: list) -> list:
    """Pad with ``False`` or truncate so ``len(whichtoinvert) == len(transformlist)``."""
    n, k = len(transformlist), len(invert_flags)
    if k == n:
        return list(invert_flags)
    if k < n:
        return list(invert_flags) + [False] * (n - k)
    return list(invert_flags[:n])


__all__ = [
    "get_image_affine", "vox_to_physical", "vox_zyx_to_physical", "physical_to_vox",
    "physical_offset_to_voxel", "voxel_gradient_to_physical",
    "image_to_tensor", "sample_tensor_at_physical",
    "dominant_axes", "axis_orientation_code", "anatomical_axis_labels", "format_axis_xlabel",
    "ortho_view_spec", "slice_from_spec", "extract_ortho_slices", "project_to_slice",
    "safe_whichtoinvert",
]
