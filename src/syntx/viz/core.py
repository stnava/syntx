"""
syntx.viz.core — oriented 2-D slice extraction used by the syntx figures
========================================================================

``AnatomicalVisualizer`` turns an ANTsImage, file path, tensor or array into a 2-D slice ready
for ``imshow``:

- ANTsImages are oriented to "LPI" display axes from their direction cosine matrix (voxel axes
  then increase towards Right, Anterior, Superior, i.e. RAS+) unless ``reorient=False``.
- Planes: sagittal = ANTs axis 0 (x), coronal = axis 1 (y), axial = axis 2 (z). The slice is
  transposed and its rows reversed so that, for an LPI image, axial shows anterior up and
  coronal / sagittal show superior up; sagittal columns are also reversed (anterior on the
  viewer's left). Vector fields get the same layout; their components are not changed.
- NumPy arrays and torch tensors are in syntx tensor layout ((z, y, x), components (z, y, x)).
- The returned aspect ratio is (row spacing) / (column spacing), for ``imshow(aspect=...)``.

``corner_watermark`` adds a bright (seeded) noise patch to a corner of an image (a test helper).
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ants


STRIATAL_AXIAL_CONTENT_FRACTION: float = 0.54


@dataclass(frozen=True)
class LpsSliceResolution:
    """Result of resolving corresponding LPI slice indices in physical LPS space."""

    fixed_slice_idx: int
    moving_slice_idx: int
    lps_coord_mm: Optional[float]
    moving_mapped_from_lps: bool


def _resolve_plane_axis(plane: Union[str, int]) -> Tuple[int, str]:
    planes = {"sagittal": 0, "coronal": 1, "axial": 2}
    if isinstance(plane, str) and plane.lower() in planes:
        plane_name = plane.lower()
        return planes[plane_name], plane_name
    if isinstance(plane, (int, np.integer)) and not isinstance(plane, bool) and 0 <= int(plane) <= 2:
        axis = int(plane)
        return axis, {0: "sagittal", 1: "coronal", 2: "axial"}[axis]
    raise ValueError(f"plane must be 'sagittal' / 'coronal' / 'axial' or 0 / 1 / 2, got {plane!r}")


def foreground_content_span(arr: np.ndarray, axis: int = 2) -> Tuple[int, int, bool]:
    """Return ``(idx_min, idx_max, has_foreground)`` along ``axis`` invariant to padding and background offset."""
    a = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    if a.ndim == 4 and axis in (0, 1, 2):
        a = np.linalg.norm(a, axis=-1)
    if a.ndim == 0 or a.size == 0:
        return (0, 0, False)
    if axis < 0 or axis >= a.ndim:
        raise ValueError(f"axis {axis} out of bounds for array of ndim {a.ndim}")

    a = np.asarray(a, dtype=np.float32)
    finite_mask = np.isfinite(a)
    if not np.any(finite_mask):
        return (0, 0, False)

    vals = a[finite_mask]
    amin = float(np.min(vals))
    amax = float(np.max(vals))
    if amax <= amin:
        return (0, 0, False)

    tol = 1e-6 * (amax - amin)
    near_min_frac = float(np.mean(vals <= amin + tol))
    if near_min_frac >= 0.15:
        bg = amin
        nonbg = vals[vals > bg + tol]
        hi = float(np.percentile(nonbg, 99)) if nonbg.size > 0 else amax
    else:
        bg = float(np.percentile(vals, 5))
        hi = float(np.percentile(vals, 99))

    if hi <= bg:
        hi = amax
    if hi <= bg:
        return (0, 0, False)

    thresh = bg + 0.05 * (hi - bg)
    fg = finite_mask & (a > thresh)
    if not np.any(fg):
        return (0, 0, False)

    other_axes = tuple(i for i in range(a.ndim) if i != axis)
    has_along = np.any(fg, axis=other_axes)
    idxs = np.where(has_along)[0]
    if idxs.size == 0:
        return (0, 0, False)
    return (int(idxs[0]), int(idxs[-1]), True)


def _content_span_fraction(arr: np.ndarray, slice_idx: int, slice_axis: int) -> float:
    """Return the clamped ``[0.0, 1.0]`` position of ``slice_idx`` inside ``foreground_content_span``."""
    a = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    if a.ndim == 0 or a.size == 0 or slice_axis < 0 or slice_axis >= a.ndim:
        return 0.50
    n = int(a.shape[slice_axis])
    if n <= 1:
        return 0.50
    idx_min, idx_max, has_fg = foreground_content_span(a, axis=slice_axis)
    if has_fg:
        if idx_max <= idx_min:
            return 0.50
        frac = (float(slice_idx) - float(idx_min)) / float(idx_max - idx_min)
    else:
        frac = float(slice_idx) / float(max(1, n - 1))
    return max(0.0, min(1.0, float(frac)))


def select_content_slice_index(
    arr: np.ndarray,
    plane: Union[str, int] = "axial",
    axial_fraction: float = STRIATAL_AXIAL_CONTENT_FRACTION,
    fraction: Optional[float] = None,
) -> int:
    """Select an anatomically consistent slice index from the foreground content span.

    Along the axial axis, ``axial_fraction`` (default ``0.54``) places the slice through
    the striatum (caudate and putamen) rather than using image-domain ``nz // 2``; sagittal
    and coronal planes center at ``0.50`` of the foreground content span unless ``fraction``
    is explicitly provided, in which case ``fraction`` (clamped to ``[0.0, 1.0]``) applies
    across all planes.
    """
    slice_axis, _ = _resolve_plane_axis(plane)
    a = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    if a.ndim == 2 or a.ndim == 0 or a.size == 0:
        return 0
    n = int(a.shape[slice_axis])
    if n <= 1:
        return 0
    if fraction is not None:
        frac = max(0.0, min(1.0, float(fraction)))
    else:
        frac = float(axial_fraction) if slice_axis == 2 else 0.50
    idx_min, idx_max, has_fg = foreground_content_span(a, axis=slice_axis)
    if has_fg:
        raw_idx = idx_min + round(frac * (idx_max - idx_min))
    else:
        raw_idx = round(frac * (n - 1))
    return max(0, min(int(raw_idx), n - 1))


class AnatomicalSlice:
    """A 2-D slice returned by ``AnatomicalVisualizer.extract_slice``.

    Attributes
    ----------
    data : np.ndarray
        Display-ready slice, shape (rows, cols) or (rows, cols, C) for vector / RGB data.
    plane : str
        Plane name, lower-cased ('axial', 'coronal', 'sagittal').
    aspect_ratio : float
        Row spacing / column spacing, for ``imshow(aspect=...)``.
    slice_idx : int
        Index along the slicing axis (0 for 2-D inputs).
    spacing : tuple of float
        Voxel spacing of the source image (ANTs (x, y, z) order, or the fallback (1, 1, 1)).
    """
    def __init__(self, data: np.ndarray, plane: str, aspect_ratio: float, slice_idx: int, spacing: Tuple[float, ...]):
        self.data = data
        self.plane = plane.lower()
        self.aspect_ratio = aspect_ratio
        self.slice_idx = slice_idx
        self.spacing = spacing

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of ``data``."""
        return self.data.shape


class AnatomicalVisualizer:
    """Single core engine for anatomical image slice extraction and standardized plotting."""

    _PLANES = {"sagittal": 0, "coronal": 1, "axial": 2}

    @staticmethod
    def prepare_image(img: Union[ants.ANTsImage, str, List, Tuple, np.ndarray], reorient: bool = True, ref_image=None) -> Tuple[ants.ANTsImage, np.ndarray, Tuple[float, ...]]:
        """Convert ``img`` to ``(ants_image, array, spacing)``; the array is ``image.numpy()``
        (ANTs (x, y, z[, C]) order).

        - ``str``: read with ``ants.image_read`` (read errors propagate; a ``.mat`` affine is
          not an image and raises ValueError).
        - list / tuple: the first .nii / .nii.gz path in it is read, else its first element
          if that is an ANTsImage; any other list is treated as an array.
        - NumPy array / torch tensor: syntx tensor layout -- spatial axes (z, y, x) / (y, x),
          optionally with a trailing component axis whose components are also in (z, y, x)
          order -- with or without ``ref_image`` (pass an ANTsImage for ANTs-order data). It
          is converted to an ANTs-order image (components reversed to (x, y, z)). The input
          is squeezed; a leading axis of size 2 or 3 is moved last as components when the
          remaining shape matches ``ref_image``'s grid (without ``ref_image``: when the last
          axis is not of size 2 or 3). With ``ref_image`` (an ANTsImage) the array must be on
          its grid (ValueError otherwise) and gets its geometry; without it, an (H, W, 2)
          array with H > 4 is a 2-D vector field, a (D, H, W, 3) array a 3-D vector field,
          and the image gets unit spacing, zero origin and identity direction.
        - 3-D images (scalar or vector) are then oriented to LPI display axes from their
          direction cosine matrix if ``reorient`` (voxels are reordered; vector components
          stay physical). 2-D images are never reoriented.

        Returns
        -------
        image : ANTsImage
        array : np.ndarray
        spacing : tuple of float
        """
        if isinstance(img, str):
            if img.endswith('.mat'):
                raise ValueError(f"prepare_image: {img!r} is an affine transform file, not an image")
            img = ants.image_read(img)

        if isinstance(img, (list, tuple)):
            warp_files = [f for f in img if isinstance(f, str) and (f.endswith('.nii.gz') or f.endswith('.nii'))]
            if warp_files:
                img = ants.image_read(warp_files[0])
            elif len(img) > 0 and isinstance(img[0], ants.ANTsImage):
                img = img[0]

        if not isinstance(img, ants.ANTsImage):
            img = AnatomicalVisualizer._array_to_image(img, ref_image)

        if reorient and img.dimension == 3:
            img = AnatomicalVisualizer._orient_lpi(img)
        return img, img.numpy(), img.spacing

    @staticmethod
    def _lpi_axis_mapping(img: ants.ANTsImage) -> Tuple[List[int], List[bool]]:
        """Return ``(perm, flips)`` mapping 3-D native voxel axes to canonical LPI axes.

        ``perm[lpi_ax] = native_ax`` and ``flips[lpi_ax]`` is True when the native axis
        is reversed to align with canonical LPI signs ``(-1, -1, +1)``.
        """
        D = np.asarray(img.direction, dtype=np.float64)
        target_signs = (-1.0, -1.0, 1.0)
        used: set[int] = set()
        perm: list[int] = []
        for p in range(3):
            order = np.argsort(-np.abs(D[p]))
            src_ax = next(int(j) for j in order if int(j) not in used)
            used.add(src_ax)
            perm.append(src_ax)
        flips: list[bool] = []
        for out_ax, src_ax in enumerate(perm):
            s = 1.0 if D[out_ax, src_ax] >= 0 else -1.0
            flips.append(bool(s != target_signs[out_ax]))
        return perm, flips

    @staticmethod
    def _orient_lpi(img: ants.ANTsImage) -> ants.ANTsImage:
        """Permute and flip 3-D voxel axes to canonical LPI display axes using ``img.direction``."""
        perm, flips = AnatomicalVisualizer._lpi_axis_mapping(img)
        D = np.asarray(img.direction, dtype=np.float64)
        arr = img.numpy()
        has_comp = bool(getattr(img, "has_components", False)) or arr.ndim == 4
        perm_axes = perm + ([3] if arr.ndim == 4 else [])
        arr_out = np.transpose(arr, perm_axes)
        corner_idx = [0, 0, 0]
        for out_ax, src_ax in enumerate(perm):
            if flips[out_ax]:
                arr_out = np.flip(arr_out, axis=out_ax)
                corner_idx[src_ax] = int(img.shape[src_ax]) - 1
        new_origin_arr = np.asarray(img.origin, dtype=np.float64) + D @ (
            np.asarray(corner_idx, dtype=np.float64) * np.asarray(img.spacing, dtype=np.float64)
        )
        new_origin = tuple(float(x) for x in new_origin_arr)
        new_spacing = tuple(float(img.spacing[j]) for j in perm)
        new_dir = np.diag([-1.0, -1.0, 1.0])
        return ants.from_numpy(
            np.ascontiguousarray(arr_out),
            origin=new_origin,
            spacing=new_spacing,
            direction=new_dir,
            has_components=has_comp,
        )

    @staticmethod
    def _array_to_image(img, ref_image=None) -> ants.ANTsImage:
        """ANTsImage of a tensor-layout array / tensor (see ``prepare_image``)."""
        arr = img.detach().cpu().numpy() if hasattr(img, 'detach') else np.asarray(img)
        arr = np.squeeze(arr).astype(np.float32)
        ref = ref_image if isinstance(ref_image, ants.ANTsImage) else None

        if ref is not None:
            grid = tuple(reversed(ref.shape))  # tensor order
            if arr.ndim == ref.dimension + 1 and arr.shape[0] in (2, 3) and arr.shape[1:] == grid:
                arr = np.moveaxis(arr, 0, -1)
            if arr.shape == grid:
                is_vector = False
            elif arr.ndim == ref.dimension + 1 and arr.shape[:-1] == grid and arr.shape[-1] == ref.dimension:
                is_vector = True
            else:
                raise ValueError(f"prepare_image: array of shape {arr.shape} is not on ref_image's grid "
                                 f"(tensor order {grid}, optionally with {ref.dimension} components)")
            geom = dict(origin=ref.origin, spacing=ref.spacing, direction=ref.direction)
        else:
            if arr.ndim in (3, 4) and arr.shape[0] in (2, 3) and arr.shape[-1] not in (2, 3):
                arr = np.moveaxis(arr, 0, -1)
            if (arr.ndim == 4 and arr.shape[-1] == 3) or (arr.ndim == 3 and arr.shape[-1] == 2 and arr.shape[0] > 4):
                is_vector = True
            elif arr.ndim in (2, 3):
                is_vector = False
            else:
                raise ValueError(f"prepare_image: cannot interpret an array of shape {arr.shape} as a 2-D / 3-D image")
            geom = {}
        if is_vector:
            d = arr.ndim - 1
            arr_itk = np.moveaxis(arr, list(range(d)), list(range(d - 1, -1, -1)))[..., ::-1]
        else:
            arr_itk = arr.T
        return ants.from_numpy(np.ascontiguousarray(arr_itk), has_components=is_vector, **geom)

    @classmethod
    def extract_slice(
        cls,
        img: Union[ants.ANTsImage, str, List, Tuple, np.ndarray],
        plane: Union[str, int] = "axial",
        slice_idx: Optional[int] = None,
        reorient: bool = True,
        ref_image=None
    ) -> AnatomicalSlice:
        """Extract one display-oriented 2-D slice from ``img`` (see ``prepare_image``).

        Parameters
        ----------
        img : ANTsImage, str, list, tuple, tensor or np.ndarray
            Image to slice (arrays / tensors in tensor layout; see ``prepare_image``).
        plane : str or int, default "axial"
            'sagittal' / 0 (slice along ANTs x; aspect sz / sy), 'coronal' / 1 (along y;
            aspect sz / sx) or 'axial' / 2 (along z; aspect sy / sx). Anything else raises
            ValueError. Ignored for 2-D images.
        slice_idx : int, optional
            Index along the slicing axis, clamped to the valid range. If None, for a scalar
            3-D volume: the mean index of voxels > 0 (for axial, plus 10 % of their z extent),
            or 60 % of the depth (axial) / the middle (others) when no voxel is > 0; for a
            vector field: 60 % (axial) / the middle (others).
        reorient : bool, default True
            Reorient 3-D images to LPI first.
        ref_image : ANTsImage, optional
            Geometry for tensor / array inputs (see ``prepare_image``).

        Returns
        -------
        AnatomicalSlice
            3-D: ``slice.T`` with rows reversed (sagittal: columns reversed too). 2-D:
            ``array.T`` (the ``ants.plot`` orientation; no row flip). Vector fields get the
            same spatial layout with a trailing component axis; the components are the
            image's physical (ANTs (x, y, z)) components, not display directions (for
            arrows use ``syntx.viz.plot_vector_field`` / ``plot_correspondence_vectors``).
        """
        if isinstance(plane, str) and plane.lower() in cls._PLANES:
            plane_name = plane.lower()
            slice_axis = cls._PLANES[plane_name]
        elif isinstance(plane, (int, np.integer)) and not isinstance(plane, bool) and 0 <= int(plane) <= 2:
            slice_axis = int(plane)
            plane_name = {0: "sagittal", 1: "coronal", 2: "axial"}[slice_axis]
        else:
            raise ValueError(f"plane must be 'sagittal' / 'coronal' / 'axial' or 0 / 1 / 2, got {plane!r}")

        image, arr, sp = cls.prepare_image(img, reorient=reorient, ref_image=ref_image)
        is_vector = image.components > 1

        if image.dimension == 2:
            asp = sp[1] / (sp[0] + 1e-8)
            data = np.swapaxes(arr, 0, 1) if is_vector else arr.T
            return AnatomicalSlice(data, plane_name, asp, 0, sp)

        n = arr.shape[slice_axis]
        if slice_idx is None:
            slice_idx = select_content_slice_index(arr, plane=slice_axis)
        slice_idx = max(0, min(int(slice_idx), n - 1))

        sl = np.take(arr, slice_idx, axis=slice_axis)
        if slice_axis == 0:  # sagittal (y-z plane)
            asp = sp[2] / (sp[1] + 1e-8)
        elif slice_axis == 1:  # coronal (x-z plane)
            asp = sp[2] / (sp[0] + 1e-8)
        else:  # axial (x-y plane)
            asp = sp[1] / (sp[0] + 1e-8)

        sl_2d = np.swapaxes(sl, 0, 1)[::-1]
        # the sagittal plane has no left/right axis of its own: an extra column flip puts
        # anterior on the viewer's LEFT, as antsxfunctional.perfusion.figures.triplanar_montage
        # (the two are mixed in the same reports; checked on real data)
        if slice_axis == 0:
            sl_2d = sl_2d[:, ::-1]
        return AnatomicalSlice(np.ascontiguousarray(sl_2d), plane_name, asp, slice_idx, sp)

    @classmethod
    def render_slice(
        cls,
        ax: plt.Axes,
        img: Union[ants.ANTsImage, str, List, Tuple, np.ndarray],
        plane: Union[str, int] = "axial",
        slice_idx: Optional[int] = None,
        reorient: bool = True,
        cmap: str = "gray",
        alpha: float = 1.0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        norm: Optional[mcolors.Normalize] = None,
        masked_zero: bool = False
    ):
        """Draw ``extract_slice(img, plane, slice_idx, reorient)`` on ``ax`` with ``imshow``.

        ``cmap``, ``alpha``, ``vmin``, ``vmax`` and ``norm`` are passed to ``imshow``; the
        slice's aspect ratio is used. ``masked_zero=True`` masks voxels equal to 0 (shown
        transparent). Turns the axes off. No ``ref_image`` can be passed, so array inputs use
        unit geometry.

        Returns
        -------
        (matplotlib.image.AxesImage, AnatomicalSlice)
        """
        slice_obj = cls.extract_slice(img, plane=plane, slice_idx=slice_idx, reorient=reorient)
        data = slice_obj.data
        if masked_zero:
            data = np.ma.masked_equal(data, 0)

        im = ax.imshow(
            data,
            cmap=cmap,
            alpha=alpha,
            aspect=slice_obj.aspect_ratio,
            vmin=vmin,
            vmax=vmax,
            norm=norm
        )
        ax.axis('off')
        return im, slice_obj


def _coerce_native_image(img, ref_image=None) -> Tuple[ants.ANTsImage, bool]:
    """Convert ``img`` to a native ``ANTsImage`` without LPI axis permutation.

    Returns ``(image, has_physical_grid)``. NumPy/torch arrays without a matching
    ``ref_image`` return ``has_physical_grid=False``.
    """
    if isinstance(img, str):
        if img.endswith(".mat"):
            raise ValueError(f"_coerce_native_image: {img!r} is an affine transform file, not an image")
        return ants.image_read(img), True
    if isinstance(img, (list, tuple)):
        warp_files = [f for f in img if isinstance(f, str) and (f.endswith(".nii.gz") or f.endswith(".nii"))]
        if warp_files:
            return ants.image_read(warp_files[0]), True
        if len(img) > 0 and isinstance(img[0], ants.ANTsImage):
            return img[0], True
    if isinstance(img, ants.ANTsImage):
        return img, True

    ref = ref_image if isinstance(ref_image, ants.ANTsImage) else None
    if ref is not None:
        arr = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
        arr_sq = np.squeeze(arr)
        grid = tuple(reversed(ref.shape))
        matches_ref = (
            arr_sq.shape == grid
            or (arr_sq.ndim == ref.dimension + 1 and arr_sq.shape[0] in (2, 3) and arr_sq.shape[1:] == grid)
            or (arr_sq.ndim == ref.dimension + 1 and arr_sq.shape[:-1] == grid and arr_sq.shape[-1] == ref.dimension)
        )
        if matches_ref:
            return AnatomicalVisualizer._array_to_image(img, ref), True
    return AnatomicalVisualizer._array_to_image(img, None), False


def _lpi_to_native_vox(img: ants.ANTsImage, lpi_vox: np.ndarray) -> np.ndarray:
    perm, flips = AnatomicalVisualizer._lpi_axis_mapping(img)
    native = np.zeros(3, dtype=np.float64)
    for out_ax, src_ax in enumerate(perm):
        val = float(lpi_vox[out_ax])
        if flips[out_ax]:
            val = float(img.shape[src_ax] - 1) - val
        native[src_ax] = val
    return native


def _native_to_lpi_vox(img: ants.ANTsImage, native_vox: np.ndarray) -> np.ndarray:
    perm, flips = AnatomicalVisualizer._lpi_axis_mapping(img)
    lpi = np.zeros(3, dtype=np.float64)
    for out_ax, src_ax in enumerate(perm):
        val = float(native_vox[src_ax])
        if flips[out_ax]:
            val = float(img.shape[src_ax] - 1) - val
        lpi[out_ax] = val
    return lpi


def _same_physical_grid_3d(a: ants.ANTsImage, b: ants.ANTsImage, tol: float = 1e-4) -> bool:
    if a.dimension != b.dimension or tuple(a.shape) != tuple(b.shape):
        return False
    return bool(
        np.allclose(a.spacing, b.spacing, atol=tol, rtol=tol)
        and np.allclose(a.origin, b.origin, atol=tol, rtol=tol)
        and np.allclose(a.direction, b.direction, atol=tol, rtol=tol)
    )


def _slice_centroid_2d(sl_2d: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(sl_2d, dtype=np.float32)
    if a.ndim == 3:
        a = np.linalg.norm(a, axis=-1)
    h, w = a.shape[:2]
    finite = np.isfinite(a)
    if np.any(finite):
        vals = a[finite]
        amin, amax = float(np.min(vals)), float(np.max(vals))
        if amax > amin:
            tol = 1e-6 * (amax - amin)
            near_min = float(np.mean(vals <= amin + tol))
            if near_min >= 0.15:
                bg = amin
                nonbg = vals[vals > bg + tol]
                hi = float(np.percentile(nonbg, 99)) if nonbg.size > 0 else amax
            else:
                bg = float(np.percentile(vals, 5))
                hi = float(np.percentile(vals, 99))
            if hi <= bg:
                hi = amax
            fg = finite & (a > bg + 0.05 * (hi - bg))
            if np.any(fg):
                r_idx, c_idx = np.where(fg)
                return float(np.mean(r_idx)), float(np.mean(c_idx))
    return (h - 1) * 0.5, (w - 1) * 0.5


def _dominant_physical_axis(img: ants.ANTsImage, native_axis: int) -> int:
    """Return the dominant physical LPS axis (0=L/R, 1=P/A, 2=I/S) for a 3-D native axis."""
    perm, _ = AnatomicalVisualizer._lpi_axis_mapping(img)
    for phys_ax, src_ax in enumerate(perm):
        if src_ax == native_axis:
            return phys_ax
    return native_axis


def resolve_lps_slice_pair(
    fixed,
    moving=None,
    plane: Union[str, int] = "axial",
    slice_idx: Optional[int] = None,
    reorient: bool = True,
    ref_image=None,
    match_physical_lps: bool = True,
) -> LpsSliceResolution:
    """Resolve content-span slice indices for ``fixed`` and ``moving`` linked in physical LPS mm.

    Uses ``syntx.spatial.vox_to_physical`` and ``syntx.spatial.physical_to_vox`` on native
    ``ANTsImage`` headers so oblique 3x3 direction cosine matrices, anisotropic spacings, and
    axis permutations map faithfully between ``fixed`` and ``moving`` when
    ``match_physical_lps=True``. When ``match_physical_lps=False`` on different physical grids
    (e.g., unregistered native scans in separate scanner coordinates), or when either input
    is 2-D, gridless, disjoint in LPS domain, or outside ``moving``'s foreground content span,
    ``moving`` is sliced at the matching fractional position inside its own foreground content
    span.
    """
    from ..spatial import physical_to_vox, vox_to_physical

    slice_axis, _ = _resolve_plane_axis(plane)
    fi_native, fi_has_grid = _coerce_native_image(fixed, ref_image=ref_image)
    if fi_native.dimension == 2:
        return LpsSliceResolution(
            fixed_slice_idx=0,
            moving_slice_idx=0,
            lps_coord_mm=None,
            moving_mapped_from_lps=False,
        )

    fi_lpi = AnatomicalVisualizer._orient_lpi(fi_native) if reorient else fi_native
    fi_lpi_arr = fi_lpi.numpy()
    n_fix = int(fi_lpi_arr.shape[slice_axis])
    if slice_idx is None:
        fixed_slice_idx = select_content_slice_index(fi_lpi_arr, plane=slice_axis)
        mov_target_fraction: Optional[float] = None
    else:
        fixed_slice_idx = max(0, min(int(slice_idx), n_fix - 1))
        mov_target_fraction = _content_span_fraction(fi_lpi_arr, fixed_slice_idx, slice_axis)

    fixed_phys_axis = slice_axis if reorient else _dominant_physical_axis(fi_native, slice_axis)
    lps_mm: Optional[np.ndarray] = None
    lps_coord_mm: Optional[float] = None
    if fi_has_grid:
        other_axes = [ax for ax in range(3) if ax != slice_axis]
        c_a, c_b = _slice_centroid_2d(np.take(fi_lpi_arr, fixed_slice_idx, axis=slice_axis))
        lpi_vox = np.zeros(3, dtype=np.float64)
        lpi_vox[slice_axis] = float(fixed_slice_idx)
        lpi_vox[other_axes[0]] = c_a
        lpi_vox[other_axes[1]] = c_b
        raw_vox = _lpi_to_native_vox(fi_native, lpi_vox) if reorient else lpi_vox
        lps_mm = vox_to_physical(fi_native, raw_vox)[0].astype(np.float64)
        if reorient or fixed_phys_axis == slice_axis:
            lps_coord_mm = float(lps_mm[fixed_phys_axis])

    if moving is None:
        return LpsSliceResolution(
            fixed_slice_idx=fixed_slice_idx,
            moving_slice_idx=fixed_slice_idx,
            lps_coord_mm=lps_coord_mm,
            moving_mapped_from_lps=False,
        )

    mov_ref = ref_image if isinstance(ref_image, ants.ANTsImage) else None
    mi_native, mi_has_grid = _coerce_native_image(moving, ref_image=mov_ref)
    if mi_native.dimension == 2:
        return LpsSliceResolution(
            fixed_slice_idx=fixed_slice_idx,
            moving_slice_idx=0,
            lps_coord_mm=lps_coord_mm,
            moving_mapped_from_lps=False,
        )

    mi_lpi = AnatomicalVisualizer._orient_lpi(mi_native) if reorient else mi_native
    mi_lpi_arr = mi_lpi.numpy()
    n_mov = int(mi_lpi_arr.shape[slice_axis])

    if not (fi_has_grid and mi_has_grid) or lps_mm is None:
        mov_idx = select_content_slice_index(mi_lpi_arr, plane=slice_axis, fraction=mov_target_fraction)
        return LpsSliceResolution(
            fixed_slice_idx=fixed_slice_idx,
            moving_slice_idx=mov_idx,
            lps_coord_mm=lps_coord_mm,
            moving_mapped_from_lps=False,
        )

    if not reorient:
        moving_phys_axis = _dominant_physical_axis(mi_native, slice_axis)
        if moving_phys_axis != fixed_phys_axis:
            mov_idx = select_content_slice_index(mi_lpi_arr, plane=slice_axis, fraction=mov_target_fraction)
            return LpsSliceResolution(
                fixed_slice_idx=fixed_slice_idx,
                moving_slice_idx=mov_idx,
                lps_coord_mm=lps_coord_mm,
                moving_mapped_from_lps=False,
            )

    if _same_physical_grid_3d(fi_native, mi_native):
        mov_idx = max(0, min(fixed_slice_idx, n_mov - 1))
        mov_min, mov_max, has_fg = foreground_content_span(mi_lpi_arr, axis=slice_axis)
        if not (has_fg and (mov_idx < mov_min or mov_idx > mov_max)):
            return LpsSliceResolution(
                fixed_slice_idx=fixed_slice_idx,
                moving_slice_idx=mov_idx,
                lps_coord_mm=lps_coord_mm,
                moving_mapped_from_lps=True,
            )
        mov_idx = select_content_slice_index(mi_lpi_arr, plane=slice_axis, fraction=mov_target_fraction)
        return LpsSliceResolution(
            fixed_slice_idx=fixed_slice_idx,
            moving_slice_idx=mov_idx,
            lps_coord_mm=lps_coord_mm,
            moving_mapped_from_lps=False,
        )

    if match_physical_lps:
        raw_mov_vox = physical_to_vox(mi_native, lps_mm[None, :])[0]
        mov_lpi_vox = _native_to_lpi_vox(mi_native, raw_mov_vox) if reorient else raw_mov_vox
        mapped = float(mov_lpi_vox[slice_axis])
        if np.isfinite(mapped) and (-0.5 <= mapped <= n_mov - 0.5):
            cand = max(0, min(int(round(mapped)), n_mov - 1))
            mov_min, mov_max, has_fg = foreground_content_span(mi_lpi_arr, axis=slice_axis)
            if not (has_fg and (cand < mov_min or cand > mov_max)):
                return LpsSliceResolution(
                    fixed_slice_idx=fixed_slice_idx,
                    moving_slice_idx=cand,
                    lps_coord_mm=lps_coord_mm,
                    moving_mapped_from_lps=True,
                )

    mov_idx = select_content_slice_index(mi_lpi_arr, plane=slice_axis, fraction=mov_target_fraction)
    return LpsSliceResolution(
        fixed_slice_idx=fixed_slice_idx,
        moving_slice_idx=mov_idx,
        lps_coord_mm=lps_coord_mm,
        moving_mapped_from_lps=False,
    )


_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")


def corner_watermark(
    img: Union[ants.ANTsImage, np.ndarray],
    patch_size: int = 10,
    corner: str = "top_left",
    seed: Optional[int] = 0,
) -> Union[ants.ANTsImage, np.ndarray]:
    """Return a copy of ``img`` with a block of bright uniform noise in one array corner.

    The block's values are drawn uniformly from [0.85 * max, max], where max is the image
    maximum (1.0 if the maximum is <= 0). ``corner`` names array-index corners of the first
    two spatial axes: "top" = the start of axis 0, "bottom" = its end, "left" = the start of
    axis 1, "right" = its end; any further spatial axis starts at index 0. Spatial axes: all
    of a 2-D array or 3-D volume, the first two of a 3-D array whose last axis is 2 or 3 and
    whose first two axes exceed ``p`` (2-D vector field; every component is overwritten),
    and axes 1-3 of a 4-D array (index 0 of axis 0 only). ``p`` is capped at each axis size.

    Parameters
    ----------
    img : ANTsImage, np.ndarray or torch.Tensor
        Input image; not modified.
    patch_size : int, default 10
        Block edge length ``p`` in voxels (>= 1).
    corner : {"top_left", "top_right", "bottom_left", "bottom_right"}, default "top_left"
    seed : int or None, default 0
        Seed of the noise (``np.random.default_rng``); None draws fresh noise.

    Returns
    -------
    Same type as ``img``: an ANTsImage with ``img``'s origin / spacing / direction, a tensor on
    ``img``'s device and dtype, or an np.ndarray.
    """
    if corner not in _CORNERS:
        raise ValueError(f"corner must be one of {_CORNERS}, got {corner!r}")
    if int(patch_size) < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    patch_size = int(patch_size)
    rng = np.random.default_rng(seed)
    is_ants = isinstance(img, ants.ANTsImage)
    is_torch = False

    if is_ants:
        arr = img.numpy().copy()
    elif hasattr(img, 'detach'):
        is_torch = True
        torch_device = img.device
        torch_dtype = img.dtype
        arr = img.detach().cpu().numpy().copy()
    else:
        arr = np.asarray(img).copy()

    max_val = float(np.max(arr))
    if max_val <= 0:
        max_val = 1.0

    ndim = arr.ndim
    if ndim == 2:
        spatial = [0, 1]
    elif ndim == 3 and arr.shape[-1] in (2, 3) and arr.shape[0] > patch_size and arr.shape[1] > patch_size:
        spatial = [0, 1]
    elif ndim == 3:
        spatial = [0, 1, 2]
    elif ndim == 4:
        spatial = [1, 2, 3]
    else:
        raise ValueError(f"corner_watermark: unsupported array shape {arr.shape}")

    from_end = {spatial[0]: corner.startswith("bottom"), spatial[1]: corner.endswith("right")}
    index = [slice(None)] * ndim
    if ndim == 4:
        index[0] = slice(0, 1)
    for ax in spatial:
        p = min(patch_size, arr.shape[ax])
        index[ax] = slice(arr.shape[ax] - p, arr.shape[ax]) if from_end.get(ax, False) else slice(0, p)
    index = tuple(index)
    arr[index] = rng.uniform(0.85 * max_val, max_val, size=arr[index].shape)

    if is_ants:
        return ants.from_numpy(arr, origin=img.origin, spacing=img.spacing, direction=img.direction)
    elif is_torch:
        import torch
        return torch.from_numpy(arr).to(device=torch_device, dtype=torch_dtype)
    return arr
