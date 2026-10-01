"""
syntx.viz.core — oriented 2-D slice extraction used by the syntx figures
========================================================================

``AnatomicalVisualizer`` turns an ANTsImage, file path, tensor or array into a 2-D slice ready
for ``imshow``:

- ANTsImages are reoriented to "LPI" with ``reorient_image2`` (ITK code; voxel axes then
  increase towards Right, Anterior, Superior, i.e. RAS+) unless ``reorient=False``.
- Planes: sagittal = ANTs axis 0 (x), coronal = axis 1 (y), axial = axis 2 (z). The slice is
  transposed and its rows reversed so that, for an LPI image, axial shows anterior up and
  coronal / sagittal show superior up; sagittal columns are also reversed (anterior on the
  viewer's left). Vector fields get the same layout; their components are not changed.
- NumPy arrays and torch tensors are in syntx tensor layout ((z, y, x), components (z, y, x)).
- The returned aspect ratio is (row spacing) / (column spacing), for ``imshow(aspect=...)``.

``corner_watermark`` adds a bright (seeded) noise patch to a corner of an image (a test helper).
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ants


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
        - 3-D images (scalar or vector) are then reoriented to LPI with ``reorient_image2``
          if ``reorient`` (voxels are reordered; vector components stay physical). 2-D
          images are never reoriented.

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
            img = img.reorient_image2("LPI")
        return img, img.numpy(), img.spacing

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
            mask = (arr > 0) if not is_vector else None
            if mask is not None and np.any(mask):
                idxs = np.where(mask)[slice_axis]
                if slice_axis == 2:  # axial: 10 % of the extent above the mean
                    slice_idx = int(np.mean(idxs) + 0.10 * (np.max(idxs) - np.min(idxs)))
                else:
                    slice_idx = int(np.mean(idxs))
            else:
                slice_idx = int(n * 0.60) if slice_axis == 2 else n // 2
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
