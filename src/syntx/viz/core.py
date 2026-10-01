"""
syntx.viz.core — oriented 2-D slice extraction used by the syntx figures
========================================================================

``AnatomicalVisualizer`` turns an ANTsImage, file path, tensor or array into a 2-D slice ready
for ``imshow``:

- ANTsImages are reoriented with ``reorient_image2("LPI")`` (ITK code; voxel axes then
  increase towards Right, Anterior, Superior, i.e. RAS+) unless ``reorient=False``.
- Planes: sagittal = ANTs axis 0 (x), coronal = axis 1 (y), axial = axis 2 (z). The slice is
  transposed and its rows reversed so that, for an LPI image, axial shows anterior up and
  coronal / sagittal show superior up; sagittal columns are also reversed (anterior on the
  viewer's left). Vector-field slices do not get the sagittal column flip.
- The returned aspect ratio is (row spacing) / (column spacing), for ``imshow(aspect=...)``.

``corner_watermark`` adds a bright noise patch to a corner of an image (a test helper).
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

    @staticmethod
    def prepare_image(img: Union[ants.ANTsImage, str, List, Tuple, np.ndarray], reorient: bool = True, ref_image=None) -> Tuple[Optional[ants.ANTsImage], np.ndarray, Tuple[float, ...]]:
        """Convert ``img`` to ``(ants_image_or_None, array, spacing)``.

        - ``str`` ending in .nii / .nii.gz / .mat: read with ``ants.image_read``; a failed read
          is silently ignored and the string falls through to the array path.
        - list / tuple: the first .nii / .nii.gz path in it is read; otherwise the first
          element is used if it is an ANTsImage (other lists go to the array path).
        - ANTsImage: reoriented to LPI if ``reorient`` (silently left as is if that fails);
          returns ``(image, image.numpy(), image.spacing)`` with the array in ANTs (x, y, z)
          order.
        - tensor / array: squeezed; a leading axis of size 2 or 3 (when the last axis is not
          2 or 3) is treated as channels and moved last. With an ANTsImage ``ref_image``:
          an array with ``ref_image.dimension`` axes is assumed to be in tensor order
          (z, y, x) / (y, x), transposed to ANTs order, given ``ref_image``'s geometry and
          reoriented like an ANTsImage; an array with one extra trailing axis of size 2 or 3
          is treated as a displacement field, transposed to ANTs order and returned without
          reorientation. Otherwise the array is returned as is (no transpose) with
          ``ref_image``'s spacing, or (1, 1, 1).

        Returns
        -------
        image : ANTsImage or None
        array : np.ndarray
        spacing : tuple of float
        """
        if isinstance(img, str) and (img.endswith('.nii.gz') or img.endswith('.nii') or img.endswith('.mat')):
            try:
                img = ants.image_read(img)
            except Exception:
                pass

        if isinstance(img, (list, tuple)):
            warp_files = [f for f in img if isinstance(f, str) and (f.endswith('.nii.gz') or f.endswith('.nii'))]
            if warp_files:
                img = ants.image_read(warp_files[0])
            elif len(img) > 0 and isinstance(img[0], ants.ANTsImage):
                img = img[0]

        ref_sp = ref_image.spacing if (ref_image is not None and isinstance(ref_image, ants.ANTsImage)) else None

        if isinstance(img, ants.ANTsImage):
            if reorient:
                try:
                    img_proc = img.reorient_image2("LPI")
                except Exception:
                    img_proc = img
            else:
                img_proc = img
            sp = img_proc.spacing
            arr = img_proc.numpy()
            return img_proc, arr, sp

        if hasattr(img, 'detach'):
            arr = img.detach().cpu().numpy()
        elif hasattr(img, 'numpy'):
            arr = img.numpy()
        else:
            arr = np.squeeze(np.asarray(img))

        arr = np.squeeze(arr)

        # Transpose PyTorch [C, H, W] or [C, D, H, W] channel-first format to channel-last format [H, W, C] / [D, H, W, C]
        if arr.ndim == 3 and arr.shape[0] in (2, 3) and arr.shape[-1] not in (2, 3):
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 4 and arr.shape[0] in (2, 3) and arr.shape[-1] not in (2, 3):
            arr = np.transpose(arr, (1, 2, 3, 0))

        if ref_image is not None and isinstance(ref_image, ants.ANTsImage):
            if arr.ndim == ref_image.dimension:
                # 3D: PyTorch/NumPy (Z, Y, X) -> ANTs (X, Y, Z) via transpose(2, 1, 0)
                # 2D: PyTorch/NumPy (H, W) -> ANTs (W, H) via arr.T
                arr_itk = arr.transpose(2, 1, 0) if arr.ndim == 3 else arr.T
                try:
                    img_ants = ants.from_numpy(arr_itk, origin=ref_image.origin, spacing=ref_image.spacing, direction=ref_image.direction)
                    if reorient:
                        try:
                            img_ants = img_ants.reorient_image2("LPI")
                        except Exception:
                            pass
                    return img_ants, img_ants.numpy(), img_ants.spacing
                except Exception:
                    pass
            elif arr.ndim == ref_image.dimension + 1 and arr.shape[-1] in (2, 3):
                # Displacement field (Z, Y, X, 3) -> (X, Y, Z, 3)
                arr_itk = arr.transpose(2, 1, 0, 3) if arr.ndim == 4 else np.transpose(arr, (1, 0, 2))
                try:
                    img_ants = ants.from_numpy(arr_itk, origin=ref_image.origin, spacing=ref_image.spacing, direction=ref_image.direction, has_components=True)
                    return img_ants, arr_itk, img_ants.spacing
                except Exception:
                    pass

        sp = ref_sp if ref_sp is not None else (1.0, 1.0, 1.0)
        return None, arr, sp

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

        Arrays are indexed in ANTs (x, y, z) order after ``prepare_image``.

        Parameters
        ----------
        img : ANTsImage, str, list, tuple, tensor or np.ndarray
            Image to slice.
        plane : str or int, default "axial"
            'sagittal' / 0 (slice along x; aspect sz / sy), 'coronal' / 1 (along y; aspect
            sz / sx) or 'axial' / 2 (along z; aspect sy / sx). Unknown strings mean axial.
        slice_idx : int, optional
            Index along the slicing axis, clamped to the valid range. If None, for a scalar
            3-D volume: the mean index of voxels > 0 (for axial, plus 10 % of their z extent),
            or 60 % of the depth (axial) / the middle (others) when no voxel is > 0; for a
            4-D vector field: 60 % (axial) / the middle (others).
        reorient : bool, default True
            Reorient ANTsImages to LPI first.
        ref_image : ANTsImage, optional
            Geometry for tensor / array inputs (see ``prepare_image``).

        Returns
        -------
        AnatomicalSlice
            Scalar 3-D: ``slice.T`` with rows reversed (sagittal: columns reversed too).
            Vector 4-D (x, y, z, C): spatial axes swapped and rows reversed, channels 0 and 1
            swapped (component values are not negated). 2-D array: transposed only (no row
            flip), ``plane`` ignored. 3-D array whose last axis is 2 or 3 and first axis > 4:
            treated as a 2-D displacement field, transposed, rows reversed and only
            channels [1, 0] kept.
        """
        _, arr, sp = cls.prepare_image(img, reorient=reorient, ref_image=ref_image)

        # Map plane parameter
        if isinstance(plane, int):
            plane_map = {0: "sagittal", 1: "coronal", 2: "axial"}
            plane_name = plane_map.get(plane, "axial")
            slice_axis = plane
        else:
            plane_name = plane.lower()
            axis_map = {"sagittal": 0, "coronal": 1, "axial": 2}
            slice_axis = axis_map.get(plane_name, 2)

        if arr.ndim <= 2:
            sl_2d = np.atleast_2d(np.squeeze(arr))
            asp = sp[1] / (sp[0] + 1e-8) if len(sp) >= 2 else 1.0
            return AnatomicalSlice(sl_2d.T, plane_name, asp, 0, sp)

        if arr.ndim == 3 and arr.shape[-1] in (2, 3) and arr.shape[0] > 4:
            asp = sp[1] / (sp[0] + 1e-8) if len(sp) >= 2 else 1.0
            # Transpose spatial dimensions 0 and 1, flip vertically [::-1, :], swapping u_y and u_x vector channels
            disp_trans = np.transpose(arr, (1, 0, 2))[::-1, :, [1, 0]]
            return AnatomicalSlice(disp_trans, plane_name, asp, 0, sp)

        if arr.ndim == 3:
            D, H, W = arr.shape
            if slice_idx is None:
                mask = (arr > 0)
                if np.any(mask):
                    idxs = np.where(mask)[slice_axis]
                    if slice_axis == 2:  # Axial: 10% more superior (5% inferior to previous 15%)
                        z_extent = np.max(idxs) - np.min(idxs)
                        slice_idx = int(np.mean(idxs) + 0.10 * z_extent)
                    else:
                        slice_idx = int(np.mean(idxs))
                else:
                    if slice_axis == 2:
                        slice_idx = int(arr.shape[slice_axis] * 0.60)
                    else:
                        slice_idx = arr.shape[slice_axis] // 2
            slice_idx = max(0, min(slice_idx, arr.shape[slice_axis] - 1))

            if slice_axis == 0:  # Sagittal (Y-Z plane)
                sl = arr[slice_idx, :, :]
                asp = sp[2] / (sp[1] + 1e-8) if len(sp) >= 3 else 1.0
            elif slice_axis == 1:  # Coronal (X-Z plane)
                sl = arr[:, slice_idx, :]
                asp = sp[2] / (sp[0] + 1e-8) if len(sp) >= 3 else 1.0
            else:  # Axial (X-Y plane)
                sl = arr[:, :, slice_idx]
                asp = sp[1] / (sp[0] + 1e-8) if len(sp) >= 2 else 1.0

            sl_2d = np.atleast_2d(np.squeeze(sl))
            # Real bug fixed here: the sagittal plane (Y-Z) has no left/right axis of its
            # own, so applying the exact same transform as axial/coronal leaves its
            # anterior-posterior direction unconstrained -- this function put anterior on
            # the viewer's RIGHT, while antsxfunctional.perfusion.figures.triplanar_montage
            # (used alongside this function in the same reports, e.g. antsxfunctional's PET
            # report mixes both) puts anterior on the viewer's LEFT. Confirmed visually on
            # real data: the same subject's sagittal midline slice rendered mirrored
            # between the two, a real cross-report inconsistency, not a cosmetic nitpick.
            # Match triplanar_montage's convention (anterior-left) with an extra column
            # flip for sagittal only.
            if slice_axis == 0:
                return AnatomicalSlice(sl_2d.T[::-1, ::-1], plane_name, asp, slice_idx, sp)
            return AnatomicalSlice(sl_2d.T[::-1, :], plane_name, asp, slice_idx, sp)

        if arr.ndim == 4:
            D, H, W, C = arr.shape
            if slice_idx is None:
                if slice_axis == 2:
                    slice_idx = int(arr.shape[slice_axis] * 0.60)
                else:
                    slice_idx = arr.shape[slice_axis] // 2
            slice_idx = max(0, min(slice_idx, arr.shape[slice_axis] - 1))

            if slice_axis == 0:
                sl = arr[slice_idx, :, :, :]
                asp = sp[2] / (sp[1] + 1e-8) if len(sp) >= 3 else 1.0
            elif slice_axis == 1:
                sl = arr[:, slice_idx, :, :]
                asp = sp[2] / (sp[0] + 1e-8) if len(sp) >= 3 else 1.0
            else:
                sl = arr[:, :, slice_idx, :]
                asp = sp[1] / (sp[0] + 1e-8) if len(sp) >= 2 else 1.0

            sl_trans = np.swapaxes(sl, 0, 1)[::-1, :]
            if C >= 2:
                sl_trans = sl_trans[..., [1, 0] + list(range(2, C))]
            return AnatomicalSlice(sl_trans, plane_name, asp, slice_idx, sp)

        sl_2d = np.atleast_2d(np.squeeze(arr))
        asp = sp[1] / (sp[0] + 1e-8) if len(sp) >= 2 else 1.0
        return AnatomicalSlice(sl_2d.T[::-1, :], plane_name, asp, 0, sp)

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
        the (1, 1, 1) spacing fallback.

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


def verify_anatomical_orientation(img_or_slice) -> bool:
    """Placeholder: always returns True without checking anything."""
    if isinstance(img_or_slice, AnatomicalSlice):
        return True
    return True


def corner_watermark(
    img: Union[ants.ANTsImage, np.ndarray],
    patch_size: int = 10,
    corner: str = "top_left"
) -> Union[ants.ANTsImage, np.ndarray]:
    """Return a copy of ``img`` with a block of bright uniform noise at array index 0.

    The block's values are drawn (unseeded ``np.random``) uniformly from
    [0.85 * max, max], where max is the image maximum (1.0 if the maximum is <= 0). The block
    always starts at index 0 of every spatial axis: ``[:p, :p]`` for 2-D arrays,
    ``[:p, :p, :p]`` for 3-D volumes, ``[:p, :p, :]`` for 3-D arrays whose last axis is 2 or 3
    and first two axes exceed ``p`` (2-D vector fields), and ``[0:1, :p, :p, :p]`` for 4-D
    arrays (``p`` capped at each axis size, including a trailing component axis).

    Parameters
    ----------
    img : ANTsImage, np.ndarray or torch.Tensor
        Input image; not modified.
    patch_size : int, default 10
        Block edge length ``p`` in voxels.
    corner : str, default "top_left"
        Ignored; the block is always at index 0.

    Returns
    -------
    Same type as ``img``: an ANTsImage with ``img``'s origin / spacing / direction, a tensor on
    ``img``'s device and dtype, or an np.ndarray.
    """
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
    # Determine slice ranges for patch_size
    s0 = slice(0, min(patch_size, arr.shape[0]))
    s1 = slice(0, min(patch_size, arr.shape[1]))

    if ndim == 2:
        noise_patch = np.random.uniform(0.85 * max_val, max_val, size=arr[s0, s1].shape)
        arr[s0, s1] = noise_patch
    elif ndim == 3:
        if arr.shape[-1] in (2, 3) and arr.shape[0] > patch_size and arr.shape[1] > patch_size:
            # 2D displacement field [H, W, dim]
            noise_patch = np.random.uniform(0.85 * max_val, max_val, size=arr[s0, s1, :].shape)
            arr[s0, s1, :] = noise_patch
        else:
            # 3D volume [D, H, W]
            s2 = slice(0, min(patch_size, arr.shape[2]))
            noise_patch = np.random.uniform(0.85 * max_val, max_val, size=arr[s0, s1, s2].shape)
            arr[s0, s1, s2] = noise_patch
    elif ndim == 4:
        # 3D displacement field or batched volume [1, D, H, W] or [1, H, W, dim]
        slices = [slice(0, 1)] + [slice(0, min(patch_size, arr.shape[i])) for i in range(1, ndim)]
        noise_patch = np.random.uniform(0.85 * max_val, max_val, size=arr[tuple(slices)].shape)
        arr[tuple(slices)] = noise_patch

    if is_ants:
        return ants.from_numpy(arr, origin=img.origin, spacing=img.spacing, direction=img.direction)
    elif is_torch:
        import torch
        return torch.from_numpy(arr).to(device=torch_device, dtype=torch_dtype)
    return arr

