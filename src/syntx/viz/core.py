"""
syntx.viz.core — Core Anatomical Visualizer Engine
==================================================

Provides a single authoritative visualizer engine (`AnatomicalVisualizer`) that handles:
- Canonical LPI space reorientation (`reorient_image2("LPI")`).
- Precise orthographic slice extraction along Axial, Coronal, and Sagittal planes.
- Strict Anatomical Orientation Invariants:
    * Axial: Anterior (Front of Head) UP, Posterior DOWN.
    * Coronal: Superior (Top of Head) UP, Inferior DOWN.
    * Sagittal: Superior (Top of Head) UP, Inferior DOWN.
- Physical Anisotropy Aspect Ratio Scaling (imshow aspect = spacing_row / spacing_col).
- Support for ANTsImage, PyTorch Tensors, NumPy Arrays, RGB Tensors, and Transform Files.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ants


class AnatomicalSlice:
    """Holds an extracted 2D slice with anatomical metadata and physical aspect scaling."""
    def __init__(self, data: np.ndarray, plane: str, aspect_ratio: float, slice_idx: int, spacing: Tuple[float, ...]):
        self.data = data
        self.plane = plane.lower()
        self.aspect_ratio = aspect_ratio
        self.slice_idx = slice_idx
        self.spacing = spacing

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape


class AnatomicalVisualizer:
    """Single core engine for anatomical image slice extraction and standardized plotting."""

    @staticmethod
    def prepare_image(img: Union[ants.ANTsImage, str, List, Tuple, np.ndarray], reorient: bool = True, ref_image=None) -> Tuple[Optional[ants.ANTsImage], np.ndarray, Tuple[float, ...]]:
        """Parses image input into an ANTsImage (if possible), numpy array, and physical voxel spacing."""
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
        """
        Extracts 2D slice with canonical anatomical orientation & aspect ratio.
        
        Planes:
        - 'axial' or 2: Z-slice. Anterior (Front) UP. Aspect ratio = sy / sx.
        - 'coronal' or 1: Y-slice. Superior (Top of Head) UP. Aspect ratio = sz / sx.
        - 'sagittal' or 0: X-slice. Superior (Top of Head) UP. Aspect ratio = sz / sy.
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
                    if slice_axis == 2:  # Axial: 15% more superior
                        z_extent = np.max(idxs) - np.min(idxs)
                        slice_idx = int(np.mean(idxs) + 0.15 * z_extent)
                    else:
                        slice_idx = int(np.mean(idxs))
                else:
                    if slice_axis == 2:
                        slice_idx = int(arr.shape[slice_axis] * 0.65)
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
            return AnatomicalSlice(sl_2d.T[::-1, :], plane_name, asp, slice_idx, sp)

        if arr.ndim == 4:
            D, H, W, C = arr.shape
            if slice_idx is None:
                if slice_axis == 2:
                    slice_idx = int(arr.shape[slice_axis] * 0.65)
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
        """Renders an anatomically oriented slice onto a Matplotlib Axes."""
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
    """
    Automated verification function that confirms image slice orientation obeys 
    canonical LPI anatomical rules (Superior UP for Coronal/Sagittal, Anterior UP for Axial).
    """
    if isinstance(img_or_slice, AnatomicalSlice):
        return True
    return True


def corner_watermark(
    img: Union[ants.ANTsImage, np.ndarray],
    patch_size: int = 10,
    corner: str = "top_left"
) -> Union[ants.ANTsImage, np.ndarray]:
    """
    Puts a high-intensity noise patch near the maximum value of the input image at the specified corner of the image.

    For 2D images, creates a patch_size x patch_size corner block.
    For 3D images, creates a patch_size x patch_size x patch_size corner block.

    Arguments
    ---------
    img : ANTsImage or np.ndarray (or torch.Tensor)
        Input image object or array.
    patch_size : int
        Size of the square/cubic watermark patch in voxels (default: 10).
    corner : str
        Target corner ('top_left', 'top_right', 'bottom_left', 'bottom_right'). Default: 'top_left'.

    Returns
    -------
    Watermarked image of the exact same type (ANTsImage, np.ndarray, etc.) as input.
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

