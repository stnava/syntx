"""
syntx.viz.figures — matplotlib figures for registration and fMRI QC
===================================================================

Image figures use ``AnatomicalVisualizer.extract_slice`` (see ``syntx.viz.core`` for the
slice orientation and default-slice rules):

- ``render_input_pair_figure``: fixed and moving images before registration.
- ``render_standard_4panel``: deformed grid, Jacobian determinant, inverse-consistency error
  and edge overlap for one slice.
- ``plot_deformation_grid``, ``plot_edge_overlay``, ``plot_correspondence_vectors``,
  ``plot_vector_field``, ``plot_deformation_tensor_rgb``: single views of a field / pair.
- ``render_label_alignment_figure``, ``render_label_overlay_figure``,
  ``render_checkerboard_figure``: label and alignment views.
- ``plot_time_varying_velocity_grid``: keyframes of a time-varying velocity field.
- ``render_correlation_matrix_figure``, ``render_carpet_plot_figure``,
  ``render_motion_parameters_figure``: fMRI / connectivity plots from arrays.

Displacement-field slices: after ``extract_slice``, channel 0 is the ANTs y component and
channel 1 the x component whatever the plane; components are not negated when the slice
rows are flipped, and displacements are added to pixel positions without dividing by the
spacing (so grids / arrows are to scale only for 1 mm pixels).

This module defines its own ``get_dkt_colormap`` (default 256 labels, saturation 0.85),
``dkt_colormap`` and ``get_dkt_label_color_dict`` (tab20 colours by position), which shadow the
``syntx.viz.colormaps`` functions inside this module; ``syntx.viz`` exports the colormaps ones.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ants


from .core import AnatomicalSlice, AnatomicalVisualizer
from .colormaps import dkt_colormap, get_dkt_colormap, get_dkt_label_color_dict


def extract_oriented_slice(img, slice_axis: int = 2, slice_idx=None, reorient: bool = True, ref_image=None):
    """Return ``(slice_array, aspect_ratio)`` from ``AnatomicalVisualizer.extract_slice``.

    ``slice_axis`` 0 / 1 / 2 = sagittal / coronal / axial; the other arguments are passed
    through (see ``extract_slice``).
    """
    slice_obj = AnatomicalVisualizer.extract_slice(img, plane=slice_axis, slice_idx=slice_idx, reorient=reorient, ref_image=ref_image)
    return slice_obj.data, slice_obj.aspect_ratio


def extract_2d_slice(img, slice_axis: int = 2, slice_idx=None, ref_image=None):
    """Return only the slice array of ``extract_slice(img, slice_axis, slice_idx,
    reorient=False)``. ``ref_image`` is accepted but ignored."""
    slice_obj = AnatomicalVisualizer.extract_slice(img, plane=slice_axis, slice_idx=slice_idx, reorient=False)
    return slice_obj.data


def plot_deformation_grid(
    warp,
    fixed=None,
    slice_axis: int = 2,
    slice_idx=None,
    grid_spacing: int = 8,
    line_color: str = '#38bdf8',
    theme: str = "dark",
    reorient: bool = True,
    ax=None,
    figsize=(7, 7),
    title="Deformed Coordinate Mesh Grid",
    filename=None,
    show=False
):
    """Draw a deformed grid for one slice of a displacement field.

    Grid nodes every ``grid_spacing`` pixels are moved by the in-plane slice components
    (column += channel 1, row += channel 0; see the module notes on sign and units) and
    joined by lines, over the ``fixed`` slice in gray (alpha 0.5).

    Parameters
    ----------
    warp : ANTsImage, str, list, tensor or np.ndarray
        Displacement field (anything ``extract_slice`` accepts).
    fixed : image, optional
        Background; sliced separately, so with ``slice_idx=None`` its default slice can differ
        from the field's. Black background if None.
    slice_axis : int, default 2
        0 sagittal, 1 coronal, 2 axial.
    slice_idx : int, optional
        Slice index (see ``extract_slice`` for the default).
    grid_spacing : int, default 8
        Pixels between grid lines.
    line_color : str, default '#38bdf8'
    theme : str, default "dark"
        "dark", otherwise light background.
    reorient : bool, default True
        Reorient ANTsImages to LPI before slicing.
    ax : matplotlib Axes, optional
        Draw here; otherwise a new figure of size ``figsize`` is made.
    figsize : tuple, default (7, 7)
    title : str, default "Deformed Coordinate Mesh Grid"
        Empty / None for no title.
    filename : str, optional
        Save the figure here (dpi 200).
    show : bool, default False
        Call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
        Not closed.
    """
    disp, aspect_ratio = extract_oriented_slice(warp, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)
    if fixed is not None:
        fi_arr, _ = extract_oriented_slice(fixed, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)
    else:
        fi_arr = np.zeros(disp.shape[:2], dtype=np.float32)

    H, W = fi_arr.shape
    grid_y, grid_x = np.mgrid[0:H:grid_spacing, 0:W:grid_spacing]

    if disp.ndim >= 2 and disp.shape[-1] >= 2:
        disp_y = disp[::grid_spacing, ::grid_spacing, 0]
        disp_x = disp[::grid_spacing, ::grid_spacing, 1]
    else:
        disp_y, disp_x = 0, 0

    def_y = grid_y + disp_y
    def_x = grid_x + disp_x

    is_dark = (theme.lower() == "dark")
    bg_color = "#0b0f17" if is_dark else "#ffffff"
    text_color = "#f8fafc" if is_dark else "#0f172a"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    else:
        fig = ax.figure

    ax.set_facecolor(bg_color)
    ax.imshow(fi_arr, cmap='gray', alpha=0.5, aspect=aspect_ratio)

    for i in range(def_y.shape[0]):
        ax.plot(def_x[i, :], def_y[i, :], color=line_color, linewidth=1.1)
    for j in range(def_x.shape[1]):
        ax.plot(def_x[:, j], def_y[:, j], color=line_color, linewidth=1.1)

    ax.axis('off')
    if title:
        ax.set_title(title, color=text_color, fontsize=12, fontweight='bold', pad=10)

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor=bg_color)
    if show:
        plt.show()

    return fig


def plot_edge_overlay(
    fixed,
    warped,
    slice_axis: int = 2,
    slice_idx=None,
    edge_color='#f85149',
    fixed_edge_color=None,
    alpha=0.85,
    theme: str = "dark",
    reorient: bool = True,
    ax=None,
    figsize=(7, 7),
    title="Canny Edge Alignment Overlap",
    filename=None,
    show=False
):
    """Show the ``fixed`` slice in gray with the edges of the ``warped`` slice on top.

    Both slices are min-max normalised; ``warped`` is resized to the fixed slice's shape if
    they differ (``skimage.transform.resize``). Edges: ``skimage.feature.canny`` with sigma
    1.2; if scikit-image cannot be imported, the warped edges are Sobel gradient magnitudes
    above their 88th percentile and no fixed edges are drawn. Each image is sliced with its
    own default when ``slice_idx`` is None, so the two slices can differ.

    Parameters
    ----------
    fixed, warped : image
        Anything ``extract_slice`` accepts.
    slice_axis : int, default 2
        0 sagittal, 1 coronal, 2 axial.
    slice_idx : int, optional
    edge_color : str, default '#f85149'
        Colour of the warped-image edges.
    fixed_edge_color : str, optional
        If given, the fixed-image edges are drawn too, in this colour.
    alpha : float, default 0.85
        Edge opacity.
    theme, reorient, ax, figsize, title, filename, show
        As in ``plot_deformation_grid`` (default title "Canny Edge Alignment Overlap").

    Returns
    -------
    matplotlib.figure.Figure
        Not closed.
    """
    fi_arr, aspect_ratio = extract_oriented_slice(fixed, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)
    mi_arr, _ = extract_oriented_slice(warped, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)

    def _norm(a):
        amin, amax = np.min(a), np.max(a)
        return (a - amin) / (amax - amin + 1e-8) if amax > amin else a.copy()

    fi_norm = _norm(fi_arr)
    mi_norm = _norm(mi_arr)

    if fi_norm.shape != mi_norm.shape:
        from skimage.transform import resize
        mi_norm = resize(mi_norm, fi_norm.shape, mode='edge', anti_aliasing=True)

    try:
        from skimage.feature import canny
        edges_warped = canny(mi_norm, sigma=1.2)
        edges_fixed = canny(fi_norm, sigma=1.2) if fixed_edge_color else None
    except ImportError:
        from scipy.ndimage import sobel
        g_w = np.hypot(sobel(mi_norm, 0), sobel(mi_norm, 1))
        edges_warped = g_w > np.percentile(g_w, 88)
        edges_fixed = None

    is_dark = (theme.lower() == "dark")
    bg_color = "#0b0f17" if is_dark else "#ffffff"
    text_color = "#f8fafc" if is_dark else "#0f172a"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    else:
        fig = ax.figure

    ax.set_facecolor(bg_color)
    ax.imshow(fi_norm, cmap='gray', aspect=aspect_ratio)

    overlay_w = np.zeros((*fi_norm.shape, 4), dtype=np.float32)
    r, g, b = mcolors.to_rgb(edge_color)
    overlay_w[edges_warped] = [r, g, b, alpha]
    ax.imshow(overlay_w, aspect=aspect_ratio)

    if fixed_edge_color and edges_fixed is not None:
        overlay_f = np.zeros((*fi_norm.shape, 4), dtype=np.float32)
        rf, gf, bf = mcolors.to_rgb(fixed_edge_color)
        overlay_f[edges_fixed] = [rf, gf, bf, alpha]
        ax.imshow(overlay_f, aspect=aspect_ratio)

    ax.axis('off')
    if title:
        ax.set_title(title, color=text_color, fontsize=12, fontweight='bold', pad=10)

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor=bg_color)
    if show:
        plt.show()

    return fig

def plot_correspondence_vectors(
    warp,
    fixed=None,
    slice_axis: int = 2,
    slice_idx=None,
    subsample_step: int = 8,
    scale: float = 1.0,
    line_color: str = '#38bdf8',
    theme: str = "dark",
    reorient: bool = True,
    ax=None,
    figsize=(7, 7),
    title="Physical Correspondence Vector Display",
    filename=None,
    show=False
):
    """Quiver plot of a displacement-field slice over the ``fixed`` slice (gray, alpha 0.5).

    Arrows start every ``subsample_step`` pixels and have components (channel 1, channel 0)
    times ``scale``, drawn in pixel units (``scale_units='xy', scale=1``; see the module
    notes on sign and units).

    Parameters
    ----------
    warp : image
        Displacement field (anything ``extract_slice`` accepts).
    fixed : image, optional
        Background, sliced separately; black if None.
    slice_axis, slice_idx
        As in ``plot_deformation_grid``.
    subsample_step : int, default 8
        Pixels between arrows.
    scale : float, default 1.0
        Multiplier on arrow length.
    line_color : str, default '#38bdf8'
    theme, reorient, ax, figsize, filename, show
        As in ``plot_deformation_grid``.
    title : str, default "Physical Correspondence Vector Display"

    Returns
    -------
    matplotlib.figure.Figure
        Not closed.
    """
    disp, aspect_ratio = extract_oriented_slice(warp, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)
    if fixed is not None:
        fi_arr, _ = extract_oriented_slice(fixed, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)
    else:
        fi_arr = np.zeros(disp.shape[:2], dtype=np.float32)

    H, W = fi_arr.shape
    grid_y, grid_x = np.mgrid[0:H:subsample_step, 0:W:subsample_step]

    if disp.ndim >= 2 and disp.shape[-1] >= 2:
        disp_y = disp[::subsample_step, ::subsample_step, 0]
        disp_x = disp[::subsample_step, ::subsample_step, 1]
    else:
        disp_y, disp_x = np.zeros_like(grid_y), np.zeros_like(grid_x)

    is_dark = (theme.lower() == "dark")
    bg_color = "#0b0f17" if is_dark else "#ffffff"
    text_color = "#f8fafc" if is_dark else "#0f172a"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    else:
        fig = ax.figure

    ax.set_facecolor(bg_color)
    ax.imshow(fi_arr, cmap='gray', alpha=0.5, aspect=aspect_ratio)

    ax.quiver(
        grid_x, grid_y, disp_x * scale, disp_y * scale,
        color=line_color, angles='xy', scale_units='xy', scale=1.0,
        width=0.005, headwidth=4, headlength=4, headaxislength=3.5
    )

    ax.axis('off')
    if title:
        ax.set_title(title, color=text_color, fontsize=12, fontweight='bold', pad=10)

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor=bg_color)
    if show:
        plt.show()

    return fig


def plot_vector_field(
    warp,
    fixed=None,
    slice_axis: int = 2,
    slice_idx=None,
    subsample_step: int = 6,
    theme: str = "dark",
    reorient: bool = True,
    ax=None,
    figsize=(7, 7),
    title="Deformation Vector Field Overlay",
    filename=None,
    show=False
):
    """Displacement-magnitude heatmap plus quiver arrows for one slice of a field.

    The heatmap (magma, alpha 0.65, with a colorbar labelled mm) is the norm over all
    channels of the slice (all 3 components for a 3-D field), over the ``fixed`` slice
    (gray, alpha 0.4). Arrows every ``subsample_step`` pixels use the in-plane components
    at pixel scale (see the module notes).

    Parameters
    ----------
    warp : image
        Displacement field (anything ``extract_slice`` accepts).
    fixed : image, optional
        Background, sliced separately; black if None.
    slice_axis, slice_idx, theme, reorient, ax, figsize, filename, show
        As in ``plot_deformation_grid``.
    subsample_step : int, default 6
        Pixels between arrows.
    title : str, default "Deformation Vector Field Overlay"

    Returns
    -------
    matplotlib.figure.Figure
        Not closed.
    """
    disp, aspect_ratio = extract_oriented_slice(warp, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)
    if fixed is not None:
        fi_arr, _ = extract_oriented_slice(fixed, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)
    else:
        fi_arr = np.zeros(disp.shape[:2], dtype=np.float32)

    mag = np.linalg.norm(disp, axis=-1) if (disp.ndim >= 2 and disp.shape[-1] >= 2) else np.zeros_like(fi_arr)

    H, W = fi_arr.shape
    grid_y, grid_x = np.mgrid[0:H:subsample_step, 0:W:subsample_step]

    if disp.ndim >= 2 and disp.shape[-1] >= 2:
        disp_y = disp[::subsample_step, ::subsample_step, 0]
        disp_x = disp[::subsample_step, ::subsample_step, 1]
    else:
        disp_y, disp_x = np.zeros_like(grid_y), np.zeros_like(grid_x)

    is_dark = (theme.lower() == "dark")
    bg_color = "#0b0f17" if is_dark else "#ffffff"
    text_color = "#f8fafc" if is_dark else "#0f172a"
    cbar_tick_color = "#c9d1d9" if is_dark else "#334155"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    else:
        fig = ax.figure

    ax.set_facecolor(bg_color)
    ax.imshow(fi_arr, cmap='gray', alpha=0.4, aspect=aspect_ratio)
    im_mag = ax.imshow(mag, cmap='magma', alpha=0.65, aspect=aspect_ratio)

    ax.quiver(
        grid_x, grid_y, disp_x, disp_y,
        color='#38bdf8', angles='xy', scale_units='xy', scale=1.0,
        width=0.004, headwidth=3.5
    )

    cbar = fig.colorbar(im_mag, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Displacement Magnitude (mm)', color=cbar_tick_color, fontsize=10)
    cbar.ax.tick_params(colors=cbar_tick_color)

    ax.axis('off')
    if title:
        ax.set_title(title, color=text_color, fontsize=12, fontweight='bold', pad=10)

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor=bg_color)
    if show:
        plt.show()

    return fig


def compute_deformation_tensor_rgb(warp):
    """Colour-code the main stretch direction of a displacement field.

    F is the deformation gradient from ``ants.deformation_gradient`` (C++ backend). The RGB
    value per voxel is the absolute value of the eigenvector of C = F^T F with the largest
    eigenvalue (components x, y, z -> R, G, B; 2-D: R, G and B = 0), multiplied by
    ``clip(1.5 * (l_max - l_min) / (l_max + 1e-6), 0, 1)`` so that near-isotropic voxels are
    dark.

    If ``ants.deformation_gradient`` fails (or ``warp`` is an array), a NumPy fallback is used
    for 3-D arrays of shape (X, Y, Z, 3) in ANTs order. Note: that fallback fills F with the
    derivative axes permuted (``F[0, 0] = 1 + du_0/d(axis 1)``), so it is not the true
    deformation gradient.

    Parameters
    ----------
    warp : ANTsImage, str, list / tuple or np.ndarray
        Displacement field. A path ending in .nii / .nii.gz is read; from a list / tuple the
        first such path, else the first ANTsImage, is used.

    Returns
    -------
    ANTsImage
        float32, 3 components, the field's spacing (origin and direction are left at their
        defaults).

    Raises
    ------
    ValueError
        If the fallback path gets anything other than a 4-D array with 3 components.
    """
    if isinstance(warp, (list, tuple)):
        warp_files = [f for f in warp if isinstance(f, str) and (f.endswith('.nii.gz') or f.endswith('.nii'))]
        if warp_files: warp = ants.image_read(warp_files[0])
        elif len(warp) > 0 and isinstance(warp[0], ants.ANTsImage): warp = warp[0]

    if isinstance(warp, str) and (warp.endswith('.nii.gz') or warp.endswith('.nii')):
        warp = ants.image_read(warp)

    F_arr = None
    sp = (1.0, 1.0, 1.0)
    if isinstance(warp, ants.ANTsImage):
        sp = warp.spacing
        try:
            dg = ants.deformation_gradient(warp)
            F_arr = dg.numpy()
        except Exception:
            F_arr = None

    if F_arr is None:
        if isinstance(warp, ants.ANTsImage):
            arr = warp.numpy()
            sp = warp.spacing
        else:
            arr = np.squeeze(np.asarray(warp))

        if arr.ndim == 4 and arr.shape[-1] == 3:
            D, H, W, C = arr.shape
            gy, gx, gz = np.gradient(arr, sp[0], sp[1], sp[2], axis=(0, 1, 2))
            F_mat = np.zeros((D, H, W, 3, 3), dtype=np.float32)
            F_mat[..., 0, 0] = 1.0 + gx[..., 0]
            F_mat[..., 0, 1] = gy[..., 0]
            F_mat[..., 0, 2] = gz[..., 0]
            F_mat[..., 1, 0] = gx[..., 1]
            F_mat[..., 1, 1] = 1.0 + gy[..., 1]
            F_mat[..., 1, 2] = gz[..., 1]
            F_mat[..., 2, 0] = gx[..., 2]
            F_mat[..., 2, 1] = gy[..., 2]
            F_mat[..., 2, 2] = 1.0 + gz[..., 2]
        else:
            raise ValueError("compute_deformation_tensor_rgb: invalid displacement field array shape.")
    else:
        if F_arr.ndim == 4 and F_arr.shape[-1] == 9:
            D, H, W, C = F_arr.shape
            F_mat = F_arr.reshape(D, H, W, 3, 3)
        elif F_arr.ndim == 3 and F_arr.shape[-1] == 4:
            H, W, C = F_arr.shape
            F_mat = F_arr.reshape(H, W, 2, 2)
        else:
            F_mat = F_arr

    if F_mat.ndim == 5:
        C_mat = np.matmul(np.swapaxes(F_mat, -1, -2), F_mat)
        evals, evecs = np.linalg.eigh(C_mat)
        v_max = evecs[..., :, -1]
        rgb = np.abs(v_max)
        aniso = (evals[..., -1] - evals[..., 0]) / (evals[..., -1] + 1e-6)
        rgb_scaled = rgb * np.clip(aniso[..., None] * 1.5, 0.0, 1.0)
        return ants.from_numpy(rgb_scaled.astype(np.float32), spacing=sp, has_components=True)
    else:
        C_mat = np.matmul(np.swapaxes(F_mat, -1, -2), F_mat)
        evals, evecs = np.linalg.eigh(C_mat)
        v_max = evecs[..., :, -1]
        rgb_2d = np.zeros((*F_mat.shape[:2], 3), dtype=np.float32)
        rgb_2d[..., :2] = np.abs(v_max)
        aniso = (evals[..., -1] - evals[..., 0]) / (evals[..., -1] + 1e-6)
        rgb_scaled = rgb_2d * np.clip(aniso[..., None] * 1.5, 0.0, 1.0)
        return ants.from_numpy(rgb_scaled.astype(np.float32), spacing=sp, has_components=True)


def plot_deformation_tensor_rgb(
    warp,
    fixed=None,
    slice_indices=None,
    output_path=None,
    filename=None,
    alpha: float = 0.85,
    overlay_edges: bool = True,
    theme: str = "dark",
    reorient: bool = True,
    dpi: int = 150,
    title=None,
    show_figure=False
):
    """Axial / coronal / sagittal views of ``compute_deformation_tensor_rgb(warp)``.

    The RGB slices are clipped to [0, 1]. With ``fixed``, each is multiplied by the
    min-max-normalised fixed slice to the power 0.65 and, if ``overlay_edges``, Canny edges
    (sigma 1.2) of the fixed slice are drawn on top (failures silently skipped). Note: the
    RGB slices pass through ``extract_slice``'s vector path, which swaps channels 0 and 1, so
    the displayed red / green are the y / x directions, not x / y as the default title says.

    Parameters
    ----------
    warp : ANTsImage, str, list / tuple or np.ndarray
        Displacement field (see ``compute_deformation_tensor_rgb``).
    fixed : image, optional
        Anatomical background, sliced separately (the RGB image has default origin /
        direction, so the two are only aligned when ``fixed`` has them too).
    slice_indices : sequence of 3 int, optional
        (sagittal, coronal, axial) indices = ANTs (x, y, z). If None each image uses
        ``extract_slice``'s default.
    output_path, filename : str, optional
        Save the figure here (``output_path`` wins).
    alpha : float, default 0.85
        Ignored.
    overlay_edges : bool, default True
        Draw fixed-image edges (only with ``fixed``).
    theme : str, default "dark"
    reorient : bool, default True
    dpi : int, default 150
    title : str, optional
        Figure title; a default describing the colour code is used if None.
    show_figure : bool, default False
        Call ``plt.show()``; otherwise the figure is closed before being returned.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if output_path is not None:
        filename = output_path

    tensor_rgb_img = compute_deformation_tensor_rgb(warp)

    is_dark = (theme.lower() == "dark")
    bg_color = "#0b0f17" if is_dark else "#ffffff"
    text_color = "#f8fafc" if is_dark else "#0f172a"
    sub_color = "#94a3b8" if is_dark else "#475569"
    edge_color = "#38bdf8" if is_dark else "#0284c7"

    # Extract 3-panel orthographic slices via AnatomicalVisualizer
    ax_rgb, asp_ax = extract_oriented_slice(tensor_rgb_img, slice_axis=2, slice_idx=slice_indices[2] if slice_indices else None, reorient=reorient)
    cor_rgb, asp_cor = extract_oriented_slice(tensor_rgb_img, slice_axis=1, slice_idx=slice_indices[1] if slice_indices else None, reorient=reorient)
    sag_rgb, asp_sag = extract_oriented_slice(tensor_rgb_img, slice_axis=0, slice_idx=slice_indices[0] if slice_indices else None, reorient=reorient)

    if fixed is not None:
        ax_bg, _ = extract_oriented_slice(fixed, slice_axis=2, slice_idx=slice_indices[2] if slice_indices else None, reorient=reorient)
        cor_bg, _ = extract_oriented_slice(fixed, slice_axis=1, slice_idx=slice_indices[1] if slice_indices else None, reorient=reorient)
        sag_bg, _ = extract_oriented_slice(fixed, slice_axis=0, slice_idx=slice_indices[0] if slice_indices else None, reorient=reorient)
    else:
        ax_bg = np.ones(ax_rgb.shape[:2], dtype=np.float32)
        cor_bg = np.ones(cor_rgb.shape[:2], dtype=np.float32)
        sag_bg = np.ones(sag_rgb.shape[:2], dtype=np.float32)

    def _norm(a):
        amin, amax = np.min(a), np.max(a)
        return (a - amin) / (amax - amin + 1e-8) if amax > amin else a.copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=dpi, facecolor=bg_color)
    fig.subplots_adjust(wspace=0.15, left=0.05, right=0.95, top=0.85, bottom=0.08)

    for ax in axes:
        ax.set_facecolor(bg_color)
        ax.axis('off')

    slices = [
        (ax_bg, np.clip(ax_rgb, 0, 1), "Axial View (Anterior UP)", asp_ax),
        (cor_bg, np.clip(cor_rgb, 0, 1), "Coronal View (Superior UP)", asp_cor),
        (sag_bg, np.clip(sag_rgb, 0, 1), "Sagittal View (Superior UP)", asp_sag)
    ]

    for idx, (bg_raw, rgb_sl, label, aspect_ratio) in enumerate(slices):
        bg_norm = _norm(bg_raw)
        
        if fixed is not None:
            render_rgb = (bg_norm[..., None] ** 0.65) * rgb_sl
        else:
            render_rgb = rgb_sl

        axes[idx].imshow(render_rgb, aspect=aspect_ratio)

        if overlay_edges and fixed is not None:
            try:
                from skimage.feature import canny
                edges = canny(bg_norm, sigma=1.2)
                edge_overlay = np.zeros((*bg_norm.shape, 4), dtype=np.float32)
                r, g, b = mcolors.to_rgb(edge_color)
                edge_overlay[edges] = [r, g, b, 0.90]
                axes[idx].imshow(edge_overlay, aspect=aspect_ratio)
            except Exception:
                pass

        axes[idx].set_title(label, fontsize=11, fontweight='bold', color=sub_color)

    if title is None:
        title = "Deformation Gradient Tensor RGB Strain Map\n[Red: Left-Right | Green: Anterior-Posterior | Blue: Superior-Inferior]"
    fig.suptitle(title, fontsize=14, fontweight='bold', color=text_color, y=0.97)

    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor=bg_color)

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig


def get_dkt_colormap(max_label=256, lightness=0.68, saturation=0.85):
    """Module-local copy of ``syntx.viz.colormaps.get_dkt_colormap`` with different
    defaults (256 labels, HSV saturation 0.85): entry 0 transparent, entry ``i`` the ``i``-th
    golden-ratio hue step, alpha 0.90. ``lightness`` is the HSV value."""
    golden_ratio = 0.618033988749895
    colors = [(0.0, 0.0, 0.0, 0.0)]  # Label 0 = transparent background
    
    h = 0.125
    for i in range(1, max_label + 1):
        h = (h + golden_ratio) % 1.0
        rgb = mcolors.hsv_to_rgb((h, saturation, lightness))
        colors.append((*rgb, 0.90))

    return mcolors.ListedColormap(colors, name="dkt_colormap")

dkt_colormap = get_dkt_colormap()


def get_dkt_label_color_dict(unique_labels):
    """Unused: shadowed by the later ``get_dkt_label_color_dict`` definition in this module.
    Maps integer labels 1..256 to ``get_dkt_colormap()`` entries and other labels (as str) to
    entry ``(position + 1) % 257``."""
    cmap = get_dkt_colormap()
    color_map = {}
    for idx, l in enumerate(unique_labels):
        try:
            val = int(l)
            if 0 < val < len(cmap.colors):
                color_map[val] = cmap.colors[val]
            else:
                color_map[str(l)] = cmap.colors[(idx + 1) % len(cmap.colors)]
        except Exception:
            color_map[str(l)] = cmap.colors[(idx + 1) % len(cmap.colors)]

    return color_map


def render_input_pair_figure(
    fixed,
    moving,
    output_path=None,
    title=None,
    slice_indices=None,
    theme: str = "dark",
    crop_background: bool = True,
    reorient: bool = True,
    show_colorbar: bool = True,
    dpi=150,
    show_figure=False,
    filename=None
):
    """Show the fixed and moving images side by side before registration.

    3-D: a 2 x 3 grid, fixed on the top row and moving on the bottom, axial / coronal /
    sagittal. 2-D: fixed left, moving right. Images are shown in gray with their own
    autoscaled intensity range per panel.

    ANTsImages are reoriented to LPI if ``reorient``; arrays are used as given and indexed as
    if in ANTs (x, y, z) order (no spacing: aspect 1). Default slices (3-D): the mean index of
    voxels > 0 in each axis (axial plus 10 % of the z extent), else the middle (axial: 60 %);
    computed separately for fixed and moving.

    Note: the string literal placed after the first statement of this function is not a
    docstring and is partly wrong (e.g. sagittal shows anterior on the left, and
    ``slice_indices`` is in (x, y, z) order).

    Parameters
    ----------
    fixed, moving : ANTsImage, tensor or np.ndarray
        2-D or 3-D images (anything not 2-D is treated as 3-D).
    output_path : str, optional
        Save the figure here (parent directories are created).
    title : str, optional
        Figure title; a layout-describing default if None.
    slice_indices : tuple of 3 int, optional
        (sagittal, coronal, axial) = ANTs (x, y, z) indices, used for both images (3-D only).
    theme : str, default "dark"
        "dark", otherwise light colours.
    crop_background : bool, default True
        2-D: crop each image to the bounding box of voxels > 0 plus 4 pixels. 3-D: the
        bounding box is only used to clamp the slice indices; the panels are not cropped.
    reorient : bool, default True
        Reorient ANTsImages to LPI.
    show_colorbar : bool, default True
        2-D: one colorbar per panel. 3-D: one per row, taken from that row's axial panel
        (the other panels have their own ranges).
    dpi : int, default 150
    show_figure : bool, default False
        Call ``plt.show()``. The figure is never closed.
    filename : str, optional
        Alias for ``output_path`` (used only when ``output_path`` is None).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if output_path is None and filename is not None:
        output_path = filename
    """
    Renders standard Figure 1 visualization of input images prior to registration.
    
    Layout Invariants:
    * 3D Images: 2x3 panel layout with Fixed Image at top (Axial, Coronal, Sagittal)
      and Moving Image at bottom (Axial, Coronal, Sagittal).
    * 2D Images: 1x2 panel layout with Fixed Image on Left and Moving Image on Right.
    
    Colorbar Invariants:
    * Exactly 1 colorbar per image (1 shared colorbar for Fixed Image row, 1 shared colorbar for Moving Image row).
    
    Anatomical Orientation Invariants:
    * Axial: Anterior (Front) UP, Posterior (Back) DOWN.
    * Coronal: Superior (Top of Head) UP, Inferior DOWN.
    * Sagittal: Superior (Top of Head) UP, Anterior RIGHT.
    
    Args:
        fixed: Fixed target image (ANTsImage, PyTorch Tensor, or NumPy array).
        moving: Moving source image (ANTsImage, PyTorch Tensor, or NumPy array).
        output_path: Optional path to save PNG figure asset.
        title: Optional figure title.
        slice_indices: Optional tuple of slice indices (slice_z, slice_y, slice_x) for 3D images.
        theme: Color theme - 'dark' (default) or 'light'.
        crop_background: If True, crops empty zero-padding tightly around brain tissue (default: True).
        reorient: If True, reorients ANTsImages to canonical LPI anatomical space (default: True).
        show_colorbar: If True, displays 1 colorbar per image (default: True).
        dpi: Output figure DPI resolution (default: 150).
        show_figure: If True, calls plt.show() (default: False).
        
    Returns:
        matplotlib.figure.Figure: Generated Figure object.
    """
    if isinstance(fixed, ants.ANTsImage) and reorient:
        try: fixed_img = fixed.reorient_image2("LPI")
        except Exception: fixed_img = fixed
    else: fixed_img = fixed

    if isinstance(moving, ants.ANTsImage) and reorient:
        try: moving_img = moving.reorient_image2("LPI")
        except Exception: moving_img = moving
    else: moving_img = moving

    raw_fi = fixed_img.numpy() if isinstance(fixed_img, ants.ANTsImage) else np.squeeze(np.asarray(fixed_img))
    raw_mi = moving_img.numpy() if isinstance(moving_img, ants.ANTsImage) else np.squeeze(np.asarray(moving_img))

    dim = 2 if (fixed.dimension == 2 if isinstance(fixed, ants.ANTsImage) else raw_fi.ndim == 2) else 3
    if dim not in (2, 3):
        raise ValueError(f"render_input_pair_figure expects 2D or 3D images, got shape {raw_fi.shape}")

    if dim == 2:
        fi_arr, aspect_f = extract_oriented_slice(fixed_img, slice_axis=2, reorient=reorient)
        mi_arr, aspect_m = extract_oriented_slice(moving_img, slice_axis=2, reorient=reorient)
    else:
        fi_arr, mi_arr = raw_fi, raw_mi
        aspect_f, aspect_m = 1.0, 1.0

    # Theme parameters
    is_dark = (theme.lower() == "dark")
    bg_color = "#090d16" if is_dark else "#ffffff"
    text_color = "#f8fafc" if is_dark else "#0f172a"
    sub_color = "#94a3b8" if is_dark else "#475569"
    fixed_label_color = "#38bdf8" if is_dark else "#0284c7"
    moving_label_color = "#fb923c" if is_dark else "#ea580c"
    cbar_tick_color = "#c9d1d9" if is_dark else "#334155"

    if dim == 2:
        if crop_background:
            mask_f = (fi_arr > 0)
            if np.any(mask_f):
                rows_f = np.any(mask_f, axis=1)
                cols_f = np.any(mask_f, axis=0)
                rmin_f, rmax_f = np.where(rows_f)[0][[0, -1]]
                cmin_f, cmax_f = np.where(cols_f)[0][[0, -1]]
                pad = 4
                fi_render = fi_arr[max(0, rmin_f - pad):min(fi_arr.shape[0], rmax_f + pad),
                                   max(0, cmin_f - pad):min(fi_arr.shape[1], cmax_f + pad)]
            else:
                fi_render = fi_arr

            mask_m = (mi_arr > 0)
            if np.any(mask_m):
                rows_m = np.any(mask_m, axis=1)
                cols_m = np.any(mask_m, axis=0)
                rmin_m, rmax_m = np.where(rows_m)[0][[0, -1]]
                cmin_m, cmax_m = np.where(cols_m)[0][[0, -1]]
                pad = 4
                mi_render = mi_arr[max(0, rmin_m - pad):min(mi_arr.shape[0], rmax_m + pad),
                                   max(0, cmin_m - pad):min(mi_arr.shape[1], cmax_m + pad)]
            else:
                mi_render = mi_arr
        else:
            fi_render, mi_render = fi_arr, mi_arr

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi, facecolor=bg_color)
        fig.subplots_adjust(wspace=0.18, left=0.08, right=0.90, top=0.88, bottom=0.05)

        for ax in axes:
            ax.set_facecolor(bg_color)
            ax.axis('off')

        im0 = axes[0].imshow(fi_render, cmap='gray', aspect=aspect_f)
        axes[0].set_title("Fixed Image (Target)", fontsize=13, fontweight='bold', color=fixed_label_color, pad=8)
        if show_colorbar:
            cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            cb0.ax.tick_params(colors=cbar_tick_color, labelsize=9)

        im1 = axes[1].imshow(mi_render, cmap='gray', aspect=aspect_m)
        axes[1].set_title("Moving Image (Source)", fontsize=13, fontweight='bold', color=moving_label_color, pad=8)
        if show_colorbar:
            cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            cb1.ax.tick_params(colors=cbar_tick_color, labelsize=9)

        if title is None:
            title = "Figure 1: Input Fixed (Left) & Moving (Right) Images"
        fig.suptitle(title, fontsize=15, fontweight='bold', color=text_color, y=0.97)

    else:  # 3D
        shape_f = fi_arr.shape
        shape_m = mi_arr.shape

        def _get_bbox_3d(arr):
            mask = (arr > 0)
            if np.any(mask):
                d0_idxs, d1_idxs, d2_idxs = np.where(mask)
                pad = 4
                return (
                    max(0, np.min(d0_idxs) - pad), min(arr.shape[0], np.max(d0_idxs) + pad),
                    max(0, np.min(d1_idxs) - pad), min(arr.shape[1], np.max(d1_idxs) + pad),
                    max(0, np.min(d2_idxs) - pad), min(arr.shape[2], np.max(d2_idxs) + pad)
                )
            return (0, arr.shape[0], 0, arr.shape[1], 0, arr.shape[2])

        if crop_background:
            f0min, f0max, f1min, f1max, f2min, f2max = _get_bbox_3d(fi_arr)
            m0min, m0max, m1min, m1max, m2min, m2max = _get_bbox_3d(mi_arr)
        else:
            f0min, f0max, f1min, f1max, f2min, f2max = 0, shape_f[0], 0, shape_f[1], 0, shape_f[2]
            m0min, m0max, m1min, m1max, m2min, m2max = 0, shape_m[0], 0, shape_m[1], 0, shape_m[2]

        if slice_indices is not None:
            s0_f, s1_f, s2_f = slice_indices
            s0_m, s1_m, s2_m = slice_indices
        else:
            fi_mask = (fi_arr > 0)
            if np.any(fi_mask):
                i0, i1, i2 = np.where(fi_mask)
                z_ext_f = np.max(i2) - np.min(i2)
                s0_f, s1_f, s2_f = int(np.mean(i0)), int(np.mean(i1)), int(np.mean(i2) + 0.10 * z_ext_f)
            else:
                s0_f, s1_f, s2_f = shape_f[0] // 2, shape_f[1] // 2, int(shape_f[2] * 0.60)

            mi_mask = (mi_arr > 0)
            if np.any(mi_mask):
                j0, j1, j2 = np.where(mi_mask)
                z_ext_m = np.max(j2) - np.min(j2)
                s0_m, s1_m, s2_m = int(np.mean(j0)), int(np.mean(j1)), int(np.mean(j2) + 0.10 * z_ext_m)
            else:
                s0_m, s1_m, s2_m = shape_m[0] // 2, shape_m[1] // 2, int(shape_m[2] * 0.60)

        s0_f = max(f0min, min(f0max - 1, s0_f))
        s1_f = max(f1min, min(f1max - 1, s1_f))
        s2_f = max(f2min, min(f2max - 1, s2_f))

        s0_m = max(m0min, min(m0max - 1, s0_m))
        s1_m = max(m1min, min(m1max - 1, s1_m))
        s2_m = max(m2min, min(m2max - 1, s2_m))

        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), dpi=dpi, facecolor=bg_color)
        fig.subplots_adjust(wspace=0.18, hspace=0.25, left=0.10, right=0.90, top=0.88, bottom=0.05)

        for ax_row in axes:
            for ax in ax_row:
                ax.set_facecolor(bg_color)
                ax.axis('off')

        # Physical voxel spacing aspect ratios:
        sp_f = fixed_img.spacing if isinstance(fixed_img, ants.ANTsImage) else (1.0, 1.0, 1.0)
        sp_m = moving_img.spacing if isinstance(moving_img, ants.ANTsImage) else (1.0, 1.0, 1.0)

        asp_ax_f = sp_f[1] / (sp_f[0] + 1e-8)
        asp_cor_f = sp_f[2] / (sp_f[0] + 1e-8)
        asp_sag_f = sp_f[2] / (sp_f[1] + 1e-8)

        asp_ax_m = sp_m[1] / (sp_m[0] + 1e-8)
        asp_cor_m = sp_m[2] / (sp_m[0] + 1e-8)
        asp_sag_m = sp_m[2] / (sp_m[1] + 1e-8)

        # Fixed Image Slices (Top Row)
        fi_ax_sl, asp_ax_f = extract_oriented_slice(fixed_img, slice_axis=2, slice_idx=s2_f, reorient=reorient)
        fi_cor_sl, asp_cor_f = extract_oriented_slice(fixed_img, slice_axis=1, slice_idx=s1_f, reorient=reorient)
        fi_sag_sl, asp_sag_f = extract_oriented_slice(fixed_img, slice_axis=0, slice_idx=s0_f, reorient=reorient)

        slices_fixed = [
            (fi_ax_sl, f"Axial (Z={s2_f})", asp_ax_f),
            (fi_cor_sl, f"Coronal (Y={s1_f})", asp_cor_f),
            (fi_sag_sl, f"Sagittal (X={s0_f})", asp_sag_f)
        ]

        im_fixed = None
        for col_idx, (sl, label, aspect_ratio) in enumerate(slices_fixed):
            im = axes[0, col_idx].imshow(sl, cmap='gray', aspect=aspect_ratio)
            if col_idx == 0: im_fixed = im
            axes[0, col_idx].set_title(f"Fixed: {label}", fontsize=11, fontweight='bold', color=sub_color)

        if show_colorbar and im_fixed is not None:
            cb_fixed = fig.colorbar(im_fixed, ax=axes[0, :].ravel().tolist(), fraction=0.015, pad=0.03)
            cb_fixed.ax.tick_params(colors=cbar_tick_color, labelsize=9)

        # Moving Image Slices (Bottom Row)
        mi_ax_sl, asp_ax_m = extract_oriented_slice(moving_img, slice_axis=2, slice_idx=s2_m, reorient=reorient)
        mi_cor_sl, asp_cor_m = extract_oriented_slice(moving_img, slice_axis=1, slice_idx=s1_m, reorient=reorient)
        mi_sag_sl, asp_sag_m = extract_oriented_slice(moving_img, slice_axis=0, slice_idx=s0_m, reorient=reorient)

        slices_moving = [
            (mi_ax_sl, f"Axial (Z={s2_m})", asp_ax_m),
            (mi_cor_sl, f"Coronal (Y={s1_m})", asp_cor_m),
            (mi_sag_sl, f"Sagittal (X={s2_m})", asp_sag_m)
        ]

        im_moving = None
        for col_idx, (sl, label, aspect_ratio) in enumerate(slices_moving):
            im = axes[1, col_idx].imshow(sl, cmap='gray', aspect=aspect_ratio)
            if col_idx == 0: im_moving = im
            axes[1, col_idx].set_title(f"Moving: {label}", fontsize=11, fontweight='bold', color=sub_color)

        if show_colorbar and im_moving is not None:
            cb_moving = fig.colorbar(im_moving, ax=axes[1, :].ravel().tolist(), fraction=0.015, pad=0.03)
            cb_moving.ax.tick_params(colors=cbar_tick_color, labelsize=9)

        # Row Labels (Fixed Top / Moving Bottom)
        axes[0, 0].text(-0.22, 0.5, "FIXED\n(Top)", transform=axes[0, 0].transAxes,
                         fontsize=13, fontweight='bold', va='center', ha='center', color=fixed_label_color, rotation=90)
        axes[1, 0].text(-0.22, 0.5, "MOVING\n(Bottom)", transform=axes[1, 0].transAxes,
                         fontsize=13, fontweight='bold', va='center', ha='center', color=moving_label_color, rotation=90)

        if title is None:
            title = "Figure 1: Input Fixed (Top) & Moving (Bottom) Images (Tri-Planar Views)"
        fig.suptitle(title, fontsize=15, fontweight='bold', color=text_color, y=0.97)

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=bg_color)

    if show_figure:
        plt.show()
    return fig


def render_standard_4panel(
    fixed,
    warped,
    warp=None,
    detJ=None,
    inv_err_map=None,
    moving=None,
    slice_axis: int = 2,
    slice_idx=None,
    show_header: bool = False,
    theme: str = "dark",
    reorient: bool = True,
    lncc_val=None,
    mi_val=None,
    inv_err_max=None,
    inv_err_mean=None,
    inv_err_p95=None,
    min_detJ=None,
    title_prefix="Registration Report",
    filename=None,
    output_path=None
):
    """Four QC panels for one slice of a registration (2 x 2; 3 x 2 with ``show_header``).

    A: deformed grid of ``warp`` (every 8 pixels, over ``fixed``; see the module notes on
    sign and units). B: ``detJ`` with a seismic colormap, ``TwoSlopeNorm`` 0 / 1 / 2.5, voxels
    <= 0 painted green; title gives the min and the percentage <= 0 *of this slice*. C:
    ``inv_err_map`` (inferno, range 0 .. max(3, max error), over ``fixed``) with max / mean /
    95th percentile *of this slice*. D: ``plot_edge_overlay(fixed, warped)`` with the optional
    LNCC / MI values in the title. With ``show_header`` a first row shows ``fixed`` and
    ``moving`` (or ``warped`` if ``moving`` is None).

    If ``fixed`` is an ANTsImage, array inputs (``warped``, ``moving``, ``detJ``,
    ``inv_err_map``, ``warp``) are assumed to be in tensor order (z, y, x[, c]), transposed
    to ANTs order and given ``fixed``'s geometry (vector components are not reordered); a
    vector ``inv_err_map`` (last axis 2 or 3) is reduced to its norm. Each input is then
    sliced separately, so with ``slice_idx=None`` the panels can show different slices (the
    field and Jacobian get different default indices than ``fixed``). A failing ``warp`` /
    ``inv_err_map`` slice is silently replaced by zeros.

    Parameters
    ----------
    fixed, warped : image
        Fixed image and warped moving image.
    warp : image, optional
        Displacement field for panel A (zeros if missing).
    detJ : image, optional
        Jacobian determinant map for panel B. Required in practice (slicing None fails).
    inv_err_map : image
        Inverse-consistency error map (mm). Required.
    moving : image, optional
        Shown only in the header row.
    slice_axis : int, default 2
        0 sagittal, 1 coronal, 2 axial.
    slice_idx : int, optional
    show_header : bool, default False
    theme : str, default "dark"
    reorient : bool, default True
    lncc_val, mi_val : float, optional
        Printed in panel D's title.
    inv_err_max, inv_err_mean, inv_err_p95 : float, optional
        Override the slice statistics printed in panel C (``inv_err_max`` also sets the
        colour range).
    min_detJ : float, optional
        Override the min printed in panel B (the folding percentage is still per slice).
    title_prefix : str, default "Registration Report"
        First line of panel A's title.
    filename, output_path : str, optional
        Save the figure here at dpi 200 (``output_path`` wins).

    Returns
    -------
    matplotlib.figure.Figure
        Not closed.

    Raises
    ------
    ValueError
        If ``inv_err_map`` is None.
    """
    if output_path is not None:
        filename = output_path

    if inv_err_map is None:
        raise ValueError(
            "render_standard_4panel requires a valid inv_err_map (ANTsImage or Tensor representing physical inverse identity error in mm). "
            "Passing None or dummy objects is strictly prohibited per GEMINI.md Section 3."
        )

    # Ensure all scalar maps and displacement fields inherit spatial metadata from fixed ANTsImage
    if isinstance(fixed, ants.ANTsImage):
        if not isinstance(warped, ants.ANTsImage) and hasattr(warped, 'shape'):
            w_arr = np.squeeze(np.asarray(warped))
            has_comp = (w_arr.ndim == fixed.dimension + 1 and w_arr.shape[-1] in (2, 3))
            if not has_comp:
                if fixed.dimension == 2 and w_arr.ndim == 2: w_arr = w_arr.T
                elif fixed.dimension == 3 and w_arr.ndim == 3: w_arr = w_arr.transpose(2, 1, 0)
            else:
                if fixed.dimension == 2 and w_arr.ndim == 3: w_arr = np.transpose(w_arr, (1, 0, 2))
                elif fixed.dimension == 3 and w_arr.ndim == 4: w_arr = w_arr.transpose(2, 1, 0, 3)
            warped = ants.from_numpy(w_arr, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction, has_components=has_comp)
        if moving is not None and not isinstance(moving, ants.ANTsImage) and hasattr(moving, 'shape'):
            m_arr = np.squeeze(np.asarray(moving))
            has_comp = (m_arr.ndim == fixed.dimension + 1 and m_arr.shape[-1] in (2, 3))
            if not has_comp:
                if fixed.dimension == 2 and m_arr.ndim == 2: m_arr = m_arr.T
                elif fixed.dimension == 3 and m_arr.ndim == 3: m_arr = m_arr.transpose(2, 1, 0)
            else:
                if fixed.dimension == 2 and m_arr.ndim == 3: m_arr = np.transpose(m_arr, (1, 0, 2))
                elif fixed.dimension == 3 and m_arr.ndim == 4: m_arr = m_arr.transpose(2, 1, 0, 3)
            moving = ants.from_numpy(m_arr, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction, has_components=has_comp)
        if not isinstance(detJ, ants.ANTsImage) and hasattr(detJ, 'shape'):
            if hasattr(detJ, 'detach'):
                dj_arr = detJ.detach().cpu().numpy()
            else:
                dj_arr = np.squeeze(np.asarray(detJ))
            dj_arr = np.squeeze(dj_arr)
            if fixed.dimension == 2 and dj_arr.ndim == 2:
                dj_arr = dj_arr.T
            elif fixed.dimension == 3 and dj_arr.ndim == 3:
                dj_arr = dj_arr.transpose(2, 1, 0)
            detJ = ants.from_numpy(dj_arr, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction)
        if not isinstance(inv_err_map, ants.ANTsImage) and hasattr(inv_err_map, 'shape'):
            if hasattr(inv_err_map, 'detach'):
                inv_err_arr_raw = inv_err_map.detach().cpu().numpy()
            else:
                inv_err_arr_raw = np.squeeze(np.asarray(inv_err_map))
            inv_err_arr_raw = np.squeeze(inv_err_arr_raw)
            if inv_err_arr_raw.ndim in (3, 4) and inv_err_arr_raw.shape[-1] in (2, 3) and inv_err_arr_raw.shape[0] > 4:
                inv_err_arr_raw = np.linalg.norm(inv_err_arr_raw, axis=-1)
            if fixed.dimension == 2 and inv_err_arr_raw.ndim == 2:
                inv_err_arr_raw = inv_err_arr_raw.T
            elif fixed.dimension == 3 and inv_err_arr_raw.ndim == 3:
                inv_err_arr_raw = inv_err_arr_raw.transpose(2, 1, 0)
            inv_err_map = ants.from_numpy(inv_err_arr_raw, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction)

        if not isinstance(warp, ants.ANTsImage) and hasattr(warp, 'shape'):
            if hasattr(warp, 'detach'):
                w_disp = warp.detach().cpu().numpy()
            else:
                w_disp = np.squeeze(np.asarray(warp))
            w_disp = np.squeeze(w_disp)
            has_comp = (w_disp.ndim == fixed.dimension + 1 and w_disp.shape[-1] in (2, 3))
            if fixed.dimension == 2 and has_comp:
                w_disp = np.transpose(w_disp, (1, 0, 2))
            elif fixed.dimension == 3 and has_comp:
                w_disp = w_disp.transpose(2, 1, 0, 3)
            warp = ants.from_numpy(w_disp, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction, has_components=has_comp)


    fi_arr, aspect_ratio = extract_oriented_slice(fixed, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient, ref_image=fixed)
    warped_arr, _ = extract_oriented_slice(warped, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient, ref_image=fixed)
    detJ_arr, _ = extract_oriented_slice(detJ, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient, ref_image=fixed)
    
    try:
        disp, _ = extract_oriented_slice(warp, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient, ref_image=fixed)
        if not isinstance(disp, np.ndarray):
            disp = np.zeros((*fi_arr.shape, 2 if fixed.dimension == 2 else 3))
    except Exception:
        disp = np.zeros((*fi_arr.shape, 2 if fixed.dimension == 2 else 3))

    if moving is not None:
        mov_arr, _ = extract_oriented_slice(moving, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient, ref_image=fixed)
    else:
        mov_arr = warped_arr

    try:
        inv_err_arr, _ = extract_oriented_slice(inv_err_map, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient, ref_image=fixed)
        if inv_err_arr.ndim == 3:
            inv_err_arr = np.linalg.norm(inv_err_arr, axis=-1)
        inv_err_arr = np.asarray(inv_err_arr, dtype=np.float32)
    except Exception:
        inv_err_arr = np.zeros_like(fi_arr, dtype=np.float32)

    if not isinstance(inv_err_arr, np.ndarray) or inv_err_arr.size == 0:
        inv_err_arr = np.zeros_like(fi_arr, dtype=np.float32)

    is_dark = (theme.lower() == "dark")
    bg_color = "#0d1117" if is_dark else "#ffffff"
    text_color = "#f8fafc" if is_dark else "#0f172a"
    sub_color = "#94a3b8" if is_dark else "#475569"
    cbar_tick_color = "#c9d1d9" if is_dark else "#334155"

    if show_header:
        fig, axes = plt.subplots(3, 2, figsize=(14, 16), facecolor=bg_color)
        for ax_row in axes:
            for ax in ax_row:
                ax.set_facecolor(bg_color)
                ax.axis('off')

        # Row 0: Far Top Inputs (Fixed Left | Moving Right)
        axes[0, 0].imshow(fi_arr, cmap='gray', aspect=aspect_ratio)
        axes[0, 0].set_title('Fixed Target Image Input', color='#38bdf8', fontsize=11, fontweight='bold')

        axes[0, 1].imshow(mov_arr, cmap='gray', aspect=aspect_ratio)
        axes[0, 1].set_title('Moving / Warped Image Input', color='#fb923c', fontsize=11, fontweight='bold')

        ax_panel_a = axes[1, 0]
        ax_panel_b = axes[1, 1]
        ax_panel_c = axes[2, 0]
        ax_panel_d = axes[2, 1]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor=bg_color)
        for ax_row in axes:
            for ax in ax_row:
                ax.set_facecolor(bg_color)
                ax.axis('off')

        ax_panel_a = axes[0, 0]
        ax_panel_b = axes[0, 1]
        ax_panel_c = axes[1, 0]
        ax_panel_d = axes[1, 1]

    # Panel A: Standard Deformed Mesh Grid
    H, W = fi_arr.shape
    grid_spacing = 8
    grid_y, grid_x = np.mgrid[0:H:grid_spacing, 0:W:grid_spacing]
    if disp.ndim >= 2 and disp.shape[-1] >= 2:
        disp_y = disp[::grid_spacing, ::grid_spacing, 0][:grid_y.shape[0], :grid_x.shape[1]]
        disp_x = disp[::grid_spacing, ::grid_spacing, 1][:grid_y.shape[0], :grid_x.shape[1]]
    else:
        disp_x = 0
        disp_y = 0
    def_x = grid_x + disp_x
    def_y = grid_y + disp_y

    ax_panel_a.imshow(fi_arr, cmap='gray', alpha=0.5, aspect=aspect_ratio)
    for i in range(def_y.shape[0]):
        ax_panel_a.plot(def_x[i, :], def_y[i, :], color='#38bdf8', linewidth=1.1)
    for j in range(def_x.shape[1]):
        ax_panel_a.plot(def_x[:, j], def_y[:, j], color='#38bdf8', linewidth=1.1)
    ax_panel_a.set_title(f'{title_prefix}\nPanel A: Standard Deformed Mesh Grid', color='#38bdf8', fontsize=11, fontweight='bold')

    # Panel B: Standard Divergent Jacobian Determinant Map (seismic centered at 1.0)
    norm_jac = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.5)
    im_jac = ax_panel_b.imshow(detJ_arr, cmap='seismic', norm=norm_jac, aspect=aspect_ratio)
    
    # Overlay folding voxels (detJ <= 0) explicitly in high-visibility bright green
    folding_mask = (detJ_arr <= 0.0)
    if np.any(folding_mask):
        fold_overlay = np.zeros((*detJ_arr.shape, 4), dtype=np.float32)
        fold_overlay[folding_mask] = [0.0, 1.0, 0.0, 1.0]  # Bright solid green
        ax_panel_b.imshow(fold_overlay, aspect=aspect_ratio)

    folding_pct = float(np.mean(detJ_arr <= 0.0) * 100.0)
    min_j_val = min_detJ if min_detJ is not None else float(np.min(detJ_arr))
    status_str = "0.00% Folding" if folding_pct == 0.0 else f"{folding_pct:.4f}% Folding"
    title_color = "#3fb950" if folding_pct == 0.0 else "#f85149"
    fold_note = " [Green = Folding det(J) ≤ 0]" if np.any(folding_mask) else ""
    ax_panel_b.set_title(f'Panel B: Standard Jacobian det(J)\nmin det(J) = {min_j_val:+.6f} ({status_str}){fold_note}', color=title_color, fontsize=11, fontweight='bold')
    cbar_j = fig.colorbar(im_jac, ax=ax_panel_b, fraction=0.046, pad=0.04)
    cbar_j.set_label('det(J)', color=cbar_tick_color, fontsize=10)
    cbar_j.ax.tick_params(colors=cbar_tick_color)

    # Panel C: Standardized Inverse Identity Error Map (mm)
    max_err_val = inv_err_max if inv_err_max is not None else float(np.max(inv_err_arr))
    mean_err_val = inv_err_mean if inv_err_mean is not None else float(np.mean(inv_err_arr))
    p95_err_val = inv_err_p95 if inv_err_p95 is not None else float(np.percentile(inv_err_arr, 95))

    ax_panel_c.imshow(fi_arr, cmap='gray', alpha=0.3, aspect=aspect_ratio)
    im_err = ax_panel_c.imshow(inv_err_arr, cmap='inferno', alpha=0.85, vmin=0.0, vmax=max(3.0, max_err_val), aspect=aspect_ratio)
    ax_panel_c.set_title(f'Panel C: Inverse Error Map (mm)\nMax: {max_err_val:.6f}mm | Mean: {mean_err_val:.6f}mm | p95: {p95_err_val:.6f}mm', color='#d29922', fontsize=11, fontweight='bold')

    cbar_e = fig.colorbar(im_err, ax=ax_panel_c, fraction=0.046, pad=0.04)
    cbar_e.set_label('Inverse Error (mm)', color=cbar_tick_color, fontsize=10)
    cbar_e.ax.tick_params(colors=cbar_tick_color)

    # Panel D: Standardized High-Contrast Canny Edge Overlap
    plot_edge_overlay(fixed, warped, slice_axis=slice_axis, slice_idx=slice_idx, edge_color='#f85149', fixed_edge_color=None, theme=theme, reorient=reorient, ax=ax_panel_d, title="")
    lncc_str = f"Target LNCC: {lncc_val:.4f}" if lncc_val is not None else ""
    mi_str = f"Mattes MI: {mi_val:.4f}" if mi_val is not None else ""
    metrics_sub = " | ".join(filter(None, [lncc_str, mi_str]))
    ax_panel_d.set_title(f'Panel D: Edge Alignment Overlap (Canny Red Contours)\n{metrics_sub}', color='#bc8cff', fontsize=11, fontweight='bold')

    plt.tight_layout()

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor=bg_color)

    return fig


def get_dkt_label_color_dict(unique_labels):
    """Map labels to RGB colours from the tab20 + tab20b + tab20c palettes (60 colours) by
    position in ``unique_labels`` (cycling after 60).

    Labels that ``int()`` converts and are > 0 are keyed as int; all others as ``str``. This
    definition is the one used inside this module (it shadows the import from
    ``syntx.viz.colormaps``)."""
    clean_labels = []
    for l in unique_labels:
        try:
            val = int(l)
            if val > 0: clean_labels.append(val)
            else: clean_labels.append(str(l))
        except Exception:
            clean_labels.append(str(l))

    palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
    
    color_map = {}
    for idx, lid in enumerate(clean_labels):
        c = palette[idx % len(palette)]
        color_map[lid] = c
    return color_map


def render_label_alignment_figure(
    fixed_labels,
    warped_labels,
    fixed_image=None,
    colormap_type: str = "discrete",
    output_path=None,
    title=None,
    slice_indices=None,
    theme: str = "dark",
    crop_background: bool = True,
    reorient: bool = True,
    show_colorbar: bool = True,
    dpi=150,
    show_figure=False
):
    """Fixed labels (top row) and warped labels (bottom row), axial / coronal / sagittal.

    3-D only. ANTsImages are reoriented to LPI if ``reorient``; arrays are indexed as ANTs
    (x, y, z). Slices are cut directly (not via ``AnatomicalVisualizer``) and shown with
    ``np.rot90``, i.e. transposed with rows reversed; unlike ``extract_slice``, the sagittal
    view is not mirrored (anterior on the right). With ``crop_background`` all panels are
    cropped to the union bounding box (plus 4 voxels) of labels > 0 in both maps. Default
    slices: per map, the mean index of labels > 0 (axial plus 10 % of the z extent), clamped
    to the crop box.

    Parameters
    ----------
    fixed_labels, warped_labels : ANTsImage or np.ndarray
        Integer label maps on the same grid.
    fixed_image : ANTsImage or np.ndarray, optional
        Gray background (alpha 0.6) under both rows; must be on the label grid.
    colormap_type : str, default "discrete"
        "discrete": colours from ``build_dkt_label_palette`` over the labels of both maps
        (by rank), looked up by label value. Anything else: 'turbo' scaled from 1 to the
        largest label, with 0 masked.
    output_path : str, optional
        Save the figure here (parent directories are created).
    title : str, optional
    slice_indices : tuple of 3 int, optional
        ANTs (x, y, z) indices used for both maps.
    theme : str, default "dark"
    crop_background : bool, default True
    reorient : bool, default True
    show_colorbar : bool, default True
        Discrete: one colorbar for the figure listing the first 16 labels. Otherwise one per
        row.
    dpi : int, default 150
    show_figure : bool, default False
        Call ``plt.show()``; otherwise the figure is closed before being returned.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(fixed_labels, ants.ANTsImage) and reorient:
        try: fl_img = fixed_labels.reorient_image2("LPI")
        except Exception: fl_img = fixed_labels
    else: fl_img = fixed_labels

    if isinstance(warped_labels, ants.ANTsImage) and reorient:
        try: wl_img = warped_labels.reorient_image2("LPI")
        except Exception: wl_img = warped_labels
    else: wl_img = warped_labels

    if fixed_image is not None and isinstance(fixed_image, ants.ANTsImage) and reorient:
        try: fi_img = fixed_image.reorient_image2("LPI")
        except Exception: fi_img = fixed_image
    else: fi_img = fixed_image

    fl_arr = fl_img.numpy() if isinstance(fl_img, ants.ANTsImage) else np.squeeze(np.asarray(fl_img))
    wl_arr = wl_img.numpy() if isinstance(wl_img, ants.ANTsImage) else np.squeeze(np.asarray(wl_img))
    fi_arr = fi_img.numpy() if isinstance(fi_img, ants.ANTsImage) else (np.squeeze(np.asarray(fi_img)) if fi_img is not None else None)

    is_dark = (theme.lower() == "dark")
    bg_color = "#090d16" if is_dark else "#ffffff"
    text_color = "#f8fafc" if is_dark else "#0f172a"
    sub_color = "#94a3b8" if is_dark else "#475569"
    fixed_label_color = "#38bdf8" if is_dark else "#0284c7"
    moving_label_color = "#fb923c" if is_dark else "#ea580c"
    cbar_tick_color = "#c9d1d9" if is_dark else "#334155"

    shape_f = fl_arr.shape
    shape_w = wl_arr.shape

    def _get_bbox_3d(arr):
        mask = (arr > 0)
        if np.any(mask):
            d0, d1, d2 = np.where(mask)
            pad = 4
            return (
                max(0, np.min(d0) - pad), min(arr.shape[0], np.max(d0) + pad),
                max(0, np.min(d1) - pad), min(arr.shape[1], np.max(d1) + pad),
                max(0, np.min(d2) - pad), min(arr.shape[2], np.max(d2) + pad)
            )
        return (0, arr.shape[0], 0, arr.shape[1], 0, arr.shape[2])

    if crop_background:
        f0min, f0max, f1min, f1max, f2min, f2max = _get_bbox_3d(fl_arr)
        w0min, w0max, w1min, w1max, w2min, w2max = _get_bbox_3d(wl_arr)
        b0min, b0max = min(f0min, w0min), max(f0max, w0max)
        b1min, b1max = min(f1min, w1min), max(f1max, w1max)
        b2min, b2max = min(f2min, w2min), max(f2max, w2max)
    else:
        b0min, b0max = 0, shape_f[0]
        b1min, b1max = 0, shape_f[1]
        b2min, b2max = 0, shape_f[2]

    if slice_indices is not None:
        s0_f, s1_f, s2_f = slice_indices
        s0_w, s1_w, s2_w = slice_indices
    else:
        mask_f = (fl_arr > 0)
        if np.any(mask_f):
            i0, i1, i2 = np.where(mask_f)
            z_ext_f = np.max(i2) - np.min(i2)
            s0_f, s1_f, s2_f = int(np.mean(i0)), int(np.mean(i1)), int(np.mean(i2) + 0.10 * z_ext_f)
        else:
            s0_f, s1_f, s2_f = shape_f[0] // 2, shape_f[1] // 2, int(shape_f[2] * 0.60)

        mask_w = (wl_arr > 0)
        if np.any(mask_w):
            j0, j1, j2 = np.where(mask_w)
            z_ext_w = np.max(j2) - np.min(j2)
            s0_w, s1_w, s2_w = int(np.mean(j0)), int(np.mean(j1)), int(np.mean(j2) + 0.10 * z_ext_w)
        else:
            s0_w, s1_w, s2_w = shape_w[0] // 2, shape_w[1] // 2, int(shape_w[2] * 0.60)

    s0_f = max(b0min, min(b0max - 1, s0_f))
    s1_f = max(b1min, min(b1max - 1, s1_f))
    s2_f = max(b2min, min(b2max - 1, s2_f))

    s0_w = max(b0min, min(b0max - 1, s0_w))
    s1_w = max(b1min, min(b1max - 1, s1_w))
    s2_w = max(b2min, min(b2max - 1, s2_w))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), dpi=dpi, facecolor=bg_color)
    fig.subplots_adjust(wspace=0.18, hspace=0.25, left=0.10, right=0.90, top=0.88, bottom=0.05)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor(bg_color)
            ax.axis('off')

    sp_f = fl_img.spacing if isinstance(fl_img, ants.ANTsImage) else (1.0, 1.0, 1.0)
    sp_w = wl_img.spacing if isinstance(wl_img, ants.ANTsImage) else (1.0, 1.0, 1.0)

    asp_ax_f = sp_f[1] / (sp_f[0] + 1e-8)
    asp_cor_f = sp_f[2] / (sp_f[0] + 1e-8)
    asp_sag_f = sp_f[2] / (sp_f[1] + 1e-8)

    asp_ax_w = sp_w[1] / (sp_w[0] + 1e-8)
    asp_cor_w = sp_w[2] / (sp_w[0] + 1e-8)
    asp_sag_w = sp_w[2] / (sp_w[1] + 1e-8)

    # Build colormap (discrete vs continuous)
    from .colormaps import build_dkt_label_palette
    unique_labels = sorted(list(set(np.unique(fl_arr[fl_arr > 0])).union(set(np.unique(wl_arr[wl_arr > 0])))))
    color_dict, lut_rgba = build_dkt_label_palette(unique_labels)
    
    if colormap_type.lower() != "discrete":
        cmap_labels = plt.get_cmap('turbo').resampled(256)
        try:
            cmap_labels.set_under(color='black', alpha=0.0)
        except Exception:
            pass
        norm_labels = mcolors.Normalize(vmin=1, vmax=max(1, max(unique_labels) if unique_labels else 1))

    def _render_label_slice(ax_obj, sl_data, aspect_r):
        if colormap_type.lower() == "discrete":
            int_sl = np.clip(sl_data.astype(np.int64), 0, lut_rgba.shape[0] - 1)
            rgba_sl = lut_rgba[int_sl]
            return ax_obj.imshow(rgba_sl, aspect=aspect_r)
        else:
            return ax_obj.imshow(np.ma.masked_equal(sl_data, 0), cmap=cmap_labels, norm=norm_labels, aspect=aspect_r)

    # Fixed Labels
    ax_fl = np.rot90(fl_arr[b0min:b0max, b1min:b1max, s2_f])
    cor_fl = np.rot90(fl_arr[b0min:b0max, s1_f, b2min:b2max])
    sag_fl = np.rot90(fl_arr[s0_f, b1min:b1max, b2min:b2max])

    slices_fl = [
        (ax_fl, f"Axial (Z={s2_f})", asp_ax_f),
        (cor_fl, f"Coronal (Y={s1_f})", asp_cor_f),
        (sag_fl, f"Sagittal (X={s0_f})", asp_sag_f)
    ]

    im_fl = None
    for col_idx, (sl, label, aspect_ratio) in enumerate(slices_fl):
        if fi_arr is not None:
            bg_sl = np.rot90(fi_arr[b0min:b0max, b1min:b1max, s2_f] if col_idx == 0
                           else (fi_arr[b0min:b0max, s1_f, b2min:b2max] if col_idx == 1
                                 else fi_arr[s0_f, b1min:b1max, b2min:b2max]))
            axes[0, col_idx].imshow(bg_sl, cmap='gray', aspect=aspect_ratio, alpha=0.6)

        im = _render_label_slice(axes[0, col_idx], sl, aspect_ratio)
        if col_idx == 0: im_fl = im
        axes[0, col_idx].set_title(f"Fixed Labels: {label}", fontsize=11, fontweight='bold', color=sub_color)

    # Warped Labels
    ax_wl = np.rot90(wl_arr[b0min:b0max, b1min:b1max, s2_w])
    cor_wl = np.rot90(wl_arr[b0min:b0max, s1_w, b2min:b2max])
    sag_wl = np.rot90(wl_arr[s0_w, b1min:b1max, b2min:b2max])

    slices_wl = [
        (ax_wl, f"Axial (Z={s2_w})", asp_ax_w),
        (cor_wl, f"Coronal (Y={s1_w})", asp_cor_w),
        (sag_wl, f"Sagittal (X={s0_w})", asp_sag_w)
    ]

    im_wl = None
    for col_idx, (sl, label, aspect_ratio) in enumerate(slices_wl):
        if fi_arr is not None:
            bg_sl = np.rot90(fi_arr[b0min:b0max, b1min:b1max, s2_w] if col_idx == 0
                           else (fi_arr[b0min:b0max, s1_w, b2min:b2max] if col_idx == 1
                                 else fi_arr[s0_w, b1min:b1max, b2min:b2max]))
            axes[1, col_idx].imshow(bg_sl, cmap='gray', aspect=aspect_ratio, alpha=0.6)
        im = _render_label_slice(axes[1, col_idx], sl, aspect_ratio)
        if col_idx == 0: im_wl = im
        axes[1, col_idx].set_title(f"Warped Labels: {label}", fontsize=11, fontweight='bold', color=sub_color)

    # DKT-specific Colorbar (Discrete vs Continuous)
    if show_colorbar:
        if colormap_type.lower() == "discrete" and unique_labels:
            disp_labels = unique_labels[:16] if len(unique_labels) > 16 else unique_labels
            colors_sub = [color_dict.get(l, color_dict.get(str(l), (0.5, 0.5, 0.5, 1.0))) for l in disp_labels]
            cmap_discrete = mcolors.ListedColormap(colors_sub)
            norm_discrete = mcolors.BoundaryNorm(np.arange(len(disp_labels) + 1) - 0.5, len(disp_labels))
            sm = plt.cm.ScalarMappable(cmap=cmap_discrete, norm=norm_discrete)
            sm.set_array([])

            cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.03, ticks=np.arange(len(disp_labels)))
            cb.ax.tick_params(colors=cbar_tick_color, labelsize=8)
            cb.ax.set_yticklabels([str(l) for l in disp_labels])
            cb.ax.set_title("DKT Labels", color=text_color, fontsize=9, fontweight='bold', pad=6)
        else:
            if im_fl is not None:
                cb_f = fig.colorbar(im_fl, ax=axes[0, :].ravel().tolist(), fraction=0.015, pad=0.03)
                cb_f.ax.tick_params(colors=cbar_tick_color, labelsize=9)
            if im_wl is not None:
                cb_w = fig.colorbar(im_wl, ax=axes[1, :].ravel().tolist(), fraction=0.015, pad=0.03)
                cb_w.ax.tick_params(colors=cbar_tick_color, labelsize=9)

    axes[0, 0].text(-0.22, 0.5, "FIXED LABELS\n(Top)", transform=axes[0, 0].transAxes,
                     fontsize=12, fontweight='bold', va='center', ha='center', color=fixed_label_color, rotation=90)
    axes[1, 0].text(-0.22, 0.5, "WARPED LABELS\n(Bottom)", transform=axes[1, 0].transAxes,
                     fontsize=12, fontweight='bold', va='center', ha='center', color=moving_label_color, rotation=90)

    mode_str = "Discrete Qualitative" if colormap_type.lower() == "discrete" else "Continuous Gradient"
    if title is None:
        title = f"Anatomical Label Alignment ({mode_str} Colormap)"
    fig.suptitle(title, fontsize=15, fontweight='bold', color=text_color, y=0.97)

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=bg_color)

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_time_varying_velocity_grid(
    tvf_model,
    fixed_image=None,
    subsample_step: int = 8,
    mode: str = "hybrid",
    grid_scale: float = 8.0,
    quiver_scale: Optional[float] = None,
    theme: str = "dark",
    reorient: bool = True,
    figsize=None,
    title: str = "Time-Varying Velocity Field Keyframes Across Time",
    output_path: str = None,
    show_figure: bool = False
):
    """One panel per velocity keyframe of a time-varying velocity field (axial slice).

    Each keyframe (tensor order (Z, Y, X, 3) with components (v_z, v_y, v_x), or 2-D
    (Y, X, 2)) is converted with ``export_ants_displacement_field`` (geometry of
    ``fixed_image`` if it is an ANTsImage, else unit spacing) and its axial slice taken with
    ``extract_oriented_slice``. Note: the panels then use channel 0 as the horizontal and
    channel 1 as the vertical component, whereas ``extract_slice`` returns (y, x), so the
    arrows / grid have x and y swapped.

    Each title shows the keyframe's normalised time k / (T - 1), the slice's max ||v|| and a
    slice-mean bending-energy value (second differences of the two in-plane components).

    If every velocity value is below 1e-4 in magnitude and ``tvf_model`` has
    ``midpoint_warp_l2r``, the keyframes are replaced by displacement fields (0.5 * midpoint,
    midpoint, full forward warp for T = 3; 0.5 * midpoint and full for T = 2), so the
    figure then shows warps, not velocities.

    Parameters
    ----------
    tvf_model : object or array
        An object with a ``velocity`` attribute (tensor / array), or the velocity array
        itself: (T, B, Z, Y, X, 3) / (T, B, Y, X, 2) with B = 1 dropped, or a single
        (Z, Y, X, 3) / (Y, X, 2) field (T = 1).
    fixed_image : ANTsImage, optional
        Geometry for the export and gray background (sliced with its own default index).
    subsample_step : int, default 8
        Pixels between arrows / grid lines.
    mode : str, default "hybrid"
        "hybrid": magnitude heatmap (plasma) plus arrows; "quiver": arrows coloured by
        magnitude; "grid": grid deformed by ``grid_scale`` times the velocity.
    grid_scale : float, default 8.0
        Displacement multiplier in "grid" mode.
    quiver_scale : float, optional
        Matplotlib quiver ``scale``; default makes the largest arrow over all keyframes
        1.25 * ``subsample_step`` pixels long.
    theme : str, default "dark"
    reorient : bool, default True
    figsize : tuple, optional
        Default (4.5 * T, 5.0).
    title : str, default "Time-Varying Velocity Field Keyframes Across Time"
    output_path : str, optional
        Save at dpi 200 (parent directories are created).
    show_figure : bool, default False
        Call ``plt.show()``. The figure is never closed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    if hasattr(tvf_model, 'velocity'):
        vel_param = tvf_model.velocity
    elif hasattr(tvf_model, 'model') and hasattr(tvf_model['model'], 'velocity'):
        vel_param = tvf_model['model'].velocity
    else:
        vel_param = tvf_model

    if hasattr(vel_param, 'detach'):
        vel_np = vel_param.detach().cpu().numpy()
    else:
        vel_np = np.asarray(vel_param)
        
    # TVF velocity fields natively have shape (T, B, Z, Y, X, 3) or (T, B, Y, X, 2)
    # We must squeeze the Batch dimension (index 1) if it is 1, but preserve T.
    if vel_np.ndim >= 5 and vel_np.shape[1] == 1:
        vel_np = vel_np[:, 0, ...]
    elif vel_np.ndim == 6 and vel_np.shape[0] == 1 and vel_np.shape[1] == 1:
        vel_np = vel_np[0, 0, ...]
        
    if vel_np.ndim == 4 and vel_np.shape[-1] == 3: # (Z, Y, X, 3) -> T=1
        vel_np = np.expand_dims(vel_np, axis=0)
    elif vel_np.ndim == 3 and vel_np.shape[-1] == 2: # (Y, X, 2) -> T=1
        vel_np = np.expand_dims(vel_np, axis=0)

    T = vel_np.shape[0]

    # Reconstruct non-zero keyframe fields if vel_np is zero
    if float(np.max(np.abs(vel_np))) < 1e-4 and hasattr(tvf_model, 'midpoint_warp_l2r'):
        mid_warp = tvf_model.midpoint_warp_l2r.squeeze().detach().cpu().numpy()
        full_warp = tvf_model.full_forward_warp.squeeze().detach().cpu().numpy() if hasattr(tvf_model, 'full_forward_warp') and tvf_model.full_forward_warp is not None else mid_warp * 2.0
        v0_warp = mid_warp * 0.5
        if T == 3:
            vel_np = np.stack([v0_warp, mid_warp, full_warp], axis=0)
        elif T == 2:
            vel_np = np.stack([v0_warp, full_warp], axis=0)

    from ..transform import export_ants_displacement_field

    # Pre-extract all keyframe 2D slices to compute global max magnitude for perceptual auto-scaling
    v_slices = []
    max_mags = []
    for k in range(T):
        vel_k = vel_np[k]
        if fixed_image is not None and isinstance(fixed_image, ants.ANTsImage):
            vel_img = export_ants_displacement_field(
                vel_k, origin=fixed_image.origin, spacing=fixed_image.spacing, direction=fixed_image.direction
            )
        else:
            dim = 2 if (vel_k.ndim == 3 and vel_k.shape[-1] == 2) else 3
            sp = (1.0,) * dim
            orig = (0.0,) * dim
            dir_mat = np.eye(dim)
            vel_img = export_ants_displacement_field(vel_k, origin=orig, spacing=sp, direction=dir_mat)

        v_sl, _ = extract_oriented_slice(vel_img, slice_axis=2, reorient=reorient, ref_image=fixed_image)
        v_slices.append(v_sl)
        v_mag = np.linalg.norm(v_sl, axis=-1)
        max_mags.append(float(np.max(v_mag)))

    global_max_v_mag = max(max_mags) if max_mags else 0.0

    # Global perceptual quiver arrow auto-scaling:
    # Target maximum arrow length = 1.25 * subsample_step display pixels for clear, uncluttered visual perception
    if quiver_scale is not None:
        auto_scale = quiver_scale
    else:
        target_max_arrow_len = 1.25 * float(subsample_step)
        auto_scale = (global_max_v_mag / (target_max_arrow_len + 1e-8)) if global_max_v_mag > 1e-4 else 1.0

    is_dark = (theme.lower() == "dark")
    bg_color = "#0b0f17" if is_dark else "#ffffff"
    text_color = "#f8fafc" if is_dark else "#0f172a"
    sub_color = "#94a3b8" if is_dark else "#475569"

    if figsize is None:
        figsize = (4.5 * T, 5.0)

    fig, axes = plt.subplots(1, T, figsize=figsize, facecolor=bg_color)
    if T == 1:
        axes = [axes]

    fi_arr = None
    aspect_ratio = 1.0
    if fixed_image is not None:
        fi_arr, aspect_ratio = extract_oriented_slice(fixed_image, slice_axis=2, reorient=reorient)

    for k in range(T):
        ax = axes[k]
        ax.set_facecolor(bg_color)
        ax.axis('off')

        v_slice = np.squeeze(v_slices[k])
        if v_slice.ndim == 3 and v_slice.shape[0] == 1:
            v_slice = v_slice[0]
        H, W = v_slice.shape[:2]
        if fi_arr is None:
            bg_arr = np.zeros((H, W), dtype=np.float32)
        else:
            bg_arr = np.squeeze(fi_arr)

        v_mag = np.linalg.norm(v_slice, axis=-1)
        max_v_mag = max_mags[k]

        grid_y, grid_x = np.mgrid[0:H:subsample_step, 0:W:subsample_step]
        disp_x = v_slice[::subsample_step, ::subsample_step, 0][:grid_y.shape[0], :grid_x.shape[1]]
        disp_y = v_slice[::subsample_step, ::subsample_step, 1][:grid_y.shape[0], :grid_x.shape[1]]

        mode_clean = str(mode).lower()
        if mode_clean == "grid":
            def_x = grid_x + disp_x * grid_scale
            def_y = grid_y + disp_y * grid_scale
            ax.imshow(bg_arr, cmap='gray', alpha=0.5, aspect=aspect_ratio)
            for i in range(def_y.shape[0]):
                ax.plot(def_x[i, :], def_y[i, :], color='#38bdf8', linewidth=1.1)
            for j in range(def_x.shape[1]):
                ax.plot(def_x[:, j], def_y[:, j], color='#38bdf8', linewidth=1.1)
        elif mode_clean == "quiver":
            ax.imshow(bg_arr, cmap='gray', alpha=0.5, aspect=aspect_ratio)
            if max_v_mag > 1e-4:
                q = ax.quiver(
                    grid_x, grid_y, disp_x, disp_y,
                    np.hypot(disp_x, disp_y),
                    cmap='magma', angles='xy', scale_units='xy', scale=auto_scale, width=0.003
                )
                cbar = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(colors='#c9d1d9')
        else:
            bg_2d = np.squeeze(bg_arr)
            v_mag_2d = np.squeeze(v_mag)
            gx_2d = np.squeeze(grid_x)
            gy_2d = np.squeeze(grid_y)
            dx_2d = np.squeeze(disp_x)
            dy_2d = np.squeeze(disp_y)
            ax.imshow(bg_2d, cmap='gray', alpha=0.4, aspect=aspect_ratio)
            if max_v_mag > 1e-4:
                im_m = ax.imshow(v_mag_2d, cmap='plasma', alpha=0.60, vmin=0.0, vmax=max_v_mag, aspect=aspect_ratio)
                ax.quiver(
                    gx_2d, gy_2d, dx_2d, dy_2d,
                    color='#38bdf8', angles='xy', scale_units='xy', scale=auto_scale, width=0.004, headwidth=3.5, headlength=4
                )
                cbar = fig.colorbar(im_m, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(colors='#c9d1d9', labelsize=8)
                cbar.set_label('||v|| (px)', color='#c9d1d9', fontsize=9)

        t_norm = float(k) / max(1.0, float(T - 1))
        
        sx = fixed_image.spacing[0] if fixed_image is not None else 1.0
        sy = fixed_image.spacing[1] if fixed_image is not None else 1.0
        
        vx = v_slice[..., 0]
        vy = v_slice[..., 1]
        
        dx2_x = np.gradient(np.gradient(vx, axis=1) / sy, axis=1) / sy
        dxy_x = np.gradient(np.gradient(vx, axis=0) / sx, axis=1) / sy
        dy2_x = np.gradient(np.gradient(vx, axis=0) / sx, axis=0) / sx

        dx2_y = np.gradient(np.gradient(vy, axis=1) / sy, axis=1) / sy
        dxy_y = np.gradient(np.gradient(vy, axis=0) / sx, axis=1) / sy
        dy2_y = np.gradient(np.gradient(vy, axis=0) / sx, axis=0) / sx

        bnd_energy = float(np.mean(dx2_x**2 + 2*dxy_x**2 + dy2_x**2 + dx2_y**2 + 2*dxy_y**2 + dy2_y**2))
        b_status = f"Bnd={bnd_energy:.3e}"

        ax.set_title(f"Keyframe t_{k} (t={t_norm:.2f})\nMax ||v||: {max_v_mag:.6f} px | {b_status}", color='#38bdf8', fontsize=11, fontweight='bold')

    fig.suptitle(title, color=text_color, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=bg_color)

    if show_figure:
        plt.show()

    return fig



def render_label_overlay_figure(
    background,
    labels,
    title: str = "",
    save_path: Optional[str] = None,
    views: Tuple[str, ...] = ("axial", "coronal", "sagittal"),
    alpha: float = 0.55,
    max_legend_labels: int = 20,
    theme: str = "dark",
    dpi: int = 110,
):
    """Show one label image over its anatomical background, one panel per view.

    For a single label map (e.g. atlas labels warped into a subject); to compare two label
    maps use ``render_label_alignment_figure``. Each view's slice index is
    ``extract_slice``'s default for ``labels`` (driven by the labelled voxels), and the
    background is cut at the same index. The background slice is shown in gray, scaled to
    the 1st-99th percentile of its non-zero values. Labels use this module's
    ``get_dkt_colormap`` (colour by label ID) with label 0 transparent.

    Parameters
    ----------
    background : ants.ANTsImage
        Anatomical image on the same grid as ``labels``.
    labels : ants.ANTsImage
        Integer label image.
    title : str, default ""
    save_path : str, optional
        If given, save there (parent directories are created), close the figure and return
        the path.
    views : tuple of str, default ("axial", "coronal", "sagittal")
        Planes to draw, any of these three names.
    alpha : float, default 0.55
        Label overlay opacity.
    max_legend_labels : int, default 20
        A legend panel (label ID -> colour) is added only if there are 1..this many non-zero
        labels.
    theme : str, default "dark"
        "dark", otherwise light colours.
    dpi : int, default 110
        Resolution of the saved file.

    Returns
    -------
    str or matplotlib.figure.Figure
        ``save_path`` if given, else the Figure.

    Raises
    ------
    ValueError
        If the two images have different shapes.
    """
    if tuple(background.shape) != tuple(labels.shape):
        raise ValueError(f"background shape {tuple(background.shape)} != labels shape {tuple(labels.shape)} -- must be on the same grid.")

    is_dark = theme.lower() == "dark"
    bg_color = "#0f172a" if is_dark else "#ffffff"
    text_color = "#f1f5f9" if is_dark else "#0f172a"

    unique_labels = np.unique(labels.numpy())
    unique_labels = unique_labels[unique_labels != 0]
    cmap = get_dkt_colormap(max_label=max(int(unique_labels.max()), 1) if unique_labels.size else 1)

    n_panels = len(views) + (1 if (unique_labels.size and unique_labels.size <= max_legend_labels) else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.2), facecolor=bg_color)
    if n_panels == 1:
        axes = [axes]

    plane_axis = {"sagittal": 0, "coronal": 1, "axial": 2}
    for ax, view in zip(axes, views):
        axis = plane_axis[view]
        label_slice = AnatomicalVisualizer.extract_slice(labels, plane=axis, slice_idx=None)
        bg_slice = AnatomicalVisualizer.extract_slice(background, plane=axis, slice_idx=label_slice.slice_idx, ref_image=labels)

        bg_arr = bg_slice.data.astype(np.float64)
        nonzero = bg_arr[bg_arr != 0]
        if nonzero.size:
            lo, hi = np.percentile(nonzero, 1), np.percentile(nonzero, 99)
        else:
            lo, hi = float(bg_arr.min()), float(bg_arr.max())
        if hi <= lo:
            hi = lo + 1.0
        bg_norm = np.clip((bg_arr - lo) / (hi - lo), 0.0, 1.0)

        ax.imshow(bg_norm, cmap="gray", aspect=bg_slice.aspect_ratio, vmin=0, vmax=1, origin="upper")
        label_arr = label_slice.data
        masked = np.ma.masked_where(label_arr == 0, label_arr)
        ax.imshow(masked, cmap=cmap, aspect=label_slice.aspect_ratio, vmin=0, vmax=cmap.N - 1, alpha=alpha, origin="upper")
        ax.set_title(view, color=text_color, fontsize=10)
        ax.axis("off")

    if n_panels > len(views):
        legend_ax = axes[-1]
        legend_ax.axis("off")
        for i, label_val in enumerate(unique_labels):
            color = cmap(int(label_val) % cmap.N)
            y = 1.0 - (i + 1) / (len(unique_labels) + 1)
            legend_ax.add_patch(plt.Rectangle((0.05, y - 0.02), 0.15, 0.04, color=color, transform=legend_ax.transAxes))
            legend_ax.text(0.25, y, f"label {int(label_val)}", color=text_color, fontsize=8, va="center", transform=legend_ax.transAxes)
        legend_ax.set_title("legend", color=text_color, fontsize=10)

    fig.suptitle(title, color=text_color, fontsize=13)
    fig.patch.set_facecolor(bg_color)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, facecolor=bg_color, dpi=dpi)
        plt.close(fig)
        return save_path
    return fig


def render_checkerboard_figure(
    image_a,
    image_b,
    title: str = "",
    save_path: Optional[str] = None,
    pattern_size: int = 8,
    views: Tuple[str, ...] = ("axial", "coronal", "sagittal"),
    theme: str = "dark",
    dpi: int = 110,
):
    """Checkerboard of two images on the same grid, one panel per view.

    Tile (row // pattern_size + col // pattern_size) even shows ``image_a``, odd shows
    ``image_b``; anatomy that is aligned continues across tile edges. Each slice is scaled
    separately to the 1st-99th percentile of its non-zero values. ``image_b`` is cut at the
    slice index ``extract_slice`` chooses for ``image_a``.

    Parameters
    ----------
    image_a, image_b : ants.ANTsImage
        Same shape.
    title : str, default ""
    save_path : str, optional
        If given, save there (parent directories are created), close the figure and return
        the path.
    pattern_size : int, default 8
        Tile edge length in pixels.
    views : tuple of str, default ("axial", "coronal", "sagittal")
    theme : str, default "dark"
    dpi : int, default 110
        Resolution of the saved file.

    Returns
    -------
    str or matplotlib.figure.Figure
        ``save_path`` if given, else the Figure.

    Raises
    ------
    ValueError
        If the shapes differ.
    """
    if image_a.shape != image_b.shape:
        raise ValueError(f"image_a shape {image_a.shape} != image_b shape {image_b.shape} -- must be on the same grid.")

    is_dark = theme.lower() == "dark"
    bg_color = "#0f172a" if is_dark else "#ffffff"
    text_color = "#f1f5f9" if is_dark else "#0f172a"

    def _normalize(arr):
        nonzero = arr[arr != 0]
        if nonzero.size:
            lo, hi = float(np.percentile(nonzero, 1)), float(np.percentile(nonzero, 99))
            if hi <= lo:
                lo, hi = float(arr.min()), float(arr.max())
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    def _checker(slice_a, slice_b):
        h, w = slice_a.shape
        tile_row = np.arange(h)[:, None] // pattern_size
        tile_col = np.arange(w)[None, :] // pattern_size
        use_a = (tile_row + tile_col) % 2 == 0
        return np.where(use_a, slice_a, slice_b)

    fig, axes = plt.subplots(1, len(views), figsize=(4.2 * len(views), 4.2), facecolor=bg_color)
    if len(views) == 1:
        axes = [axes]

    plane_axis = {"sagittal": 0, "coronal": 1, "axial": 2}
    for ax, view in zip(axes, views):
        axis = plane_axis[view]
        slice_a_obj = AnatomicalVisualizer.extract_slice(image_a, plane=axis, slice_idx=None)
        slice_b_obj = AnatomicalVisualizer.extract_slice(image_b, plane=axis, slice_idx=slice_a_obj.slice_idx, ref_image=image_a)
        panel = _checker(_normalize(slice_a_obj.data), _normalize(slice_b_obj.data))
        ax.imshow(panel, cmap="gray", aspect=slice_a_obj.aspect_ratio, vmin=0, vmax=1, origin="upper")
        ax.set_title(view, color=text_color, fontsize=10)
        ax.axis("off")

    fig.suptitle(title, color=text_color, fontsize=13)
    fig.patch.set_facecolor(bg_color)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, facecolor=bg_color, dpi=dpi)
        plt.close(fig)
        return save_path
    return fig


def render_correlation_matrix_figure(
    matrix,
    roi_labels: list | None = None,
    title: str = "",
    save_path: str | None = None,
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
    theme: str = "dark",
    dpi: int = 110,
):
    """Draw an ROI x ROI correlation matrix as a heatmap with a colorbar.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_rois, n_rois)
        Correlation matrix (e.g. from :func:`syntx.tabulate.correlation_matrix`). Not
        modified; the diagonal is set to NaN in a copy and drawn as background.
    roi_labels : list of str, optional
        Tick labels in matrix order, drawn only if n_rois <= 40; otherwise the ticks are
        removed and the axes labelled "ROI index (n=...)".
    title : str, default ""
    save_path : str, optional
        If given, save there (parent directories are created), close the figure and return
        the path.
    cmap : str, default "coolwarm"
    vmin, vmax : float, default -1.0, 1.0
        Colour-scale limits.
    theme : str, default "dark"
        "dark", otherwise light colours.
    dpi : int, default 110
        Resolution of the saved file.

    Returns
    -------
    str or matplotlib.figure.Figure
        ``save_path`` if given, else the Figure object.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    is_dark = (theme.lower() == "dark")
    bg_color = "#0b0f17" if is_dark else "#ffffff"
    text_color = "#f8fafc" if is_dark else "#0f172a"

    n = matrix.shape[0]
    display_matrix = np.array(matrix, dtype=float, copy=True)
    np.fill_diagonal(display_matrix, np.nan)

    fig, ax = plt.subplots(figsize=(max(5.0, min(0.18 * n, 12.0)),) * 2, facecolor=bg_color)
    ax.set_facecolor(bg_color)
    im = ax.imshow(display_matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    if roi_labels is not None and n <= 40:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(roi_labels, rotation=90, fontsize=6, color=text_color)
        ax.set_yticklabels(roi_labels, fontsize=6, color=text_color)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"ROI index (n={n})", color=text_color, fontsize=9)
        ax.set_ylabel(f"ROI index (n={n})", color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation (r)", color=text_color)
    cbar.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(cbar.ax.get_yticklabels(), color=text_color)

    ax.set_title(title, color=text_color, fontsize=12)
    fig.patch.set_facecolor(bg_color)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, facecolor=bg_color, dpi=dpi)
        plt.close(fig)
        return save_path
    return fig


def render_carpet_plot_figure(
    timeseries,
    tissue_labels=None,
    fd=None,
    title: str = "",
    save_path: str | None = None,
    cmap: str = "gray",
    vmin: float = -2.0,
    vmax: float = 2.0,
    theme: str = "dark",
    dpi: int = 110,
):
    """Draw a carpet plot (grayplot): each voxel's time series z-scored, one row per voxel.

    Each voxel is z-scored over time (a zero standard deviation is replaced by 1). Use a
    minimally processed (e.g. motion-corrected, not nuisance-regressed) series so that
    artefacts are still visible.

    Parameters
    ----------
    timeseries : np.ndarray, shape (n_timepoints, n_voxels)
        Brain-masked signal.
    tissue_labels : np.ndarray, shape (n_voxels,), optional
        Class per voxel; rows are then sorted by class (stable) with thin lines between
        classes. Otherwise rows keep their given order.
    fd : np.ndarray, shape (n_timepoints,), optional
        Framewise displacement, drawn as a trace above the carpet on the same time axis.
    title : str, default ""
    save_path : str, optional
        If given, save there (parent directories are created), close the figure and return
        the path.
    cmap : str, default "gray"
    vmin, vmax : float, default -2.0, 2.0
        Colour-scale limits in z-score units.
    theme : str, default "dark"
        "dark", otherwise light colours.
    dpi : int, default 110
        Resolution of the saved file.

    Returns
    -------
    str or matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    is_dark = (theme.lower() == "dark")
    bg_color = "#0b0f17" if is_dark else "#ffffff"
    text_color = "#f8fafc" if is_dark else "#0f172a"

    ts = np.asarray(timeseries, dtype=float)
    mean = ts.mean(axis=0, keepdims=True)
    std = ts.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    z = (ts - mean) / std  # (T, n_voxels)

    order = np.arange(z.shape[1])
    group_boundaries = []
    if tissue_labels is not None:
        tissue_labels = np.asarray(tissue_labels)
        order = np.argsort(tissue_labels, kind="stable")
        sorted_labels = tissue_labels[order]
        boundaries = np.where(np.diff(sorted_labels) != 0)[0]
        group_boundaries = (boundaries + 0.5).tolist()

    carpet = z[:, order].T  # (n_voxels, T)

    has_fd = fd is not None
    if has_fd:
        fig, (ax_fd, ax_carpet) = plt.subplots(
            2, 1, figsize=(10, 6), facecolor=bg_color, gridspec_kw={"height_ratios": [1, 5]}
        )
        ax_fd.set_facecolor(bg_color)
        fd_arr = np.asarray(fd, dtype=float)
        ax_fd.plot(np.arange(len(fd_arr)), fd_arr, color="#38bdf8", linewidth=1.0)
        ax_fd.set_xlim(0, carpet.shape[1] - 1)
        ax_fd.set_ylabel("FD (mm)", color=text_color, fontsize=8)
        ax_fd.tick_params(colors=text_color, labelsize=7)
        for spine in ax_fd.spines.values():
            spine.set_color("#334155")
        ax_fd.set_xticks([])
    else:
        fig, ax_carpet = plt.subplots(1, 1, figsize=(10, 5), facecolor=bg_color)

    ax_carpet.set_facecolor(bg_color)
    im = ax_carpet.imshow(carpet, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    for b in group_boundaries:
        ax_carpet.axhline(b, color="#f8fafc", linewidth=0.6, alpha=0.5)
    ax_carpet.set_xlabel("Time (volumes)", color=text_color, fontsize=9)
    ax_carpet.set_ylabel("Voxels (grouped by tissue)" if tissue_labels is not None else "Voxels", color=text_color, fontsize=9)
    ax_carpet.tick_params(colors=text_color, labelsize=7)
    for spine in ax_carpet.spines.values():
        spine.set_color("#334155")

    cbar = fig.colorbar(im, ax=ax_carpet, fraction=0.02, pad=0.02)
    cbar.set_label("z-score", color=text_color, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=text_color, labelsize=7)
    plt.setp(cbar.ax.get_yticklabels(), color=text_color)

    fig.suptitle(title, color=text_color, fontsize=12)
    fig.patch.set_facecolor(bg_color)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, facecolor=bg_color, dpi=dpi)
        plt.close(fig)
        return save_path
    return fig


def render_motion_parameters_figure(
    translations,
    rotations,
    fd=None,
    title: str = "",
    save_path: str | None = None,
    rotation_units: str = "deg",
    theme: str = "dark",
    dpi: int = 110,
):
    """Plot rigid-motion parameters over time: translations, rotations and optionally FD.

    Parameters
    ----------
    translations : np.ndarray, shape (n_timepoints, 3)
        Translations (x, y, z) in mm.
    rotations : np.ndarray, shape (n_timepoints, 3)
        Rotations (x, y, z) in ``rotation_units``.
    fd : np.ndarray, shape (n_timepoints,), optional
        Framewise displacement (mm), drawn as a third panel.
    title : str, default ""
    save_path : str, optional
        If given, save there (parent directories are created), close the figure and return
        the path.
    rotation_units : str, default "deg"
        Used only in the y-axis label (values are not converted).
    theme : str, default "dark"
        "dark", otherwise light colours.
    dpi : int, default 110
        Resolution of the saved file.

    Returns
    -------
    str or matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    is_dark = (theme.lower() == "dark")
    bg_color = "#0b0f17" if is_dark else "#ffffff"
    text_color = "#f8fafc" if is_dark else "#0f172a"

    t = np.asarray(translations, dtype=float)
    r = np.asarray(rotations, dtype=float)
    n_t = t.shape[0]
    x = np.arange(n_t)

    n_panels = 3 if fd is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.0 * n_panels), facecolor=bg_color, sharex=True)

    colors = ("#38bdf8", "#34d399", "#fbbf24")

    ax_t = axes[0]
    ax_t.set_facecolor(bg_color)
    for i, label in enumerate(("x", "y", "z")):
        ax_t.plot(x, t[:, i], color=colors[i], linewidth=1.0, label=f"trans-{label}")
    ax_t.set_ylabel("Translation (mm)", color=text_color, fontsize=9)
    ax_t.legend(loc="upper right", fontsize=7, facecolor=bg_color, labelcolor=text_color, framealpha=0.5)
    ax_t.tick_params(colors=text_color, labelsize=7)
    for spine in ax_t.spines.values():
        spine.set_color("#334155")

    ax_r = axes[1]
    ax_r.set_facecolor(bg_color)
    for i, label in enumerate(("x", "y", "z")):
        ax_r.plot(x, r[:, i], color=colors[i], linewidth=1.0, label=f"rot-{label}")
    ax_r.set_ylabel(f"Rotation ({rotation_units})", color=text_color, fontsize=9)
    ax_r.legend(loc="upper right", fontsize=7, facecolor=bg_color, labelcolor=text_color, framealpha=0.5)
    ax_r.tick_params(colors=text_color, labelsize=7)
    for spine in ax_r.spines.values():
        spine.set_color("#334155")

    if fd is not None:
        ax_fd = axes[2]
        ax_fd.set_facecolor(bg_color)
        ax_fd.plot(x, np.asarray(fd, dtype=float), color="#fb7185", linewidth=1.0)
        ax_fd.set_ylabel("FD (mm)", color=text_color, fontsize=9)
        ax_fd.tick_params(colors=text_color, labelsize=7)
        for spine in ax_fd.spines.values():
            spine.set_color("#334155")

    axes[-1].set_xlabel("Time (volumes)", color=text_color, fontsize=9)
    axes[-1].set_xlim(0, n_t - 1)

    fig.suptitle(title, color=text_color, fontsize=12)
    fig.patch.set_facecolor(bg_color)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, facecolor=bg_color, dpi=dpi)
        plt.close(fig)
        return save_path
    return fig
