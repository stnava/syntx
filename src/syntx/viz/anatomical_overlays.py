"""
syntx.viz.anatomical_overlays — Generalized anatomical overlay and flow visualization hierarchy
==============================================================================================

This module formalizes a structured visualization hierarchy for medical image registration:
- Level 1 (Primitives):
    * ``visualize_segmentation_on_anatomy``: Single label map or subset overlaid on anatomy.
    * ``visualize_segmentation_pair_on_anatomy``: Dual label overlay (target vs moving, overlap/TP/FP/FN).
    * ``visualize_flow_on_anatomy``: Deformation vector flows (quiver, magnitude, mesh, quiver+magnitude).
    * ``visualize_flow_differential_on_anatomy``: Vector and magnitude delta (u2 - u1) between two flows.
- Level 2 (Series Composers):
    * ``render_segmentation_alignment_series``: Multi-model label progression on target anatomy.
    * ``render_flow_series``: Comparative multi-model flow dynamics.
- Level 3 (Hierarchical Dissection Figure):
    * ``render_gainer_anatomical_dissection``: Publication-grade 2xN dissection combining label alignment
      and flow mechanics for any anatomical structure or gainer parcel.
- Class Interface:
    * ``AnatomicalOverlayVisualizer``: Object-oriented stateful visualizer for coordinate-aware ROI extraction.

Strictly adheres to GEMINI.md Rules 1-6 (light theme, physical aspect ratios, syntx.viz orientation).
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ants

from .core import AnatomicalVisualizer
from .figures import extract_oriented_slice, _as_displacement_image, _display_displacement


# =========================================================================
# THEME CONSTANTS & PALETTES (GEMINI.md Compliant)
# =========================================================================
LIGHT_THEME_RC = {
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.family': 'sans-serif',
    'mathtext.fontset': 'dejavusans',
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FFFFFF',
    'text.color': '#1E293B',
    'axes.labelcolor': '#1E293B',
    'xtick.color': '#475569',
    'ytick.color': '#475569',
    'axes.edgecolor': '#CBD5E1',
    'axes.linewidth': 1.0,
}

COLORS = {
    'target': '#0284C7',          # Cyan / Primary
    'target_contour': '#F59E0B',  # Bright Amber / Yellow outline
    'affine': '#94A3B8',          # Slate Gray
    'sobolev': '#0284C7',         # Primary Blue
    'hyperelastic': '#059669',    # Emerald Green
    'divcurl': '#D97706',         # Amber
    'diff_pos': '#EF4444',        # Red
    'diff_neg': '#3B82F6',        # Blue
    'mesh': '#059669',            # Green
    'edge_dark': '#0F172A',       # Deep Slate
}


# =========================================================================
# LEVEL 1: PRIMITIVES (Plotting on single axes)
# =========================================================================

def _prepare_slice_data(
    image: Union[ants.ANTsImage, str, np.ndarray],
    slice_axis: int = 2,
    slice_idx: Optional[int] = None,
    reorient: bool = True
) -> Tuple[np.ndarray, float]:
    """Helper to extract a 2D oriented slice and aspect ratio."""
    if isinstance(image, str):
        image = ants.image_read(image)
    return extract_oriented_slice(image, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)


def visualize_segmentation_on_anatomy(
    anatomy: Union[ants.ANTsImage, str, np.ndarray],
    segmentation: Union[ants.ANTsImage, str, np.ndarray],
    slice_axis: int = 2,
    slice_idx: Optional[int] = None,
    labels: Optional[Union[int, List[int]]] = None,
    color_fill: str = '#0284C7',
    fill_alpha: float = 0.55,
    color_edge: Optional[str] = '#0369A1',
    edge_width: float = 1.8,
    target_contour: Optional[Union[ants.ANTsImage, str, np.ndarray]] = None,
    target_labels: Optional[Union[int, List[int]]] = None,
    target_contour_color: str = '#F59E0B',
    target_contour_width: float = 2.2,
    roi: Optional[Tuple[int, int, int, int]] = None,
    badge_text: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    reorient: bool = True
) -> plt.Axes:
    """Visualize a segmentation overlaid on an anatomical slice.

    Parameters
    ----------
    anatomy : ANTsImage, str or array
        Background grayscale anatomy image.
    segmentation : ANTsImage, str or array
        Integer segmentation label map.
    slice_axis : int, default 2
        0=sagittal, 1=coronal, 2=axial.
    slice_idx : int, optional
        Slice index along slice_axis.
    labels : int or list of int, optional
        Specific label IDs to display (binary mask formed). If None, all voxels > 0 are shown.
    color_fill : str, default '#0284C7'
        Hex or CSS color for label fill.
    fill_alpha : float, default 0.55
        Transparency of label fill.
    color_edge : str, optional
        Contour edge color for the segmentation.
    edge_width : float, default 1.8
        Line width of segmentation contour.
    target_contour : ANTsImage, str or array, optional
        Optional reference target label map to draw as an outline contour.
    target_labels : int or list of int, optional
        Specific label IDs for the target contour.
    target_contour_color : str, default '#F59E0B'
        Color of the target contour outline (e.g. amber).
    target_contour_width : float, default 2.2
        Line width of the target contour outline.
    roi : tuple of int (ymin, ymax, xmin, xmax), optional
        Bounding box to zoom in on the slice.
    badge_text : str, optional
        Text badge to render in corner (e.g. "Dice: 0.614").
    title : str, optional
        Subplot title.
    ax : plt.Axes, optional
        Matplotlib axis. Created if None.
    reorient : bool, default True
        Reorient to LPI.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='#FFFFFF')

    bg_arr, aspect = _prepare_slice_data(anatomy, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)
    seg_arr, _ = _prepare_slice_data(segmentation, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)

    # Form label mask
    if labels is None:
        seg_mask = (seg_arr > 0)
    else:
        lbl_list = [labels] if isinstance(labels, int) else labels
        seg_mask = np.isin(seg_arr, lbl_list)

    # Apply ROI crop if provided
    if roi is not None:
        ymin, ymax, xmin, xmax = roi
        bg_plot = bg_arr[ymin:ymax, xmin:xmax]
        seg_plot = seg_mask[ymin:ymax, xmin:xmax]
    else:
        bg_plot = bg_arr
        seg_plot = seg_mask

    # Plot anatomical background
    ax.imshow(bg_plot, cmap='gray', aspect=aspect, interpolation='bilinear')

    # Target reference contour if provided
    if target_contour is not None:
        tgt_arr, _ = _prepare_slice_data(target_contour, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)
        if target_labels is None:
            tgt_mask = (tgt_arr > 0)
        else:
            tgt_lbl_list = [target_labels] if isinstance(target_labels, int) else target_labels
            tgt_mask = np.isin(tgt_arr, tgt_lbl_list)

        tgt_plot = tgt_mask[ymin:ymax, xmin:xmax] if roi is not None else tgt_mask
        if tgt_plot.sum() > 0:
            ax.contour(tgt_plot, levels=[0.5], colors=[target_contour_color], linewidths=[target_contour_width])

    # Overlay moving/warped segmentation
    if seg_plot.sum() > 0:
        overlay = np.zeros((*bg_plot.shape, 4), dtype=np.float32)
        r, g, b = mcolors.to_rgb(color_fill)
        overlay[seg_plot] = [r, g, b, fill_alpha]
        ax.imshow(overlay, aspect=aspect)

        if color_edge:
            ax.contour(seg_plot, levels=[0.5], colors=[color_edge], linewidths=[edge_width])

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8, color='#1E293B')

    if badge_text:
        ax.text(0.04, 0.06, badge_text, transform=ax.transAxes, fontsize=9.5, fontweight='bold',
                color='#1E293B', bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec="#CBD5E1", alpha=0.92))

    ax.axis('off')
    return ax


def visualize_segmentation_pair_on_anatomy(
    anatomy: Union[ants.ANTsImage, str, np.ndarray],
    target_seg: Union[ants.ANTsImage, str, np.ndarray],
    moving_seg: Union[ants.ANTsImage, str, np.ndarray],
    slice_axis: int = 2,
    slice_idx: Optional[int] = None,
    target_labels: Optional[Union[int, List[int]]] = None,
    moving_labels: Optional[Union[int, List[int]]] = None,
    mode: str = 'contour_and_fill',
    target_contour_color: str = '#F59E0B',
    moving_fill_color: str = '#0284C7',
    fill_alpha: float = 0.55,
    roi: Optional[Tuple[int, int, int, int]] = None,
    badge_text: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    reorient: bool = True
) -> plt.Axes:
    """Visualize a pair of segmentations (target vs moving) on the anatomy.

    Parameters
    ----------
    anatomy : ANTsImage, str or array
        Background anatomy.
    target_seg : ANTsImage, str or array
        Target / reference ground-truth segmentation.
    moving_seg : ANTsImage, str or array
        Moving / transformed segmentation.
    mode : str, default 'contour_and_fill'
        'contour_and_fill': target is drawn as reference contour, moving as fill.
        'confusion': TP (green), FP (red), FN (blue).
    """
    if mode == 'contour_and_fill':
        return visualize_segmentation_on_anatomy(
            anatomy=anatomy,
            segmentation=moving_seg,
            slice_axis=slice_axis,
            slice_idx=slice_idx,
            labels=moving_labels,
            color_fill=moving_fill_color,
            fill_alpha=fill_alpha,
            target_contour=target_seg,
            target_labels=target_labels,
            target_contour_color=target_contour_color,
            roi=roi,
            badge_text=badge_text,
            title=title,
            ax=ax,
            reorient=reorient
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='#FFFFFF')

    bg_arr, aspect = _prepare_slice_data(anatomy, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)
    tgt_arr, _ = _prepare_slice_data(target_seg, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)
    mov_arr, _ = _prepare_slice_data(moving_seg, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)

    t_lbls = [target_labels] if isinstance(target_labels, int) else (target_labels or [1])
    m_lbls = [moving_labels] if isinstance(moving_labels, int) else (moving_labels or [1])

    t_mask = np.isin(tgt_arr, t_lbls) if target_labels else (tgt_arr > 0)
    m_mask = np.isin(mov_arr, m_lbls) if moving_labels else (mov_arr > 0)

    if roi is not None:
        ymin, ymax, xmin, xmax = roi
        bg_arr = bg_arr[ymin:ymax, xmin:xmax]
        t_mask = t_mask[ymin:ymax, xmin:xmax]
        m_mask = m_mask[ymin:ymax, xmin:xmax]

    ax.imshow(bg_arr, cmap='gray', aspect=aspect, interpolation='bilinear')

    # Confusion colors: TP=Green, FN (target only)=Amber, FP (moving only)=Red
    tp = t_mask & m_mask
    fn = t_mask & (~m_mask)
    fp = (~t_mask) & m_mask

    overlay = np.zeros((*bg_arr.shape, 4), dtype=np.float32)
    overlay[tp] = [0.02, 0.59, 0.41, 0.6]   # Emerald
    overlay[fn] = [0.96, 0.62, 0.04, 0.6]   # Amber
    overlay[fp] = [0.94, 0.27, 0.27, 0.6]   # Red
    ax.imshow(overlay, aspect=aspect)

    ax.contour(t_mask, levels=[0.5], colors=['#0284C7'], linewidths=[1.8])

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8, color='#1E293B')
    if badge_text:
        ax.text(0.04, 0.06, badge_text, transform=ax.transAxes, fontsize=9.5, fontweight='bold',
                color='#1E293B', bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec="#CBD5E1", alpha=0.92))

    ax.axis('off')
    return ax


def visualize_flow_on_anatomy(
    anatomy: Union[ants.ANTsImage, str, np.ndarray],
    flow: Union[ants.ANTsImage, str, list, tuple, np.ndarray],
    slice_axis: int = 2,
    slice_idx: Optional[int] = None,
    mode: str = 'quiver_magnitude',
    subsample_step: int = 4,
    scale: float = 50.0,
    quiver_color: str = '#FFFFFF',
    mag_cmap: str = 'viridis',
    vmax_mm: float = 15.0,
    grid_step: int = 6,
    grid_color: str = '#059669',
    target_contour: Optional[Union[ants.ANTsImage, str, np.ndarray]] = None,
    target_labels: Optional[Union[int, List[int]]] = None,
    target_contour_color: str = '#F59E0B',
    roi: Optional[Tuple[int, int, int, int]] = None,
    show_colorbar: bool = True,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    reorient: bool = True
) -> Tuple[plt.Axes, Dict[str, float]]:
    """Visualize displacement flow fields overlaid on an anatomical background slice.

    Parameters
    ----------
    anatomy : ANTsImage, str or array
        Reference anatomical image.
    flow : ANTsImage, str, transform list or tensor
        Displacement vector field.
    slice_axis : int, default 2
        0=sagittal, 1=coronal, 2=axial.
    slice_idx : int, optional
        Slice index.
    mode : str, default 'quiver_magnitude'
        'quiver_magnitude': flow magnitude heatmap with superimposed displacement arrows.
        'quiver': vector arrows on grayscale anatomy.
        'magnitude': magnitude heatmap on anatomy.
        'mesh': deformed coordinate mesh grid on anatomy.
    subsample_step : int, default 4
        Subsampling step for vector arrows.
    scale : float, default 50.0
        Matplotlib quiver scale factor (larger value = shorter arrows).
    vmax_mm : float, default 15.0
        Maximum displacement magnitude for colormap clamping (mm).
    grid_step : int, default 6
        Spacing between coordinate mesh lines (pixels).
    roi : tuple of int (ymin, ymax, xmin, xmax), optional
        Bounding box to zoom into.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='#FFFFFF')

    ref_img = anatomy if isinstance(anatomy, ants.ANTsImage) else ants.image_read(anatomy) if isinstance(anatomy, str) else None
    dcol, drow, fi_arr, aspect, mag_mm = _display_displacement(flow, ref_img, slice_axis, slice_idx, reorient=reorient)

    H, W = fi_arr.shape
    if roi is not None:
        ymin, ymax, xmin, xmax = roi
    else:
        ymin, ymax, xmin, xmax = 0, H, 0, W

    bg_crop = fi_arr[ymin:ymax, xmin:xmax]
    mag_crop = mag_mm[ymin:ymax, xmin:xmax]
    dcol_crop = dcol[ymin:ymax, xmin:xmax]
    drow_crop = drow[ymin:ymax, xmin:xmax]

    im_cbar = None
    if mode in ('quiver_magnitude', 'magnitude'):
        im_cbar = ax.imshow(mag_crop, cmap=mag_cmap, aspect=aspect, vmin=0, vmax=vmax_mm)
    elif mode in ('quiver', 'mesh'):
        ax.imshow(bg_crop, cmap='gray', aspect=aspect, alpha=0.6 if mode == 'mesh' else 1.0, interpolation='bilinear')

    if mode in ('quiver_magnitude', 'quiver'):
        sub_y, sub_x = np.mgrid[0:(ymax - ymin), 0:(xmax - xmin)]
        X_sub = sub_x[::subsample_step, ::subsample_step]
        Y_sub = sub_y[::subsample_step, ::subsample_step]
        U_sub = dcol_crop[::subsample_step, ::subsample_step]
        V_sub = drow_crop[::subsample_step, ::subsample_step]

        ax.quiver(X_sub, Y_sub, U_sub, V_sub, color=quiver_color, scale=scale, width=0.006, headwidth=4)

    elif mode == 'mesh':
        g_y, g_x = np.mgrid[0:(ymax - ymin):grid_step, 0:(xmax - xmin):grid_step]
        d_y = g_y + drow_crop[g_y, g_x]
        d_x = g_x + dcol_crop[g_y, g_x]
        for i in range(d_y.shape[0]):
            ax.plot(d_x[i, :], d_y[i, :], color=grid_color, linewidth=1.1, alpha=0.85)
        for j in range(d_x.shape[1]):
            ax.plot(d_x[:, j], d_y[:, j], color=grid_color, linewidth=1.1, alpha=0.85)

    # Target contour outline
    if target_contour is not None:
        tgt_arr, _ = _prepare_slice_data(target_contour, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)
        if target_labels is None:
            tgt_mask = (tgt_arr > 0)
        else:
            tgt_lbl_list = [target_labels] if isinstance(target_labels, int) else target_labels
            tgt_mask = np.isin(tgt_arr, tgt_lbl_list)
        tgt_plot = tgt_mask[ymin:ymax, xmin:xmax]
        if tgt_plot.sum() > 0:
            ax.contour(tgt_plot, levels=[0.5], colors=[target_contour_color], linewidths=[1.8])

    if show_colorbar and im_cbar is not None:
        cbar = plt.colorbar(im_cbar, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Displacement (mm)", fontsize=8.5)

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8, color='#1E293B')

    ax.axis('off')
    stats = {
        'max_displacement_mm': float(np.max(mag_crop)),
        'mean_displacement_mm': float(np.mean(mag_crop)),
    }
    return ax, stats


def visualize_flow_differential_on_anatomy(
    anatomy: Union[ants.ANTsImage, str, np.ndarray],
    flow1: Union[ants.ANTsImage, str, list, tuple, np.ndarray],
    flow2: Union[ants.ANTsImage, str, list, tuple, np.ndarray],
    slice_axis: int = 2,
    slice_idx: Optional[int] = None,
    subsample_step: int = 4,
    scale: float = 30.0,
    quiver_color: str = '#0F172A',
    vmax_mm: float = 6.0,
    target_contour: Optional[Union[ants.ANTsImage, str, np.ndarray]] = None,
    target_labels: Optional[Union[int, List[int]]] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    title: Optional[str] = None,
    cbar_label: str = "||u2|| - ||u1|| (mm)",
    ax: Optional[plt.Axes] = None,
    reorient: bool = True
) -> Tuple[plt.Axes, Dict[str, float]]:
    """Visualize the displacement and vector differential (flow2 - flow1) between two flows.

    Highlights where flow2 applied active additional displacement or tangential slip.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='#FFFFFF')

    ref_img = anatomy if isinstance(anatomy, ants.ANTsImage) else ants.image_read(anatomy) if isinstance(anatomy, str) else None
    dcol1, drow1, fi_arr, aspect, mag1 = _display_displacement(flow1, ref_img, slice_axis, slice_idx, reorient=reorient)
    dcol2, drow2, _, _, mag2 = _display_displacement(flow2, ref_img, slice_axis, slice_idx, reorient=reorient)

    diff_mag = mag2 - mag1
    diff_dcol = dcol2 - dcol1
    diff_drow = drow2 - drow1

    H, W = fi_arr.shape
    ymin, ymax, xmin, xmax = roi if roi is not None else (0, H, 0, W)

    diff_mag_crop = diff_mag[ymin:ymax, xmin:xmax]
    dcol_crop = diff_dcol[ymin:ymax, xmin:xmax]
    drow_crop = diff_drow[ymin:ymax, xmin:xmax]

    im_diff = ax.imshow(diff_mag_crop, cmap='bwr', vmin=-vmax_mm, vmax=vmax_mm, aspect=aspect)

    sub_y, sub_x = np.mgrid[0:(ymax - ymin), 0:(xmax - xmin)]
    X_sub = sub_x[::subsample_step, ::subsample_step]
    Y_sub = sub_y[::subsample_step, ::subsample_step]
    U_sub = dcol_crop[::subsample_step, ::subsample_step]
    V_sub = drow_crop[::subsample_step, ::subsample_step]

    ax.quiver(X_sub, Y_sub, U_sub, V_sub, color=quiver_color, scale=scale, width=0.007, headwidth=4)

    if target_contour is not None:
        tgt_arr, _ = _prepare_slice_data(target_contour, slice_axis=slice_axis, slice_idx=slice_idx, reorient=reorient)
        tgt_mask = np.isin(tgt_arr, target_labels) if target_labels else (tgt_arr > 0)
        tgt_plot = tgt_mask[ymin:ymax, xmin:xmax]
        if tgt_plot.sum() > 0:
            ax.contour(tgt_plot, levels=[0.5], colors=['#0F172A'], linewidths=[1.8])

    cbar = plt.colorbar(im_diff, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=8.5)

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8, color='#1E293B')

    ax.axis('off')
    stats = {
        'max_diff_mm': float(np.max(diff_mag_crop)),
        'min_diff_mm': float(np.min(diff_mag_crop)),
        'mean_diff_mm': float(np.mean(diff_mag_crop)),
    }
    return ax, stats


# =========================================================================
# LEVEL 2: COMPOSERS & SERIES
# =========================================================================

def render_segmentation_alignment_series(
    anatomy: Union[ants.ANTsImage, str],
    target_label: Union[ants.ANTsImage, str],
    model_labels: Dict[str, Dict[str, Any]],
    slice_axis: int = 2,
    slice_idx: Optional[int] = None,
    target_label_ids: Optional[Union[int, List[int]]] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    axes: Optional[List[plt.Axes]] = None,
    reorient: bool = True
) -> List[plt.Axes]:
    """Render a horizontal multi-model progression of segmentations overlaid on anatomy.

    Parameters
    ----------
    model_labels : dict of {model_key: {'image': label_img, 'name': str, 'badge': str, 'color': str, 'edge': str}}
    """
    n_models = len(model_labels) + 1  # +1 for target GT
    if axes is None:
        fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4.5), facecolor='#FFFFFF')

    # Panel 0: Ground Truth Target
    visualize_segmentation_on_anatomy(
        anatomy=anatomy,
        segmentation=target_label,
        slice_axis=slice_axis,
        slice_idx=slice_idx,
        labels=target_label_ids,
        color_fill='#0284C7',
        fill_alpha=0.50,
        color_edge='#0284C7',
        roi=roi,
        badge_text="Ground Truth Target",
        title="Target Reference",
        ax=axes[0],
        reorient=reorient
    )

    # Panels 1..N: Model labels
    for idx, (m_key, m_info) in enumerate(model_labels.items(), start=1):
        visualize_segmentation_on_anatomy(
            anatomy=anatomy,
            segmentation=m_info['image'],
            slice_axis=slice_axis,
            slice_idx=slice_idx,
            labels=target_label_ids,
            color_fill=m_info.get('color', '#0284C7'),
            fill_alpha=0.55,
            color_edge=m_info.get('edge', '#0369A1'),
            target_contour=target_label,
            target_labels=target_label_ids,
            target_contour_color='#F59E0B',
            roi=roi,
            badge_text=m_info.get('badge', ''),
            title=m_info.get('name', m_key),
            ax=axes[idx],
            reorient=reorient
        )

    return axes


# =========================================================================
# LEVEL 3: HIERARCHICAL DISSECTION FIGURE
# =========================================================================

def render_gainer_anatomical_dissection(
    anatomy: Union[ants.ANTsImage, str],
    target_label: Union[ants.ANTsImage, str],
    models_config: Dict[str, Dict[str, Any]],
    flow_sobolev: Union[ants.ANTsImage, str, list, tuple],
    flow_continuum: Union[ants.ANTsImage, str, list, tuple],
    slice_axis: int = 2,
    slice_idx: Optional[int] = None,
    target_label_ids: Optional[Union[int, List[int]]] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    structure_name: str = "Structure",
    gain_pct: float = 0.0,
    sobolev_name: str = "Sobolev SyN",
    continuum_name: str = "Hyperelastic SyN",
    flow_vmax: float = 15.0,
    delta_vmax: float = 6.0,
    output_filename: Optional[str] = None,
    dpi: int = 200,
    reorient: bool = True
) -> plt.Figure:
    """Generate the standardized 2x4 publication-grade anatomical dissection figure:
    Row 1: Label alignments on target anatomy across models (Target, Affine, Sobolev, Continuum).
    Row 2: Flow field mechanics (Sobolev Flow, Continuum Flow, Deformed Grid, Flow Delta).

    Parameters
    ----------
    anatomy : ANTsImage, str or array
        Target background anatomy.
    target_label : ANTsImage, str or array
        Target ground truth segmentation.
    models_config : dict
        Dict with keys ['affine', 'sobolev', 'continuum'], each specifying {'image', 'badge', 'color', 'edge', 'title'}.
    flow_sobolev, flow_continuum : warps
        Displacement fields for baseline and continuum models.
    structure_name : str
        Name of anatomical parcel (e.g. "Middle Temporal Gyrus").
    gain_pct : float
        Dice gain percentage (e.g. 6.93).
    output_filename : str, optional
        Path to save figure PNG.
    """
    plt.rcParams.update(LIGHT_THEME_RC)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), facecolor='#FFFFFF')

    # Row 1: Labels
    # Panel 0: Target GT
    visualize_segmentation_on_anatomy(
        anatomy=anatomy,
        segmentation=target_label,
        slice_axis=slice_axis,
        slice_idx=slice_idx,
        labels=target_label_ids,
        color_fill='#0284C7',
        fill_alpha=0.50,
        color_edge='#0284C7',
        roi=roi,
        badge_text="Ground Truth Target",
        title=f"A: Target Anatomy & Ground Truth {structure_name}\n(Reference Slice)",
        ax=axes[0, 0],
        reorient=reorient
    )

    # Panel 1: Affine
    aff_cfg = models_config.get('affine', {})
    visualize_segmentation_on_anatomy(
        anatomy=anatomy,
        segmentation=aff_cfg['image'],
        slice_axis=slice_axis,
        slice_idx=slice_idx,
        labels=target_label_ids,
        color_fill=aff_cfg.get('color', '#94A3B8'),
        color_edge=aff_cfg.get('edge', '#475569'),
        target_contour=target_label,
        target_labels=target_label_ids,
        target_contour_color='#F59E0B',
        roi=roi,
        badge_text=aff_cfg.get('badge', 'Affine Initial'),
        title=f"B: Original Moving Label (Affine)\nAmber Contour = Target Boundary",
        ax=axes[0, 1],
        reorient=reorient
    )

    # Panel 2: Sobolev
    sob_cfg = models_config.get('sobolev', {})
    visualize_segmentation_on_anatomy(
        anatomy=anatomy,
        segmentation=sob_cfg['image'],
        slice_axis=slice_axis,
        slice_idx=slice_idx,
        labels=target_label_ids,
        color_fill=sob_cfg.get('color', '#0284C7'),
        color_edge=sob_cfg.get('edge', '#0369A1'),
        target_contour=target_label,
        target_labels=target_label_ids,
        target_contour_color='#F59E0B',
        roi=roi,
        badge_text=sob_cfg.get('badge', 'Sobolev Transformed'),
        title=f"C: {sobolev_name} Transformed Label\nChoked Deformation / Sulcal Under-reach",
        ax=axes[0, 2],
        reorient=reorient
    )

    # Panel 3: Continuum
    con_cfg = models_config.get('continuum', {})
    visualize_segmentation_on_anatomy(
        anatomy=anatomy,
        segmentation=con_cfg['image'],
        slice_axis=slice_axis,
        slice_idx=slice_idx,
        labels=target_label_ids,
        color_fill=con_cfg.get('color', '#059669'),
        color_edge=con_cfg.get('edge', '#047857'),
        target_contour=target_label,
        target_labels=target_label_ids,
        target_contour_color='#F59E0B',
        roi=roi,
        badge_text=con_cfg.get('badge', f'{continuum_name} (+{gain_pct:.2f}%)'),
        title=f"D: {continuum_name} Transformed Label\nConformal Filling of Gyral Bank",
        ax=axes[0, 3],
        reorient=reorient
    )

    # Row 2: Flows
    # Panel E: Sobolev Flow
    visualize_flow_on_anatomy(
        anatomy=anatomy,
        flow=flow_sobolev,
        slice_axis=slice_axis,
        slice_idx=slice_idx,
        mode='quiver_magnitude',
        vmax_mm=flow_vmax,
        target_contour=target_label,
        target_labels=target_label_ids,
        roi=roi,
        title=f"E: {sobolev_name} Flow Field (Vectors + Mag)\nMuted Flow Vectors into Target Sulcus",
        ax=axes[1, 0],
        reorient=reorient
    )

    # Panel F: Continuum Flow
    visualize_flow_on_anatomy(
        anatomy=anatomy,
        flow=flow_continuum,
        slice_axis=slice_axis,
        slice_idx=slice_idx,
        mode='quiver_magnitude',
        vmax_mm=flow_vmax,
        target_contour=target_label,
        target_labels=target_label_ids,
        roi=roi,
        title=f"F: {continuum_name} Flow Field (Vectors + Mag)\nStrong Directed Flow Expanding Structure",
        ax=axes[1, 1],
        reorient=reorient
    )

    # Panel G: Deformed Grid
    visualize_flow_on_anatomy(
        anatomy=anatomy,
        flow=flow_continuum,
        slice_axis=slice_axis,
        slice_idx=slice_idx,
        mode='mesh',
        grid_color='#059669',
        target_contour=target_label,
        target_labels=target_label_ids,
        roi=roi,
        title=f"G: {continuum_name} Deformed Mesh Grid\nSmooth Isochoric Shear without Tearing",
        ax=axes[1, 2],
        reorient=reorient
    )

    # Panel H: Flow Differential
    visualize_flow_differential_on_anatomy(
        anatomy=anatomy,
        flow1=flow_sobolev,
        flow2=flow_continuum,
        slice_axis=slice_axis,
        slice_idx=slice_idx,
        vmax_mm=delta_vmax,
        target_contour=target_label,
        target_labels=target_label_ids,
        roi=roi,
        title=f"H: Active Flow Differential (Δu)\nRed = Extra Flow Driving +{gain_pct:.2f}% Gain",
        ax=axes[1, 3],
        reorient=reorient
    )

    fig.suptitle(
        f"Anatomical Dissection: {structure_name} (+{gain_pct:.2f}% Dice Gain)\nOriginal & Transformed Labels on Target Anatomy with Flow Field Mechanics",
        fontsize=13, fontweight='bold', y=0.98, color='#1E293B'
    )
    plt.tight_layout()

    if output_filename:
        os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
        fig.savefig(output_filename, dpi=dpi, bbox_inches='tight', facecolor='#FFFFFF')
        plt.close(fig)

    return fig


# =========================================================================
# STATEFUL OBJECT-ORIENTED INTERFACE
# =========================================================================

class AnatomicalOverlayVisualizer:
    """Stateful visualizer for coordinate-aware anatomical and flow overlays."""

    def __init__(
        self,
        anatomy: Union[ants.ANTsImage, str],
        slice_axis: int = 2,
        slice_idx: Optional[int] = None,
        reorient: bool = True
    ):
        self.anatomy = anatomy if isinstance(anatomy, ants.ANTsImage) else ants.image_read(anatomy)
        self.slice_axis = slice_axis
        self.slice_idx = slice_idx
        self.reorient = reorient
        self.roi: Optional[Tuple[int, int, int, int]] = None

    def set_roi(self, ymin: int, ymax: int, xmin: int, xmax: int):
        """Set zoom ROI bounding box."""
        self.roi = (ymin, ymax, xmin, xmax)
        return self

    def plot_segmentation(self, ax: plt.Axes, segmentation: Any, **kwargs) -> plt.Axes:
        """Plot segmentation on current anatomy."""
        return visualize_segmentation_on_anatomy(
            anatomy=self.anatomy,
            segmentation=segmentation,
            slice_axis=self.slice_axis,
            slice_idx=self.slice_idx,
            roi=self.roi,
            ax=ax,
            reorient=self.reorient,
            **kwargs
        )

    def plot_segmentation_pair(self, ax: plt.Axes, target_seg: Any, moving_seg: Any, **kwargs) -> plt.Axes:
        """Plot segmentation pair on current anatomy."""
        return visualize_segmentation_pair_on_anatomy(
            anatomy=self.anatomy,
            target_seg=target_seg,
            moving_seg=moving_seg,
            slice_axis=self.slice_axis,
            slice_idx=self.slice_idx,
            roi=self.roi,
            ax=ax,
            reorient=self.reorient,
            **kwargs
        )

    def plot_flow(self, ax: plt.Axes, flow: Any, **kwargs) -> Tuple[plt.Axes, Dict[str, float]]:
        """Plot deformation flow field on current anatomy."""
        return visualize_flow_on_anatomy(
            anatomy=self.anatomy,
            flow=flow,
            slice_axis=self.slice_axis,
            slice_idx=self.slice_idx,
            roi=self.roi,
            ax=ax,
            reorient=self.reorient,
            **kwargs
        )

    def plot_flow_differential(self, ax: plt.Axes, flow1: Any, flow2: Any, **kwargs) -> Tuple[plt.Axes, Dict[str, float]]:
        """Plot flow differential on current anatomy."""
        return visualize_flow_differential_on_anatomy(
            anatomy=self.anatomy,
            flow1=flow1,
            flow2=flow2,
            slice_axis=self.slice_axis,
            slice_idx=self.slice_idx,
            roi=self.roi,
            ax=ax,
            reorient=self.reorient,
            **kwargs
        )
