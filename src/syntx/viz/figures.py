"""
Standard Figure Generators for Syntx Medical Image Registration.

Provides publication-grade 2D and 3D figure rendering tools:
- Figure 1: render_input_pair_figure (Fixed Top / Moving Bottom for 3D, Side-by-Side for 2D)
- Figure 2: render_standard_4panel (Mesh Grid, Jacobian Map, Inverse Error Map, Edge Overlap)
- plot_deformation_grid & plot_edge_overlay helpers
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ants


def extract_2d_slice(img, slice_axis: int = 2, slice_idx=None, ref_image=None):
    """
    Extracts 2D slice from ANTsImage, PyTorch Tensor, or NumPy array.
    """
    if isinstance(img, ants.ANTsImage):
        arr = img.numpy()
    elif hasattr(img, 'detach'):
        arr = img.squeeze(0).detach().cpu().numpy()
        if arr.ndim == 4 and arr.shape[0] in (2, 3):
            arr = np.moveaxis(arr, 0, -1)
    elif hasattr(img, 'numpy'):
        arr = img.numpy()
    else:
        arr = np.squeeze(np.asarray(img))

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if arr.shape[-1] in (2, 3) and arr.shape[0] > 4:
            # 2D displacement field (H, W, 2) or (H, W, 3)
            return arr

        D, H, W = arr.shape
        if slice_idx is None:
            if slice_axis == 0: slice_idx = D // 2
            elif slice_axis == 1: slice_idx = H // 2
            else: slice_idx = W // 2
        slice_idx = max(0, min(slice_idx, arr.shape[slice_axis] - 1))

        if slice_axis == 0: return arr[slice_idx, :, :]
        elif slice_axis == 1: return arr[:, slice_idx, :]
        else: return arr[:, :, slice_idx]

    if arr.ndim == 4:
        # 3D displacement field (D, H, W, 3)
        D, H, W, C = arr.shape
        if slice_idx is None:
            if slice_axis == 0: slice_idx = D // 2
            elif slice_axis == 1: slice_idx = H // 2
            else: slice_idx = W // 2
        slice_idx = max(0, min(slice_idx, arr.shape[slice_axis] - 1))

        if slice_axis == 0: sl = arr[slice_idx, :, :, :]
        elif slice_axis == 1: sl = arr[:, slice_idx, :, :]
        else: sl = arr[:, :, slice_idx, :]

        if C == 3:
            if slice_axis == 0: return sl[..., 1:]
            elif slice_axis == 1: return sl[..., [0, 2]]
            else: return sl[..., :2]
        return sl[..., :2]

    return np.squeeze(arr)


def plot_deformation_grid(
    warp,
    fixed=None,
    slice_axis: int = 2,
    slice_idx=None,
    grid_spacing: int = 8,
    line_color: str = '#38bdf8',
    ax=None,
    figsize=(7, 7),
    title="Deformed Coordinate Mesh Grid",
    filename=None,
    show=False
):
    """
    Renders 2D slice of deformed mesh grid overlay.
    """
    disp = extract_2d_slice(warp, slice_axis=slice_axis, slice_idx=slice_idx, ref_image=fixed)
    if fixed is not None:
        fi_arr = extract_2d_slice(fixed, slice_axis=slice_axis, slice_idx=slice_idx)
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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor='#0b0f17')
    else:
        fig = ax.figure

    ax.set_facecolor('#0b0f17')
    ax.imshow(fi_arr, cmap='gray', alpha=0.5, origin='upper')

    for i in range(def_y.shape[0]):
        ax.plot(def_x[i, :], def_y[i, :], color=line_color, linewidth=1.1)
    for j in range(def_x.shape[1]):
        ax.plot(def_x[:, j], def_y[:, j], color=line_color, linewidth=1.1)

    ax.axis('off')
    if title:
        ax.set_title(title, color='#f8fafc', fontsize=12, fontweight='bold', pad=10)

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='#0b0f17')
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
    ax=None,
    figsize=(7, 7),
    title="Canny Edge Alignment Overlap",
    filename=None,
    show=False
):
    """
    Renders high-contrast Canny edge alignment contour overlay.
    """
    fi_arr = extract_2d_slice(fixed, slice_axis=slice_axis, slice_idx=slice_idx)
    mi_arr = extract_2d_slice(warped, slice_axis=slice_axis, slice_idx=slice_idx, ref_image=fixed)

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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor='#0b0f17')
    else:
        fig = ax.figure

    ax.set_facecolor('#0b0f17')
    ax.imshow(fi_norm, cmap='gray', origin='upper')

    overlay_w = np.zeros((*fi_norm.shape, 4), dtype=np.float32)
    r, g, b = mcolors.to_rgb(edge_color)
    overlay_w[edges_warped] = [r, g, b, alpha]
    ax.imshow(overlay_w, origin='upper')

    if fixed_edge_color and edges_fixed is not None:
        overlay_f = np.zeros((*fi_norm.shape, 4), dtype=np.float32)
        rf, gf, bf = mcolors.to_rgb(fixed_edge_color)
        overlay_f[edges_fixed] = [rf, gf, bf, alpha]
        ax.imshow(overlay_f, origin='upper')

    ax.axis('off')
    if title:
        ax.set_title(title, color='#f8fafc', fontsize=12, fontweight='bold', pad=10)

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='#0b0f17')
    if show:
        plt.show()

    return fig


def render_input_pair_figure(
    fixed,
    moving,
    output_path=None,
    title=None,
    slice_indices=None,
    theme: str = "dark",
    crop_background: bool = True,
    reorient: bool = True,
    dpi=150,
    show_figure=False
):
    """
    Renders standard Figure 1 visualization of input images prior to registration.
    
    Layout Invariants:
    * 3D Images: 2x3 panel layout with Fixed Image at top (Axial, Coronal, Sagittal)
      and Moving Image at bottom (Axial, Coronal, Sagittal).
    * 2D Images: 1x2 panel layout with Fixed Image on Left and Moving Image on Right.
    
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

    fi_arr = fixed_img.numpy() if isinstance(fixed_img, ants.ANTsImage) else np.squeeze(np.asarray(fixed_img))
    mi_arr = moving_img.numpy() if isinstance(moving_img, ants.ANTsImage) else np.squeeze(np.asarray(moving_img))

    dim = fi_arr.ndim
    if dim not in (2, 3):
        raise ValueError(f"render_input_pair_figure expects 2D or 3D images, got shape {fi_arr.shape}")

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
        fig.subplots_adjust(wspace=0.18, left=0.08, right=0.92, top=0.88, bottom=0.05)

        for ax in axes:
            ax.set_facecolor(bg_color)
            ax.axis('off')

        im0 = axes[0].imshow(np.rot90(fi_render), cmap='gray')
        axes[0].set_title("Fixed Image (Target)", fontsize=13, fontweight='bold', color=fixed_label_color, pad=8)
        cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cb0.ax.tick_params(colors=cbar_tick_color)

        im1 = axes[1].imshow(np.rot90(mi_render), cmap='gray')
        axes[1].set_title("Moving Image (Source)", fontsize=13, fontweight='bold', color=moving_label_color, pad=8)
        cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cb1.ax.tick_params(colors=cbar_tick_color)

        if title is None:
            title = "Figure 1: Input Fixed (Left) & Moving (Right) Images"
        fig.suptitle(title, fontsize=15, fontweight='bold', color=text_color, y=0.97)

    else:  # 3D
        # In LPI space or standard 3D array:
        # If shape is (X, Y, Z): X=Left-Right, Y=Posterior-Anterior, Z=Inferior-Superior
        # If shape is (D, H, W): dim 0=Z/X, dim 1=Y, dim 2=X/Z
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
                s0_f, s1_f, s2_f = int(np.mean(i0)), int(np.mean(i1)), int(np.mean(i2))
            else:
                s0_f, s1_f, s2_f = shape_f[0] // 2, shape_f[1] // 2, shape_f[2] // 2

            mi_mask = (mi_arr > 0)
            if np.any(mi_mask):
                j0, j1, j2 = np.where(mi_mask)
                s0_m, s1_m, s2_m = int(np.mean(j0)), int(np.mean(j1)), int(np.mean(j2))
            else:
                s0_m, s1_m, s2_m = shape_m[0] // 2, shape_m[1] // 2, shape_m[2] // 2

        s0_f = max(f0min, min(f0max - 1, s0_f))
        s1_f = max(f1min, min(f1max - 1, s1_f))
        s2_f = max(f2min, min(f2max - 1, s2_f))

        s0_m = max(m0min, min(m0max - 1, s0_m))
        s1_m = max(m1min, min(m1max - 1, s1_m))
        s2_m = max(m2min, min(m2max - 1, s2_m))

        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), dpi=dpi, facecolor=bg_color)
        fig.subplots_adjust(wspace=0.18, hspace=0.25, left=0.10, right=0.95, top=0.88, bottom=0.05)

        for ax_row in axes:
            for ax in ax_row:
                ax.set_facecolor(bg_color)
                ax.axis('off')

        # Slice extraction with proper anatomical orientation (LPI space):
        # Dim 0: Left-Right (X), Dim 1: Posterior-Anterior (Y), Dim 2: Inferior-Superior (Z)
        # Axial: Slice along Z (dim 2=s2). Matrix (X, Y). np.rot90 makes Anterior UP.
        # Coronal: Slice along Y (dim 1=s1). Matrix (X, Z). np.rot90 makes Superior UP.
        # Sagittal: Slice along X (dim 0=s0). Matrix (Y, Z). np.rot90 makes Superior UP.
        
        # Fixed Image Slices
        ax_f = np.rot90(fi_arr[f0min:f0max, f1min:f1max, s2_f])
        cor_f = np.rot90(fi_arr[f0min:f0max, s1_f, f2min:f2max])
        sag_f = np.rot90(fi_arr[s0_f, f1min:f1max, f2min:f2max])

        slices_fixed = [
            (ax_f, f"Axial (Z={s2_f})"),
            (cor_f, f"Coronal (Y={s1_f})"),
            (sag_f, f"Sagittal (X={s0_f})")
        ]

        for col_idx, (sl, label) in enumerate(slices_fixed):
            im = axes[0, col_idx].imshow(sl, cmap='gray')
            axes[0, col_idx].set_title(f"Fixed: {label}", fontsize=11, fontweight='bold', color=sub_color)
            cb = plt.colorbar(im, ax=axes[0, col_idx], fraction=0.046, pad=0.04)
            cb.ax.tick_params(colors=cbar_tick_color)

        # Moving Image Slices
        ax_m = np.rot90(mi_arr[m0min:m0max, m1min:m1max, s2_m])
        cor_m = np.rot90(mi_arr[m0min:m0max, s1_m, m2min:m2max])
        sag_m = np.rot90(mi_arr[s0_m, m1min:m1max, m2min:m2max])

        slices_moving = [
            (ax_m, f"Axial (Z={s2_m})"),
            (cor_m, f"Coronal (Y={s1_m})"),
            (sag_m, f"Sagittal (X={s2_m})")
        ]

        for col_idx, (sl, label) in enumerate(slices_moving):
            im = axes[1, col_idx].imshow(sl, cmap='gray')
            axes[1, col_idx].set_title(f"Moving: {label}", fontsize=11, fontweight='bold', color=sub_color)
            cb = plt.colorbar(im, ax=axes[1, col_idx], fraction=0.046, pad=0.04)
            cb.ax.tick_params(colors=cbar_tick_color)

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
    warp,
    detJ,
    inv_err_map,
    slice_axis: int = 2,
    slice_idx=None,
    lncc_val=None,
    mi_val=None,
    inv_err_max=None,
    inv_err_mean=None,
    inv_err_p95=None,
    min_detJ=None,
    title_prefix="Registration Report",
    filename=None
):
    """
    Renders standardized 4-panel registration visual report in 2D or 3D:
      Panel A: Standard Deformed Mesh Grid
      Panel B: Standard Divergent Jacobian Determinant Map
      Panel C: Standardized Inverse Identity Error Map (mm)
      Panel D: High-Contrast Canny Edge Alignment Overlap
    """
    if inv_err_map is None:
        raise ValueError(
            "render_standard_4panel requires a valid inv_err_map (ANTsImage or Tensor representing physical inverse identity error in mm). "
            "Passing None or dummy objects is strictly prohibited per GEMINI.md Section 3."
        )

    fi_arr = extract_2d_slice(fixed, slice_axis=slice_axis, slice_idx=slice_idx)
    mi_arr = extract_2d_slice(warped, slice_axis=slice_axis, slice_idx=slice_idx, ref_image=fixed)
    detJ_arr = extract_2d_slice(detJ, slice_axis=slice_axis, slice_idx=slice_idx, ref_image=fixed)
    inv_err_arr = extract_2d_slice(inv_err_map, slice_axis=slice_axis, slice_idx=slice_idx, ref_image=fixed)
    disp = extract_2d_slice(warp, slice_axis=slice_axis, slice_idx=slice_idx, ref_image=fixed)

    if not isinstance(inv_err_arr, np.ndarray) or inv_err_arr.size == 0:
        raise ValueError("render_standard_4panel: inv_err_map slice extraction failed or produced empty array.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 13), facecolor='#0d1117')
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('#0d1117')
            ax.axis('off')

    # Panel A: Standard Deformed Mesh Grid
    H, W = fi_arr.shape
    grid_spacing = 8
    grid_y, grid_x = np.mgrid[0:H:grid_spacing, 0:W:grid_spacing]
    disp_y = disp[::grid_spacing, ::grid_spacing, 0] if disp.ndim >= 2 else 0
    disp_x = disp[::grid_spacing, ::grid_spacing, 1] if disp.ndim >= 2 else 0
    def_y = grid_y + disp_y
    def_x = grid_x + disp_x

    axes[0, 0].imshow(fi_arr, cmap='gray', alpha=0.5, origin='upper')
    for i in range(def_y.shape[0]):
        axes[0, 0].plot(def_x[i, :], def_y[i, :], color='#38bdf8', linewidth=1.1)
    for j in range(def_x.shape[1]):
        axes[0, 0].plot(def_x[:, j], def_y[:, j], color='#38bdf8', linewidth=1.1)
    axes[0, 0].set_title(f'{title_prefix}\nPanel A: Standard Deformed Mesh Grid', color='#38bdf8', fontsize=11, fontweight='bold')

    # Panel B: Standard Jacobian Determinant Map
    colors_jac = [(0.0, '#00ff00'), (0.001, '#f85149'), (0.5, '#161b22'), (1.0, '#58a6ff')]
    cmap_jac = mcolors.LinearSegmentedColormap.from_list('diffeo_cmap', colors_jac)
    im_jac = axes[0, 1].imshow(detJ_arr, cmap=cmap_jac, vmin=-0.1, vmax=2.5, origin='upper')
    folding_pct = float(np.mean(detJ_arr <= 0.0) * 100.0)
    min_j_val = min_detJ if min_detJ is not None else float(np.min(detJ_arr))
    status_str = "0.00% Folding" if folding_pct == 0.0 else f"{folding_pct:.2f}% Folding"
    axes[0, 1].set_title(f'Panel B: Standard Jacobian det(J)\nmin det(J) = {min_j_val:+.2f} ({status_str})', color='#3fb950', fontsize=11, fontweight='bold')
    cbar_j = fig.colorbar(im_jac, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar_j.ax.tick_params(colors='#c9d1d9')

    # Panel C: Standardized Inverse Identity Error Map (mm)
    max_err_val = inv_err_max if inv_err_max is not None else float(np.max(inv_err_arr))
    mean_err_val = inv_err_mean if inv_err_mean is not None else float(np.mean(inv_err_arr))
    p95_err_val = inv_err_p95 if inv_err_p95 is not None else float(np.percentile(inv_err_arr, 95))

    axes[1, 0].imshow(fi_arr, cmap='gray', alpha=0.3, origin='upper')
    im_err = axes[1, 0].imshow(inv_err_arr, cmap='inferno', alpha=0.85, vmin=0.0, vmax=max(3.0, max_err_val), origin='upper')
    axes[1, 0].set_title(f'Panel C: Inverse Error Map (mm)\nMax: {max_err_val:.2f}mm | Mean: {mean_err_val:.3f}mm | p95: {p95_err_val:.2f}mm', color='#d29922', fontsize=11, fontweight='bold')
    cbar_e = fig.colorbar(im_err, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar_e.set_label('Inverse Error (mm)', color='#c9d1d9', fontsize=10)
    cbar_e.ax.tick_params(colors='#c9d1d9')

    # Panel D: Standardized High-Contrast Canny Edge Overlap
    plot_edge_overlay(fixed, warped, slice_axis=slice_axis, slice_idx=slice_idx, edge_color='#f85149', fixed_edge_color=None, ax=axes[1, 1], title="")
    lncc_str = f"Target LNCC: {lncc_val:.4f}" if lncc_val is not None else ""
    mi_str = f"Mattes MI: {mi_val:.4f}" if mi_val is not None else ""
    metrics_sub = " | ".join(filter(None, [lncc_str, mi_str]))
    axes[1, 1].set_title(f'Panel D: Edge Alignment Overlap (Canny Red Contours)\n{metrics_sub}', color='#bc8cff', fontsize=11, fontweight='bold')

    plt.tight_layout()

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='#0d1117')

    return fig
