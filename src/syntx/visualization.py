"""
Publication-Grade Visualization Engine for Syntx Medical Image Registration.

Provides standard, modular, software-engineered plotting utilities for 2D and 3D
registration comparisons, structural overlays, deformation grids, Jacobian maps,
and multi-method benchmark grids.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import ants


def _to_numpy(img):
    """Converts ANTsImage, PyTorch Tensor, NumPy array, or file path to NumPy array."""
    if isinstance(img, str) and os.path.exists(img):
        img = ants.image_read(img)
    if isinstance(img, ants.ANTsImage):
        return img.numpy(), img
    if hasattr(img, 'detach'):
        arr = img.detach().cpu().numpy()
        return np.squeeze(arr), None
    if hasattr(img, 'numpy'):
        return np.squeeze(img.numpy()), None
    arr = np.asarray(img)
    return np.squeeze(arr), None


def extract_2d_slice(img, slice_axis: int = 2, slice_idx=None, ref_image=None):
    """
    Extracts a 2D scalar NumPy slice (H, W) or 2D vector slice (H, W, 2)
    from 2D or 3D images, tensors, ANTsImages, file paths, or vector fields 
    strictly respecting standard ANTs LAI anatomical orientation conventions.

    Parameters
    ----------
    img : ANTsImage, Tensor, ndarray, or str path
        Input 2D or 3D scalar volume or vector displacement field.
    slice_axis : int
        Axis along which to slice 3D volumes (0: Sagittal, 1: Coronal, 2: Axial). Default: 2.
    slice_idx : int or float, optional
        Slice index or relative fraction [0, 1]. Defaults to midpoint.
    ref_image : ANTsImage, optional
        Reference image for physical space matching.

    Returns
    -------
    ndarray : 2D slice formatted in standard ANTs anatomical viewing orientation.
    """
    arr, ants_obj = _to_numpy(img)
    ref_ants = ref_image if isinstance(ref_image, ants.ANTsImage) else (ants_obj if isinstance(ants_obj, ants.ANTsImage) else None)

    if isinstance(img, ants.ANTsImage):
        if img.dimension == 3:
            img_lai = img.reorient_image2('LAI')
            arr = img_lai.numpy()
        else:
            arr = img.numpy()
    elif ref_ants is not None and ref_ants.dimension == 3 and arr.ndim in (3, 4):
        try:
            if arr.ndim == 3:
                img_ants = ants.from_numpy(arr, origin=ref_ants.origin, spacing=ref_ants.spacing, direction=ref_ants.direction)
                arr = img_ants.reorient_image2('LAI').numpy()
            elif arr.ndim == 4:
                if arr.shape[0] in (2, 3) and arr.shape[1] > 4:
                    arr = np.moveaxis(arr, 0, -1)
                comps = []
                for c in range(arr.shape[-1]):
                    img_c = ants.from_numpy(arr[..., c], origin=ref_ants.origin, spacing=ref_ants.spacing, direction=ref_ants.direction)
                    comps.append(img_c.reorient_image2('LAI').numpy())
                arr = np.stack(comps, axis=-1)
        except Exception:
            pass

    if arr.ndim < 2:
        if ref_ants is not None:
            ref_slice = extract_2d_slice(ref_ants, slice_axis=slice_axis, slice_idx=slice_idx)
            return np.full(ref_slice.shape, float(arr), dtype=np.float32)
        return np.array([[float(arr)]], dtype=np.float32)

    # 2D Scalar image (H, W) -> ANTs 2D plot convention: arr.T
    if arr.ndim == 2:
        return arr.T

    # 2D Vector displacement field: (2, H, W) or (H, W, 2)
    if arr.ndim == 3 and arr.shape[0] in (2, 3) and arr.shape[1] > 4 and arr.shape[2] > 4:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] in (2, 3):
        return np.transpose(arr[..., :2], (1, 0, 2))

    # 3D Scalar volume (H, W, D)
    if arr.ndim == 3:
        depth = arr.shape[slice_axis]
        if slice_idx is None:
            s_idx = depth // 2
        elif isinstance(slice_idx, float) and 0.0 <= slice_idx <= 1.0:
            s_idx = int(round(slice_idx * (depth - 1)))
        else:
            s_idx = max(0, min(depth - 1, int(slice_idx)))

        if slice_axis == 2:    # Axial: Transpose
            return arr[:, :, s_idx].T
        elif slice_axis == 1:  # Coronal: Transpose + flip vertical
            return arr[:, s_idx, :].T[::-1, :]
        else:                  # Sagittal: Transpose + flip vertical
            return arr[s_idx, :, :].T[::-1, :]

    # 3D Vector field (3, H, W, D) or (H, W, D, 3)
    if arr.ndim == 4:
        if arr.shape[0] in (2, 3) and arr.shape[1] > 4:
            arr = np.moveaxis(arr, 0, -1)
        depth = arr.shape[slice_axis]
        if slice_idx is None:
            s_idx = depth // 2
        elif isinstance(slice_idx, float) and 0.0 <= slice_idx <= 1.0:
            s_idx = int(round(slice_idx * (depth - 1)))
        else:
            s_idx = max(0, min(depth - 1, int(slice_idx)))

        if slice_axis == 2:
            vec_2d = arr[:, :, s_idx, :2]
        elif slice_axis == 1:
            vec_2d = arr[:, s_idx, :, :2]
        else:
            vec_2d = arr[s_idx, :, :, :2]
        return np.transpose(vec_2d, (1, 0, 2))

    return arr


def _compute_canny_or_sobel_edges(img_2d, sigma=1.2):
    """Computes binary Canny or Sobel edge map on a 2D scalar array."""
    amin, amax = np.min(img_2d), np.max(img_2d)
    norm = (img_2d - amin) / (amax - amin + 1e-8) if amax > amin else img_2d.copy()

    try:
        from skimage.feature import canny
        return canny(norm, sigma=sigma)
    except ImportError:
        from scipy.ndimage import sobel
        g = np.hypot(sobel(norm, 0), sobel(norm, 1))
        return g > np.percentile(g, 88)


def _compute_jacobian_2d(disp_2d, spacing=(1.0, 1.0)):
    """Computes 2D Jacobian determinant map det(J) from 2D displacement field (H, W, 2)."""
    if disp_2d.ndim != 3 or disp_2d.shape[-1] < 2:
        return np.ones(disp_2d.shape[:2], dtype=np.float32)
    du_dx = np.gradient(disp_2d[..., 0], axis=0) / spacing[0]
    du_dy = np.gradient(disp_2d[..., 1], axis=1) / spacing[1]
    detJ = (1.0 + du_dx) * (1.0 + du_dy)
    return detJ


def plot_comparison(
    images,
    mode: str = "side_by_side",
    slice_axis: int = 2,
    slice_idx=None,
    titles=None,
    subtitles=None,
    main_title: str = None,
    main_subtitle: str = None,
    metrics: dict = None,
    badges: list = None,
    cmap: str = "gray",
    diff_cmap: str = "magma",
    jac_cmap: str = "turbo",
    edge_color: str = "#f85149",
    fixed_edge_color: str = "#38bdf8",
    grid_spacing: int = 8,
    grid_color: str = "#38bdf8",
    linewidth: float = 1.2,
    alpha: float = 0.85,
    robust_scaling: bool = True,
    quantile_range=(0.01, 0.99),
    figsize=None,
    dpi: int = 150,
    cbar: bool = False,
    theme: str = "dark",
    ncols: int = None,
    ax=None,
    show: bool = False,
    filename: str = None,
):
    """
    Standard, Modular, Software-Engineered Plotting Core Engine for Syntx Registration Comparisons.

    Supports both 2D and 3D medical images with strict anatomical orientation verification,
    flexible layout modes (side-by-side, difference, edge overlay, deformed grid, jacobian, orthogonal/triplanar),
    custom metric badges, headers, dark/light themes, and rich subtitle annotations.

    Parameters
    ----------
    images : list, tuple, dict, or ndarray
        Image(s) or dictionary of images to plot.
        - Single image or list of images: [fixed, moving, warped1, ...]
        - Dict of labeled images: {"Target (Fixed)": fi, "Source (Moving)": mi, ...}
        - Tuple of 2D grid matrix of images: [[row1_cols], [row2_cols]]
    mode : str
        Display mode:
        - 'side_by_side' (or 'grid'): Grid layout of images.
        - 'orthogonal' (or 'triplanar'): 3-slice orthogonal view (Axial, Coronal, Sagittal) per image (for 3D volumes).
        - 'difference' (or 'diff'): Displays [Target, Candidate, |Target - Candidate|].
        - 'edge_overlay' (or 'edges'): Displays Canny edge alignment overlay (Target edges in Cyan, Candidate in Red).
        - 'deformed_grid' (or 'mesh'): Displays spatial displacement grid overlaid on target image.
        - 'jacobian' (or 'detJ'): Displays Jacobian determinant map det(J) with divergent colormap.
    slice_axis : int
        Slice orientation for 3D volumes (0: Sagittal, 1: Coronal, 2: Axial). Default: 2.
    slice_idx : int or float, optional
        Slice index or relative fraction along slice_axis. Defaults to volume midpoint.
    titles : list of str, optional
        Panel titles for each image/panel.
    subtitles : list of str, optional
        Secondary subtitles / metrics for each panel.
    main_title : str, optional
        Main figure title rendered at top header.
    main_subtitle : str, optional
        Main figure subtitle rendered below main_title.
    metrics : dict, optional
        Summary metrics dict rendered as high-contrast top header metric cards (e.g. {"MSE": 0.0015, "DICE": 0.88}).
    badges : list of str, optional
        List of badge strings rendered at top right.
    cmap : str
        Colormap for grayscale images (default: 'gray').
    diff_cmap : str
        Colormap for difference heatmaps (default: 'magma').
    jac_cmap : str
        Colormap for Jacobian determinant maps (default: 'turbo').
    edge_color : str
        Color for candidate/warped image edges in edge_overlay mode (default: '#f85149').
    fixed_edge_color : str
        Color for target/fixed image edges in edge_overlay mode (default: '#38bdf8').
    grid_spacing : int
        Subsampling interval for coordinate deformation mesh grid lines (default: 8 voxels).
    grid_color : str
        Mesh grid line color (default: '#38bdf8').
    linewidth : float
        Line width for mesh grid or contour plots.
    alpha : float
        Alpha transparency level for overlays.
    robust_scaling : bool
        If True, scales image contrast dynamically using quantile_range (default: True).
    quantile_range : tuple of float
        Quantile bounds (lower, upper) for robust intensity scaling (default: (0.01, 0.99)).
    figsize : tuple of int, optional
        Matplotlib figure size. Auto-computed if None.
    dpi : int
        Resolution dots per inch (default: 150).
    cbar : bool
        If True, displays colorbars for heatmaps or difference/jacobian maps.
    theme : str
        Visual theme: 'dark' (default, #090d16 background) or 'light' (white background).
    ncols : int, optional
        Number of columns for grid layout mode.
    ax : Matplotlib Axes, optional
        Existing Axes object for single-panel embedding.
    show : bool
        If True, calls plt.show().
    filename : str, optional
        File path to save the generated output figure image.

    Returns
    -------
    fig : Matplotlib Figure object
    """
    mode = mode.lower()
    theme = theme.lower()

    # Colors & Theme Setup
    if theme == "dark":
        bg_color = "#090d16"
        card_bg = "#111622"
        border_color = "#212636"
        text_main = "#f8fafc"
        text_muted = "#94a3b8"
        accent_blue = "#38bdf8"
    else:
        bg_color = "#ffffff"
        card_bg = "#f8fafc"
        border_color = "#e2e8f0"
        text_main = "#0f172a"
        text_muted = "#64748b"
        accent_blue = "#0284c7"

    # --- 1. Normalize Input Image Data Structure ---
    img_list = []
    labels_list = []
    ref_image = None

    if isinstance(images, dict):
        for k, v in images.items():
            labels_list.append(k)
            img_list.append(v)
            if ref_image is None and isinstance(v, ants.ANTsImage):
                ref_image = v
    elif isinstance(images, (list, tuple)):
        # Check if 2D grid matrix of images
        if len(images) > 0 and isinstance(images[0], (list, tuple)):
            for r_idx, row in enumerate(images):
                for c_idx, item in enumerate(row):
                    img_list.append(item)
                    labels_list.append(f"Row {r_idx+1}, Col {c_idx+1}")
                    if ref_image is None and isinstance(item, ants.ANTsImage):
                        ref_image = item
        else:
            for idx, item in enumerate(images):
                img_list.append(item)
                labels_list.append(f"Image {idx+1}")
                if ref_image is None and isinstance(item, ants.ANTsImage):
                    ref_image = item
    else:
        img_list.append(images)
        labels_list.append("Image")
        if isinstance(images, ants.ANTsImage):
            ref_image = images

    if titles is not None:
        if isinstance(titles, str):
            titles = [titles]
        for i in range(min(len(titles), len(labels_list))):
            labels_list[i] = titles[i]

    # --- 2. Handle Specific Specialized Modes ---
    if mode in ("orthogonal", "triplanar"):
        # Orthogonal triplanar 3-slice views (Axial, Coronal, Sagittal) per image
        n_imgs = len(img_list)
        if figsize is None:
            figsize = (13, 4.2 * n_imgs)

        fig, axes = plt.subplots(n_imgs, 3, figsize=figsize, facecolor=bg_color)
        if n_imgs == 1:
            axes = np.expand_dims(axes, 0)

        for i, img in enumerate(img_list):
            label = labels_list[i]
            # Extract 3 orthogonal slices using extract_2d_slice
            slice_ax = extract_2d_slice(img, slice_axis=2, slice_idx=slice_idx, ref_image=ref_image)  # Axial
            slice_co = extract_2d_slice(img, slice_axis=1, slice_idx=slice_idx, ref_image=ref_image)  # Coronal
            slice_sa = extract_2d_slice(img, slice_axis=0, slice_idx=slice_idx, ref_image=ref_image)  # Sagittal

            views = [
                (slice_ax, f"{label} (Axial)"),
                (slice_co, f"{label} (Coronal)"),
                (slice_sa, f"{label} (Sagittal)")
            ]

            for j, (slice_data, v_title) in enumerate(views):
                ax_curr = axes[i, j]
                ax_curr.set_facecolor(bg_color)

                if robust_scaling:
                    vmin, vmax = np.quantile(slice_data, quantile_range)
                else:
                    vmin, vmax = slice_data.min(), slice_data.max()

                im = ax_curr.imshow(slice_data, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
                ax_curr.set_title(v_title, color=text_main, fontsize=11, fontweight='semibold', pad=8)
                ax_curr.axis('off')

                if cbar and j == 2:
                    cb = fig.colorbar(im, ax=ax_curr, fraction=0.046, pad=0.04)
                    cb.ax.tick_params(colors=text_muted)

        # Header Titles
        if main_title:
            fig.suptitle(main_title, color=text_main, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        if filename:
            fig.savefig(filename, dpi=dpi, facecolor=bg_color, bbox_inches='tight')
            if not show:
                plt.close(fig)
        elif show:
            plt.show()
        return fig

    elif mode in ("difference", "diff"):
        # Expects pair: [fixed, moving/warped]
        fi_slice = extract_2d_slice(img_list[0], slice_axis=slice_axis, slice_idx=slice_idx, ref_image=ref_image)
        mi_slice = extract_2d_slice(img_list[1] if len(img_list) > 1 else img_list[0], slice_axis=slice_axis, slice_idx=slice_idx, ref_image=ref_image)
        diff_slice = np.abs(fi_slice - mi_slice)

        if figsize is None:
            figsize = (14, 4.8)

        fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor=bg_color)
        for ax_c in axes:
            ax_c.set_facecolor(bg_color)
            ax_c.axis('off')

        vmin_f, vmax_f = (np.quantile(fi_slice, quantile_range) if robust_scaling else (fi_slice.min(), fi_slice.max()))
        vmin_m, vmax_m = (np.quantile(mi_slice, quantile_range) if robust_scaling else (mi_slice.min(), mi_slice.max()))

        t0 = titles[0] if (titles and len(titles) > 0) else "Target (Fixed)"
        t1 = titles[1] if (titles and len(titles) > 1) else "Candidate (Warped)"
        t2 = titles[2] if (titles and len(titles) > 2) else f"Absolute Difference (MSE: {np.mean(diff_slice**2):.4f})"

        axes[0].imshow(fi_slice, cmap=cmap, vmin=vmin_f, vmax=vmax_f, origin='upper')
        axes[0].set_title(t0, color=text_main, fontsize=12, fontweight='semibold', pad=8)

        axes[1].imshow(mi_slice, cmap=cmap, vmin=vmin_m, vmax=vmax_m, origin='upper')
        axes[1].set_title(t1, color=text_main, fontsize=12, fontweight='semibold', pad=8)

        im_d = axes[2].imshow(diff_slice, cmap=diff_cmap, origin='upper')
        axes[2].set_title(t2, color=text_main, fontsize=12, fontweight='semibold', pad=8)
        cb = fig.colorbar(im_d, ax=axes[2], fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=text_muted)

        if main_title:
            fig.suptitle(main_title, color=text_main, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if filename:
            fig.savefig(filename, dpi=dpi, facecolor=bg_color, bbox_inches='tight')
            if not show:
                plt.close(fig)
        elif show:
            plt.show()
        return fig

    elif mode in ("edge_overlay", "edges"):
        # Canny / Sobel edge overlay of candidate edges on fixed background
        fi_slice = extract_2d_slice(img_list[0], slice_axis=slice_axis, slice_idx=slice_idx, ref_image=ref_image)
        mi_slice = extract_2d_slice(img_list[1] if len(img_list) > 1 else img_list[0], slice_axis=slice_axis, slice_idx=slice_idx, ref_image=ref_image)

        edges_w = _compute_canny_or_sobel_edges(mi_slice)
        edges_f = _compute_canny_or_sobel_edges(fi_slice) if fixed_edge_color else None

        if ax is None:
            if figsize is None:
                figsize = (7.5, 7.5)
            fig, ax_curr = plt.subplots(figsize=figsize, facecolor=bg_color)
        else:
            fig = ax.figure
            ax_curr = ax

        ax_curr.set_facecolor(bg_color)
        vmin_f, vmax_f = (np.quantile(fi_slice, quantile_range) if robust_scaling else (fi_slice.min(), fi_slice.max()))
        ax_curr.imshow(fi_slice, cmap=cmap, vmin=vmin_f, vmax=vmax_f, origin='upper')

        # Candidate Edges Overlay (Red / edge_color)
        overlay_w = np.zeros((*fi_slice.shape, 4), dtype=np.float32)
        r, g, b = mcolors.to_rgb(edge_color)
        overlay_w[edges_w] = [r, g, b, alpha]
        ax_curr.imshow(overlay_w, origin='upper')

        if fixed_edge_color and edges_f is not None:
            overlay_f = np.zeros((*fi_slice.shape, 4), dtype=np.float32)
            rf, gf, bf = mcolors.to_rgb(fixed_edge_color)
            overlay_f[edges_f] = [rf, gf, bf, alpha]
            ax_curr.imshow(overlay_f, origin='upper')

        ax_curr.axis('off')
        t0 = titles[0] if (titles and len(titles) > 0) else "Canny Edge Alignment Overlap"
        if main_title:
            t0 = f"{main_title}\n{t0}"
        ax_curr.set_title(t0, color=text_main, fontsize=12, fontweight='bold', pad=10)

        plt.tight_layout()
        if filename:
            fig.savefig(filename, dpi=dpi, facecolor=bg_color, bbox_inches='tight')
            if not show:
                plt.close(fig)
        elif show:
            plt.show()
        return fig

    elif mode in ("deformed_grid", "mesh"):
        # Spatial coordinate deformation mesh grid overlay
        warp_data = img_list[0]
        bg_data = img_list[1] if len(img_list) > 1 else None

        disp_2d = extract_2d_slice(warp_data, slice_axis=slice_axis, slice_idx=slice_idx, ref_image=ref_image)
        bg_slice = extract_2d_slice(bg_data, slice_axis=slice_axis, slice_idx=slice_idx, ref_image=ref_image) if bg_data is not None else None

        if ax is None:
            if figsize is None:
                figsize = (7.5, 7.5)
            fig, ax_curr = plt.subplots(figsize=figsize, facecolor=bg_color)
        else:
            fig = ax.figure
            ax_curr = ax

        ax_curr.set_facecolor(bg_color)
        H, W = disp_2d.shape[:2]
        step = max(1, grid_spacing)

        grid_y, grid_x = np.mgrid[0:H:step, 0:W:step]
        disp_y = disp_2d[::step, ::step, 0] if disp_2d.ndim >= 3 else np.zeros_like(grid_y)
        disp_x = disp_2d[::step, ::step, 1] if disp_2d.ndim >= 3 else np.zeros_like(grid_x)

        def_y = grid_y + disp_y
        def_x = grid_x + disp_x

        if bg_slice is not None:
            vmin_b, vmax_b = (np.quantile(bg_slice, quantile_range) if robust_scaling else (bg_slice.min(), bg_slice.max()))
            ax_curr.imshow(bg_slice, cmap=cmap, alpha=0.5, vmin=vmin_b, vmax=vmax_b, origin='upper')

        for i in range(def_y.shape[0]):
            ax_curr.plot(def_x[i, :], def_y[i, :], color=grid_color, lw=linewidth)
        for j in range(def_x.shape[1]):
            ax_curr.plot(def_x[:, j], def_y[:, j], color=grid_color, lw=linewidth)

        ax_curr.set_aspect('equal')
        ax_curr.axis('off')

        t0 = titles[0] if (titles and len(titles) > 0) else "Deformation Mesh Grid"
        if main_title:
            t0 = f"{main_title}\n{t0}"
        ax_curr.set_title(t0, color=text_main, fontsize=12, fontweight='bold', pad=10)

        plt.tight_layout()
        if filename:
            fig.savefig(filename, dpi=dpi, facecolor=bg_color, bbox_inches='tight')
            if not show:
                plt.close(fig)
        elif show:
            plt.show()
        return fig

    elif mode in ("jacobian", "detj"):
        # Jacobian determinant map det(J)
        warp_data = img_list[0]
        disp_2d = extract_2d_slice(warp_data, slice_axis=slice_axis, slice_idx=slice_idx, ref_image=ref_image)

        if disp_2d.ndim == 3 and disp_2d.shape[-1] >= 2:
            detJ = _compute_jacobian_2d(disp_2d)
        else:
            detJ = disp_2d

        if ax is None:
            if figsize is None:
                figsize = (7.5, 7.5)
            fig, ax_curr = plt.subplots(figsize=figsize, facecolor=bg_color)
        else:
            fig = ax.figure
            ax_curr = ax

        ax_curr.set_facecolor(bg_color)
        colors_jac = [(0.0, '#00ff00'), (0.001, '#f85149'), (0.5, '#161b22'), (1.0, '#58a6ff')]
        cmap_j = mcolors.LinearSegmentedColormap.from_list('diffeo_cmap', colors_jac)

        im = ax_curr.imshow(detJ, cmap=cmap_j, vmin=-0.1, vmax=2.5, origin='upper')
        ax_curr.axis('off')

        folding_pct = float(np.mean(detJ <= 0.0) * 100.0)
        status_str = "0.00% Folding" if folding_pct == 0.0 else f"{folding_pct:.2f}% Folding"
        t0 = titles[0] if (titles and len(titles) > 0) else f"Jacobian det(J) [{np.min(detJ):+.2f}, {np.max(detJ):.2f}] ({status_str})"
        if main_title:
            t0 = f"{main_title}\n{t0}"
        ax_curr.set_title(t0, color=text_main, fontsize=12, fontweight='bold', pad=10)

        cb = fig.colorbar(im, ax=ax_curr, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=text_muted)

        plt.tight_layout()
        if filename:
            fig.savefig(filename, dpi=dpi, facecolor=bg_color, bbox_inches='tight')
            if not show:
                plt.close(fig)
        elif show:
            plt.show()
        return fig

    # --- 3. Default Grid Layout Mode ('side_by_side' / 'grid') ---
    n_panels = len(img_list)
    if ncols is None:
        if n_panels <= 4:
            ncols = n_panels
        else:
            ncols = min(4, math.ceil(math.sqrt(n_panels)))
    nrows = math.ceil(n_panels / ncols)

    if figsize is None:
        figsize = (3.8 * ncols, 4.0 * nrows + (1.2 if (main_title or metrics) else 0))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=bg_color)
    axes_flat = np.array(axes).flatten() if n_panels > 1 else [axes]

    # Render Header Metrics Cards / Title if supplied
    if main_title or main_subtitle or metrics:
        header_text = main_title if main_title else "Syntx Image Registration Comparison"
        if main_subtitle:
            header_text += f"\n{main_subtitle}"
        fig.suptitle(header_text, color=text_main, fontsize=14, fontweight='bold', y=0.98)

    for i in range(len(axes_flat)):
        ax_curr = axes_flat[i]
        ax_curr.set_facecolor(bg_color)

        if i < n_panels:
            slice_2d = extract_2d_slice(img_list[i], slice_axis=slice_axis, slice_idx=slice_idx, ref_image=ref_image)
            label = labels_list[i]
            sub = subtitles[i] if (subtitles and i < len(subtitles)) else None

            if robust_scaling:
                vmin, vmax = np.quantile(slice_2d, quantile_range)
            else:
                vmin, vmax = slice_2d.min(), slice_2d.max()

            im = ax_curr.imshow(slice_2d, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')

            full_label = f"{label}\n{sub}" if sub else label
            ax_curr.set_title(full_label, color=text_main, fontsize=11, fontweight='semibold', pad=6)

            if cbar:
                cb = fig.colorbar(im, ax=ax_curr, fraction=0.046, pad=0.04)
                cb.ax.tick_params(colors=text_muted)
        ax_curr.axis('off')

    plt.tight_layout()

    if filename:
        fig.savefig(filename, dpi=dpi, facecolor=bg_color, bbox_inches='tight')
        if not show:
            plt.close(fig)
    elif show:
        plt.show()

    return fig


# --- Layered Specialized Plotting Helpers Built on Core Visualization Engine ---

def plot_structural_comparison(
    fixed,
    moving,
    title: str = "Structural Image Comparison",
    labels=("Fixed Target Image", "Moving Input Image"),
    figsize=None,
    robust_scaling: bool = True,
    cmap: str = 'gray',
    ax=None,
    show: bool = False,
    filename=None
):
    """
    Renders a standard side-by-side structural image comparison panel
    (orthogonal views for 3D, side-by-side for 2D) using plot_comparison core engine.
    """
    f_arr, _ = _to_numpy(fixed)
    is_3d = (f_arr.ndim == 3 and min(f_arr.shape) > 4)

    mode = "orthogonal" if is_3d else "side_by_side"
    dict_imgs = {labels[0]: fixed, labels[1]: moving}

    return plot_comparison(
        images=dict_imgs,
        mode=mode,
        main_title=title,
        figsize=figsize,
        robust_scaling=robust_scaling,
        cmap=cmap,
        show=show,
        filename=filename
    )


def plot_edge_overlay(
    fixed,
    warped,
    slice_axis: int = 2,
    slice_idx=None,
    edge_color: str = '#f85149',
    fixed_edge_color='#38bdf8',
    alpha: float = 0.85,
    ax=None,
    figsize=(8, 8),
    title: str = "Edge Alignment Overlap",
    show: bool = False,
    filename=None
):
    """
    Plots high-contrast anatomical Canny edge overlays using plot_comparison core engine.
    """
    return plot_comparison(
        images=[fixed, warped],
        mode="edge_overlay",
        slice_axis=slice_axis,
        slice_idx=slice_idx,
        edge_color=edge_color,
        fixed_edge_color=fixed_edge_color,
        alpha=alpha,
        ax=ax,
        figsize=figsize,
        main_title=title,
        show=show,
        filename=filename
    )


def plot_deformation_grid(
    warp,
    fixed=None,
    slice_axis: int = 2,
    slice_idx=None,
    grid_spacing: int = 8,
    color: str = '#38bdf8',
    linewidth: float = 1.2,
    background_cmap: str = 'gray',
    figsize=(8, 8),
    ax=None,
    title: str = "Deformation Grid",
    show: bool = False,
    filename=None
):
    """
    Plots spatial coordinate deformation mesh grid using plot_comparison core engine.
    """
    imgs = [warp, fixed] if fixed is not None else [warp]
    return plot_comparison(
        images=imgs,
        mode="deformed_grid",
        slice_axis=slice_axis,
        slice_idx=slice_idx,
        grid_spacing=grid_spacing,
        grid_color=color,
        linewidth=linewidth,
        cmap=background_cmap,
        figsize=figsize,
        ax=ax,
        main_title=title,
        show=show,
        filename=filename
    )


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
    fi_arr = extract_2d_slice(fixed, slice_axis=slice_axis, slice_idx=slice_idx)
    detJ_arr = extract_2d_slice(detJ, slice_axis=slice_axis, slice_idx=slice_idx, ref_image=fixed)
    inv_err_arr = extract_2d_slice(inv_err_map, slice_axis=slice_axis, slice_idx=slice_idx, ref_image=fixed)
    disp = extract_2d_slice(warp, slice_axis=slice_axis, slice_idx=slice_idx, ref_image=fixed)

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

    # Panel D: Standardized High-Contrast Canny Edge Overlap (Cyan=Fixed, Red=Warped)
    plot_edge_overlay(fixed, warped, slice_axis=slice_axis, slice_idx=slice_idx, edge_color='#f85149', fixed_edge_color='#38bdf8', ax=axes[1, 1], title="")
    lncc_str = f"Target LNCC: {lncc_val:.4f}" if lncc_val is not None else ""
    mi_str = f"Mattes MI: {mi_val:.4f}" if mi_val is not None else ""
    metrics_sub = " | ".join(filter(None, [lncc_str, mi_str]))
    axes[1, 1].set_title(f'Panel D: Edge Alignment Overlap (Canny Red Contours)\n{metrics_sub}', color='#bc8cff', fontsize=11, fontweight='bold')

    plt.tight_layout()
    if filename:
        fig.savefig(filename, dpi=160, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)

    return fig
