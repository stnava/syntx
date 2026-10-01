"""
syntx.viz.stats — summary plots of registration quality numbers
===============================================================

- ``plot_label_overlap_stats``: Dice box plots (fixed / moving / symmetric) and a per-region
  bar chart or histogram.
- ``plot_jacobian_distribution``: histogram of Jacobian determinant values with the
  fraction <= 0.
- ``plot_loss_convergence``: a loss curve (caller-supplied axis / legend labels).

All take ``theme`` ("dark" or anything else for light), optionally save to ``output_path``,
and return the matplotlib Figure.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_label_overlap_stats(
    dice_scores,
    labels_dict=None,
    title="Mindboggle Cortical DKT31 Label Overlap Benchmark",
    theme: str = "dark",
    output_path=None,
    dpi=150,
    show_figure=False
):
    """Two-panel Dice summary figure.

    Panel A: box plots -- fixed-space, moving-space and symmetric Dice when both directions
    are given, otherwise one "Dice" box -- titled with the mean, median and IQR of the
    symmetric (or only) values. Panel B: horizontal bars of the per-region mean Dice (the 15
    highest regions, sorted ascending, coloured with ``get_dkt_label_color_dict``), or, when
    there is no per-region data, a 12-bin histogram of the symmetric (or only) values.

    Parameters
    ----------
    dice_scores : dict or array-like
        One of:

        - dict with ``"fixed_dice"`` and ``"moving_dice"`` (arrays), optional ``"sym_dice"``
          (default their mean) and optional ``"per_region"`` ({region: float or list});
        - dict {region: float or list of floats}: panel A shows the per-region means,
          panel B the regions;
        - array-like of Dice values: panel A shows them, panel B is a histogram.
    labels_dict : dict, optional
        Region key -> display name (missing keys shown as "Region <key>"). If None the key is
        shown as is.
    title : str, default "Mindboggle Cortical DKT31 Label Overlap Benchmark"
        Figure title.
    theme : str, default "dark"
        "dark", otherwise light colours.
    output_path : str, optional
        Save the figure here (parent directories are created).
    dpi : int, default 150
        Figure and saved-image resolution.
    show_figure : bool, default False
        Call ``plt.show()``; otherwise the figure is closed before being returned.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If there are no finite Dice values.
    """
    is_dark = (theme.lower() == "dark")
    bg_color = "#090d16" if is_dark else "#ffffff"
    card_bg = "#161b22" if is_dark else "#f8fafc"
    text_color = "#f8fafc" if is_dark else "#0f172a"
    sub_color = "#94a3b8" if is_dark else "#475569"
    fixed_color = "#38bdf8" if is_dark else "#0284c7"
    moving_color = "#fb923c" if is_dark else "#ea580c"
    sym_color = "#3fb950" if is_dark else "#16a34a"
    grid_color = "#21262d" if is_dark else "#e2e8f0"

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=dpi, facecolor=bg_color)
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.95, top=0.88, bottom=0.12)

    for ax in axes:
        ax.set_facecolor(card_bg)
        ax.grid(True, linestyle='--', alpha=0.4, color=grid_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(sub_color)
        ax.spines['bottom'].set_color(sub_color)
        ax.tick_params(colors=text_color)

    def _finite(a):
        a = np.asarray(a, dtype=float).ravel()
        return a[np.isfinite(a)]

    def _region_mean(v):
        return float(np.mean(v)) if np.ndim(v) else float(v)

    # Process input dice data
    if isinstance(dice_scores, dict) and "fixed_dice" in dice_scores and "moving_dice" in dice_scores:
        f_dice = _finite(dice_scores["fixed_dice"])
        m_dice = _finite(dice_scores["moving_dice"])
        s_dice = _finite(dice_scores["sym_dice"] if "sym_dice" in dice_scores else
                         (np.asarray(dice_scores["fixed_dice"], float) + np.asarray(dice_scores["moving_dice"], float)) / 2.0)
        region_dict = dice_scores.get("per_region", {}) or {}
        boxes = [f_dice, m_dice, s_dice]
        box_labels = ["Fixed Space\n(Moving → Fixed)", "Moving Space\n(Fixed → Moving)", "Symmetric Mean\n(Dice Sym)"]
        colors = [fixed_color, moving_color, sym_color]
        panel_a = "Panel A: Symmetric Space Evaluation"
    else:
        if isinstance(dice_scores, dict):
            region_dict = dice_scores
            s_dice = _finite([_region_mean(v) for v in dice_scores.values()])
        else:
            region_dict = {}
            s_dice = _finite(dice_scores)
        boxes, box_labels, colors = [s_dice], ["Dice"], [sym_color]
        panel_a = "Panel A: Dice Distribution"
    if s_dice.size == 0:
        raise ValueError("plot_label_overlap_stats: no finite Dice values")

    # Panel A: Dice distributions
    bplot = axes[0].boxplot(
        boxes,
        tick_labels=box_labels,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color='#ffffff', linewidth=2.0)
    )

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor(text_color)

    mean_sym = float(np.mean(s_dice))
    median_sym = float(np.median(s_dice))
    iqr_sym = float(np.percentile(s_dice, 75) - np.percentile(s_dice, 25))

    axes[0].set_ylabel("Sørensen-Dice Score (MeanOverlap)", color=text_color, fontsize=11, fontweight='bold')
    axes[0].set_title(f"{panel_a}\nMean: {mean_sym:.4f} | Median: {median_sym:.4f} | IQR: {iqr_sym:.4f}",
                      color=text_color, fontsize=12, fontweight='bold', pad=10)
    axes[0].set_ylim([max(0.0, float(np.min(s_dice)) - 0.08), min(1.0, float(np.max(s_dice)) + 0.05)])

    # Panel B: Per-Region DKT Cortical Label Dice Bar Chart
    if region_dict:
        sorted_items = sorted(region_dict.items(), key=lambda x: _region_mean(x[1]))
        if len(sorted_items) > 15:
            sorted_items = sorted_items[-15:]

        from .colormaps import get_dkt_label_color_dict
        raw_lids = [k for k, _ in sorted_items]
        color_dict = get_dkt_label_color_dict(raw_lids)

        reg_names = []
        reg_means = []
        bar_colors = []
        for k, v in sorted_items:
            name = labels_dict.get(k, f"Region {k}") if labels_dict else str(k)
            reg_names.append(name)
            val = _region_mean(v)
            reg_means.append(val)

            c = color_dict.get(k, color_dict.get(str(k), None))
            if c is None:
                try:
                    lid_int = int(str(k).replace("DKT", "").strip())
                    c = color_dict.get(lid_int, color_dict.get(str(lid_int), sym_color))
                except Exception:
                    c = sym_color
            bar_colors.append(c)

        y_pos = np.arange(len(reg_names))
        bars = axes[1].barh(y_pos, reg_means, height=0.6, color=bar_colors, alpha=0.85, edgecolor=text_color)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(reg_names, fontsize=9.5, color=text_color)
        axes[1].set_xlabel("Mean Dice Score", color=text_color, fontsize=11, fontweight='bold')
        axes[1].set_title("Panel B: Per-Region DKT Cortical Label Overlap", color=text_color, fontsize=12, fontweight='bold', pad=10)
        axes[1].set_xlim([0.0, 1.0])

        for bar, val in zip(bars, reg_means):
            axes[1].text(val + 0.015, bar.get_y() + bar.get_height() / 2.0, f"{val:.3f}",
                         va='center', ha='left', color=text_color, fontsize=9, fontweight='bold')
    else:
        n, bins, patches = axes[1].hist(s_dice, bins=12, color=sym_color, alpha=0.8, edgecolor=text_color)
        axes[1].set_xlabel("Dice Score Bins", color=text_color, fontsize=11, fontweight='bold')
        axes[1].set_ylabel("Frequency / Count", color=text_color, fontsize=11, fontweight='bold')
        axes[1].set_title("Panel B: Label Overlap Distribution", color=text_color, fontsize=12, fontweight='bold', pad=10)

    fig.suptitle(title, fontsize=15, fontweight='bold', color=text_color, y=0.97)

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=bg_color)

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_jacobian_distribution(
    detJ,
    title="Jacobian Determinant det(J) Distribution & Singularities",
    theme: str = "dark",
    output_path=None,
    dpi=150,
    show_figure=False,
    mask=None,
):
    """Histogram (60 bins, density) of Jacobian-determinant values, with fold statistics.

    The values inside ``mask`` (all values if None) that are finite are used; the number of
    non-finite values is reported. Bins whose left edge is <= 0 are red; dashed / solid lines
    mark det(J) = 1 and 0. The title adds min, mean, 5th / 95th percentiles and the
    percentage of values <= 0 ("no det(J) <= 0" when there is none -- a statement about these
    sampled values, not a proof that the map is diffeomorphic).

    Parameters
    ----------
    detJ : ANTsImage, np.ndarray, list or torch.Tensor
        Determinant values (any shape; flattened). Tensors on any device are detached first.
    title : str, default "Jacobian Determinant det(J) Distribution & Singularities"
        First title line.
    theme : str, default "dark"
        "dark", otherwise light colours.
    output_path : str, optional
        Save the figure here (parent directories are created).
    dpi : int, default 150
        Figure and saved-image resolution.
    show_figure : bool, default False
        Call ``plt.show()``; otherwise the figure is closed before being returned.
    mask : ANTsImage, np.ndarray or torch.Tensor, optional
        Region of interest (nonzero = inside), same number of values as ``detJ``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``mask`` does not match ``detJ`` or no finite value is left.
    """
    def _np(x):
        if hasattr(x, 'detach'):
            return x.detach().cpu().numpy()
        if hasattr(x, 'numpy'):
            return x.numpy()
        return np.asarray(x)

    arr = np.asarray(_np(detJ), dtype=float).ravel()
    if mask is not None:
        m = np.asarray(_np(mask)).ravel() != 0
        if m.size != arr.size:
            raise ValueError(f"plot_jacobian_distribution: mask has {m.size} values, detJ {arr.size}")
        arr = arr[m]
    finite = np.isfinite(arr)
    n_nonfinite = int((~finite).sum())
    arr = arr[finite]
    if arr.size == 0:
        raise ValueError("plot_jacobian_distribution: no finite det(J) values")

    arr_flat = arr.ravel()

    is_dark = (theme.lower() == "dark")
    bg_color = "#090d16" if is_dark else "#ffffff"
    card_bg = "#161b22" if is_dark else "#f8fafc"
    text_color = "#f8fafc" if is_dark else "#0f172a"
    sub_color = "#94a3b8" if is_dark else "#475569"
    grid_color = "#21262d" if is_dark else "#e2e8f0"

    min_j = float(np.min(arr_flat))
    max_j = float(np.max(arr_flat))
    mean_j = float(np.mean(arr_flat))
    folding_pct = float(np.mean(arr_flat <= 0.0) * 100.0)
    p05 = float(np.percentile(arr_flat, 5))
    p95 = float(np.percentile(arr_flat, 95))

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=dpi, facecolor=bg_color)
    ax.set_facecolor(card_bg)
    ax.grid(True, linestyle='--', alpha=0.4, color=grid_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(sub_color)
    ax.spines['bottom'].set_color(sub_color)
    ax.tick_params(colors=text_color)

    counts, bins, patches = ax.hist(arr_flat, bins=60, density=True, alpha=0.75, edgecolor='none')

    for bin_left, patch in zip(bins[:-1], patches):
        if bin_left <= 0.0:
            patch.set_facecolor('#f85149')
        else:
            patch.set_facecolor('#38bdf8')

    ax.axvline(1.0, color='#3fb950', linestyle='--', linewidth=1.8, label='Identity det(J)=1.0')
    ax.axvline(0.0, color='#f85149', linestyle='-', linewidth=2.0, label='Singularity Limit det(J)=0.0')

    status_str = "0.00% Folding (no det(J) <= 0)" if folding_pct == 0.0 else f"{folding_pct:.3f}% Grid Folding"
    if n_nonfinite:
        status_str += f" | {n_nonfinite} non-finite values excluded"
    status_color = "#3fb950" if folding_pct == 0.0 else "#f85149"

    ax.set_xlabel("Jacobian Determinant det(J)", color=text_color, fontsize=11, fontweight='bold')
    ax.set_ylabel("Probability Density", color=text_color, fontsize=11, fontweight='bold')
    ax.set_title(f"{title}\nMin: {min_j:+.3f} | Mean: {mean_j:.3f} | p5: {p05:.2f} | p95: {p95:.2f}\nStatus: {status_str}",
                 color=status_color if folding_pct > 0 else text_color, fontsize=12, fontweight='bold', pad=10)

    ax.legend(facecolor=card_bg, edgecolor=sub_color, labelcolor=text_color, loc='upper right')

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=bg_color)

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig

def plot_loss_convergence(
    losses,
    output_path=None,
    title="Similarity Loss Convergence",
    theme: str = "dark",
    dpi=150,
    show_figure=False,
    xlabel: str = "Iteration",
    ylabel: str = "Loss",
    label=None,
):
    """Plot ``losses`` against their index.

    Parameters
    ----------
    losses : sequence of float
    output_path : str, optional
        Save the figure here (parent directories are created).
    title : str, default "Similarity Loss Convergence"
    theme : str, default "dark"
        "dark", otherwise light colours.
    dpi : int, default 150
    show_figure : bool, default False
        Call ``plt.show()`` before the figure is closed.
    xlabel, ylabel : str, default "Iteration" / "Loss"
        Axis labels.
    label : str, optional
        Legend entry for the curve (e.g. the metric name); no legend if None.

    Returns
    -------
    matplotlib.figure.Figure
        Closed.
    """
    is_dark = (theme.lower() == "dark")
    bg_color = "#090d16" if is_dark else "#ffffff"
    card_bg = "#161b22" if is_dark else "#f8fafc"
    text_color = "#f8fafc" if is_dark else "#0f172a"
    sub_color = "#94a3b8" if is_dark else "#475569"
    grid_color = "#21262d" if is_dark else "#e2e8f0"
    line_color = "#38bdf8" if is_dark else "#0284c7"

    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi, facecolor=bg_color)
    ax.set_facecolor(card_bg)
    ax.grid(True, linestyle='--', alpha=0.4, color=grid_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(grid_color)
    ax.tick_params(colors=sub_color)

    ax.plot(losses, color=line_color, linewidth=2, label=label)
    ax.set_title(title, color=text_color, pad=10, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, color=sub_color, fontweight='bold')
    ax.set_ylabel(ylabel, color=sub_color, fontweight='bold')

    if label is not None:
        legend = ax.legend(facecolor=card_bg, edgecolor=grid_color)
        for text in legend.get_texts():
            text.set_color(sub_color)

    fig.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor=bg_color, bbox_inches='tight')
    if show_figure:
        plt.show()
    plt.close(fig)
    return fig
