"""
Statistical Visualization Tools for Syntx Anatomical Labeling and Registration Quality.

Provides publication-grade statistical distribution plots:
- plot_label_overlap_stats: Mindboggle DKT cortical label overlap distributions (Dice box/violin plots & per-region bar charts).
- plot_jacobian_distribution: Jacobian determinant det(J) histograms & diffeomorphic fold stats.
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
    """
    Renders standard 2-panel statistical summary figure for anatomical segmentations (Mindboggle DKT).
    
    Panel A: Symmetric Dice Score Distributions (Fixed Space, Moving Space, Symmetric Mean).
    Panel B: Per-Region DKT Cortical Label Dice Bar Chart (sorted by mean performance).
    
    Args:
        dice_scores: dict mapping label_id/name -> float (or list/array of subject Dice scores),
                     or dict with keys {'fixed_dice': [...], 'moving_dice': [...], 'sym_dice': [...]}.
        labels_dict: dict mapping label_id -> label_name string (optional).
        title: Figure title.
        theme: 'dark' (default) or 'light'.
        output_path: Optional path to save PNG asset.
        dpi: Output figure DPI resolution (default: 150).
        show_figure: If True, calls plt.show() (default: False).
        
    Returns:
        matplotlib.figure.Figure: Generated Figure object.
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

    # Process input dice data
    if isinstance(dice_scores, dict) and "fixed_dice" in dice_scores and "moving_dice" in dice_scores:
        f_dice = np.asarray(dice_scores["fixed_dice"])
        m_dice = np.asarray(dice_scores["moving_dice"])
        s_dice = np.asarray(dice_scores.get("sym_dice", (f_dice + m_dice) / 2.0))
        region_dict = dice_scores.get("per_region", {})
    elif isinstance(dice_scores, dict):
        f_dice = np.fromiter(dice_scores.values(), dtype=float)
        m_dice = f_dice
        s_dice = f_dice
        region_dict = dice_scores
    else:
        s_dice = np.asarray(dice_scores)
        f_dice, m_dice = s_dice, s_dice
        region_dict = {}

    # Panel A: Symmetric Dice Distributions
    bplot = axes[0].boxplot(
        [f_dice, m_dice, s_dice],
        tick_labels=["Fixed Space\n(Moving → Fixed)", "Moving Space\n(Fixed → Moving)", "Symmetric Mean\n(Dice Sym)"],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color='#ffffff', linewidth=2.0)
    )

    colors = [fixed_color, moving_color, sym_color]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor(text_color)

    mean_sym = float(np.mean(s_dice))
    median_sym = float(np.median(s_dice))
    iqr_sym = float(np.percentile(s_dice, 75) - np.percentile(s_dice, 25))

    axes[0].set_ylabel("Dice Overlap Score (TargetOverlap)", color=text_color, fontsize=11, fontweight='bold')
    axes[0].set_title(f"Panel A: Symmetric Space Evaluation\nMean: {mean_sym:.4f} | Median: {median_sym:.4f} | IQR: {iqr_sym:.4f}",
                      color=text_color, fontsize=12, fontweight='bold', pad=10)
    axes[0].set_ylim([max(0.0, float(np.min(s_dice)) - 0.08), min(1.0, float(np.max(s_dice)) + 0.05)])

    # Panel B: Per-Region DKT Cortical Label Dice Bar Chart
    if region_dict:
        sorted_items = sorted(region_dict.items(), key=lambda x: np.mean(x[1]) if hasattr(x[1], '__iter__') else x[1])
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
            val = float(np.mean(v)) if hasattr(v, '__iter__') else float(v)
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
    show_figure=False
):
    """
    Renders publication-grade Jacobian determinant det(J) distribution histogram & fold statistics.
    
    Args:
        detJ: 2D/3D array, ANTsImage, or list of det(J) values.
        title: Figure title.
        theme: 'dark' (default) or 'light'.
        output_path: Optional output path.
        dpi: Output resolution.
        show_figure: If True, calls plt.show().
        
    Returns:
        matplotlib.figure.Figure: Generated Figure object.
    """
    if hasattr(detJ, 'numpy'):
        arr = detJ.numpy()
    elif hasattr(detJ, 'detach'):
        arr = detJ.detach().cpu().numpy()
    else:
        arr = np.asarray(detJ)

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

    status_str = "0.00% Folding (Fully Diffeomorphic)" if folding_pct == 0.0 else f"{folding_pct:.3f}% Grid Folding"
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
    show_figure=False
):
    """
    Renders a standard convergence curve for similarity loss.
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

    ax.plot(losses, color=line_color, linewidth=2, label="LNCC Loss")
    ax.set_title(title, color=text_color, pad=10, fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch", color=sub_color, fontweight='bold')
    ax.set_ylabel("Loss", color=sub_color, fontweight='bold')
    
    legend = ax.legend(facecolor=card_bg, edgecolor=grid_color)
    for text in legend.get_texts():
        text.set_color(sub_color)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi, facecolor=bg_color, bbox_inches='tight')
    if show_figure:
        plt.show()
    plt.close(fig)
    return fig
