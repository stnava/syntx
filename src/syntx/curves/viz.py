"""Dimension-adaptive visualization of N-dimensional elastic curve matching.

Follows syntx light-theme aesthetics:
- Pure white background (#ffffff) and deep slate (#1e293b) typography.
- For 3-D curves: Direction-Encoded Color (DEC: Red=X, Green=Y, Blue=Z).
- For 2-D curves: High-contrast planar coordinate rendering with correspondence rays.
- For d > 3: Principal component projection into R^3.
"""

from __future__ import annotations

import base64
import io
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from syntx.curves.geometry import curve_tangents
from syntx.curves.matching import CurveMatch

# Theme palette
BG = "#ffffff"
PANEL_BG = "#f8fafc"
TEXT = "#1e293b"
MUTED = "#64748b"
BORDER = "#e2e8f0"
CYAN = "#0ea5e9"
EMERALD = "#10b981"
AMBER = "#f59e0b"
ROSE = "#f43f5e"
PURPLE = "#8b5cf6"

_STYLE = {
    "font.family": "sans-serif",
    "font.size": 9.5,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 10.5,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": BORDER,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.6,
}


def compute_dec_colors(points: np.ndarray, mode: str = "open") -> np.ndarray:
    """Compute Direction-Encoded Colors (DEC) for 3-D curve points.

    Standard DTI physical space convention:
    Red = Left-Right (X), Green = Anterior-Posterior (Y), Blue = Inferior-Superior (Z).
    """
    T = curve_tangents(points, mode=mode)
    dec = np.abs(T)
    norms = np.linalg.norm(dec, axis=1, keepdims=True)
    return np.divide(dec, np.maximum(norms, 1e-12))


def fig_to_base64(fig: matplotlib.figure.Figure, dpi: int = 130) -> str:
    """Export figure to an inline Base64 data URI string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_curve_match(
    match: CurveMatch,
    *,
    label_P: str = "Curve P",
    label_Q: str = "Curve Q",
    every: int = 4,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (18, 10.5),
    savepath: Optional[str] = None,
    dpi: int = 130,
) -> matplotlib.figure.Figure:
    """Six-panel diagnostic figure for an N-dimensional elastic curve match."""
    N = match.n_points
    d = match.dimension
    P = match.P_resampled
    Q = match.Q_resampled
    if P is None or Q is None:
        raise ValueError("Match must carry P_resampled and Q_resampled")

    Pa = match.warp_P()
    Qm = match.warp_Q()
    t1 = match.t1
    t2 = match.t2_unwrapped
    ramp = np.linspace(0.0, 1.0, N) if match.mode == "open" else match.t1 + match.offset_fraction
    dev = t2 - ramp

    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig, width_ratios=[1.25, 1, 1], hspace=0.35, wspace=0.28)

        # Panel 1: Trajectory plot in ambient space
        if d == 1:
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(t1, Pa[:, 0], color=CYAN, lw=2.2, label=label_P)
            ax1.plot(t1, Q[:, 0], color=EMERALD, lw=2.0, ls="--", label=label_Q)
            for i in range(0, N, every):
                ax1.plot([t1[i], t1[i]], [Pa[i, 0], Qm[i, 0]], color=MUTED, lw=0.8, alpha=0.6)
            ax1.set_xlabel("Arc fraction t1")
            ax1.set_ylabel("Coordinate (mm)")
            ax1.set_title(
                f"1-D Trajectory in Physical Space\n"
                f"Baseline: {match.baseline_rms_mm:.2f} mm -> Elastic: {match.elastic_rms_mm:.2f} mm "
                f"(Gain: {match.elastic_gain_mm:+.2f} mm)"
            )
            ax1.legend(loc="best", frameon=False, fontsize=8)

        elif d == 2:
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(Pa[:, 0], Pa[:, 1], color=CYAN, lw=2.2, label=label_P)
            ax1.plot(Q[:, 0], Q[:, 1], color=EMERALD, lw=2.0, ls="--", label=label_Q)
            for i in range(0, N, every):
                ax1.plot([Pa[i, 0], Qm[i, 0]], [Pa[i, 1], Qm[i, 1]], color=MUTED, lw=0.8, alpha=0.6)
            ax1.scatter(Pa[0, 0], Pa[0, 1], color=ROSE, s=40, zorder=5, label="Start (s=0)")
            ax1.scatter(Pa[-1, 0], Pa[-1, 1], color=PURPLE, s=40, zorder=5, label="End (s=1)")
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")
            ax1.set_title(
                f"2-D Trajectories in Physical Space\n"
                f"Baseline: {match.baseline_rms_mm:.2f} mm -> Elastic: {match.elastic_rms_mm:.2f} mm "
                f"(Gain: {match.elastic_gain_mm:+.2f} mm)"
            )
            ax1.legend(loc="best", frameon=False, fontsize=8)
            ax1.set_aspect("equal", adjustable="datalim")

        elif d == 3:
            ax1 = fig.add_subplot(gs[0, 0], projection="3d")
            dec_P = compute_dec_colors(Pa, mode=match.mode)
            dec_Q = compute_dec_colors(Q, mode=match.mode)
            for i in range(len(Pa) - 1):
                ax1.plot(Pa[i : i + 2, 0], Pa[i : i + 2, 1], Pa[i : i + 2, 2], color=dec_P[i], lw=2.2)
            for i in range(len(Q) - 1):
                ax1.plot(Q[i : i + 2, 0], Q[i : i + 2, 1], Q[i : i + 2, 2], color=dec_Q[i], lw=1.8, ls="--")
            for i in range(0, N, every):
                ax1.plot(
                    [Pa[i, 0], Qm[i, 0]],
                    [Pa[i, 1], Qm[i, 1]],
                    [Pa[i, 2], Qm[i, 2]],
                    color=MUTED,
                    lw=0.7,
                    alpha=0.6,
                )
            ax1.scatter(*Pa[0], color=ROSE, s=40, label="Start (s=0)")
            ax1.scatter(*Pa[-1], color=PURPLE, s=40, label="End (s=1)")
            ax1.set_xlabel("X / L (mm)")
            ax1.set_ylabel("Y / P (mm)")
            ax1.set_zlabel("Z / S (mm)")
            ax1.set_title(
                f"3-D Trajectories (DEC: R=X, G=Y, B=Z)\n"
                f"Baseline: {match.baseline_rms_mm:.2f} mm -> Elastic: {match.elastic_rms_mm:.2f} mm "
                f"(Gain: {match.elastic_gain_mm:+.2f} mm)"
            )
            ax1.xaxis.pane.fill = False
            ax1.yaxis.pane.fill = False
            ax1.zaxis.pane.fill = False
            ax1.grid(False)
            stacked_3d = np.vstack([Pa, Q])
            ax1.set_box_aspect(np.maximum(np.ptp(stacked_3d, axis=0), 1e-3))

        else:
            # d > 3: Project to first 3 principal components
            ax1 = fig.add_subplot(gs[0, 0], projection="3d")
            stacked = np.vstack([Pa, Q])
            stacked_c = stacked - stacked.mean(axis=0)
            _, _, vt = np.linalg.svd(stacked_c, full_matrices=False)
            proj_Pa = Pa @ vt[:3].T
            proj_Q = Q @ vt[:3].T
            proj_Qm = Qm @ vt[:3].T
            ax1.plot(proj_Pa[:, 0], proj_Pa[:, 1], proj_Pa[:, 2], color=CYAN, lw=2.2, label=f"{label_P} (PCA)")
            ax1.plot(proj_Q[:, 0], proj_Q[:, 1], proj_Q[:, 2], color=EMERALD, lw=2.0, ls="--", label=f"{label_Q} (PCA)")
            for i in range(0, N, every):
                ax1.plot(
                    [proj_Pa[i, 0], proj_Qm[i, 0]],
                    [proj_Pa[i, 1], proj_Qm[i, 1]],
                    [proj_Pa[i, 2], proj_Qm[i, 2]],
                    color=MUTED,
                    lw=0.7,
                    alpha=0.6,
                )
            ax1.set_xlabel("PC1 (mm)")
            ax1.set_ylabel("PC2 (mm)")
            ax1.set_zlabel("PC3 (mm)")
            ax1.set_title(
                f"{d}-D Trajectories Projected to PCA R^3\n"
                f"Baseline: {match.baseline_rms_mm:.2f} -> Elastic: {match.elastic_rms_mm:.2f} "
                f"(Gain: {match.elastic_gain_mm:+.2f})"
            )
            ax1.set_box_aspect(np.maximum(np.ptp(np.vstack([proj_Pa, proj_Q]), axis=0), 1e-3))
            ax1.legend(loc="best", frameon=False, fontsize=8)
            ax1.xaxis.pane.fill = False
            ax1.yaxis.pane.fill = False
            ax1.zaxis.pane.fill = False
            ax1.grid(False)

        # Panel 2: Parameter Square
        ax2 = fig.add_subplot(gs[0, 1])
        if match.cost_field is not None:
            C = match.cost_field
            ax2.imshow(
                C.T,
                origin="lower",
                cmap="Blues",
                extent=[0, 1, 0, 1],
                aspect="equal",
                vmin=np.percentile(C, 2),
                vmax=np.percentile(C, 98),
                interpolation="bilinear",
            )
        ax2.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls="--", label="Linear ramp")
        ax2.plot(
            match.path[:, 0] / max(match.path[-1, 0], 1),
            match.path[:, 1] / max(match.path[-1, 1], 1),
            color=AMBER,
            lw=2.0,
            label="DP Path g",
        )
        ax2.set_xlabel(f"t1 ({label_P} arc fraction)")
        ax2.set_ylabel(f"t2 ({label_Q} arc fraction{' [flipped]' if match.reversed else ''})")
        ax2.set_title("Parameter Grid: Cost Field & Elastic Path")
        ax2.legend(loc="upper left", frameon=False, fontsize=8)
        ax2.grid(False)

        # Panel 3: Reparameterization strain
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(t1, dev, color=AMBER, lw=2.0, label="Elastic Deviation (t2 - ramp)")
        ax3.axhline(0, color=MUTED, lw=0.8, ls="--")
        ax3.set_xlabel(f"t1 ({label_P} arc fraction)")
        ax3.set_ylabel("Reparameterization Strain")
        ax3.set_title(
            f"Local Speed Strain (Max stretch: {match.energies.get('max_speed_stretch', 1.0):.2f}x)\n"
            f"Smoothness sigma: {match.smooth_sigma:g} samples"
        )
        ax3.legend(loc="best", frameon=False, fontsize=8)

        # Panel 4: Local slope
        ax4 = fig.add_subplot(gs[1, 0])
        dt1 = 1.0 / max(N - (0 if match.mode == "closed" else 1), 1)
        if match.mode == "closed":
            dt2_arr = np.diff(np.concatenate([t2, [t2[0] + 1.0]]))
        else:
            dt2_arr = np.diff(t2)
        slope = dt2_arr / dt1
        x_slope = t1 if match.mode == "closed" else t1[:-1] + 0.5 * dt1
        ax4.plot(x_slope, slope, color=CYAN, lw=1.8, label="Local Speed dt2/dt1")
        ax4.axhline(1.0, color=MUTED, lw=0.8, ls="--", label="Uniform (1.0)")
        ax4.set_xlabel(f"t1 ({label_P} arc fraction)")
        ax4.set_ylabel("Speed Ratio (dt2 / dt1)")
        ax4.set_title("Local Arc-Speed Distortion along Curve")
        ax4.legend(loc="best", frameon=False, fontsize=8)

        # Panel 5: Scale-Space Curvature Striping
        ax5 = fig.add_subplot(gs[1, 1])
        if match.features_P is not None and match.features_Q_oriented is not None:
            FP = match.features_P
            FQ = match.features_Q_oriented
            S = FP.shape[0]
            if match.mode == "closed":
                idx_q = np.mod(t2, 1.0) * N
                i0 = np.floor(idx_q).astype(int) % N
                i1 = (i0 + 1) % N
                w = (idx_q - np.floor(idx_q))
            else:
                idx_q = np.clip(t2, 0.0, 1.0) * (N - 1)
                i0 = np.floor(idx_q).astype(int)
                i1 = np.clip(i0 + 1, 0, N - 1)
                w = (idx_q - i0)
            FQw = FQ[:, i0] * (1.0 - w) + FQ[:, i1] * w
            img = np.vstack([FP, np.full((1, N), np.nan), FQw])
            vmax = np.percentile(np.concatenate([FP.ravel(), FQw.ravel()]), 98)
            ax5.imshow(
                img, aspect="auto", cmap="Blues", vmin=0, vmax=vmax,
                extent=[0, 1, 2 * S + 1, 0], interpolation="nearest"
            )
            scales = match.scales_mm if match.scales_mm is not None else np.arange(S)
            ax5.set_yticks(list(np.arange(S) + 0.5) + list(np.arange(S) + S + 1.5))
            ax5.set_yticklabels([f"P sigma {s:g}" for s in scales] + [f"Q sigma {s:g}" for s in scales], fontsize=7)
            ax5.set_xlabel("t1 (Q features warped by match)")
            ax5.set_title("Scale-Space Curvature Profiles (stripes align when matched)")
            ax5.grid(False)
        else:
            ax5.axis("off")

        # Panel 6: Energy breakdown
        ax6 = fig.add_subplot(gs[1, 2])
        e = match.energies
        b = match.energies_baseline
        names = ["feature", "tangent", "distance", "membrane", "bending"]
        x_bar = np.arange(len(names))
        bw = 0.35
        ax6.bar(x_bar - bw / 2, [b.get(k, 0.0) for k in names], bw, color=BORDER, edgecolor=MUTED, label="Baseline Uniform")
        ax6.bar(x_bar + bw / 2, [e.get(k, 0.0) for k in names], bw, color=AMBER, label="Elastic Match")
        for idx_b, k in enumerate(names):
            ax6.text(idx_b - bw / 2, b.get(k, 0.0), f"{b.get(k, 0.0):.1f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
            ax6.text(idx_b + bw / 2, e.get(k, 0.0), f"{e.get(k, 0.0):.1f}", ha="center", va="bottom", fontsize=7.5, color=TEXT)
        ax6.set_xticks(x_bar)
        ax6.set_xticklabels(names)
        ax6.set_ylabel("Energy (dimensionless)")
        ax6.set_title("Energy Terms: Baseline vs Elastic Match")
        ax6.legend(loc="upper right", frameon=False, fontsize=8)
        ax6.grid(axis="x", visible=False)

        if title:
            fig.suptitle(title, fontsize=13, fontweight="bold", color=TEXT, y=0.995)
        if savepath:
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight", facecolor=BG)

    return fig
