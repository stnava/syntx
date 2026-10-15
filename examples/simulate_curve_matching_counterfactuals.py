"""Simulated counterfactual evaluation for N-dimensional continuous elastic curve matching.

Scenarios (each reports ground-truth landmark error, the primary accuracy metric):
 1. Displaced-feature recovery, default parameters (3-D open)
 2. Large SO(3) rotation with tangent covariance (3-D open, rigid)
 3. Traversal inversion (3-D open)
 4. Sub-sample cyclic phase (2-D closed)
 5. Opposite-bending tangent discrimination (2-D open)
 6. 4-D trajectory with measurement noise

Sweeps compare "legacy-like" settings (distance scale NOT floored, ramp allowed as a candidate,
wide smoothing grid; the closest reproduction of the pre-repair behaviour reachable through
public arguments) against the repaired defaults, on identical inputs:
 A. measurement noise (without and with arc-length pre-smoothing)
 B. rotation angle
 C. sub-sample cyclic phase
 D. input resolution / n_points
 E. dimension d in {2, 3, 4, 6}

Output: reports/curve_matching_counterfactual_evaluation.html (light theme, self-contained).
Execution: CPU, NumPy/SciPy, single process, fixed seeds.
"""

import platform
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from syntx.curves import (
    correspondence_error,
    curve_arc_fraction,
    fig_to_base64,
    generate_curve_matching_html_report,
    match_curves,
    resample_curve,
)

warnings.simplefilter("ignore")

LEGACY = dict(distance_scale_rel=0.0, smooth_sigma_grid=(0.0, 1.0, 2.0, 4.0, 8.0, None))
SLATE, CYAN, EMERALD, AMBER = "#1e293b", "#0891b2", "#059669", "#d97706"


def bump_pair(n=400, d=3, c_p=(0.3, 0.7), c_q=(0.4, 0.6)):
    """Displaced Gaussian bumps in R^d with ground-truth landmark arc fractions."""
    t = np.linspace(0, 1, n)

    def make(c1, c2):
        y = 10.0 * np.exp(-((t - c1) / 0.05) ** 2) + 10.0 * np.exp(-((t - c2) / 0.05) ** 2)
        cols = [100.0 * t, y] + [3.0 * np.sin((k + 1) * np.pi * t) for k in range(max(d - 2, 0))]
        return np.column_stack(cols[:d])

    P, Q = make(*c_p), make(*c_q)
    afP, afQ = curve_arc_fraction(P), curve_arc_fraction(Q)
    gp = np.array([afP[np.argmin(abs(t - c))] for c in c_p])
    gq = np.array([afQ[np.argmin(abs(t - c))] for c in c_q])
    return P, Q, gp, gq


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=SLATE, fontsize=11)
    ax.set_xlabel(xlabel, color=SLATE)
    ax.set_ylabel(ylabel, color=SLATE)
    ax.grid(alpha=0.25)
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_color(SLATE)
    ax.tick_params(colors=SLATE)


def _sweep_fig(xs, series, title, xlabel, ylabel="max landmark error (arc fraction)", logx=False, hline=None):
    fig, ax = plt.subplots(figsize=(5.4, 3.6), facecolor="white")
    for (label, ys, color, style) in series:
        ax.plot(xs, ys, style, color=color, lw=2, ms=5, label=label)
    if hline is not None:
        ax.axhline(hline, color=SLATE, lw=0.8, ls=":")
    if logx:
        ax.set_xscale("log")
    _style(ax, title, xlabel, ylabel)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def _card(title, note, fig):
    img = fig_to_base64(fig, dpi=120)
    plt.close(fig)
    return (f'<div class="figure-container"><div class="sweep-note"><strong>{title}</strong> &mdash; {note}</div>'
            f'<img class="report-fig" src="{img}" alt="{title}"/></div>')


def run():
    t_start = time.time()
    print("=" * 70)
    print("  syntx.curves: counterfactual simulation & evaluation (R^d)")
    print("=" * 70)
    matches = []

    # 1. displaced features, default parameters
    P1, Q1, gp, gq = bump_pair(d=3)
    m1 = match_curves(P1, Q1, n_points=80)
    e1 = correspondence_error(m1, gp, gq)
    matches.append(("1. Displaced-feature recovery, default parameters (3-D)", m1, e1))
    ramp_err = float(np.max(np.abs(gp - gq)))
    print(f"[1] GT err {e1['max_error_fraction']:.4f} (uniform ramp would give {ramp_err:.4f})")

    # 2. large rotation
    t = np.linspace(0, 1, 90)
    P2 = np.column_stack([75 * t, 18 * np.sin(np.pi * t) + 4 * np.sin(3 * np.pi * t), 8 * np.cos(2 * np.pi * t)])
    th = np.deg2rad(45.0)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    m2 = match_curves(P2, P2 @ R.T + [12.0, -8.0, 5.0], n_points=60, rigid=True)
    matches.append(("2. Large SO(3) rotation (45 deg), rigid", m2))
    print(f"[2] tangent energy {m2.energies['tangent']:.2e}, rms {m2.elastic_rms_mm:.2e} mm")

    # 3. traversal inversion
    m3 = match_curves(P2, P2[::-1].copy(), n_points=60)
    matches.append(("3. Traversal inversion (3-D)", m3))
    print(f"[3] reversed={m3.reversed}, rms {m3.elastic_rms_mm:.2e} mm")

    # 4. sub-sample phase on a closed loop
    tt = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    rr = 20 * (1 + 0.25 * np.cos(3 * tt) + 0.12 * np.sin(5 * tt))
    loop = np.column_stack([rr * np.cos(tt), rr * np.sin(tt)])
    base = resample_curve(loop, 60, mode="closed")
    fine = resample_curve(loop, 6000, mode="closed")
    Q4 = resample_curve(np.roll(fine, -35, axis=0), 60, mode="closed")  # 0.35 of a sample
    m4 = match_curves(base, Q4, mode="closed", n_points=60)
    matches.append(("4. Sub-sample cyclic phase, closed (2-D)", m4))
    print(f"[4] rms {m4.elastic_rms_mm:.3f} mm vs spacing {m4.spacing_mm:.2f} mm")

    # 5. opposite bending
    t5 = np.linspace(0, 1, 80)
    P5 = np.column_stack([50 * t5, 14 * np.sin(np.pi * t5)])
    Q5 = np.column_stack([50 * t5, -14 * np.sin(np.pi * t5)])
    m5 = match_curves(P5, Q5, n_points=60, tangent_weight=2.0)
    matches.append(("5. Opposite-bending tangent discrimination (2-D)", m5))
    print(f"[5] tangent energy {m5.energies['tangent']:.3f}")

    # 6. 4-D with noise
    P6, Q6, gp6, gq6 = bump_pair(n=100, d=4)
    r6 = np.random.default_rng(99)
    P6n, Q6n = P6 + r6.normal(scale=0.2, size=P6.shape), Q6 + r6.normal(scale=0.2, size=Q6.shape)
    P6n[[0, -1]], Q6n[[0, -1]] = P6[[0, -1]], Q6[[0, -1]]
    m6 = match_curves(P6n, Q6n, n_points=80, bidirectional=True)
    e6 = correspondence_error(m6, gp6, gq6)
    matches.append(("6. 4-D trajectory, 0.2 mm noise, bidirectional", m6, e6))
    print(f"[6] GT err {e6['max_error_fraction']:.4f}, inv consistency {m6.inverse_consistency_rms:.4f} frac")

    # ------------------------------------------------------------------ sweeps
    cards = []

    # A. noise
    print("[A] noise sweep ...")
    P, Q, gp, gq = bump_pair(n=100, d=3)
    sig_list = [0.0, 0.1, 0.3, 0.5, 1.0]
    rows = {"legacy-like": [], "repaired": [], "repaired + presmooth 3 mm": []}
    for s in sig_list:
        acc = {k: [] for k in rows}
        for seed in range(6):
            r = np.random.default_rng(seed)
            Pn, Qn = P + r.normal(scale=s, size=P.shape), Q + r.normal(scale=s, size=Q.shape)
            Pn[[0, -1]], Qn[[0, -1]] = P[[0, -1]], Q[[0, -1]]
            for k, kw in (("legacy-like", LEGACY), ("repaired", {}), ("repaired + presmooth 3 mm", dict(presmooth_mm=3.0))):
                acc[k].append(correspondence_error(match_curves(Pn, Qn, n_points=80, **kw), gp, gq)["max_error_fraction"])
        for k in rows:
            rows[k].append(float(np.median(acc[k])))
    cards.append(_card(
        "A. Measurement noise", "median over 6 seeds; point spacing 1 mm; ramp error shown dotted. Noise inflates "
        "polyline arc length, so noise approaching the spacing needs pre-smoothing and eventually fails.",
        _sweep_fig(sig_list, [("legacy-like", rows["legacy-like"], AMBER, "o--"),
                              ("repaired", rows["repaired"], CYAN, "s-"),
                              ("repaired + presmooth 3 mm", rows["repaired + presmooth 3 mm"], EMERALD, "^-")],
                   "Noise robustness", "noise sigma (mm)", hline=float(np.max(np.abs(gp - gq))))))

    # B. rotation angle
    print("[B] rotation sweep ...")
    angs = [0, 10, 25, 40, 60, 90, 135]
    rig = {"unbounded (default)": [], "max_rotation_deg=25": []}
    for a in angs:
        th = np.deg2rad(a)
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        Qr = P2 @ Rz.T + [5.0, -3.0, 2.0]
        for k, kw in (("unbounded (default)", {}), ("max_rotation_deg=25", dict(max_rotation_deg=25.0))):
            rig[k].append(match_curves(P2, Qr, n_points=60, rigid=True, **kw).elastic_rms_mm)
    cards.append(_card(
        "B. Rotation angle", "residual RMS after rigid alignment; the bounded variant warns and flags "
        "<code>rotation_clamped</code> once the bound is exceeded.",
        _sweep_fig(angs, [("unbounded (default)", rig["unbounded (default)"], CYAN, "s-"),
                          ("max_rotation_deg=25", rig["max_rotation_deg=25"], AMBER, "o--")],
                   "Rigid rotation recovery", "true rotation (deg)", ylabel="residual RMS (mm)")))

    # C. sub-sample phase
    print("[C] phase sweep ...")
    phases = [0.0, 0.1, 0.25, 0.5, 0.75]
    ph = []
    for p in phases:
        Qp = resample_curve(np.roll(fine, -int(round(p * 100)), axis=0), 60, mode="closed")
        ph.append(match_curves(base, Qp, mode="closed", n_points=60).elastic_rms_mm / m4.spacing_mm)
    cards.append(_card(
        "C. Sub-sample cyclic phase", "closed-loop residual RMS as a fraction of sample spacing "
        "(whole-sample alignment alone would leave up to 0.5 spacing).",
        _sweep_fig(phases, [("repaired", ph, CYAN, "s-")], "Sub-sample phase recovery",
                   "phase shift (fraction of one sample)", ylabel="residual RMS / spacing", hline=0.0)))

    # D. resolution
    print("[D] resolution sweep ...")
    P, Q, gp, gq = bump_pair(n=400, d=3)
    ns = [30, 60, 120, 200]
    res = {"legacy-like": [], "repaired": []}
    for n in ns:
        for k, kw in (("legacy-like", LEGACY), ("repaired", {})):
            res[k].append(correspondence_error(match_curves(P, Q, n_points=n, **kw), gp, gq)["max_error_fraction"])
    cards.append(_card(
        "D. Resolution (n_points)", "ground-truth error should be stable as the sampling changes.",
        _sweep_fig(ns, [("legacy-like", res["legacy-like"], AMBER, "o--"), ("repaired", res["repaired"], CYAN, "s-")],
                   "Resolution consistency", "n_points", hline=float(np.max(np.abs(gp - gq))))))

    # E. dimension
    print("[E] dimension sweep ...")
    dims = [2, 3, 4, 6]
    dd = {"legacy-like": [], "repaired": []}
    for d in dims:
        P, Q, gp, gq = bump_pair(n=400, d=d)
        for k, kw in (("legacy-like", LEGACY), ("repaired", {})):
            dd[k].append(correspondence_error(match_curves(P, Q, n_points=80, **kw), gp, gq)["max_error_fraction"])
    cards.append(_card(
        "E. Dimension", "same displaced-feature truth embedded in R^d.",
        _sweep_fig(dims, [("legacy-like", dd["legacy-like"], AMBER, "o--"), ("repaired", dd["repaired"], CYAN, "s-")],
                   "Dimension invariance", "dimension d", hline=float(np.max(np.abs(gp - gq))))))

    def _tab(name, xs, d):
        print(f"\n{name}")
        print("  x: " + "  ".join(f"{x:>8}" for x in xs))
        for k, v in d.items():
            print(f"  {k:<26}" + "  ".join(f"{y:8.3f}" for y in v))

    _tab("A. noise sigma (mm) -> median max GT error (frac)", sig_list, rows)
    _tab("B. rotation (deg) -> residual RMS (mm)", angs, rig)
    _tab("C. phase (sample fraction) -> RMS/spacing", phases, {"repaired": ph})
    _tab("D. n_points -> max GT error (frac)", ns, res)
    _tab("E. dimension -> max GT error (frac)", dims, dd)

    extra = ('<div class="bundle-card"><div class="bundle-header"><h2>Counterfactual sweeps: legacy-like vs repaired</h2></div>'
             '<p class="sweep-note">Legacy-like = <code>distance_scale_rel=0</code> and a smoothing grid that includes the '
             'unwarped ramp, i.e. the closest reproduction of the pre-repair selection behaviour through public arguments '
             '(the earlier scale-selection rule and slope defaults are not toggled here). Dotted line = error of the '
             'identity map.</p><div class="sweep-grid">' + "".join(cards) + "</div></div>")

    runtime = time.time() - t_start
    info = (f"Device: CPU (NumPy/SciPy, single process) &bull; Python {platform.python_version()} on "
            f"{platform.system()} {platform.machine()} &bull; runtime {runtime:.0f} s &bull; fixed seeds (noise: 0-5, 99)")
    path = generate_curve_matching_html_report(
        Path("reports/curve_matching_counterfactual_evaluation.html"),
        matches=matches,
        title="Continuous Elastic Curve Matching Evaluation Report",
        subtitle="Ground-truth counterfactual analysis across 1-D to 6-D curves in syntx.curves",
        extra_html=extra,
        execution_info=info,
    )
    print(f"\nReport: {path}  ({runtime:.0f} s)")
    return path


if __name__ == "__main__":
    run()
