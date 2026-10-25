"""Benchmark: removing encoding-locked systematic noise from motion estimates (ASL, slow flow, DWI, BOLD).

Ground-truth simulations from ``syntx.motion_simulation``; every strategy is scored by its head-motion error against the
true trajectory. ``floor`` rows run the same motion without the systematic noise (the registration's own precision).

    python benchmarks/motion_encoding_bias_simulation.py --out <dir> [--seeds 3] [--modalities asl slowflow dwi bold]

Writes ``results.csv``, ``results.json`` and a self-contained ``report.html`` to ``--out``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from syntx import (  # noqa: E402
    build_bilateral_window_references,
    encoding_reference_predictor,
    motion_correct_grouped,
    motion_correction,
    shell_groups,
)
from syntx.motion_simulation import (  # noqa: E402
    evaluate_motion_estimate,
    simulate_asl_series,
    simulate_bold_series,
    simulate_dwi_series,
    simulate_slowflow_series,
)


def _frames(img):
    import ants

    return [ants.slice_image(img, axis=3, idx=t) for t in range(img.shape[3])]


def _row(modality, scenario, strategy, seed, sim, res, seconds, extra=None):
    ev = evaluate_motion_estimate(res["fwdtransforms"], sim)
    row = {"modality": modality, "scenario": scenario, "strategy": strategy, "seed": seed, "seconds": round(seconds, 1)}
    row.update({k: v for k, v in ev.items() if k not in ("estimated_pose", "jump_recovery")})
    row["jump_recovery_min"] = min(ev["jump_recovery"]) if ev["jump_recovery"] else float("nan")
    if extra:
        row.update(extra)
    row["_est"] = ev["estimated_pose"]
    row["_true"] = sim.head_pose
    return row


def _timed(fn):
    t0 = time.time()
    out = fn()
    return out, time.time() - t0


def run_asl(seed):
    rows = []
    for scen, perf in (("floor", 0.0), ("perfusion_10pct", 0.10)):
        sim = simulate_asl_series(24, seed=seed, perfusion=perf)
        s = sim.encoding_covariates
        res, sec = _timed(lambda: motion_correction(sim.image, preset="bold"))
        rows.append(_row("asl", scen, "mean_ref", seed, sim, res, sec))
        if perf == 0.0:
            continue
        res, sec = _timed(lambda: motion_correction(sim.image, preset="asl", encoding_covariates=s))
        rows.append(_row("asl", scen, "mean_ref+encoding_projection", seed, sim, res, sec))
        refs = build_bilateral_window_references(_frames(sim.image), encoding_covariates=s)
        res, sec = _timed(lambda: motion_correction(sim.image, preset="bold", reference=refs))
        rows.append(_row("asl", scen, "state_matched_ref", seed, sim, res, sec))
        res, sec = _timed(lambda: motion_correction(sim.image, preset="asl", reference=refs, encoding_covariates=s))
        rows.append(_row("asl", scen, "state_matched_ref+encoding_projection", seed, sim, res, sec))
        res, sec = _timed(lambda: motion_correction(sim.image, preset="asl", encoding_covariates=s, window_size=len(s)))
        rows.append(_row("asl", scen, "mean_ref+encoding_projection(global window)", seed, sim, res, sec))
    return rows


def run_slowflow(seed):
    rows = []
    for scen, b_flow, eddy in (("floor", 0.0, 0.0), ("flow_contrast", 500.0, 0.0), ("flow_contrast+eddy_0.4mm", 500.0, 0.4)):
        sim = simulate_slowflow_series(6, seed=seed, b_flow=b_flow, eddy_mm=eddy)
        st = sim.state_labels
        res, sec = _timed(lambda: motion_correction(sim.image, preset="bold"))
        rows.append(_row("slowflow", scen, "mean_ref", seed, sim, res, sec))
        if scen == "floor":
            continue
        res, sec = _timed(lambda: motion_correction(sim.image, preset="bold", encoding_covariates=st))
        rows.append(_row("slowflow", scen, "mean_ref+encoding_projection", seed, sim, res, sec))
        refs = build_bilateral_window_references(_frames(sim.image), encoding_covariates=st)
        res, sec = _timed(lambda: motion_correction(sim.image, preset="bold", reference=refs))
        rows.append(_row("slowflow", scen, "state_matched_ref", seed, sim, res, sec))
        res, sec = _timed(lambda: motion_correction(sim.image, preset="slowflow", encoding_covariates=st))
        rows.append(_row("slowflow", scen, "state_matched_ref+encoding_projection (slowflow preset)", seed, sim, res, sec))
        res, sec = _timed(lambda: motion_correction(sim.image, preset="slowflow", encoding_covariates=st, window_size=len(st)))
        rows.append(_row("slowflow", scen, "slowflow preset (global window)", seed, sim, res, sec))
    return rows


def _l2_covariates(bvals, bvecs):
    """Even (l = 2) gradient-direction terms: the span of the direction-dependent white-matter contrast."""
    g = np.asarray(bvecs, dtype=float)
    n = np.linalg.norm(g, axis=1)
    gu = np.zeros_like(g)
    dw = (np.asarray(bvals) > 50) & (n > 0)
    gu[dw] = g[dw] / n[dw, None]
    x, y, z = gu.T
    cov = np.c_[x * x - y * y, 2 * z * z - x * x - y * y, x * y, x * z, y * z]
    cov[~dw] = 0.0
    return cov


def run_dwi(seed):
    rows = []
    for scen, eddy in (("floor_b0only", None), ("dwi_contrast", 0.0), ("dwi_contrast+eddy_0.6mm", 0.6)):
        if scen == "floor_b0only":
            sim = simulate_dwi_series(24, seed=seed, b=0.0)
            res, sec = _timed(lambda: motion_correction(sim.image, preset="bold"))
            rows.append(_row("dwi", scen, "mean_ref", seed, sim, res, sec))
            continue
        sim = simulate_dwi_series(24, seed=seed, eddy_mm_per_unit=eddy)
        bvals, bvecs = sim.extra["bvals"], sim.extra["bvecs"]
        groups = shell_groups(bvals)
        pe_cov = bvecs * (bvals[:, None] / 1000.0)

        def eddy_err(r):
            pe = r.get("pe_affine")
            if pe is None or sim.eddy is None:
                return {}
            return {"eddy_shift_rms_err_mm": float(np.sqrt(np.mean((np.asarray(pe["a"])[:, 0] - sim.eddy[:, 0]) ** 2)))}

        res, sec = _timed(lambda: motion_correction(sim.image, preset="bold"))
        rows.append(_row("dwi", scen, "single_mean_ref", seed, sim, res, sec))
        l2 = _l2_covariates(bvals, bvecs)
        kw = dict(preset="dwi", pe_affine_covariates=pe_cov, pe_axis=sim.pe_axis, bias_covariates=l2)
        variants = [
            ("grouped+eddy+l2_bias (antsxdwi recipe)", dict(cross_group_debias=False)),
            ("+cross_group_debias", dict(cross_group_debias=True)),
            ("+cross_group_debias+encoding_projection(l2)", dict(cross_group_debias=True, encoding_covariates=l2)),
            ("+cross_group_debias+predicted_ref(l2)", dict(cross_group_debias=True, reference_predictor=encoding_reference_predictor(l2))),
            ("+cross_group_debias+predicted_ref(l2)+encoding_projection(l2)",
             dict(cross_group_debias=True, reference_predictor=encoding_reference_predictor(l2), encoding_covariates=l2)),
            ("+predicted_ref(l2) (evaluated, not adopted)", dict(reference_predictor=encoding_reference_predictor(l2))),
        ]
        for name, extra in variants:
            res, sec = _timed(lambda: motion_correct_grouped(sim.image, groups, **kw, **extra))
            rows.append(_row("dwi", scen, name, seed, sim, res, sec, eddy_err(res)))
    return rows


def run_bold(seed):
    rows = []
    for scen, resp, nod in (("floor", 0.0, 0.0), ("resp_0.4mm", 0.4, 0.0), ("resp_0.4mm+nodding", 0.4, 0.5)):
        sim = simulate_bold_series(48, seed=seed, resp_mm=resp, nod_mm=nod, nod_deg=nod)
        res, sec = _timed(lambda: motion_correction(sim.image, preset="bold"))
        rows.append(_row("bold", scen, "mean_ref", seed, sim, res, sec))
        if scen == "floor":
            continue
        res, sec = _timed(lambda: motion_correction(sim.image, preset="bold", pe_axis=sim.pe_axis, debias_pe_pseudomotion=True))
        # the PE debias changes the REPORTED trajectory only: score motion_parameters, not the (unchanged) transforms
        ev_rows = _row("bold", scen, "mean_ref+pe_debias(reported)", seed, sim, res, sec)
        if res.get("pe_pseudomotion_debias") is not None:
            from syntx.motion_simulation import matrix_to_pose  # noqa: F401
            mp = np.asarray(res["motion_parameters"], dtype=float)
            est = ev_rows["_est"].copy()
            shift = np.asarray(res["pe_b0_shift_mm"], dtype=float)
            est[:, :3] = est[:, :3] - np.outer(shift, sim.pe_axis)
            tru = sim.head_pose
            e = (est - est.mean(0)) - (tru - tru.mean(0))
            ev_rows["translation_rms_mm"] = float(np.sqrt((e[:, :3] ** 2).sum(1).mean()))
            ev_rows["pose_error_mm"] = float(np.sqrt((e[:, :3] ** 2).sum(1) + 2500.0 * (e[:, 3:] ** 2).sum(1)).mean())
            fd_fn = lambda y: np.r_[0.0, np.abs(np.diff(y[:, :3], axis=0)).sum(1) + 50.0 * np.abs(np.diff(y[:, 3:], axis=0)).sum(1)]  # noqa: E731
            ev_rows["fd_error_mm"] = float(np.abs(fd_fn(est) - fd_fn(tru))[1:].mean())
            ev_rows["_est"] = est
            ev_rows["removed_pe_rms_mm"] = float(np.sqrt(np.mean(shift ** 2)))
        rows.append(ev_rows)
    return rows


RUNNERS = {"asl": run_asl, "slowflow": run_slowflow, "dwi": run_dwi, "bold": run_bold}


def _summary(rows):
    import pandas as pd

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    keys = ["translation_rms_mm", "rotation_rms_deg", "pose_error_mm", "fd_error_mm", "jump_recovery_min"]
    keys += [k for k in ("eddy_shift_rms_err_mm", "removed_pe_rms_mm") if k in df]
    return df, df.groupby(["modality", "scenario", "strategy"], sort=False)[keys].mean().reset_index()


def _figure(rows, modality):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = [r for r in rows if r["modality"] == modality and r["seed"] == rows[0]["seed"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    true = sel[0]["_true"]
    for ax, j, lab in ((axes[0], slice(0, 3), "translation norm of demeaned pose (mm)"), (axes[1], slice(3, 6), "rotation norm (deg)")):
        scale = 1.0 if j.start == 0 else 57.2958
        tv = true[:, j] - true[:, j].mean(0)
        ax.plot(np.sign(tv[:, 0]) * np.linalg.norm(tv, axis=1) * scale, "k-", lw=2.5, label="truth")
        for r in sel:
            ev = r["_est"][:, j] - r["_est"][:, j].mean(0)
            ax.plot(np.sign(ev[:, 0]) * np.linalg.norm(ev, axis=1) * scale, lw=1, label=f"{r['scenario']}: {r['strategy']}")
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("frame")
    axes[1].legend(fontsize=6, loc="best")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--modalities", nargs="+", default=list(RUNNERS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in args.modalities:
        for seed in range(args.seeds):
            t0 = time.time()
            rows += RUNNERS[m](seed)
            print(f"{m} seed {seed}: {time.time() - t0:.0f}s", flush=True)
    df, summ = _summary(rows)
    df.to_csv(out / "results.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.to_string(index=False))
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>Encoding-bias motion simulation</title>",
            "<style>body{font-family:system-ui,sans-serif;max-width:1150px;margin:24px auto;padding:0 16px}"
            "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}"
            "td:nth-child(-n+3),th:nth-child(-n+3){text-align:left}</style></head><body>",
            "<h1>Encoding-locked systematic noise in motion estimates: simulation benchmark</h1>",
            f"<p>{args.seeds} seed(s) per cell; errors are against the simulated head trajectory after demeaning (the "
            "reference's mean pose is a gauge). <code>floor</code> = same motion without the systematic noise.</p>",
            summ.to_html(index=False, float_format=lambda v: f"{v:.3f}")]
    for m in args.modalities:
        html.append(f"<h2>{m}</h2><img style='max-width:100%' src='data:image/png;base64,{_figure(rows, m)}'>")
    html.append("</body></html>")
    (out / "report.html").write_text("\n".join(html))
    print("report:", out / "report.html")


if __name__ == "__main__":
    main()
