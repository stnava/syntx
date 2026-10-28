"""Gate for the default head prior of the joint engine: the FORM of the surround (second-difference) penalty and its weight.

Three phases, each resumable from its JSON:

  preflight  a 24-frame excerpt of the first cohort series; forms x weights must move the poses by more than 0.01 mm against
             no prior and against each other at the same weight (the rule of 2026-10-28: a flat response is a bug until
             proven otherwise). Aborts otherwise.
  phantoms   the three drifting phantoms with known truth (ASL tag/control with drift, DWI with eddy currents, slow-flow
             encoding states): FD error, translation and rotation RMS error per setting from ``evaluate_motion_estimate``.
             Pass: no phantom loses more than 10 % FD error against no prior AND the DWI phantom gains.
  cohort     the five real series of ``cohort_shrinkage_report`` for the settings that passed the phantoms: FD, the B0-immune
             readout step, the phase-encode step, alignment (tsd_rel), convergence. Pass: readout step within 0.02 mm of no
             prior, alignment within 1 %, FD mean within max(2 %, twice the measured run-to-run noise of the no-prior solver).

The report (HTML, self-contained) shows, per setting: phantom error bars, pose traces with and without the prior, the
second-difference histograms with the knee, FD traces per series, and the response curves over weight.

Usage: python benchmarks/prior_form_gate.py <motion_cohort dir> <out dir> --phase preflight|phantoms|cohort|report|all
       [--settings none,sparse:0.1,sparse:0.3,sparse:1,cauchy:0.1,cauchy:0.3,charbonnier:0.1] [--cohort-settings ...]
       [--only bold,asl,asl60,dwi031,dwi11] [--device auto]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cohort_shrinkage_report import CASES, _design  # noqa: E402
from motion_cohort_gate import _pe_axis, _signatures  # noqa: E402

PHANTOM_BUDGET_LOSS = 0.10       # a phantom may lose at most 10 % FD error against no prior
COHORT_READOUT_TOL_MM = 0.02     # the B0-immune readout step may move at most this much
COHORT_TSD_TOL = 0.01            # alignment may worsen at most 1 %
COHORT_FD_TOL = 0.02             # FD mean may rise at most 2 % (author, 2026-10-28: 'any FD increase is too strict'), or the measured
                                 # run-to-run noise of the no-prior solver when that is larger (repeat_none.json next to cohort.json)
KNEE_MM = 0.05                   # surround_delta_mm of the joint engine (JOINT_SURROUND_DELTA_MM)


def parse_setting(s: str):
    """'none' -> (None, 0.0); 'sparse:0.3' -> ('sparse', 0.3)."""
    if s == "none":
        return None, 0.0
    form, w = s.split(":")
    return form, float(w)


def prior_kwargs(setting: str) -> dict:
    form, w = parse_setting(setting)
    if form is None:
        return {}
    return {"surround_weight": w, "surround_form": form}


def _poses(r):
    if "translations_mm" in r:
        return np.asarray(r["translations_mm"], float), np.asarray(r["rotations_deg"], float)
    mp = r["motion_parameters"]
    return np.asarray(mp.translations, float), np.asarray(mp.rotations_deg, float)


def _second_differences_mm(t, rot_deg):
    m = np.c_[t, np.radians(rot_deg) * 50.0]
    return np.sqrt((((m[1:-1] - 0.5 * (m[:-2] + m[2:])) ** 2).sum(1)))


# ------------------------------------------------------------------------------------------------------------------- preflight
def preflight(args, settings):
    import ants
    import syntx

    key = args.only.split(",")[0]
    rel, label, kind = CASES[key]
    path = rel if rel.startswith("/") else os.path.join(args.root, rel)
    img = ants.image_read(path)
    pe = _pe_axis(path, img)
    n = min(24, img.shape[-1])
    sub = ants.from_numpy(img.numpy()[..., :n].copy(), spacing=img.spacing, origin=img.origin, direction=img.direction)
    if kind == "shells":
        bv = np.loadtxt(path.replace(".nii.gz", ".bval"))[:n]; g = np.loadtxt(path.replace(".nii.gz", ".bvec")).T[:n]
        design = syntx.make_design("shells", bvals=bv, bvecs=g, pe_axis=pe)
    elif kind == "tag_control":
        design = syntx.make_design("tag_control", volume_types=[("control" if i % 2 == 0 else "label") for i in range(n)])
    else:
        design = _design(key, kind, path, sub, pe)
    poses = {}
    for s in settings:
        r = syntx.motion_correct_design(sub, design, engine="joint", device=args.device, verbose=False, **prior_kwargs(s))
        poses[s] = _poses(r)[0]
    base = [s for s in settings if s == "none"]
    report = {}
    ok = True
    for s in settings:
        if s == "none":
            continue
        moved = float(np.abs(poses[s] - poses[base[0]]).max()) if base else float("nan")
        report[s] = {"max_dt_vs_none_mm": moved}
        print(f"PREFLIGHT {key} ({n} frames): {s} vs none: max |dt| = {moved:.4f} mm", flush=True)
        if base and moved < 0.01:
            ok = False
    by_w: dict = {}
    for s in settings:
        f, w = parse_setting(s)
        if f is not None:
            by_w.setdefault(w, []).append(s)
    for w, group in by_w.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                moved = float(np.abs(poses[group[i]] - poses[group[j]]).max())
                report[f"{group[i]} vs {group[j]}"] = {"max_dt_mm": moved}
                print(f"PREFLIGHT {key}: {group[i]} vs {group[j]} (same weight): max |dt| = {moved:.4f} mm", flush=True)
                if moved < 0.01:
                    ok = False
    json.dump(report, open(os.path.join(args.out, "preflight.json"), "w"), indent=1)
    if not ok:
        raise SystemExit("PREFLIGHT FAILED: a setting does not move the outcome by 0.01 mm; fix before the gate")


# -------------------------------------------------------------------------------------------------------------------- phantoms
def phantoms(args, settings):
    import syntx
    from syntx.motion_simulation import evaluate_motion_estimate, simulate_asl_series, simulate_dwi_series, simulate_slowflow_series

    out_json = os.path.join(args.out, "phantoms.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}
    sims = {}

    def asl():
        sim = simulate_asl_series(n_frames=24, seed=2)
        return sim, syntx.make_design("tag_control", volume_types=["control" if s > 0 else "label" for s in sim.encoding_covariates])

    def dwi():
        sim = simulate_dwi_series(16, seed=42, eddy_mm_per_unit=0.4)
        return sim, syntx.make_design("shells", bvals=sim.extra["bvals"], bvecs=sim.extra["bvecs"], pe_axis=sim.pe_axis)

    def slowflow():
        sim = simulate_slowflow_series(6, seed=0, eddy_mm=0.4)
        return sim, syntx.make_design("encoding_states", state_labels=np.asarray(sim.state_labels, dtype=int))

    for name, maker in (("asl_drift", asl), ("dwi_eddy", dwi), ("slowflow_trajectory", slowflow)):
        res.setdefault(name, {})
        for s in settings:
            if s in res[name]:
                continue
            if name not in sims:
                sims[name] = maker()
            sim, design = sims[name]
            t0 = time.time()
            r = syntx.motion_correct_design(sim.image, design, engine="joint", device=args.device, verbose=False,
                                            **({"fd_method": "jenkinson"} if name == "slowflow_trajectory" else {}), **prior_kwargs(s))
            ev = evaluate_motion_estimate(r["fwdtransforms"], sim)
            t, rot = _poses(r)
            res[name][s] = {"fd_error_mm": float(ev["fd_error_mm"]), "translation_rms_mm": float(ev["translation_rms_mm"]),
                            "rotation_rms_deg": float(ev["rotation_rms_deg"]), "pose_error_mm": float(ev.get("pose_error_mm", np.nan)),
                            "seconds": time.time() - t0, "t": t.tolist(), "rot": rot.tolist(),
                            "true_pose": np.asarray(sim.head_pose, float).tolist(), "second_diff_mm": _second_differences_mm(t, rot).tolist()}
            json.dump(res, open(out_json, "w"), indent=1)
            print(f"PHANTOM {name:20s} {s:16s} fd_err {ev['fd_error_mm']:.3f}  t_rms {ev['translation_rms_mm']:.3f}  rot_rms {ev['rotation_rms_deg']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
    return res


def phantom_verdict(res, settings):
    """Per setting: passes if no phantom loses more than PHANTOM_BUDGET_LOSS against 'none' and the DWI phantom gains."""
    verdict = {}
    for s in settings:
        if s == "none":
            continue
        losses = {}
        ok = True
        for name, runs in res.items():
            if s not in runs or "none" not in runs:
                ok = False
                continue
            rel = runs[s]["fd_error_mm"] / runs["none"]["fd_error_mm"] - 1.0
            losses[name] = rel
            if rel > PHANTOM_BUDGET_LOSS:
                ok = False
        if "dwi_eddy" in losses and losses["dwi_eddy"] >= 0.0:
            ok = False
        verdict[s] = {"pass": ok, "fd_error_change": losses}
    return verdict


# ---------------------------------------------------------------------------------------------------------------------- cohort
def cohort(args, settings):
    import ants
    import syntx

    out_json = os.path.join(args.out, "cohort.json")
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for key in args.only.split(","):
        rel, label, kind = CASES[key]
        path = rel if rel.startswith("/") else os.path.join(args.root, rel)
        img = ants.image_read(path)
        pe = _pe_axis(path, img)
        sc = json.load(open(path.replace(".nii.gz", ".json"))) if os.path.exists(path.replace(".nii.gz", ".json")) else {}
        tr = float(sc.get("RepetitionTime", img.spacing[3]))
        design = _design(key, kind, path, img, pe); levels = np.asarray(design.levels); D = np.asarray(img.direction)[:3, :3]
        results.setdefault(key, {"label": label, "T": int(img.shape[-1]), "tr": tr, "runs": {}})
        for s in settings:
            if s in results[key]["runs"]:
                continue
            t0 = time.time()
            r = syntx.motion_correct_design(img, design, engine="joint", device=args.device, verbose=False, tr=tr, **prior_kwargs(s))
            secs = time.time() - t0
            sig = _signatures(r, img, levels)
            t, rot = _poses(r)
            results[key]["runs"][s] = {
                "seconds": secs, "fd_mean": sig["fd_mean"], "fd_max": sig["fd_max"], "step_mad_mm": sig["step_mad_mm"], "tsd_rel": sig["tsd_rel"],
                "frame_converged_frac": sig["frame_converged_frac"], "design_locked_mm": sig["design_locked_mm"],
                "fd": np.asarray(r["fd"], float).tolist(),
                "fd_bandlimited": (np.asarray(r["fd_bandlimited"], float).tolist() if r.get("fd_bandlimited") is not None else None),
                "t_ro": (t @ D[:, 0]).tolist(), "t_pe": (t @ D[:, 1]).tolist(), "t_sl": (t @ D[:, 2]).tolist(), "rot": rot.tolist(),
                "second_diff_mm": _second_differences_mm(t, rot).tolist(),
                "frac_gt_0p5": float(np.mean(np.asarray(r["fd"], float) > 0.5))}
            json.dump(results, open(out_json, "w"), indent=1, default=float)
            print(key, s, json.dumps({"fd": round(sig["fd_mean"], 3), "ro": round(sig["step_mad_mm"]["t_ro"], 3), "pe": round(sig["step_mad_mm"]["t_pe"], 3),
                                      "tsd": round(sig["tsd_rel"], 4), "conv": round(sig["frame_converged_frac"], 2), "s": int(secs)}), flush=True)
    return results


def _fd_tolerance(results, out_dir=None):
    """Per series: the larger of COHORT_FD_TOL and twice the measured run-to-run relative FD noise (``repeat_none.json``)."""
    tol = {key: COHORT_FD_TOL for key in results}
    rp = os.path.join(out_dir, "repeat_none.json") if out_dir else None
    if rp and os.path.exists(rp):
        rep = json.load(open(rp))
        for key, r in rep.items():
            if key in results and "none" in results[key]["runs"]:
                noise = abs(r["fd_mean"] / results[key]["runs"]["none"]["fd_mean"] - 1.0)
                tol[key] = max(COHORT_FD_TOL, 2.0 * noise)
    return tol


def cohort_verdict(results, settings, out_dir=None):
    verdict = {}
    fd_tol = _fd_tolerance(results, out_dir)
    for s in settings:
        if s == "none":
            continue
        rows = {}
        ok = True
        for key, rec in results.items():
            runs = rec["runs"]
            if s not in runs or "none" not in runs:
                ok = False
                continue
            b, r = runs["none"], runs[s]
            d_ro = r["step_mad_mm"]["t_ro"] - b["step_mad_mm"]["t_ro"]
            d_tsd = r["tsd_rel"] / b["tsd_rel"] - 1.0
            d_fd = r["fd_mean"] / b["fd_mean"] - 1.0
            rows[key] = {"readout_step_change_mm": d_ro, "tsd_change": d_tsd, "fd_change": d_fd, "fd_tolerance": fd_tol[key]}
            if abs(d_ro) > COHORT_READOUT_TOL_MM or d_tsd > COHORT_TSD_TOL or d_fd > fd_tol[key]:
                ok = False
        verdict[s] = {"pass": ok, "series": rows}
    return verdict


# ---------------------------------------------------------------------------------------------------------------------- report
def _b64(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.out
    ph = json.load(open(os.path.join(out, "phantoms.json"))) if os.path.exists(os.path.join(out, "phantoms.json")) else {}
    co = json.load(open(os.path.join(out, "cohort.json"))) if os.path.exists(os.path.join(out, "cohort.json")) else {}
    pf = json.load(open(os.path.join(out, "preflight.json"))) if os.path.exists(os.path.join(out, "preflight.json")) else {}
    settings_ph = sorted({s for runs in ph.values() for s in runs}, key=lambda s: (parse_setting(s)[0] or "", parse_setting(s)[1]))
    settings_co = sorted({s for rec in co.values() for s in rec["runs"]}, key=lambda s: (parse_setting(s)[0] or "", parse_setting(s)[1]))
    pv = phantom_verdict(ph, settings_ph) if ph else {}
    cv = cohort_verdict(co, settings_co, out) if co else {}
    figs = []
    # phantom bars + traces + second-difference histograms
    if ph:
        names = list(ph)
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
        for ax, metric in zip(axes, ("fd_error_mm", "translation_rms_mm", "rotation_rms_deg")):
            width = 0.8 / max(len(settings_ph), 1)
            for i, s in enumerate(settings_ph):
                ax.bar(np.arange(len(names)) + i * width, [ph[n][s][metric] if s in ph[n] else np.nan for n in names], width, label=s)
            ax.set_xticks(np.arange(len(names)) + 0.4 - width / 2); ax.set_xticklabels(names, fontsize=8); ax.set_title(metric, fontsize=9)
            ax.legend(fontsize=6)
        fig.tight_layout(); f = os.path.join(out, "phantom_bars.png"); fig.savefig(f, dpi=110); plt.close(fig); figs.append(("Phantoms: error against the known trajectory per setting", f))
        for n in names:
            runs = ph[n]
            fig, axes = plt.subplots(2, 4, figsize=(19, 7))
            true = np.asarray(runs["none"]["true_pose"]) if "none" in runs else None
            for k in range(3):
                for s in settings_ph:
                    if s in runs:
                        t = np.asarray(runs[s]["t"]); axes[0, k].plot(t[:, k] - t[:, k].mean(), lw=0.9, label=s)
                        rot = np.asarray(runs[s]["rot"]); axes[1, k].plot(rot[:, k] - rot[:, k].mean(), lw=0.9, label=s)
                if true is not None:
                    axes[0, k].plot(true[:, k] - true[:, k].mean(), "k--", lw=1.2, label="truth")
                    axes[1, k].plot(np.degrees(true[:, 3 + k] - true[:, 3 + k].mean()), "k--", lw=1.2, label="truth")
                axes[0, k].set_title(f"translation {'xyz'[k]} (demeaned, mm)", fontsize=8); axes[1, k].set_title(f"rotation {'xyz'[k]} (demeaned, deg)", fontsize=8)
            for s in settings_ph:
                if s in runs:
                    axes[0, 3].hist(runs[s]["second_diff_mm"], bins=30, histtype="step", lw=1.0, label=s)
            axes[0, 3].axvline(KNEE_MM, color="r", ls=":", lw=0.8); axes[0, 3].set_title("|second difference| of the pose (mm at 50 mm); knee dotted", fontsize=8)
            if true is not None:
                tt = np.c_[true[:, :3], true[:, 3:] * 50.0]; ud = np.sqrt((((tt[1:-1] - 0.5 * (tt[:-2] + tt[2:])) ** 2).sum(1)))
                axes[0, 3].hist(ud, bins=30, histtype="step", lw=1.4, color="k", label="truth")
            axes[1, 3].bar(range(len(settings_ph)), [runs[s]["fd_error_mm"] if s in runs else np.nan for s in settings_ph]); axes[1, 3].set_xticks(range(len(settings_ph)))
            axes[1, 3].set_xticklabels(settings_ph, rotation=30, fontsize=7); axes[1, 3].set_title("FD error (mm)", fontsize=8)
            for ax in axes.ravel():
                if ax.lines or ax.patches:
                    ax.legend(fontsize=6)
            fig.suptitle(n); fig.tight_layout(); f = os.path.join(out, f"phantom_{n}.png"); fig.savefig(f, dpi=100); plt.close(fig); figs.append((f"Phantom {n}: poses, second differences, FD error", f))
    # cohort traces and response
    if co:
        for key, rec in co.items():
            runs = rec["runs"]; x = np.arange(rec["T"])
            fig, axes = plt.subplots(2, 3, figsize=(19, 7.5))
            for s in settings_co:
                if s not in runs:
                    continue
                r = runs[s]
                axes[0, 0].plot(x, r["fd"], lw=0.8, label=s); axes[0, 1].plot(x, r["t_ro"], lw=0.8, label=s); axes[0, 2].plot(x, r["t_pe"], lw=0.8, label=s)
                axes[1, 0].hist(r["second_diff_mm"], bins=40, histtype="step", lw=1.0, label=s)
                if "none" in runs and s != "none":
                    axes[1, 1].plot(x, np.asarray(r["fd"]) - np.asarray(runs["none"]["fd"]), lw=0.8, label=f"FD({s}) - FD(none)")
                    axes[1, 2].plot(x, np.asarray(r["t_ro"]) - np.asarray(runs["none"]["t_ro"]), lw=0.8, label=f"readout({s}) - readout(none)")
            axes[0, 0].set_yscale("symlog", linthresh=0.5); axes[0, 0].axhline(0.5, color="r", ls=":", lw=0.6)
            axes[1, 0].axvline(KNEE_MM, color="r", ls=":", lw=0.8)
            for ax, t in zip(axes.ravel(), ("FD (rigid, mm; symlog)", "translation readout (B0-immune), mm", "translation phase encode, mm",
                                            "|second difference| (mm at 50 mm); knee dotted", "FD difference from none (mm)", "readout difference from none (mm)")):
                ax.set_title(t, fontsize=8)
                if ax.lines or ax.patches:
                    ax.legend(fontsize=6)
            fig.suptitle(rec["label"]); fig.tight_layout(); f = os.path.join(out, f"cohort_{key}.png"); fig.savefig(f, dpi=100); plt.close(fig); figs.append((rec["label"], f))
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        forms = sorted({parse_setting(s)[0] for s in settings_co if s != "none"})
        for key, rec in co.items():
            runs = rec["runs"]
            if "none" not in runs:
                continue
            for form in forms:
                ss = sorted([s for s in settings_co if parse_setting(s)[0] == form and s in runs], key=lambda s: parse_setting(s)[1])
                ws = [0.0] + [parse_setting(s)[1] for s in ss]
                axes[0].plot(ws, [0.0] + [100 * (runs[s]["fd_mean"] / runs["none"]["fd_mean"] - 1) for s in ss], "o-", label=f"{key} {form}")
                axes[1].plot(ws, [0.0] + [1000 * (runs[s]["step_mad_mm"]["t_ro"] - runs["none"]["step_mad_mm"]["t_ro"]) for s in ss], "o-", label=f"{key} {form}")
                axes[2].plot(ws, [0.0] + [100 * (runs[s]["tsd_rel"] / runs["none"]["tsd_rel"] - 1) for s in ss], "o-", label=f"{key} {form}")
        for ax, t in zip(axes, ("FD mean, % change from none", "readout step change, micrometres", "alignment tsd, % change (negative = better)")):
            ax.set_title(t, fontsize=9); ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("weight (curvature units)"); ax.legend(fontsize=6)
        axes[1].axhline(1000 * COHORT_READOUT_TOL_MM, color="r", ls=":", lw=0.6); axes[1].axhline(-1000 * COHORT_READOUT_TOL_MM, color="r", ls=":", lw=0.6)
        fig.tight_layout(); f = os.path.join(out, "cohort_response.png"); fig.savefig(f, dpi=110); plt.close(fig); figs.insert(0, ("Cohort response across series", f))

    def table(rows, header):
        return "<table><tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"

    ph_rows = []
    for s in settings_ph:
        cells = [s]
        for n in ph:
            r = ph[n].get(s)
            cells.append("-" if r is None else f"{r['fd_error_mm']:.3f} / {r['translation_rms_mm']:.3f} / {r['rotation_rms_deg']:.3f}")
        cells.append("" if s == "none" else ("PASS" if pv.get(s, {}).get("pass") else "fail"))
        ph_rows.append(cells)
    co_rows = []
    for s in settings_co:
        cells = [s]
        for key in co:
            r = co[key]["runs"].get(s)
            cells.append("-" if r is None else f"{r['fd_mean']:.3f} / {r['step_mad_mm']['t_ro']:.3f} / {r['step_mad_mm']['t_pe']:.3f} / {r['tsd_rel']:.4f}")
        cells.append("" if s == "none" else ("PASS" if cv.get(s, {}).get("pass") else "fail"))
        co_rows.append(cells)
    both = [s for s in settings_co if s != "none" and pv.get(s, {}).get("pass") and cv.get(s, {}).get("pass")]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Prior form gate</title>
<style>body{{font-family:sans-serif;margin:2em;max-width:1500px}} table{{border-collapse:collapse;margin:6px 0}} td,th{{border:1px solid #999;padding:3px 6px;font-size:12px}} th{{background:#eee}}</style></head>
<body><h1>Default head prior for the joint engine: form and weight of the surround penalty</h1>
<p>Settings are <code>form:weight</code> in curvature units (1 = prior curvature equal to the measured image curvature at the knee);
<code>none</code> is the 6.27.0 default. Forms (<code>syntx.motion_batched.surround_rho</code>): charbonnier (quadratic below the knee,
gradient vanishes at zero), sparse (gradient ten times steeper at zero: sub-knee jitter driven to zero, drift and jumps free), cauchy
(re-descending). Knee {KNEE_MM} mm.</p>
<h2>Verdict</h2><p>Settings passing BOTH gates: <b>{', '.join(both) if both else 'none'}</b>.</p>
<h2>Pre-flight</h2><pre>{json.dumps(pf, indent=1)}</pre>
<h2>Phantoms (FD error mm / translation RMS mm / rotation RMS deg; pass = no phantom loses more than {int(100 * PHANTOM_BUDGET_LOSS)} % FD error against none and DWI gains)</h2>
{table(ph_rows, ['setting'] + list(ph) + ['gate'])}
<h2>Cohort (FD mean mm / readout step mm / PE step mm / tsd_rel; pass = readout step within {COHORT_READOUT_TOL_MM} mm, alignment within {int(100 * COHORT_TSD_TOL)} %, FD mean within max({int(100 * COHORT_FD_TOL)} %, 2 x repeat noise))</h2>
{table(co_rows, ['setting'] + [co[k]['label'] for k in co] + ['gate'])}
{''.join(f'<h3>{t}</h3><img src="{_b64(f)}" style="max-width:100%">' for t, f in figs)}
</body></html>"""
    path = os.path.join(out, "prior_form_gate.html")
    open(path, "w").write(html)
    json.dump({"phantom_verdict": pv, "cohort_verdict": cv, "pass_both": both}, open(os.path.join(out, "verdict.json"), "w"), indent=1)
    print("DONE", path, "pass_both:", both)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out")
    ap.add_argument("--phase", default="all", choices=["preflight", "phantoms", "cohort", "report", "all"])
    ap.add_argument("--settings", default="none,sparse:0.1,sparse:0.3,sparse:1,cauchy:0.1,cauchy:0.3,charbonnier:0.1")
    ap.add_argument("--cohort-settings", default=None, help="default: the settings that passed the phantoms")
    ap.add_argument("--only", default="bold,asl,asl60,dwi031,dwi11"); ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    settings = args.settings.split(",")
    if args.phase in ("preflight", "all"):
        preflight(args, settings)
    ph = None
    if args.phase in ("phantoms", "all"):
        ph = phantoms(args, settings)
        print(json.dumps(phantom_verdict(ph, settings), indent=1))
    if args.phase in ("cohort", "all"):
        if args.cohort_settings:
            cs = args.cohort_settings.split(",")
        else:
            ph = ph or json.load(open(os.path.join(args.out, "phantoms.json")))
            cs = ["none"] + [s for s, v in phantom_verdict(ph, settings).items() if v["pass"]]
            if len(cs) == 1:
                print("no setting passed the phantoms; cohort phase skipped")
                cs = []
        if cs:
            co = cohort(args, cs)
            print(json.dumps(cohort_verdict(co, cs, args.out), indent=1))
    if args.phase in ("report", "all"):
        report(args)


if __name__ == "__main__":
    main()
