"""Cohort scan of the shrinkage weight (curvature units; 1 = prior curvature equal to the measured image curvature) with a
visual report: per series, the FD trace and band-limited FD per weight, the six pose traces per weight, the eddy term where
it exists; per series and weight the readout / PE / slice step MAD, FD, alignment, convergence; a cross-series summary.
Joint engine, per-frame gain on.  Usage: python benchmarks/cohort_shrinkage_report.py <motion_cohort dir> <out dir>
[--weights 0,0.01,0.03,0.1] [--only bold,asl,asl60,dwi031,dwi11]
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from motion_cohort_gate import _pe_axis, _signatures  # noqa: E402

CASES = {
    "bold": ("breacher/sub-3315/ses-initial-day2/func/sub-3315_ses-initial-day2_task-rest_run-01_bold.nii.gz", "resting BOLD, breacher sub-3315, TR 1.0 s", "none"),
    "asl": ("ExpArt/sub-EAS020/ses-01/asl/sub-EAS020_ses-01_run-001_dir-PA_asl.nii.gz", "ASL, ExpArt sub-EAS020, 25 vol", "tag_control"),
    "asl60": ("/Users/stnava/data/ds004856/sub-1003/ses-wave1/perf/sub-1003_ses-wave1_acq-ASL_run-1_asl.nii.gz", "ASL, OpenNeuro ds004856 sub-1003, 60 vol", "tag_control"),
    "dwi031": ("FPA/sub-BLAST031/ses-01/dwi/sub-BLAST031_ses-01_run-001_dir-AP_dwi.nii.gz", "DWI, FPA BLAST031 dir-AP, high motion", "shells"),
    "dwi11": ("/Users/stnava/data/blast_cohorts/BIDS/SOCOM/sub-Blast-11/ses-01/dwi/sub-Blast-11_ses-01_run-001_dir-PA_dwi.nii.gz", "DWI, SOCOM Blast-11 dir-PA, low SNR", "shells"),
}


def _design(key, kind, path, img, pe):
    import syntx
    if kind == "none":
        return syntx.make_design("none", n_frames=img.shape[-1])
    if kind == "tag_control":
        ctx = path.replace("_asl.nii.gz", "_aslcontext.tsv")
        if not os.path.exists(ctx):
            ctx = glob.glob(os.path.join(os.path.dirname(path), "*_aslcontext.tsv"))[0]
        return syntx.make_design("tag_control", volume_types=[l.strip() for l in open(ctx).read().splitlines()[1:]])
    bv = np.loadtxt(path.replace(".nii.gz", ".bval")); g = np.loadtxt(path.replace(".nii.gz", ".bvec")).T
    return syntx.make_design("shells", bvals=bv, bvecs=g, pe_axis=pe)


def preflight(args):
    """Verify that the scanned parameter changes the outcome on a short excerpt of the first series (the rule of
    2026-10-28: a flat response is a bug until proven otherwise). Aborts the scan when the top weight moves no pose by
    more than 0.01 mm."""
    import ants
    import syntx
    key = args.only.split(",")[0]
    rel, label, kind = CASES[key]
    path = rel if rel.startswith("/") else os.path.join(args.root, rel)
    img = ants.image_read(path); pe, _ = _pe_axis(path, img), None
    n = min(24, img.shape[-1]); sub = ants.from_numpy(img.numpy()[..., :n].copy(), spacing=img.spacing, origin=img.origin, direction=img.direction)
    design = _design(key, kind, path, sub, pe) if kind != "shells" else None
    if kind == "shells":
        bv = np.loadtxt(path.replace(".nii.gz", ".bval"))[:n]; g = np.loadtxt(path.replace(".nii.gz", ".bvec")).T[:n]
        design = syntx.make_design("shells", bvals=bv, bvecs=g, pe_axis=pe)
    elif kind == "tag_control":
        design = syntx.make_design("tag_control", volume_types=[("control" if i % 2 == 0 else "label") for i in range(n)])
    weights = [float(w) for w in args.weights.split(",")]
    poses = {}
    for w in (weights[0], weights[-1]):
        r = syntx.motion_correct_design(sub, design, engine="joint", device=args.device, verbose=False, pose_shrinkage=w)
        poses[w] = np.asarray(r["translations_mm"] if "translations_mm" in r else r["motion_parameters"].translations, float)
    moved = float(np.abs(poses[weights[-1]] - poses[weights[0]]).max())
    print(f"PREFLIGHT {key} ({n} frames): max |dt| between mu={weights[0]:g} and mu={weights[-1]:g} is {moved:.4f} mm", flush=True)
    if moved < 0.01:
        raise SystemExit("PREFLIGHT FAILED: the parameter does not move the outcome; fix before scanning")


def run(args):
    import ants
    import syntx

    os.makedirs(args.out, exist_ok=True)
    preflight(args)
    out_json = os.path.join(args.out, "cohort_shrinkage.json")
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    weights = [float(w) for w in args.weights.split(",")]
    for key in args.only.split(","):
        rel, label, kind = CASES[key]
        path = rel if rel.startswith("/") else os.path.join(args.root, rel)
        img = ants.image_read(path); pe, sc = _pe_axis(path, img), (json.load(open(path.replace(".nii.gz", ".json"))) if os.path.exists(path.replace(".nii.gz", ".json")) else {})
        tr = float(sc.get("RepetitionTime", img.spacing[3]))
        design = _design(key, kind, path, img, pe); levels = np.asarray(design.levels); D = np.asarray(img.direction)[:3, :3]
        results.setdefault(key, {"label": label, "T": int(img.shape[-1]), "tr": tr, "levels": levels.tolist(), "runs": {}})
        for w in weights:
            wl = f"{w:g}"
            if wl in results[key]["runs"]:
                continue
            t0 = time.time()
            r = syntx.motion_correct_design(img, design, engine="joint", device=args.device, verbose=False, tr=tr, pose_shrinkage=w)
            secs = time.time() - t0
            sig = _signatures(r, img, levels)
            if "translations_mm" in r:
                t = np.asarray(r["translations_mm"], float); rot = np.asarray(r["rotations_deg"], float)
            else:
                mp = r["motion_parameters"]; t = np.asarray(mp.translations, float); rot = np.asarray(mp.rotations_deg, float)
            pe1 = r.get("pe_affine"); eddy = np.asarray(pe1["a"], float)[:, 0].tolist() if isinstance(pe1, dict) and pe1.get("a") is not None else None
            pr = r.get("prior"); pr = pr[0] if isinstance(pr, list) else (pr or {})
            sl = pr.get("scale_log") or []
            corr = r["motion_corrected"].numpy(); zc = corr.shape[2] // 2; yc = corr.shape[1] // 2
            mean_img = corr.mean(-1); cv = corr.std(-1) / (np.abs(mean_img) + 1e-6)
            np.savez_compressed(os.path.join(args.out, f"{key}_mu{wl}_slices.npz"), mean_ax=mean_img[:, :, zc], mean_cor=mean_img[:, yc, :],
                                cv_ax=np.clip(cv[:, :, zc], 0, 1), cv_cor=np.clip(cv[:, yc, :], 0, 1))
            if wl == f"{weights[0]:g}":
                raw = img.numpy(); rm = raw.mean(-1); rcv = raw.std(-1) / (np.abs(rm) + 1e-6)
                np.savez_compressed(os.path.join(args.out, f"{key}_raw_slices.npz"), mean_ax=rm[:, :, zc], mean_cor=rm[:, yc, :], cv_ax=np.clip(rcv[:, :, zc], 0, 1), cv_cor=np.clip(rcv[:, yc, :], 0, 1))
            results[key]["runs"][wl] = {
                "seconds": secs, "fd_mean": sig["fd_mean"], "fd_max": sig["fd_max"], "step_mad_mm": sig["step_mad_mm"], "tsd_rel": sig["tsd_rel"],
                "frame_converged_frac": sig["frame_converged_frac"], "design_locked_mm": sig["design_locked_mm"],
                "fd": np.asarray(r["fd"], float).tolist(), "fd_bandlimited": (np.asarray(r["fd_bandlimited"], float).tolist() if r.get("fd_bandlimited") is not None else None),
                "band": r.get("fd_band"), "t_ro": (t @ D[:, 0]).tolist(), "t_pe": (t @ D[:, 1]).tolist(), "t_sl": (t @ D[:, 2]).tolist(),
                "rot": rot.tolist(), "eddy": eddy, "gain": r.get("frame_gain"), "curvature_scale": (sl[-1].get("scale") if sl else None),
                "frac_gt_0p5": float(np.mean(np.asarray(r["fd"], float) > 0.5))}
            json.dump(results, open(out_json, "w"), indent=1, default=float)
            print(key, "mu", wl, json.dumps({"fd": round(sig["fd_mean"], 3), "ro": round(sig["step_mad_mm"]["t_ro"], 3), "pe": round(sig["step_mad_mm"]["t_pe"], 3), "tsd": round(sig["tsd_rel"], 4), "conv": sig["frame_converged_frac"], "s": round(secs)}), flush=True)
    report(results, args.out)


def _b64(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def report(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sections = []
    for key, rec in results.items():
        runs = rec["runs"]; wls = sorted(runs, key=float)
        T = rec["T"]; x = np.arange(T); r_base = runs[wls[0]]
        # ---- figure 1: traces per weight, with differences from mu = 0
        fig, axes = plt.subplots(5, 2, figsize=(17, 16))
        for wl in wls:
            r = runs[wl]; lab = f"mu = {wl}"
            axes[0, 0].plot(x, r["fd"], lw=0.8, label=lab)
            if r["fd_bandlimited"] is not None:
                axes[0, 1].plot(x, r["fd_bandlimited"], lw=0.8, label=lab)
            axes[1, 0].plot(x, r["t_ro"], lw=0.8, label=lab); axes[1, 1].plot(x, r["t_pe"], lw=0.8, label=lab)
            axes[2, 0].plot(x, r["t_sl"], lw=0.8, label=lab)
            rot = np.asarray(r["rot"]); axes[2, 1].plot(x, rot[:, 0], lw=0.8, label=lab)
            axes[3, 0].plot(x, rot[:, 1], lw=0.8, label=lab); axes[3, 1].plot(x, rot[:, 2], lw=0.8, label=lab)
            if wl != wls[0]:
                axes[4, 0].plot(x, np.asarray(r["fd"]) - np.asarray(r_base["fd"]), lw=0.8, label=f"FD(mu={wl}) - FD(mu={wls[0]})")
                axes[4, 1].plot(x, np.asarray(r["t_ro"]) - np.asarray(r_base["t_ro"]), lw=0.8, label=f"readout(mu={wl}) - readout(mu={wls[0]})")
        if r_base["eddy"] is not None:
            axes[1, 1].plot(x, r_base["eddy"], lw=0.7, ls="--", color="k", label="eddy PE shift (applied, not in FD)")
        axes[0, 0].axhline(0.5, color="r", lw=0.6, ls=":"); axes[0, 0].set_yscale("symlog", linthresh=0.5)
        titles = [["FD (rigid, mm; symlog)", "FD band-limited (censoring series, mm)"], ["translation readout (B0-immune), mm", "translation phase encode, mm"],
                  ["translation slice, mm", "rotation x, deg"], ["rotation y, deg", "rotation z, deg"],
                  [f"difference from mu = {wls[0]}: FD (mm)", f"difference from mu = {wls[0]}: readout translation (mm)"]]
        for i in range(5):
            for j in range(2):
                axes[i, j].set_title(titles[i][j], fontsize=9)
                if axes[i, j].lines:
                    axes[i, j].legend(fontsize=6, ncol=2)
        for j in range(2):
            axes[4, j].axhline(0, color="k", lw=0.5)
        axes[4, 0].set_xlabel("volume"); axes[4, 1].set_xlabel("volume")
        fig.suptitle(rec["label"], fontsize=11); fig.tight_layout(); f1 = os.path.join(out, f"{key}_traces.png"); fig.savefig(f1, dpi=100); plt.close(fig)
        # ---- figure 2: pose-difference heat maps (frames x 6 DOF) per weight, FD histograms, FD scatter vs mu = 0
        nw = len(wls) - 1
        fig, axes = plt.subplots(2, max(nw, 1) + 1, figsize=(4.2 * (max(nw, 1) + 1), 7))
        axes = np.atleast_2d(axes)
        P0 = np.c_[r_base["t_ro"], r_base["t_pe"], r_base["t_sl"], np.asarray(r_base["rot"]) * 50 * np.pi / 180]
        for i, wl in enumerate(wls[1:]):
            r = runs[wl]; P = np.c_[r["t_ro"], r["t_pe"], r["t_sl"], np.asarray(r["rot"]) * 50 * np.pi / 180]
            d = (P - P0).T
            v = max(float(np.abs(d).max()), 1e-6)
            im = axes[0, i].imshow(d, aspect="auto", cmap="coolwarm", vmin=-v, vmax=v); axes[0, i].set_yticks(range(6)); axes[0, i].set_yticklabels(["t ro", "t pe", "t sl", "rot x", "rot y", "rot z"], fontsize=7)
            axes[0, i].set_title(f"pose(mu={wl}) - pose(mu={wls[0]}), mm (rot at 50 mm)", fontsize=8); plt.colorbar(im, ax=axes[0, i], fraction=0.03)
            axes[1, i].scatter(r_base["fd"][1:], r["fd"][1:], s=6, alpha=0.6); lim = max(max(r_base["fd"]), max(r["fd"]))
            axes[1, i].plot([0, lim], [0, lim], "k-", lw=0.5); axes[1, i].set_xlabel(f"FD mu={wls[0]} (mm)"); axes[1, i].set_ylabel(f"FD mu={wl} (mm)"); axes[1, i].set_title("per-volume FD", fontsize=8)
        for wl in wls:
            axes[0, -1].hist(np.asarray(runs[wl]["fd"])[1:], bins=40, histtype="step", lw=1.0, label=f"mu = {wl}")
        axes[0, -1].axvline(0.5, color="r", lw=0.6, ls=":"); axes[0, -1].set_title("FD histogram per weight", fontsize=8); axes[0, -1].legend(fontsize=6); axes[0, -1].set_xlabel("FD (mm)")
        ws = [float(w) for w in wls]
        for q, lab in (("fd_mean", "FD mean"), ):
            axes[1, -1].plot(ws, [runs[w][q] for w in wls], "o-", label=lab)
        axes[1, -1].plot(ws, [runs[w]["step_mad_mm"]["t_ro"] for w in wls], "s-", label="readout step MAD")
        axes[1, -1].plot(ws, [runs[w]["step_mad_mm"]["t_pe"] for w in wls], "^-", label="PE step MAD")
        axes[1, -1].plot(ws, [runs[w]["tsd_rel"] / r_base["tsd_rel"] * runs[wls[0]]["fd_mean"] for w in wls], "d--", label="alignment tsd (scaled to FD at mu 0)")
        axes[1, -1].set_xscale("symlog", linthresh=min([w for w in ws if w > 0] or [0.01])); axes[1, -1].set_xlabel("mu"); axes[1, -1].set_title("response to the weight (mm)", fontsize=8); axes[1, -1].legend(fontsize=6)
        fig.tight_layout(); f2 = os.path.join(out, f"{key}_diffs.png"); fig.savefig(f2, dpi=100); plt.close(fig)
        # ---- figure 3: images: raw and corrected mean and temporal CV per weight, axial and coronal mid-slices
        cols = ["raw"] + wls
        fig, axes = plt.subplots(4, len(cols), figsize=(3.4 * len(cols), 12))
        axes = np.atleast_2d(axes)
        for j, c in enumerate(cols):
            fpz = os.path.join(out, f"{key}_raw_slices.npz" if c == "raw" else f"{key}_mu{c}_slices.npz")
            if not os.path.exists(fpz):
                for i in range(4): axes[i, j].axis("off")
                continue
            z = np.load(fpz)
            axes[0, j].imshow(np.rot90(z["mean_ax"]), cmap="gray"); axes[0, j].set_title(("raw" if c == "raw" else f"mu = {c}") + ": mean, axial", fontsize=8)
            axes[1, j].imshow(np.rot90(z["cv_ax"]), cmap="magma", vmin=0, vmax=0.5); axes[1, j].set_title("temporal CV, axial", fontsize=8)
            axes[2, j].imshow(np.rot90(z["mean_cor"]), cmap="gray", aspect="auto"); axes[2, j].set_title("mean, coronal", fontsize=8)
            axes[3, j].imshow(np.rot90(z["cv_cor"]), cmap="magma", vmin=0, vmax=0.5, aspect="auto"); axes[3, j].set_title("temporal CV, coronal", fontsize=8)
            for i in range(4): axes[i, j].axis("off")
        fig.tight_layout(); f3 = os.path.join(out, f"{key}_images.png"); fig.savefig(f3, dpi=90); plt.close(fig)
        rows = "".join(f"<tr><td>{wl}</td><td>{runs[wl]['fd_mean']:.3f}</td><td>{runs[wl]['fd_max']:.2f}</td><td>{100 * runs[wl]['frac_gt_0p5']:.0f} %</td>"
                       f"<td>{runs[wl]['step_mad_mm']['t_ro']:.3f}</td><td>{runs[wl]['step_mad_mm']['t_pe']:.3f}</td><td>{runs[wl]['step_mad_mm']['t_sl']:.3f}</td><td>{runs[wl]['step_mad_mm']['pitch']:.3f}</td>"
                       f"<td>{runs[wl]['design_locked_mm']['t_pe']:.3f}</td><td>{runs[wl]['tsd_rel']:.4f}</td><td>{runs[wl]['frame_converged_frac']:.2f}</td>"
                       f"<td>{('%.2e' % runs[wl]['curvature_scale']) if runs[wl]['curvature_scale'] else '-'}</td><td>{runs[wl]['seconds']:.0f}</td></tr>" for wl in wls)
        sections.append(f"""<h2>{rec['label']} (T = {T}, TR = {rec['tr']:.2f} s)</h2>
<table><tr><th>mu</th><th>FD mean</th><th>FD max</th><th>&gt; 0.5 mm</th><th>step readout</th><th>step PE</th><th>step slice</th><th>step pitch</th><th>locked PE</th><th>tsd_rel</th><th>conv</th><th>curvature / mm^2</th><th>s</th></tr>{rows}</table>
<h3>Traces per weight</h3><img src="{_b64(f1)}" style="max-width:100%">
<h3>Differences from mu = {wls[0]}, FD distributions, response curves</h3><img src="{_b64(f2)}" style="max-width:100%">
<h3>Images: raw and corrected mean and temporal coefficient of variation</h3><img src="{_b64(f3)}" style="max-width:100%">""")
    # ---- response across series: percent change from mu = 0 of FD mean, readout step, PE step, alignment
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for key, rec in results.items():
        wls = sorted(rec["runs"], key=float); ws = [float(w) for w in wls]; b = rec["runs"][wls[0]]
        axes[0].plot(ws, [100 * (rec["runs"][w]["fd_mean"] / b["fd_mean"] - 1) for w in wls], "o-", label=key)
        axes[1].plot(ws, [100 * (rec["runs"][w]["step_mad_mm"]["t_ro"] / b["step_mad_mm"]["t_ro"] - 1) for w in wls], "o-", label=key)
        axes[2].plot(ws, [100 * (rec["runs"][w]["step_mad_mm"]["t_pe"] / b["step_mad_mm"]["t_pe"] - 1) for w in wls], "o-", label=key)
        axes[3].plot(ws, [100 * (rec["runs"][w]["tsd_rel"] / b["tsd_rel"] - 1) for w in wls], "o-", label=key)
    for ax, t in zip(axes, ("FD mean, % change from mu = 0", "readout step (B0-immune floor), % change", "phase-encode step, % change", "alignment tsd, % change (negative = better)")):
        ax.set_title(t, fontsize=9); ax.axhline(0, color="k", lw=0.5); ax.set_xscale("symlog", linthresh=0.01); ax.set_xlabel("shrinkage mu (curvature units)"); ax.legend(fontsize=7)
    fig.tight_layout(); sp = os.path.join(out, "response.png"); fig.savefig(sp, dpi=110); plt.close(fig)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Cohort shrinkage scan</title>
<style>body{{font-family:sans-serif;margin:2em;max-width:1400px}} table{{border-collapse:collapse;margin:6px 0}} td,th{{border:1px solid #999;padding:3px 6px;font-size:12px}} th{{background:#eee}}</style></head>
<body><h1>Shrinkage toward identity on the cohort: joint engine, per-frame gain on, Newton unit conversion fixed</h1>
<p>mu is in curvature units: the prior's curvature relative to the measured curvature of the image term (median over frames and
translation axes at every stage start); a weight of 1 pulls a pose the data pin halfway to identity, and fully in any direction the
data do not pin. Step MAD: robust frame-to-frame step per axis in mm; the readout axis is the B0-immune floor (real jitter the prior
must not remove). locked PE: period-2 pose amplitude along phase encode relative to the anchor level. tsd_rel: alignment, lower is
better. The scan ran only after a pre-flight check that the top weight moves the poses on a 24-frame excerpt.</p>
<h2>Response across series (percent change from mu = 0)</h2><img src="{_b64(sp)}" style="max-width:100%">{''.join(sections)}</body></html>"""
    open(os.path.join(out, "cohort_shrinkage_report.html"), "w").write(html)
    print("DONE", os.path.join(out, "cohort_shrinkage_report.html"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--weights", default="0,0.03,0.1,0.3")
    ap.add_argument("--only", default="bold,asl,asl60,dwi031,dwi11"); ap.add_argument("--device", default="auto")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        report(json.load(open(os.path.join(a.out, "cohort_shrinkage.json"))), a.out)
    else:
        run(a)
