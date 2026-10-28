"""The four ISMRM cases (docs/MOTION_NEXT_STEPS_PLAN_2026-10-28.md, step 1): the released joint engine on one resting BOLD,
one ASL and two multi-shell DWI series, with band-limited FD and the pose- and image-domain decompositions
(``syntx.motion_decomposition``). Writes <out>/ismrm_cases.json, .html and _figure.png.

Usage: python benchmarks/ismrm_cases.py <motion_cohort dir> <out dir> [--only bold,asl,dwi031,dwi11] [--device auto]
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import time

import numpy as np

from antsxqc.metrics import FD_SCRUBBING_THRESHOLD_MM

CASES = {
    "bold": ("breacher/sub-3315/ses-initial-day2/func/sub-3315_ses-initial-day2_task-rest_run-01_bold.nii.gz", "resting BOLD, breacher sub-3315, TR 1.0 s"),
    "asl": ("ExpArt/sub-EAS020/ses-01/asl/sub-EAS020_ses-01_run-001_dir-PA_asl.nii.gz", "ASL, ExpArt sub-EAS020"),
    "asl60": ("/Users/stnava/data/ds004856/sub-1003/ses-wave1/perf/sub-1003_ses-wave1_acq-ASL_run-1_asl.nii.gz", "ASL, OpenNeuro ds004856 sub-1003, PCASL 30 pairs"),
    "dwi031": ("FPA/sub-BLAST031/ses-01/dwi/sub-BLAST031_ses-01_run-001_dir-AP_dwi.nii.gz", "DWI, FPA sub-BLAST031 dir-AP, high motion"),
    "dwi11": ("/Users/stnava/data/blast_cohorts/BIDS/SOCOM/sub-Blast-11/ses-01/dwi/sub-Blast-11_ses-01_run-001_dir-PA_dwi.nii.gz", "DWI, SOCOM sub-Blast-11 dir-PA, low SNR"),
}


def _pe_axis(path, img):
    sc = json.load(open(path.replace(".nii.gz", ".json"))) if os.path.exists(path.replace(".nii.gz", ".json")) else {}
    pe = sc.get("PhaseEncodingDirection", "j")
    ax = {"i": 0, "j": 1, "k": 2}[pe[0]]
    v = np.asarray(img.direction)[:3, ax] * (-1.0 if pe.endswith("-") else 1.0)
    return v / np.linalg.norm(v), sc


def run_case(key, path, device, shrinkage=0.0):
    import ants
    import syntx
    from syntx.motion_decomposition import design_blocks, image_variance_decomposition, pose_step_decomposition

    img = ants.image_read(path)
    pe, sc = _pe_axis(path, img)
    tr = float(sc.get("RepetitionTime", img.spacing[3]))
    if key == "bold":
        design = syntx.make_design("none", n_frames=img.shape[-1]); bv = g = None
    elif key.startswith("asl"):
        ctx = path.replace("_asl.nii.gz", "_aslcontext.tsv")
        if not os.path.exists(ctx):
            ctx = glob.glob(os.path.join(os.path.dirname(path), "*_aslcontext.tsv"))[0]
        vt = [l.strip() for l in open(ctx).read().splitlines()[1:]]
        design = syntx.make_design("tag_control", volume_types=vt); bv = g = None
    else:
        bv = np.loadtxt(path.replace(".nii.gz", ".bval")); g = np.loadtxt(path.replace(".nii.gz", ".bvec")).T
        design = syntx.make_design("shells", bvals=bv, bvecs=g, pe_axis=pe)
    t0 = time.time()
    T = img.shape[-1]
    extra = {"pose_shrinkage": float(shrinkage), "surround_scale": "likelihood"} if shrinkage > 0 else {}
    r = syntx.motion_correct_design(img, design, engine="joint", device=device, verbose=False, tr=tr, **extra)
    secs = time.time() - t0
    fd = np.asarray(r["fd"], float)
    # the censoring series: band-limited FD for BOLD and ASL (periodic physiology faster than the sampling); the rigid FD
    # itself for DWI (no steady physiological oscillation at a DWI TR, and a shell alternation must not be notched)
    fdb = np.asarray(r["fd_bandlimited"], float) if (key in ("bold", "asl") and r.get("fd_bandlimited") is not None) else fd
    if "translations_mm" in r:
        t = np.asarray(r["translations_mm"], float); rot = np.asarray(r["rotations_deg"], float)
        conv = np.asarray(r["frame_converged"], bool)
    else:   # one-level design: the plain MotionCorrectionResult
        mp = r["motion_parameters"]; t = np.asarray(mp.translations, float); rot = np.asarray(mp.rotations_deg, float)
        conv = np.asarray((r.get("convergence") or {}).get("frame_converged", [True] * T), bool)
    enc = r.get("encoding_pose_bias_mm_rad"); enc = enc[0] if isinstance(enc, list) else enc
    bias = np.asarray(enc["delta_Y_mm_rad"], float) if isinstance(enc, dict) and enc.get("delta_Y_mm_rad") is not None else None
    pe1 = r.get("pe_affine"); eddy = np.asarray(pe1["a"], float)[:, 0] if isinstance(pe1, dict) and pe1.get("a") is not None else None
    pose = pose_step_decomposition(t, rot, np.asarray(img.direction)[:3, :3], applied_pe_shift_mm=eddy, pe_axis=pe, removed_bias_mm_rad=bias)
    levels = np.asarray(design.levels)
    blocks = design_blocks(levels, design.name, bvecs=g, bvals=bv, lmax=4, min_frames_per_column=3)   # a measurement, not an estimator: l = 4 with 46 frames per shell
    corr = r["motion_corrected"].numpy(); raw = img.numpy()
    m = corr.mean(-1); mask = m > 0.3 * np.percentile(m, 99)
    # motion regressors after the design: the six poses and FD when the series is long enough to spend seven degrees of
    # freedom on them (T >= 60), FD alone otherwise (a 16-frame ASL series would be overfitted)
    motion = np.c_[t, rot, fd] if T >= 60 else fd[:, None]
    gain = None if r.get("frame_gain") is None else np.asarray(r["frame_gain"], float)
    dec_c = image_variance_decomposition(corr, blocks, motion, mask, fd=fd, frame_gain=gain)
    dec_r = image_variance_decomposition(raw, blocks, motion, mask, fd=fd, frame_gain=gain)
    rec = {"series": path, "T": T, "tr": tr, "design": design.name, "levels": int(design.n_levels), "seconds": secs,
           "fd_mean": float(fd[1:].mean()), "fd_max": float(fd.max()), "fd_bandlimited_mean": float(fdb[1:].mean()), "fd_band": r.get("fd_band"),
           "frac_gt_thr_raw": float(np.mean(fd > FD_SCRUBBING_THRESHOLD_MM)), "frac_gt_thr_bandlimited": float(np.mean(fdb > FD_SCRUBBING_THRESHOLD_MM)),
           "frame_converged_frac": float(np.mean(conv)), "shrinkage": float(shrinkage),
           "frame_gain_range": None if gain is None else [float(gain.min()), float(gain.max())],
           "pose": pose, "image_corrected": {k: v for k, v in dec_c.items() if k != "residual_rms_per_frame"},
           "image_raw": {k: v for k, v in dec_r.items() if k != "residual_rms_per_frame"},
           "traces": {"t_ro": (t @ np.asarray(img.direction)[:3, 0]).tolist(), "t_pe": (t @ np.asarray(img.direction)[:3, 1]).tolist(),
                      "eddy": None if eddy is None else eddy.tolist(), "bias_pe": None if bias is None else (bias[:, :3] @ np.asarray(img.direction)[:3, 1]).tolist(),
                      "fd": fd.tolist(), "fd_bandlimited": fdb.tolist(), "levels": levels.tolist()}}
    return rec


def _figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in CASES if k in results]
    fig, axes = plt.subplots(len(keys), 3, figsize=(16, 3.2 * len(keys)))
    axes = np.atleast_2d(axes)
    for i, k in enumerate(keys):
        r = results[k]; tr = r["traces"]; ax = axes[i]
        ax[0].plot(tr["t_ro"], lw=0.8, label="readout (head)"); ax[0].plot(tr["t_pe"], lw=0.8, label="phase encode (head)")
        if tr["eddy"] is not None:
            ax[0].plot(tr["eddy"], lw=0.7, ls="--", color="k", label="eddy PE shift (applied)")
        if tr["bias_pe"] is not None:
            ax[0].plot(tr["bias_pe"], lw=0.7, ls=":", color="r", label="design-locked PE (removed)")
        ax[0].set_ylabel("mm"); ax[0].set_title(CASES[k][1], fontsize=9, loc="left"); ax[0].legend(fontsize=6, ncol=2)
        ax[1].plot(tr["fd"], lw=0.8, label="FD (rigid)"); ax[1].plot(tr["fd_bandlimited"], lw=0.8, label="FD band-limited")
        ax[1].axhline(FD_SCRUBBING_THRESHOLD_MM, color="r", lw=0.6, ls=":"); ax[1].set_ylabel("mm"); ax[1].legend(fontsize=6)
        sel = [i for i, t_ in enumerate(r["image_corrected"]["terms"]) if not (t_["term"] == "level intercepts" and r["levels"] == 1)]
        terms = [r["image_corrected"]["terms"][i]["term"] for i in sel]
        fc = [100 * r["image_corrected"]["terms"][i]["fraction"] for i in sel]; fr = [100 * r["image_raw"]["terms"][i]["fraction"] for i in sel]
        x = np.arange(len(terms)); ax[2].bar(x - 0.2, fr, 0.4, label="raw"); ax[2].bar(x + 0.2, fc, 0.4, label="corrected")
        ax[2].set_xticks(x); ax[2].set_xticklabels([t_.replace(" direction terms per shell", "").replace(" regressors after the design", "").replace("level ", "") for t_ in terms], fontsize=7, rotation=20)
        ax[2].set_ylabel("% variance"); ax[2].set_yscale("log"); ax[2].set_ylim(bottom=0.05); ax[2].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _html(results, fig_path, path):
    rows = []
    for k, r in results.items():
        ro = r["pose"]["axes"]["readout"]; pe = r["pose"]["axes"]["phase_encode"]
        tc = {t_["term"]: t_["fraction"] for t_ in r["image_corrected"]["terms"]}; trw = {t_["term"]: t_["fraction"] for t_ in r["image_raw"]["terms"]}
        design_c = sum(v for n, v in tc.items() if n not in ("residual", "motion regressors after the design", "global gain drift (solver term)"))
        design_r = sum(v for n, v in trw.items() if n not in ("residual", "motion regressors after the design", "global gain drift (solver term)"))
        drift = f"{100 * trw.get('global gain drift (solver term)', 0):.1f} / {100 * tc.get('global gain drift (solver term)', 0):.1f}"
        rows.append(f"<tr><td>{CASES[k][1]}</td><td>{r['T']}</td><td>{r['fd_mean']:.3f} / {r['fd_max']:.2f}</td><td>{r['fd_bandlimited_mean']:.3f}</td>"
                    f"<td>{100 * r['frac_gt_thr_raw']:.0f} / {100 * r['frac_gt_thr_bandlimited']:.0f}</td>"
                    f"<td>{100 * ro['head']['share']:.0f} / {100 * ro['applied_acquisition_term']['share']:.0f} / {100 * ro['removed_design_locked_term']['share']:.0f}</td>"
                    f"<td>{100 * pe['head']['share']:.0f} / {100 * pe['applied_acquisition_term']['share']:.0f} / {100 * pe['removed_design_locked_term']['share']:.0f}</td>"
                    f"<td>{drift}</td><td>{100 * design_r:.1f} / {100 * design_c:.1f}</td><td>{100 * trw['motion regressors after the design']:.2f} / {100 * tc['motion regressors after the design']:.2f}</td>"
                    f"<td>{100 * trw['residual']:.1f} / {100 * tc['residual']:.1f}</td><td>{r['image_raw']['residual_rms_fd_corr']:.2f} / {r['image_corrected']['residual_rms_fd_corr']:.2f}</td>"
                    f"<td>{r['frame_converged_frac']:.2f}</td><td>{r['seconds']:.0f}</td></tr>")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>One motion solver, four modalities</title>
<style>body{{font-family:sans-serif;margin:2em;max-width:1250px}} table{{border-collapse:collapse}} td,th{{border:1px solid #999;padding:3px 6px;font-size:12px}} th{{background:#eee}}</style></head>
<body><h1>One motion solver, four cases (joint engine, syntx 6.26)</h1>
<p>FD: Power FD at 50 mm from the rigid pose. Band-limited: the near-Nyquist physiological alternation removed from the pose series (the censoring
series). Censored: percent of frames above {FD_SCRUBBING_THRESHOLD_MM} mm, raw / band-limited. Pose shares: percent of the frame-to-frame step variance on
the readout (B0-immune) and phase-encode axes held by the head pose / the applied acquisition term (eddy shift) / the removed design-locked term.
Image: percent of the per-voxel temporal variance at foreground voxels explained by the acquisition design (level intercepts, l = 2 and l = 4 direction
terms per shell), by the motion regressors after the design, and the residual, raw / corrected; r: correlation of the per-frame residual RMS with FD.</p>
<table><tr><th>case</th><th>T</th><th>FD mean / max</th><th>FD band-lim.</th><th>censored % raw / bl</th><th>readout: head / acq / design %</th>
<th>PE: head / acq / design %</th><th>drift % raw / corr</th><th>design % raw / corr</th><th>motion % raw / corr</th><th>residual % raw / corr</th><th>resid~FD r raw / corr</th><th>conv</th><th>s</th></tr>
{''.join(rows)}</table><img src="data:image/png;base64,{base64.b64encode(open(fig_path, 'rb').read()).decode()}" style="max-width:100%"></body></html>"""
    open(path, "w").write(html)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out")
    ap.add_argument("--only", default="bold,asl,asl60,dwi031,dwi11"); ap.add_argument("--device", default="auto")
    ap.add_argument("--shrinkage", type=float, default=0.0, help="pose_shrinkage in likelihood units (decision 7); 0 = off")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "ismrm_cases.json")
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for key in args.only.split(","):
        if key in results:
            continue
        rel, _ = CASES[key]
        path = rel if rel.startswith("/") else os.path.join(args.root, rel)
        results[key] = run_case(key, path, args.device, shrinkage=args.shrinkage)
        json.dump(results, open(out_json, "w"), indent=1, default=float)
        r = results[key]
        print(key, json.dumps({k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in ("T", "fd_mean", "fd_bandlimited_mean", "frac_gt_thr_raw", "frac_gt_thr_bandlimited", "seconds")}), flush=True)
    fig = os.path.join(args.out, "ismrm_cases_figure.png"); _figure(results, fig); _html(results, fig, os.path.join(args.out, "ismrm_cases.html"))
    print("DONE", os.path.join(args.out, "ismrm_cases.html"))


if __name__ == "__main__":
    main()
