"""Ten-minute run answering the open questions after the Newton unit fix (2026-10-28). Real series on the default device,
phantoms on CPU in a parallel process. Writes <out>/ten_minute.json, .html and figures.

Q1 projected engine on real DWI (BLAST031, SOCOM-11): readout step vs joint, and the fraction of its readout translation
   explained by the gradient terms (g, g g^T). Rule: within 0.02 mm of joint and < 30 % explained -> the earlier excess was the bug.
Q2 roughness prior (Charbonnier, curvature units 0.1) on BLAST031 joint. Rule: readout step unchanged (+-0.02), FD down, alignment within 0.5 %.
Q3 roughness 0.1 + shrinkage 0.1 together on BLAST031 joint.
Q4 phantom table (truth): unified, joint, projected, joint+shrink 0.1, joint+rough 0.1 on the ASL and DWI phantoms.
"""
from __future__ import annotations

import base64
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from motion_cohort_gate import _pe_axis, _signatures  # noqa: E402

REAL = {
    "dwi031": "/private/tmp/claude-501/-Users-stnava-data-repos-syntx/ccf9c564-6e2c-4844-bc09-c6d5c2c8a889/scratchpad/motion_cohort_v2/motion_cohort/FPA/sub-BLAST031/ses-01/dwi/sub-BLAST031_ses-01_run-001_dir-AP_dwi.nii.gz",
    "dwi11": "/Users/stnava/data/blast_cohorts/BIDS/SOCOM/sub-Blast-11/ses-01/dwi/sub-Blast-11_ses-01_run-001_dir-PA_dwi.nii.gz",
}
RUNS = [("dwi031", "joint", {}), ("dwi031", "projected", {}), ("dwi031", "joint rough 0.1", {"surround_weight": 0.1, "surround_scale": "curvature"}),
        ("dwi031", "joint rough 0.1 + shrink 0.1", {"surround_weight": 0.1, "surround_scale": "curvature", "pose_shrinkage": 0.1}),
        ("dwi11", "joint", {}), ("dwi11", "projected", {})]


def r2(y, X):
    X = np.c_[np.ones(len(y)), X]; b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - ((y - X @ b) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))


def phantoms(out):
    from syntx.motion_simulation import evaluate_motion_estimate, simulate_asl_series, simulate_dwi_series
    from syntx.motion_design import make_design, motion_correct_design
    simA = simulate_asl_series(n_frames=24, seed=1); vt = ["control" if i % 2 == 0 else "label" for i in range(24)]; dA = make_design("tag_control", volume_types=vt)
    simD = simulate_dwi_series(16, seed=42, eddy_mm_per_unit=0.4); dD = make_design("shells", bvals=simD.extra["bvals"], bvecs=simD.extra["bvecs"], pe_axis=simD.pe_axis)
    rows = {}
    for lab, eng, kw in (("unified", "unified", {}), ("joint", "joint", {}), ("projected", "projected", {}),
                         ("joint shrink 0.1", "joint", {"pose_shrinkage": 0.1}), ("joint rough 0.1", "joint", {"surround_weight": 0.1, "surround_scale": "curvature"})):
        eA = evaluate_motion_estimate(motion_correct_design(simA.image, dA, engine=eng, device="cpu", verbose=False, **kw)["fwdtransforms"], simA)
        eD = evaluate_motion_estimate(motion_correct_design(simD.image, dD, engine=eng, device="cpu", verbose=False, **kw)["fwdtransforms"], simD)
        rows[lab] = {"asl_fd_err": eA["fd_error_mm"], "asl_rot": eA["rotation_rms_deg"], "dwi_fd_err": eD["fd_error_mm"], "dwi_rot": eD["rotation_rms_deg"]}
        print("phantom", lab, json.dumps({k: round(v, 3) for k, v in rows[lab].items()}), flush=True)
    json.dump(rows, open(os.path.join(out, "phantoms.json"), "w"), indent=1)


def main(out):
    import ants
    import syntx
    os.makedirs(out, exist_ok=True)
    proc = mp.get_context("spawn").Process(target=phantoms, args=(out,)); proc.start()
    res = {}
    imgs = {}
    for key, label, kw in RUNS:
        path = REAL[key]
        if key not in imgs:
            img = ants.image_read(path); bv = np.loadtxt(path.replace(".nii.gz", ".bval")); g = np.loadtxt(path.replace(".nii.gz", ".bvec")).T
            pe = _pe_axis(path, img); imgs[key] = (img, bv, g, pe, syntx.make_design("shells", bvals=bv, bvecs=g, pe_axis=pe))
        img, bv, g, pe, design = imgs[key]; levels = np.asarray(design.levels); D = np.asarray(img.direction)[:3, :3]
        t0 = time.time(); eng = "projected" if label == "projected" else "joint"
        r = syntx.motion_correct_design(img, design, engine=eng, verbose=False, **kw)
        sig = _signatures(r, img, levels); t = np.asarray(r["translations_mm"], float); ro = t @ D[:, 0]
        gu = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9); gu[bv < 50] = 0; dw = bv > 50
        G = gu[dw]; GG = np.c_[G[:, 0] ** 2, G[:, 1] ** 2, G[:, 2] ** 2, G[:, 0] * G[:, 1], G[:, 0] * G[:, 2], G[:, 1] * G[:, 2]]
        res.setdefault(key, {})[label] = {"fd_mean": sig["fd_mean"], "step": sig["step_mad_mm"], "tsd": sig["tsd_rel"], "conv": sig["frame_converged_frac"], "seconds": time.time() - t0,
                                           "readout_r2_odd_g": r2(ro[dw], G), "readout_r2_even_gg": r2(ro[dw], GG), "readout_r2_both": r2(ro[dw], np.c_[G, GG]), "t_ro": ro.tolist(), "fd": np.asarray(r["fd"], float).tolist()}
        print(key, label, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[key][label].items() if k in ("fd_mean", "tsd", "readout_r2_odd_g", "readout_r2_even_gg", "seconds")}), "ro %.3f pe %.3f" % (sig["step_mad_mm"]["t_ro"], sig["step_mad_mm"]["t_pe"]), flush=True)
        json.dump(res, open(os.path.join(out, "ten_minute.json"), "w"), indent=1, default=float)
    proc.join()
    ph = json.load(open(os.path.join(out, "phantoms.json")))
    report(res, ph, out)


def report(res, ph, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(15, 7))
    for i, key in enumerate(("dwi031", "dwi11")):
        for label, r in res.get(key, {}).items():
            axes[i].plot(r["t_ro"], lw=0.8, label=f"{label}: readout step {r['step']['t_ro']:.3f} mm, R2(g g^T) {r['readout_r2_even_gg']:.2f}")
        axes[i].set_title(f"{key}: readout translation (B0-immune axis)", fontsize=9); axes[i].legend(fontsize=7); axes[i].set_ylabel("mm")
    fig.tight_layout(); fp = os.path.join(out, "readout.png"); fig.savefig(fp, dpi=110); plt.close(fig)
    j = res["dwi031"]["joint"]
    def verdict_q1(key):
        p = res[key]["projected"]; jj = res[key]["joint"]
        ok = abs(p["step"]["t_ro"] - jj["step"]["t_ro"]) <= 0.02 and p["readout_r2_even_gg"] < 0.3
        return f"{key}: projected readout {p['step']['t_ro']:.3f} vs joint {jj['step']['t_ro']:.3f} mm; R2 even-in-g {p['readout_r2_even_gg']:.2f} (joint {jj['readout_r2_even_gg']:.2f}) -> {'the earlier excess was the Newton bug' if ok else 'an excess remains'}"
    q2 = res["dwi031"]["joint rough 0.1"]; q3 = res["dwi031"]["joint rough 0.1 + shrink 0.1"]
    rows = "".join(f"<tr><td>{k}</td><td>{l}</td><td>{r['fd_mean']:.3f}</td><td>{r['step']['t_ro']:.3f}</td><td>{r['step']['t_pe']:.3f}</td><td>{r['step']['t_sl']:.3f}</td><td>{r['tsd']:.4f}</td><td>{r['readout_r2_odd_g']:.2f}</td><td>{r['readout_r2_even_gg']:.2f}</td><td>{r['conv']:.2f}</td><td>{r['seconds']:.0f}</td></tr>" for k in res for l, r in res[k].items())
    prow = "".join(f"<tr><td>{l}</td><td>{v['asl_fd_err']:.3f}</td><td>{v['asl_rot']:.3f}</td><td>{v['dwi_fd_err']:.3f}</td><td>{v['dwi_rot']:.3f}</td></tr>" for l, v in ph.items())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Ten-minute questions</title>
<style>body{{font-family:sans-serif;margin:2em;max-width:1200px}} table{{border-collapse:collapse;margin:6px 0}} td,th{{border:1px solid #999;padding:3px 6px;font-size:12px}} th{{background:#eee}}</style></head>
<body><h1>Open questions after the Newton unit fix</h1>
<h2>Q1: projected engine on real DWI</h2><p>{verdict_q1('dwi031')}<br>{verdict_q1('dwi11')}</p>
<h2>Q2: roughness prior 0.1 (curvature units) on BLAST031 joint</h2><p>readout step {j['step']['t_ro']:.3f} -> {q2['step']['t_ro']:.3f} mm; FD {j['fd_mean']:.3f} -> {q2['fd_mean']:.3f}; alignment tsd {j['tsd']:.4f} -> {q2['tsd']:.4f} ({100 * (q2['tsd'] / j['tsd'] - 1):+.2f} %)
-> {'acts as designed' if abs(q2['step']['t_ro'] - j['step']['t_ro']) <= 0.02 and q2['fd_mean'] < j['fd_mean'] and q2['tsd'] <= 1.005 * j['tsd'] else 'rule not met'}</p>
<h2>Q3: roughness 0.1 + shrinkage 0.1</h2><p>readout {q3['step']['t_ro']:.3f}, FD {q3['fd_mean']:.3f}, tsd {q3['tsd']:.4f} ({100 * (q3['tsd'] / j['tsd'] - 1):+.2f} %)</p>
<table><tr><th>series</th><th>engine</th><th>FD</th><th>step readout</th><th>step PE</th><th>step slice</th><th>tsd</th><th>R2 readout~g</th><th>R2 readout~g g^T</th><th>conv</th><th>s</th></tr>{rows}</table>
<img src="data:image/png;base64,{base64.b64encode(open(fp, 'rb').read()).decode()}" style="max-width:100%">
<h2>Q4: phantoms with known motion (CPU; head-FD error mm, rotation RMS deg)</h2>
<table><tr><th>engine</th><th>ASL FD err</th><th>ASL rot</th><th>DWI FD err</th><th>DWI rot</th></tr>{prow}</table></body></html>"""
    open(os.path.join(out, "ten_minute.html"), "w").write(html); print("DONE", os.path.join(out, "ten_minute.html"))


if __name__ == "__main__":
    main(sys.argv[1])
