#!/usr/bin/env python
"""
mbhard benchmark: do SIFT3D landmarks help (1) affine initialisation and (2) Sobolev SyN?

Part 1 — affine: standard robust_affine (A) vs least-squares affine on RANSAC-inlier landmark
         pairs (B) vs landmark affine refined by intensity (C).  Metrics: symmetric DKT31 Dice,
         and image similarity of the affinely resampled moving T1 vs fixed T1 (lncc, ncc, mattes_mi).
Part 2 — deformable: syntx.syn + Sobolev from affine A, single-channel cc2 baseline vs
         multi-channel landmark-cluster guidance.  Guidance mirrors sulcal guidance: K soft
         membership channels (k-means clusters of the inlier landmarks; Gaussian blobs at the
         fixed / matched moving positions; softmax-like normalisation across clusters), each
         scored with soft Dice, T1 with cc2.  Arms: K=8 w=0.2, K=8 w=0.5, K=16 w=0.2,
         oracle (label-agreeing inliers only), and landmark-affine (C) + guidance.

Outputs: results/mbhard_landmark_guidance.json, docs/reports/mbhard_landmark_guidance_report.html
"""
from __future__ import annotations

import argparse, base64, io, json, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import ants
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2

import syntx
from syntx.landmarks import spatial as S
from syntx.landmarks import preprocess_for_landmarks, match_sift3d_with_rotation_search
from syntx.robust_affine import robust_affine
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics
from syntx.image_compare import image_compare

OUT_JSON = "results/mbhard_landmark_guidance.json"
OUT_HTML = "docs/reports/mbhard_landmark_guidance_report.html"
WORK = "results/mbhard_landmark_guidance"


def fig_b64(fig, dpi=100):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=dpi, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def label_at(lbl, pts):
    idx = np.clip(np.rint(S.physical_to_vox(lbl, pts[:, :3])).astype(int), 0, np.array(lbl.shape) - 1)
    return lbl.numpy()[tuple(idx.T)].astype(int)


def write_affine(M4, path):
    """fixed-mm -> moving-mm 4x4 as an ITK AffineTransform (.mat) usable as a forward transform."""
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3,
                                    matrix=np.asarray(M4[:3, :3], float), offset=tuple(float(x) for x in M4[:3, 3]))
    ants.write_transform(tx, path)
    return path


def fit_affine_lstsq(src, dst):
    A = np.hstack([src, np.ones((len(src), 1))]); coef = np.linalg.lstsq(A, dst, rcond=None)[0]
    M = np.eye(4); M[:3, :3] = coef[:3].T; M[:3, 3] = coef[3]; return M


def similarity(fi, warped, mask):
    out = {}
    for m in ("lncc", "ncc", "mattes_mi", "mse"):
        try:
            out[m] = float(image_compare(fi, warped, m))
        except Exception as e:
            out[m] = f"n/a ({e})"
    a, b = fi.numpy()[mask], warped.numpy()[mask]
    out["ncc_foreground"] = float(np.corrcoef(a, b)[0, 1])
    return out


def eval_affine(name, mat, fi, mi, fl, ml, fi_p, mi_p, mask, t):
    d_fix, d_mov, d_sym = compute_bidirectional_dice(fl, ml, fi, mi, [mat], [mat], [True])
    warped = ants.apply_transforms(fixed=fi_p, moving=mi_p, transformlist=[mat], interpolator="linear")
    sim = similarity(fi_p, warped, mask)
    r = dict(name=name, mat=mat, dice_sym=float(d_sym), dice_fixed=float(d_fix), dice_moving=float(d_mov), time=t, **sim)
    print(f"  [{name}] Dice sym {d_sym:.4f} (fix {d_fix:.4f}, mov {d_mov:.4f}) | lncc {sim['lncc']} ncc {sim['ncc']} mi {sim['mattes_mi']} ncc_fg {sim['ncc_foreground']:.4f} | {t:.1f}s", flush=True)
    return r, warped


def build_cluster_channels(fi, mi, pf, pm, K, sigma_mm=3.0, seed=0):
    """K soft membership channels in fixed and moving space from matched landmark pairs."""
    np.random.seed(seed)
    K = int(min(K, len(pf)))
    cent, lab = kmeans2(pf.astype(np.float64), K, minit="++", seed=seed)
    def blobs(img, pts):
        chans = []
        for k in range(K):
            arr = np.zeros(img.shape, np.float32)
            idx = np.clip(np.rint(S.physical_to_vox(img, pts[lab == k])).astype(int), 0, np.array(img.shape) - 1)
            if len(idx):
                arr[tuple(idx.T)] = 1.0
            im = ants.smooth_image(ants.from_numpy(arr, origin=img.origin, spacing=img.spacing, direction=img.direction), sigma_mm)
            a = im.numpy(); a = a / (a.max() + 1e-12) if a.max() > 0 else a
            chans.append(a)
        stack = np.stack(chans)                                  # [K, ...]
        tot = stack.sum(0)
        stack = stack / np.maximum(1.0, tot)[None]              # softmax-like: memberships sum <= 1, background implicit
        return [ants.from_numpy(np.ascontiguousarray(c), origin=img.origin, spacing=img.spacing, direction=img.direction) for c in stack]
    return blobs(fi, pf), blobs(mi, pm), lab, cent


def run_syn(name, fi_p, mi_p, aff, fi, mi, fl, ml, chF=None, chM=None, w=0.0):
    fixed = fi_p if chF is None else [fi_p] + chF
    moving = mi_p if chM is None else [mi_p] + chM
    kw = dict(initial_transform=[aff], reg_iterations=[100, 100, 20], levels=[4, 2, 1], affine_iterations=0,
              regularizer="sobolev", sobolev_alpha=1.5, grad_step=0.25, verbose=0)
    if chF is None:
        kw["similarity_metric"] = "cc2"
    else:
        K = len(chF)
        kw["similarity_metric"] = ["cc2"] + ["dice"] * K
        kw["syn_metric_weights"] = [1.0 - w] + [w / K] * K
    t0 = time.time(); res = syntx.syn(fixed=fixed, moving=moving, **kw); dt = time.time() - t0
    fwd, inv = res["fwdtransforms"], res["invtransforms"]; winv = res.get("whichtoinvert_inv", [True, False])
    d_fix, d_mov, d_sym = compute_bidirectional_dice(fl, ml, fi, mi, fwd, inv, winv)
    jac = compute_jacobian_metrics(fi, ants.image_read(fwd[0]))
    r = dict(name=name, dice_sym=float(d_sym), dice_fixed=float(d_fix), dice_moving=float(d_mov),
             folding_pct=float(jac["folding_pct"]), min_jac=float(jac["min"]), time=dt, fwd=fwd)
    print(f"  [{name}] Dice sym {d_sym:.4f} (fix {d_fix:.4f}, mov {d_mov:.4f}) | folds {jac['folding_pct']:.5f}% minJ {jac['min']:.3f} | {dt:.0f}s", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-syn", action="store_true")
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True); os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    ds = syntx.benchmark_data("mbhard"); fi, mi, fl, ml = ds["fixed"], ds["moving"], ds["fixed_label"], ds["moving_label"]
    assert fi.shape[0] > 100
    t0 = time.time(); fi_p, mi_p = preprocess_for_landmarks(fi), preprocess_for_landmarks(mi); t_pre = time.time() - t0
    mask = fi_p.numpy() > 0.01
    print(f"preprocess {t_pre:.1f}s", flush=True)

    # landmarks -----------------------------------------------------------------
    t0 = time.time()
    lm = match_sift3d_with_rotation_search(fi_p, mi_p, max_keypoints=2000, threshold=0.002, n_scales=10,
                                           sigma_min=1.0, sigma_max=8.0, min_distance_mm=3.0)
    t_lm = time.time() - t0
    cf, cm, inl, M_lm = lm["coords_fixed"], lm["coords_moving"], lm["inliers"], lm["affine"].astype(np.float64)
    pf, pm = cf[inl[:, 0], :3].astype(np.float64), cm[inl[:, 1], :3].astype(np.float64)
    base = "/Users/stnava/data/mindboggle/volumes"
    fl_all = ants.image_read(f"{base}/NKI-TRT-20_volumes/NKI-TRT-20-2/labels.DKT31.manual+aseg.nii.gz")
    ml_all = ants.image_read(f"{base}/MMRR-21_volumes/MMRR-21-2/labels.DKT31.manual+aseg.nii.gz")
    lf, lmv = label_at(fl_all, pf), label_at(ml_all, pm)
    agree = (lf == lmv) & (lf > 0)
    print(f"landmarks: {len(cf)}/{len(cm)} kps, {len(inl)} inliers, label agreement {agree.mean():.2f}, {t_lm:.0f}s", flush=True)

    # Part 1 --------------------------------------------------------------------
    print("\nPart 1 — affine", flush=True)
    part1 = {}
    t0 = time.time(); ra = robust_affine(fi_p, mi_p, mode="auto", seed=42); tA = time.time() - t0
    matA = os.path.join(WORK, "affine_A_standard.mat"); import shutil; shutil.copyfile(ra["fwdtransforms"][0], matA)
    part1["A_standard"], wA = eval_affine("A standard robust_affine", matA, fi, mi, fl, ml, fi_p, mi_p, mask, tA)
    M_B = fit_affine_lstsq(pf, pm)
    matB = write_affine(M_B, os.path.join(WORK, "affine_B_landmark.mat"))
    part1["B_landmark"], wB = eval_affine("B landmark least-squares", matB, fi, mi, fl, ml, fi_p, mi_p, mask, t_lm)
    t0 = time.time()
    try:
        rc = robust_affine(fi_p, mi_p, initial_transform=matB, mode="auto", multi_start=False, seed=42)
        matC = os.path.join(WORK, "affine_C_landmark_refined.mat"); shutil.copyfile(rc["fwdtransforms"][0], matC)
        part1["C_landmark_refined"], wC = eval_affine("C landmark + intensity refine", matC, fi, mi, fl, ml, fi_p, mi_p, mask, time.time() - t0 + t_lm)
    except Exception as e:
        print("  arm C failed:", e); matC = None
    # A vs B difference
    A4 = np.eye(4); txA = ants.read_transform(matA); A4[:3, :3] = np.array(txA.parameters[:9]).reshape(3, 3); A4[:3, 3] = txA.parameters[9:12]
    from syntx.landmarks.orient import _project_to_so3, rotation_geodesic_deg
    rot_diff = float(rotation_geodesic_deg(_project_to_so3(A4[:3, :3])[None], _project_to_so3(M_B[:3, :3])[None])[0, 0])
    com = np.array(ants.get_center_of_mass(fi))
    trans_diff = float(np.linalg.norm((A4[:3, :3] @ com + A4[:3, 3]) - (M_B[:3, :3] @ com + M_B[:3, 3])))
    part1["A_vs_B"] = dict(rotation_diff_deg=rot_diff, com_mapping_diff_mm=trans_diff)
    print(f"  A vs B: rotation differs by {rot_diff:.2f} deg, COM mapping by {trans_diff:.2f} mm", flush=True)

    # Part 2 --------------------------------------------------------------------
    part2 = {}
    figs = {}
    if not args.skip_syn:
        print("\nPart 2 — Sobolev SyN", flush=True)
        part2["baseline"] = run_syn("baseline cc2", fi_p, mi_p, matA, fi, mi, fl, ml)
        ch8F, ch8M, lab8, _ = build_cluster_channels(fi_p, mi_p, pf, pm, 8)
        part2["K8_w0.2"] = run_syn("guided K=8 w=0.2", fi_p, mi_p, matA, fi, mi, fl, ml, ch8F, ch8M, 0.2)
        part2["K8_w0.5"] = run_syn("guided K=8 w=0.5", fi_p, mi_p, matA, fi, mi, fl, ml, ch8F, ch8M, 0.5)
        ch16F, ch16M, _, _ = build_cluster_channels(fi_p, mi_p, pf, pm, 16)
        part2["K16_w0.2"] = run_syn("guided K=16 w=0.2", fi_p, mi_p, matA, fi, mi, fl, ml, ch16F, ch16M, 0.2)
        chOF, chOM, _, _ = build_cluster_channels(fi_p, mi_p, pf[agree], pm[agree], 8)
        part2["oracle_K8_w0.2"] = run_syn(f"oracle ({int(agree.sum())} label-agreeing) K=8 w=0.2", fi_p, mi_p, matA, fi, mi, fl, ml, chOF, chOM, 0.2)
        if matC is not None:
            part2["affC_K8_w0.2"] = run_syn("landmark affine C + guided K=8 w=0.2", fi_p, mi_p, matC, fi, mi, fl, ml, ch8F, ch8M, 0.2)
        # figures: cluster channels overlay, label alignment baseline vs best guided
        sl = S.extract_ortho_slices(fi_p)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
        cmap = plt.get_cmap("tab10")
        for ax, view in zip(axes, ("ax", "cor", "sag")):
            spec = sl["views"][view]; ax.imshow(sl[view], cmap="gray", origin="lower")
            for k in range(8):
                u, v, m = S.project_to_slice(fi_p, pf[lab8 == k], view=spec, slab_half_mm=10.0)
                ax.scatter(u[m], v[m], s=10, color=cmap(k), label=f"c{k}" if view == "ax" else None)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{spec['name']} — inlier landmarks coloured by cluster (K=8)", fontsize=9)
        figs["clusters"] = fig_b64(fig)
        try:
            from syntx.viz import render_label_alignment_figure
            best = max((v for k, v in part2.items() if k != "baseline"), key=lambda r: r["dice_sym"])
            for key, r in (("baseline", part2["baseline"]), ("best_guided", best)):
                mlw = ants.apply_transforms(fixed=fi, moving=ml, transformlist=r["fwd"], interpolator="nearestNeighbor")
                p = os.path.join(WORK, f"labels_{key}.png")
                render_label_alignment_figure(fl, mlw, fixed_image=fi, output_path=p, title=f"{r['name']}  Dice={r['dice_sym']:.4f}")
                figs[key] = base64.b64encode(open(p, "rb").read()).decode()
        except Exception as e:
            print("  [viz]", e)

    # save ------------------------------------------------------------------------
    out = dict(landmarks=dict(n_fixed=int(len(cf)), n_moving=int(len(cm)), n_inliers=int(len(inl)), label_agreement=float(agree.mean()),
                              rotation_search_deg=float(lm["rotation_deg"]), time=t_lm),
               part1=part1, part2={k: {kk: vv for kk, vv in v.items() if kk != "fwd"} for k, v in part2.items()},
               settings=dict(preprocess="NLM denoise (antstorch, MPS), no N4, 2-98% norm", syn="levels [4,2,1], iters [100,100,20], sobolev alpha 1.5, grad_step 0.25",
                             guidance="k-means clusters of inlier landmarks, 3 mm Gaussian blobs, softmax-like normalisation, soft Dice per cluster channel"))
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2, default=str)

    def tr(*c): return "<tr>" + "".join(f"<td>{x}</td>" for x in c) + "</tr>"
    def f4(x): return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)
    p1rows = "".join(tr(v["name"], f4(v["dice_sym"]), f4(v["dice_fixed"]), f4(v["dice_moving"]), f4(v["lncc"]), f4(v["ncc"]), f4(v["mattes_mi"]), f4(v["ncc_foreground"]), f"{v['time']:.0f}")
                     for k, v in part1.items() if k != "A_vs_B")
    p2rows = "".join(tr(v["name"], f4(v["dice_sym"]), f4(v["dice_fixed"]), f4(v["dice_moving"]), f"{v['dice_sym'] - part2['baseline']['dice_sym']:+.4f}" if "baseline" in part2 else "",
                        f"{v['folding_pct']:.5f}", f"{v['min_jac']:.3f}", f"{v['time']:.0f}") for v in part2.values())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>mbhard landmark guidance benchmark</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}table{{border-collapse:collapse;margin:8px 0}}
td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}}th{{background:#f2f2f2}}img{{max-width:100%}}code{{background:#f4f4f4;padding:1px 4px}}</style></head><body>
<h1>mbhard — do SIFT3D landmarks help affine initialisation and Sobolev SyN?</h1>
<p>Generated {time.strftime('%Y-%m-%d %H:%M')} by <code>scripts/benchmark_mbhard_landmark_guidance.py</code>. Pair: NKI-TRT-20-2 (fixed) → MMRR-21-2 (moving), DKT31 manual labels.
Preprocessing: {out['settings']['preprocess']}. Landmarks: <code>match_sift3d_with_rotation_search</code> → {len(inl)} RANSAC-affine inlier pairs
({agree.mean():.0%} share a whole-brain label; global rotation {lm['rotation_deg']:.1f}°), {t_lm:.0f} s.</p>
<h2>Part 1 — affine initialisation</h2>
<p>A: <code>robust_affine(mode='auto')</code>. B: least-squares affine on the inlier landmark pairs. C: B refined by intensity (<code>robust_affine(initial_transform=B, multi_start=False)</code>).
Similarity is measured between the fixed T1 and the affinely resampled moving T1 (image_compare; ncc_foreground = Pearson on fixed foreground). Dice = bidirectional symmetric DKT31.</p>
<table><tr><th>affine</th><th>Dice sym</th><th>Dice fixed</th><th>Dice moving</th><th>lncc</th><th>ncc</th><th>mattes MI</th><th>ncc (fg)</th><th>time s</th></tr>{p1rows}</table>
<p>A vs B: rotation differs by {rot_diff:.2f}°, centre-of-mass mapping by {trans_diff:.2f} mm.</p>
<h2>Part 2 — Sobolev SyN with landmark-cluster guidance</h2>
<p>All arms start from affine A unless stated; {out['settings']['syn']}. Guidance: {out['settings']['guidance']}; channels = [T1 (cc2)] + K cluster maps (soft Dice), weights [1−w] + [w/K]·K.</p>
<table><tr><th>arm</th><th>Dice sym</th><th>Dice fixed</th><th>Dice moving</th><th>Δ vs baseline</th><th>folding %</th><th>min J</th><th>time s</th></tr>{p2rows}</table>
{''.join(f'<img src="data:image/png;base64,{b}">' for b in figs.values())}
</body></html>"""
    open(OUT_HTML, "w").write(html)
    print("wrote", OUT_JSON, OUT_HTML, flush=True)


if __name__ == "__main__":
    main()
