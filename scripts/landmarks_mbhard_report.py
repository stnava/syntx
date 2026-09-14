#!/usr/bin/env python
"""
Landmark detection / matching / physical-space display report on Mindboggle `mbhard`.

Two examples in one HTML report (all coordinates in physical LPS mm via syntx.landmarks.spatial):

  Example 1 — native pair: fixed (NKI-TRT-20-2, LAS array, 1 mm) vs moving (MMRR-21-2, RPS array, 1.2 mm x).
  Example 2 — same pair after resampling fixed by +30° and moving by −30° about the superior (z) axis
              through each centre of mass.  The recovered fixed→moving affine is compared with the one
              predicted from Example 1 and the known rotations:  M2 ≈ T_m⁻¹ ∘ M1 ∘ T_f.

Pipeline per example:
  GPU/MPS preprocessing (ANTsTorch N4 + NLM + foreground 2–98% normalisation)
  → detect_sift3d (volumetric DoG in mm scale, physical-space gradient descriptors)
  → match_landmarks (Lowe ratio + mutual NN) → ransac_filter (affine)
  → score with DKT31+aseg whole-brain labels (fraction of matched pairs sharing a label)
  → figures: medical-standard orthogonal views with keypoints; side-by-side match lines.

Usage:  python scripts/landmarks_mbhard_report.py [--out reports/mbhard_spatial_report.html]
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import ants                       # noqa: E402
import matplotlib                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

import syntx                      # noqa: E402
from syntx.landmarks import spatial as S   # noqa: E402
from syntx.landmarks import (     # noqa: E402
    preprocess_for_landmarks, detect_sift3d, match_landmarks, ransac_filter,
    match_sift3d_with_rotation_search,
)

CONVENTION = "radiological"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fig_to_b64(fig, dpi=110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def label_at(lbl_img, pts_mm) -> np.ndarray:
    idx = np.rint(S.physical_to_vox(lbl_img, pts_mm[:, :3])).astype(int)
    arr = lbl_img.numpy()
    idx = np.clip(idx, 0, np.array(arr.shape) - 1)
    return arr[tuple(idx.T)].astype(int)


def rotation_about_z(image, degrees: float):
    """ANTs affine rotating physical content by `degrees` about the +z axis through the centre of mass."""
    ctr = np.array(ants.get_center_of_mass(image), dtype=np.float64)
    a = np.deg2rad(degrees)
    R = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    tx.set_parameters(np.concatenate([R.ravel(), np.zeros(3)]))
    tx.set_fixed_parameters(ctr)
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = ctr - R @ ctr           # x -> R (x - c) + c
    return tx, M


def rotation_angle_deg(A: np.ndarray) -> float:
    """Angle of the rotation part of a 3x3 linear map (polar decomposition)."""
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def show_view(ax, img, sl, view, pts_mm=None, color="lime", slab=6.0, title=None, size=10):
    spec = sl["views"][view]
    ext = [0, spec["n_u"] * spec["spacing_u"], 0, spec["n_v"] * spec["spacing_v"]]
    ax.imshow(sl[view], cmap="gray", origin="lower", extent=ext, interpolation="nearest")
    if pts_mm is not None and len(pts_mm):
        u, v, m = S.project_to_slice(img, pts_mm[:, :3], view=spec, slab_half_mm=slab)
        ax.scatter((u[m] + 0.5) * spec["spacing_u"], (v[m] + 0.5) * spec["spacing_v"],
                   s=size, facecolors="none", edgecolors=color, linewidths=0.7)
    lab = spec["labels"]
    kw = dict(color="yellow", fontsize=9, fontweight="bold")
    ax.text(0.01, 0.5, lab["left"], transform=ax.transAxes, ha="left", va="center", **kw)
    ax.text(0.99, 0.5, lab["right"], transform=ax.transAxes, ha="right", va="center", **kw)
    ax.text(0.5, 0.01, lab["bottom"], transform=ax.transAxes, ha="center", va="bottom", **kw)
    ax.text(0.5, 0.99, lab["top"], transform=ax.transAxes, ha="center", va="top", **kw)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return spec


def side_by_side_matches(fi, mi, sl_f, sl_m, view, cf, cm, matches, agree, slab=8.0, title=""):
    spec_f, spec_m = sl_f["views"][view], sl_m["views"][view]
    wf, hf = spec_f["n_u"] * spec_f["spacing_u"], spec_f["n_v"] * spec_f["spacing_v"]
    wm, hm = spec_m["n_u"] * spec_m["spacing_u"], spec_m["n_v"] * spec_m["spacing_v"]
    gap = 20.0
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.imshow(sl_f[view], cmap="gray", origin="lower", extent=[0, wf, 0, hf], interpolation="nearest")
    ax.imshow(sl_m[view], cmap="gray", origin="lower", extent=[wf + gap, wf + gap + wm, 0, hm], interpolation="nearest")
    uf, vf, mf_ = S.project_to_slice(fi, cf[matches[:, 0], :3], view=spec_f, slab_half_mm=slab)
    um, vm, _ = S.project_to_slice(mi, cm[matches[:, 1], :3], view=spec_m, slab_half_mm=1e9)
    n_drawn = 0
    for i in np.where(mf_)[0]:
        x0, y0 = (uf[i] + 0.5) * spec_f["spacing_u"], (vf[i] + 0.5) * spec_f["spacing_v"]
        x1, y1 = wf + gap + (um[i] + 0.5) * spec_m["spacing_u"], (vm[i] + 0.5) * spec_m["spacing_v"]
        c = "lime" if agree[i] else "orangered"
        ax.plot([x0, x1], [y0, y1], "-", color=c, lw=0.6, alpha=0.8)
        ax.plot([x0, x1], [y0, y1], "o", color=c, ms=2.5, mfc="none")
        n_drawn += 1
    for spec, x0 in ((spec_f, 0.0), (spec_m, wf + gap)):
        lab = spec["labels"]
        w = spec["n_u"] * spec["spacing_u"]; h = spec["n_v"] * spec["spacing_v"]
        kw = dict(color="yellow", fontsize=9, fontweight="bold")
        ax.text(x0 + 2, h / 2, lab["left"], ha="left", va="center", **kw)
        ax.text(x0 + w - 2, h / 2, lab["right"], ha="right", va="center", **kw)
        ax.text(x0 + w / 2, 2, lab["bottom"], ha="center", va="bottom", **kw)
        ax.text(x0 + w / 2, h - 2, lab["top"], ha="center", va="top", **kw)
    ax.set_xlim(0, wf + gap + wm); ax.set_ylim(0, max(hf, hm))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}  —  fixed (left) vs moving (right), {n_drawn} inlier matches with fixed keypoint within ±{slab:g} mm of the slice; "
                 f"green = same label, red = different / unlabelled", fontsize=9)
    return fig, n_drawn


def tr(*cells):
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


# ---------------------------------------------------------------------------
# one example
# ---------------------------------------------------------------------------

def run_example(tag, fi, mi, fl, ml, args, fig_offset, rotation_invariant=False, render=True, pre=None,
                rotation_search=False):
    """Preprocess, detect, match, score and (optionally) render one fixed/moving pair.
    Returns (stats dict, html).  ``pre`` = cached (fi_p, mi_p, timings) to skip preprocessing.
    ``rotation_search=True`` uses ``match_sift3d_with_rotation_search`` (PCA-seeded iterative
    global rotation estimate, coarse-to-fine) instead of plain axis-aligned matching."""
    if pre is None:
        timings = {}
        t = time.time(); fi_p = preprocess_for_landmarks(fi); timings["preprocess fixed"] = time.time() - t
        t = time.time(); mi_p = preprocess_for_landmarks(mi); timings["preprocess moving"] = time.time() - t
    else:
        fi_p, mi_p, timings = pre[0], pre[1], dict(pre[2])

    det_kw = dict(preprocess=False, max_keypoints=args.max_keypoints, threshold=args.threshold,
                  n_scales=args.n_scales, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                  min_distance_mm=args.min_dist, rotation_invariant=rotation_invariant)
    t = time.time()
    search = None
    if rotation_search:
        search = match_sift3d_with_rotation_search(fi_p, mi_p, ratio_thresh=args.ratio, mutual=not args.no_mutual,
                                                   ransac_model="affine", inlier_thresh_mm=args.inlier_mm,
                                                   **{k: v for k, v in det_kw.items() if k != "rotation_invariant"})
        cf, df, cm, dm = search["coords_fixed"], search["descs_fixed"], search["coords_moving"], search["descs_moving"]
        timings["sift3d + rotation search (both)"] = time.time() - t
    else:
        cf, df = detect_sift3d(fi_p, **det_kw)
        cm, dm = detect_sift3d(mi_p, **det_kw)
        timings["sift3d (both)"] = time.time() - t

    inside = {}
    for name, im, c in (("fixed", fi_p, cf), ("moving", mi_p, cm)):
        idx = np.rint(S.physical_to_vox(im, c[:, :3])).astype(int)
        ok = (idx >= 0).all(1) & (idx < np.array(im.shape)).all(1)
        vals = np.zeros(len(c)); vals[ok] = im.numpy()[tuple(idx[ok].T)]
        inside[name] = float(np.mean(ok & (vals > 0)))

    t = time.time()
    if search is not None:
        m0, mf, M = search["matches"], search["inliers"], search["affine"]
    else:
        m0 = match_landmarks(cf, cm, df, dm, ratio_thresh=args.ratio, mutual=not args.no_mutual)
        mf, M = ransac_filter(cf, cm, m0, model="affine", inlier_thresh_mm=args.inlier_mm, min_inliers=6)
    timings["match + RANSAC"] = time.time() - t

    lab_f, lab_m = label_at(fl, cf), label_at(ml, cm)
    labeled = (float(np.mean(lab_f > 0)), float(np.mean(lab_m > 0)))
    agree0 = (lab_f[m0[:, 0]] == lab_m[m0[:, 1]]) & (lab_f[m0[:, 0]] > 0)
    agree = (lab_f[mf[:, 0]] == lab_m[mf[:, 1]]) & (lab_f[mf[:, 0]] > 0)
    rng = np.random.default_rng(0)
    rp = np.column_stack([rng.integers(0, len(cf), 20000), rng.integers(0, len(cm), 20000)])
    agree_rand = (lab_f[rp[:, 0]] == lab_m[rp[:, 1]]) & (lab_f[rp[:, 0]] > 0)
    pred = (M[:3, :3] @ cf[mf[:, 0], :3].T).T + M[:3, 3]
    resid = np.linalg.norm(pred - cm[mf[:, 1], :3], axis=1)
    com_f, com_m = np.array(ants.get_center_of_mass(fi)), np.array(ants.get_center_of_mass(mi))
    com_err = float(np.linalg.norm(M[:3, :3] @ com_f + M[:3, 3] - com_m))
    sv = np.linalg.svd(M[:3, :3], compute_uv=False)
    if rotation_search:
        desc_name = (f"axis-aligned in a globally estimated frame (PCA-seeded iterative search, winner "
                     f"'{search['winner']}', rotation {search['rotation_deg']:.1f}°)")
    else:
        desc_name = "rotation-invariant (local structure-tensor frame)" if rotation_invariant else "scanner-axis-aligned"
    print(f"[{tag} | {desc_name}] kps {len(cf)}/{len(cm)} inside {inside}; matches {len(m0)} -> inliers {len(mf)}; "
          f"label agree raw {agree0.mean():.2f} inliers {agree.mean():.2f} chance {agree_rand.mean():.3f}; "
          f"resid median {np.median(resid):.2f} mm; COM err {com_err:.1f} mm; sv {np.round(sv, 3)}; "
          f"rotation {rotation_angle_deg(M[:3, :3]):.1f} deg")

    stats = dict(M=M, n_kp=(len(cf), len(cm)), n_match=len(m0), n_inlier=len(mf),
                 agree=float(agree.mean()), agree0=float(agree0.mean()), chance=float(agree_rand.mean()),
                 resid_med=float(np.median(resid)), com_err=com_err, sv=sv,
                 rot=rotation_angle_deg(M[:3, :3]), desc=desc_name, pre=(fi_p, mi_p, timings),
                 search=None if search is None else {k: search[k] for k in ("winner", "rotation_deg", "candidates")})
    if not render:
        return stats, ""

    # figures
    sl_f = S.extract_ortho_slices(fi_p, convention=CONVENTION)
    sl_m = S.extract_ortho_slices(mi_p, convention=CONVENTION)
    k = fig_offset
    fig1, axes = plt.subplots(2, 3, figsize=(12, 8))
    for r, (name, im, sl, c) in enumerate((("fixed", fi_p, sl_f, cf), ("moving", mi_p, sl_m, cm))):
        for j, view in enumerate(("ax", "cor", "sag")):
            show_view(axes[r, j], im, sl, view, c, slab=4.0,
                      title=f"{name} — {sl['views'][view]['name']}  (array {S.axis_orientation_code(im)})")
    fig1.suptitle(f"Figure {k+1} [{tag}] — SIFT3D keypoints within ±4 mm of the centre-of-mass slices, {CONVENTION} convention "
                  f"(patient Left on viewer's right); native arrays, no reorientation", fontsize=10)
    img1 = fig_to_b64(fig1)
    fig2, _ = side_by_side_matches(fi_p, mi_p, sl_f, sl_m, "ax", cf, cm, mf, agree, title=f"Figure {k+2} [{tag}] — axial")
    img2 = fig_to_b64(fig2)
    fig3, _ = side_by_side_matches(fi_p, mi_p, sl_f, sl_m, "cor", cf, cm, mf, agree, title=f"Figure {k+3} [{tag}] — coronal")
    img3 = fig_to_b64(fig3)
    fig4, ax4 = plt.subplots(1, 2, figsize=(10, 3.4))
    ax4[0].hist(resid, bins=25, color="steelblue"); ax4[0].set_xlabel("affine residual of inlier match (mm)"); ax4[0].set_ylabel("count")
    ax4[0].set_title("RANSAC affine residuals", fontsize=9)
    ax4[1].bar(["random pairs", "ratio+mutual matches", "RANSAC inliers"],
               [agree_rand.mean(), agree0.mean(), agree.mean()], color=["gray", "orange", "seagreen"])
    ax4[1].set_ylim(0, 1); ax4[1].set_ylabel("fraction same label"); ax4[1].set_title("Whole-brain label agreement of matched pairs", fontsize=9)
    fig4.suptitle(f"Figure {k+4} [{tag}]", fontsize=10)
    img4 = fig_to_b64(fig4)

    geom_rows = []
    for name, im in (("fixed", fi), ("moving", mi)):
        org, sp, D = S.get_image_affine(im)
        geom_rows.append((name, im.shape, np.round(sp, 2).tolist(), np.round(org, 1).tolist(),
                          S.axis_orientation_code(im), np.round(ants.get_center_of_mass(im), 1).tolist()))

    search_html = ""
    if search is not None:
        rows = "".join(tr(c["name"], f"{c['history'][0]['rotation_deg']:.1f}", f"{c['final_rotation_deg']:.1f}",
                          c["iterations"], " → ".join(f"{h['n_inliers']}@{int(100*h['fraction'])}%" for h in c["history"]))
                       for c in search["candidates"])
        search_html = ("<h3>Global rotation search</h3><p>Hypotheses fixed→moving: identity plus the four principal-axis "
                       "alignments of the two landmark clouds. Each is refined iteratively (descriptors rebuilt in the "
                       "current frame → match → rigid RANSAC → rotation), coarse-to-fine over the strongest 25% → 50% → 100% "
                       "of keypoints; the converged hypothesis with most rigid inliers wins.</p>"
                       "<table><tr><th>hypothesis</th><th>start (deg)</th><th>converged (deg)</th><th>iterations</th>"
                       "<th>rigid inliers per iteration (@ keypoint fraction)</th></tr>" + rows + "</table>")
    html = f"""
<p><b>Descriptor / matching:</b> {desc_name}.</p>
{search_html}
<h3>Geometry</h3>
<table><tr><th>image</th><th>shape</th><th>spacing (mm)</th><th>origin (mm)</th><th>array axes point to</th><th>centre of mass (mm)</th></tr>
{''.join(tr(*r) for r in geom_rows)}</table>
<h3>Timings</h3>
<table><tr><th>step</th><th>seconds</th></tr>{''.join(tr(kk, f'{v:.1f}') for kk, v in timings.items())}</table>
<h3>Results</h3>
<table><tr><th>quantity</th><th>fixed</th><th>moving</th></tr>
{tr('SIFT3D keypoints', len(cf), len(cm))}
{tr('fraction of keypoints inside brain foreground', f"{inside['fixed']:.3f}", f"{inside['moving']:.3f}")}
{tr('fraction of keypoints carrying a whole-brain label', f'{labeled[0]:.2f}', f'{labeled[1]:.2f}')}</table>
<table><tr><th>quantity</th><th>value</th></tr>
{tr('ratio-test + mutual matches', len(m0))}{tr('RANSAC affine inliers', len(mf))}
{tr('label agreement — random pairs (chance)', f'{agree_rand.mean():.3f}')}
{tr('label agreement — ratio+mutual matches', f'{agree0.mean():.3f}')}
{tr('label agreement — RANSAC inliers', f'{agree.mean():.3f}')}
{tr('median / max affine residual on inliers (mm)', f'{np.median(resid):.2f} / {resid.max():.2f}')}
{tr('centre-of-mass prediction error of recovered affine (mm)', f'{com_err:.1f}')}
{tr('singular values of recovered linear part', np.round(sv, 3).tolist())}
{tr('rotation angle of recovered linear part (deg)', f'{rotation_angle_deg(M[:3, :3]):.1f}')}</table>
<img src="data:image/png;base64,{img1}">
<img src="data:image/png;base64,{img2}">
<img src="data:image/png;base64,{img3}">
<img src="data:image/png;base64,{img4}">
"""
    return stats, html


def variant_table(rows):
    """Compact comparison of descriptor variants: list of (label, stats)."""
    head = "<tr><th>descriptor</th><th>keypoints f/m</th><th>matches</th><th>RANSAC inliers</th><th>inlier label agreement (chance)</th><th>median resid (mm)</th><th>COM err (mm)</th><th>singular values</th><th>rotation (deg)</th></tr>"
    body = "".join(tr(lab, f"{st['n_kp'][0]}/{st['n_kp'][1]}", st['n_match'], st['n_inlier'],
                      f"{st['agree']:.2f} ({st['chance']:.2f})", f"{st['resid_med']:.2f}", f"{st['com_err']:.1f}",
                      np.round(st['sv'], 3).tolist(), f"{st['rot']:.1f}") for lab, st in rows)
    return f"<table>{head}{body}</table>"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/mbhard_spatial_report.html")
    ap.add_argument("--max-keypoints", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.002)
    ap.add_argument("--n-scales", type=int, default=10)
    ap.add_argument("--sigma-min", type=float, default=1.0)
    ap.add_argument("--sigma-max", type=float, default=8.0)
    ap.add_argument("--min-dist", type=float, default=3.0)
    ap.add_argument("--ratio", type=float, default=0.95)
    ap.add_argument("--inlier-mm", type=float, default=8.0)
    ap.add_argument("--no-mutual", action="store_true", help="disable mutual nearest-neighbour check")
    ap.add_argument("--rotation-deg", type=float, default=30.0, help="Example 2: fixed +deg, moving -deg about z")
    args = ap.parse_args()

    ds = syntx.benchmark_data("mbhard")
    fi, mi, fl, ml = ds["fixed"], ds["moving"], ds["fixed_label"], ds["moving_label"]
    assert fi.shape[0] > 100, "loader fell back to synthetic spheres — real Mindboggle data not found"
    label_name = "DKT31 cortical"
    try:
        from syntx.benchmark.data import resolve_data_dir
        base = resolve_data_dir()
        pf = os.path.join(base, "NKI-TRT-20_volumes", "NKI-TRT-20-2", "labels.DKT31.manual+aseg.nii.gz")
        pm = os.path.join(base, "MMRR-21_volumes", "MMRR-21-2", "labels.DKT31.manual+aseg.nii.gz")
        if os.path.exists(pf) and os.path.exists(pm):
            fl, ml = ants.image_read(pf), ants.image_read(pm)
            label_name = "DKT31+aseg whole-brain"
    except Exception:
        pass
    print("label set:", label_name)

    # ---- Example 1: native pair — rotation-search pipeline rendered; plain variants for comparison
    s1, html1 = run_example("Ex1 native", fi, mi, fl, ml, args, fig_offset=0, rotation_search=True)
    s1a, _ = run_example("Ex1 native", fi, mi, fl, ml, args, fig_offset=0, rotation_invariant=False, render=False, pre=s1["pre"])
    s1r, _ = run_example("Ex1 native", fi, mi, fl, ml, args, fig_offset=0, rotation_invariant=True, render=False, pre=s1["pre"])
    html1 += "<h3>Matching variants</h3>" + variant_table([("rotation search (shown above)", s1),
                                                            ("plain scanner-axis-aligned", s1a),
                                                            ("plain rotation-invariant (structure-tensor frame)", s1r)])

    # ---- Example 2: opposite rotations about z through each centre of mass (resampled, frames unchanged)
    deg = args.rotation_deg
    tx_f, Mf = rotation_about_z(fi, +deg)
    tx_m, Mm = rotation_about_z(mi, -deg)
    fi_r = tx_f.apply_to_image(fi, fi, interpolation="linear")
    mi_r = tx_m.apply_to_image(mi, mi, interpolation="linear")
    fl_r = tx_f.apply_to_image(fl, fl, interpolation="nearestneighbor")
    ml_r = tx_m.apply_to_image(ml, ml, interpolation="nearestneighbor")
    # rotation-search pipeline rendered; plain variants for comparison (aligned cannot survive ~60° of yaw)
    s2, html2 = run_example(f"Ex2 rotated ±{deg:g}°", fi_r, mi_r, fl_r, ml_r, args, fig_offset=4, rotation_search=True)
    s2a, _ = run_example(f"Ex2 rotated ±{deg:g}°", fi_r, mi_r, fl_r, ml_r, args, fig_offset=4, rotation_invariant=False,
                         render=False, pre=s2["pre"])
    s2r, _ = run_example(f"Ex2 rotated ±{deg:g}°", fi_r, mi_r, fl_r, ml_r, args, fig_offset=4, rotation_invariant=True,
                         render=False, pre=s2["pre"])
    html2 += "<h3>Matching variants</h3>" + variant_table([("rotation search (shown above)", s2),
                                                            ("plain scanner-axis-aligned", s2a),
                                                            ("plain rotation-invariant (structure-tensor frame)", s2r)])

    # consistency: F'(p) = F(T_f p) so point p in F' ↔ T_f p in F ↔ M1 T_f p in M ↔ T_m⁻¹ M1 T_f p in M'
    M1, M2 = s1["M"].astype(np.float64), s2["M"].astype(np.float64)
    M2_pred = np.linalg.inv(Mm) @ M1 @ Mf
    com_fr = np.array(ants.get_center_of_mass(fi_r))
    test_pts = com_fr[None] + np.array([[0, 0, 0], [40, 0, 0], [-40, 0, 0], [0, 40, 0], [0, -40, 0], [0, 0, 40], [0, 0, -40]], dtype=float)
    def ap4(M, P):
        return (M[:3, :3] @ P.T).T + M[:3, 3]
    disc = np.linalg.norm(ap4(M2, test_pts) - ap4(M2_pred, test_pts), axis=1)
    rot_rel = rotation_angle_deg(M2[:3, :3] @ np.linalg.inv(M1[:3, :3]))
    print(f"Ex2 vs prediction from Ex1 + known rotations: mean discrepancy over 7 test points {disc.mean():.2f} mm "
          f"(max {disc.max():.2f}); relative rotation between recovered affines {rot_rel:.1f} deg (expected ≈ {2*deg:g})")

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>mbhard landmark physical-space report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;margin:8px 0}} td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}} th{{background:#f2f2f2}}
img{{max-width:100%}} code{{background:#f4f4f4;padding:1px 4px}} h2{{margin-top:36px;border-top:2px solid #ddd;padding-top:12px}}</style></head><body>
<h1>Mindboggle <code>mbhard</code> — landmark detection, matching and physical-space display</h1>
<p>Generated {time.strftime('%Y-%m-%d %H:%M')} by <code>scripts/landmarks_mbhard_report.py</code>.
All coordinates are physical LPS millimetres computed through <code>syntx.landmarks.spatial</code>; images are never reoriented —
display orientation and keypoint overlays are derived from the direction cosine matrices.
Preprocessing: ANTsTorch N4 + non-local-means denoising on MPS, then foreground 2–98% normalisation.
Detection: <code>detect_sift3d</code>, {args.n_scales} scales from {args.sigma_min:g} to {args.sigma_max:g} mm, relative threshold {args.threshold:g},
NMS radius {args.min_dist:g} mm, at most {args.max_keypoints} keypoints per image; foreground-masked; 512-D physical-space descriptors.
Matching: Lowe ratio {args.ratio}{'' if args.no_mutual else ' + mutual nearest neighbour'} → RANSAC affine, inlier threshold {args.inlier_mm} mm.
Inter-subject pairs are related non-rigidly, so the affine is a consensus filter; agreement of {label_name} labels at matched
endpoints (chance level shown) is the accuracy proxy.</p>

<h2>Example 1 — native pair</h2>
<p>Fixed and moving are stored in different native frames (LAS vs RPS arrays, 1.0 vs 1.2 mm x-spacing).</p>
{html1}

<h2>Example 2 — fixed rotated +{deg:g}°, moving rotated −{deg:g}° about the superior axis</h2>
<p>Both volumes (and their label maps) were resampled onto their own grids after a rigid rotation about the z axis through each
centre of mass, so the two heads now differ by ≈{2*deg:g}° of yaw in physical space. The axis-aligned SIFT3D descriptor tolerates
roughly 30° of relative rotation, so plain matching fails here; <code>match_sift3d_with_rotation_search</code> seeds a global
fixed→moving rotation from the principal axes of the two landmark clouds and refines it iteratively (descriptors rebuilt in the
current frame → match → rigid RANSAC → rotation; coarse-to-fine over keypoint strength). The figures use that pipeline; the plain
variants are tabulated for comparison. The recovered affine is also compared with the one predicted from Example 1 and the known
rotations (M₂ ≈ T<sub>m</sub>⁻¹ · M₁ · T<sub>f</sub>).</p>
{html2}
<h3>Consistency between the two examples</h3>
<table><tr><th>quantity</th><th>value</th></tr>
{tr('relative rotation between recovered affines M₂·M₁⁻¹ (deg), expected ≈ ' + f'{2*deg:g}', f'{rot_rel:.1f}')}
{tr('mean / max discrepancy between M₂ and T_m⁻¹·M₁·T_f over 7 test points ±40 mm from the centre (mm)', f'{disc.mean():.2f} / {disc.max():.2f}')}
{tr('RANSAC inliers, Ex1 → Ex2 (rotation search)', f"{s1['n_inlier']} → {s2['n_inlier']}")}
{tr('global rotation found by the search, Ex1 / Ex2 (deg)', f"{s1['search']['rotation_deg']:.1f} / {s2['search']['rotation_deg']:.1f}")}
{tr('inlier label agreement, Ex1 → Ex2 (chance)', f"{s1['agree']:.2f} → {s2['agree']:.2f} ({s1['chance']:.2f} / {s2['chance']:.2f})")}</table>
</body></html>"""
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
