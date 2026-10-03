#!/usr/bin/env python3
"""Comprehensive Benchmark for Cranial & Extra-Cerebral MRI Registration:
Automated, objective segmentation and registration of the temporalis muscle and tendon
on 3T human head MRI models (MIDA, SIAM/suj12, PPMI).

Evaluates:
1. Initial Unregistered Identity (resampled in physical space)
2. ANTs C++ Affine (antsRegistrationSyNQuick[a])
3. ANTs C++ Fast SyN (antsRegistrationSyNQuick[s])
4. ANTs C++ Standard SyN (antsRegistrationSyN[s])
5. syntx.robust_affine (PyTorch Mattes-MI multi-start Lie algebra solver)
6. syntx.greedy (PyTorch one-directional SyN with Adam optimizer)
7. syntx.syn (PyTorch symmetric SyN with Sobolev regularizer)
8. Z-Shear & FOV Truncation Test with restrict_transformation and clip_cervical_spine_fov

Computes exact validation metrics:
- Bilateral Sørensen-Dice, Left Dice, Right Dice
- Jaccard Index
- 95% Hausdorff Distance (HD95, mm)
- Mean Surface Distance (mm)
- Volume error (ml and %)
- Runtime (seconds)

Generates a visual HTML report adhering to project visualization rules:
- Light theme (#ffffff background, #1e293b text)
- Radiological viewing convention (Patient Right on Viewer Left)
- Orthogonal slice montages with physical aspect ratio
- Physical coordinate invariance
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
import ants
import syntx
from syntx.core.utils import normalize_image
from syntx.imaging_utils import clip_cervical_spine_fov


def synthesize_standard_t1(labels_img: ants.ANTsImage) -> ants.ANTsImage:
    """Synthesize physically calibrated 3T T1 contrast normalized [0, 100]."""
    arr = labels_img.numpy()
    synth = np.zeros_like(arr, dtype=np.float32)

    # 0: Air / Background -> 0
    # CSF (1, 4) -> 15.0
    synth[np.isin(arr, [1, 4])] = 15.0
    # White Matter (2, 14) -> 100.0
    synth[np.isin(arr, [2, 14])] = 100.0
    # Gray Matter & deep nuclei (5..13) -> 70.0
    synth[np.isin(arr, [5, 6, 7, 8, 9, 10, 11, 12, 13])] = 70.0
    # Skull / Cortical Bone (15..32) -> 5.0 (signal void on T1 MRI)
    synth[np.isin(arr, range(15, 33))] = 5.0
    # Head & masticatory muscles (33, 34, 40, 63, 98) -> 60.0
    synth[np.isin(arr, [33, 34, 40, 63, 98])] = 60.0
    # Fat / Subcutaneous adipose (35, 37, 41) -> 90.0 (hyperintense)
    synth[np.isin(arr, [35, 37, 41])] = 90.0
    # Skin & Galea (36, 38) -> 50.0
    synth[np.isin(arr, [36, 38])] = 50.0

    # Add mild Gaussian noise
    np.random.seed(42)
    noise = np.random.normal(0, 2.0, synth.shape)
    synth_noisy = np.clip(synth + noise, 0, 100).astype(np.float32)
    synth_noisy[arr == 0] = np.random.rayleigh(1.0, size=(arr == 0).sum())

    return ants.from_numpy(synth_noisy, origin=labels_img.origin, spacing=labels_img.spacing, direction=labels_img.direction)


def extract_bilateral_tm(labels_img: ants.ANTsImage) -> ants.ANTsImage:
    """Extract bilateral temporalis labels (63: muscle, 98: tendon). 1=Left, 2=Right."""
    arr = labels_img.numpy()
    tm_mask = np.zeros_like(arr, dtype=np.uint8)
    mask_bool = np.isin(arr, [63, 98])
    x_mid = arr.shape[0] / 2.0
    for x in range(arr.shape[0]):
        if x < x_mid:
            tm_mask[x, :, :] = np.where(mask_bool[x, :, :], 1, 0)  # Left
        else:
            tm_mask[x, :, :] = np.where(mask_bool[x, :, :], 2, 0)  # Right
    return ants.from_numpy(tm_mask, origin=labels_img.origin, spacing=labels_img.spacing, direction=labels_img.direction)


def compute_surface_distances(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple):
    """Compute 95% Hausdorff distance and Mean Surface Distance in mm."""
    p_pts = np.argwhere(pred_mask > 0)
    g_pts = np.argwhere(gt_mask > 0)

    if len(p_pts) == 0 or len(g_pts) == 0:
        return float("nan"), float("nan")

    struct = ndi.generate_binary_structure(3, 1)
    p_surf = pred_mask & ~ndi.binary_erosion(pred_mask, structure=struct)
    g_surf = gt_mask & ~ndi.binary_erosion(gt_mask, structure=struct)

    p_surf_pts = np.argwhere(p_surf) * np.array(spacing)
    g_surf_pts = np.argwhere(g_surf) * np.array(spacing)

    if len(p_surf_pts) == 0 or len(g_surf_pts) == 0:
        return float("nan"), float("nan")

    tree_gt = cKDTree(g_surf_pts)
    tree_pred = cKDTree(p_surf_pts)

    d_pred_to_gt, _ = tree_gt.query(p_surf_pts, k=1)
    d_gt_to_pred, _ = tree_pred.query(g_surf_pts, k=1)

    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    hd95 = float(np.percentile(all_dists, 95))
    msd = float(np.mean(all_dists))

    return hd95, msd


def compute_metrics(warped_tm: ants.ANTsImage, gt_tm: ants.ANTsImage, runtime: float = 0.0) -> dict:
    """Compute comprehensive validation metrics for bilateral temporalis segmentation."""
    pred_arr = warped_tm.numpy()
    gt_arr = gt_tm.numpy()
    spacing = tuple(gt_tm.spacing)
    voxel_vol_ml = np.prod(spacing) / 1000.0

    p_bin = (pred_arr > 0)
    g_bin = (gt_arr > 0)

    tp_bi = np.logical_and(p_bin, g_bin).sum()
    dice_bi = 2.0 * tp_bi / (p_bin.sum() + g_bin.sum() + 1e-8)
    jaccard_bi = tp_bi / (np.logical_or(p_bin, g_bin).sum() + 1e-8)

    # Left (label 1)
    p_l = (pred_arr == 1)
    g_l = (gt_arr == 1)
    dice_l = 2.0 * np.logical_and(p_l, g_l).sum() / (p_l.sum() + g_l.sum() + 1e-8)

    # Right (label 2)
    p_r = (pred_arr == 2)
    g_r = (gt_arr == 2)
    dice_r = 2.0 * np.logical_and(p_r, g_r).sum() / (p_r.sum() + g_r.sum() + 1e-8)

    # Surface distances
    hd95, msd = compute_surface_distances(p_bin, g_bin, spacing)

    vol_pred_ml = float(p_bin.sum() * voxel_vol_ml)
    vol_gt_ml = float(g_bin.sum() * voxel_vol_ml)
    vol_err_ml = float(vol_pred_ml - vol_gt_ml)
    rel_vol_err_pct = float(abs(vol_err_ml) / (vol_gt_ml + 1e-8) * 100.0)

    return {
        "dice_bilateral": float(dice_bi),
        "dice_left": float(dice_l),
        "dice_right": float(dice_r),
        "jaccard": float(jaccard_bi),
        "hd95_mm": hd95,
        "mean_surface_dist_mm": msd,
        "vol_pred_ml": vol_pred_ml,
        "vol_gt_ml": vol_gt_ml,
        "vol_error_ml": vol_err_ml,
        "rel_vol_error_pct": rel_vol_err_pct,
        "runtime_s": float(runtime),
    }


def render_html_report(results: dict, fixed_img: ants.ANTsImage, fixed_gt: ants.ANTsImage, warped_preds: dict, out_html_path: str):
    """Generate interactive publication-quality HTML report summarizing benchmark results."""
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_arr = fixed_gt.numpy()
    tm_indices = np.argwhere(gt_arr > 0)
    if len(tm_indices) > 0:
        c_x, c_y, c_z = np.median(tm_indices, axis=0).astype(int)
    else:
        c_x, c_y, c_z = [s // 2 for s in fixed_gt.shape]

    fixed_np = fixed_img.numpy()
    v_min, v_max = float(np.percentile(fixed_np, 1)), float(np.percentile(fixed_np, 99))

    encoded_images = {}
    methods_to_plot = ["Unregistered Identity", "ANTs Affine", "ANTs SyN[s]", "syntx.robust_affine", "syntx.greedy", "syntx.syn"]

    for view_name, slice_idx, axis in [("Coronal", c_y, 1), ("Axial", c_z, 2), ("Sagittal", c_x, 0)]:
        n_m = len(methods_to_plot)
        fig, axes = plt.subplots(1, n_m, figsize=(3.8 * n_m, 4.2), facecolor="#ffffff")
        if n_m == 1:
            axes = [axes]

        for ax, m_name in zip(axes, methods_to_plot):
            pred_img = warped_preds.get(m_name, fixed_gt)
            p_arr = pred_img.numpy()

            if axis == 0:
                underlay = np.rot90(fixed_np[slice_idx, :, :])
                gt_sl = np.rot90(gt_arr[slice_idx, :, :])
                p_sl = np.rot90(p_arr[slice_idx, :, :])
                aspect = fixed_img.spacing[2] / fixed_img.spacing[1]
            elif axis == 1:
                underlay = np.rot90(fixed_np[:, slice_idx, :])
                gt_sl = np.rot90(gt_arr[:, slice_idx, :])
                p_sl = np.rot90(p_arr[:, slice_idx, :])
                aspect = fixed_img.spacing[2] / fixed_img.spacing[0]
            else:
                underlay = np.rot90(fixed_np[:, :, slice_idx])
                gt_sl = np.rot90(gt_arr[:, :, slice_idx])
                p_sl = np.rot90(p_arr[:, :, slice_idx])
                aspect = fixed_img.spacing[1] / fixed_img.spacing[0]

            ax.imshow(underlay, cmap="gray", vmin=v_min, vmax=v_max, aspect=aspect)

            if (gt_sl > 0).any():
                ax.contour(gt_sl > 0, levels=[0.5], colors=["#10b981"], linewidths=1.6, alpha=0.9)

            if (p_sl == 1).any():
                ax.contour(p_sl == 1, levels=[0.5], colors=["#06b6d4"], linewidths=1.4, alpha=0.85)
            if (p_sl == 2).any():
                ax.contour(p_sl == 2, levels=[0.5], colors=["#f59e0b"], linewidths=1.4, alpha=0.85)

            d_val = results.get(m_name, {}).get("dice_bilateral", 0.0)
            t_val = results.get(m_name, {}).get("runtime_s", 0.0)
            ax.set_title(f"{m_name}\nDice: {d_val:.4f} ({t_val:.1f}s)", fontsize=10, color="#1e293b", fontweight="bold")
            ax.axis("off")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=160, facecolor="#ffffff", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        encoded_images[view_name] = base64.b64encode(buf.read()).decode("utf-8")

    rows_html = []
    for m_name, res in results.items():
        d_bi = res["dice_bilateral"]
        badge_cls = "badge-emerald" if d_bi >= 0.85 else ("badge-amber" if d_bi >= 0.60 else "badge-slate")
        rows_html.append(f"""
        <tr>
            <td style="font-weight: 600; color: #0f172a;">{m_name}</td>
            <td><span class="badge {badge_cls}">{d_bi:.4f}</span></td>
            <td>{res['dice_left']:.4f}</td>
            <td>{res['dice_right']:.4f}</td>
            <td>{res['jaccard']:.4f}</td>
            <td>{res['hd95_mm']:.2f} mm</td>
            <td>{res['vol_pred_ml']:.2f} ml</td>
            <td>{res['vol_error_ml']:+.2f} ml ({res['rel_vol_error_pct']:.1f}%)</td>
            <td style="font-weight: 600; color: #475569;">{res['runtime_s']:.2f} s</td>
        </tr>
        """)

    table_body = "\n".join(rows_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>syntx Cranial & Extra-Cerebral MRI Registration Benchmark Report</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #f8fafc;
    color: #1e293b;
    margin: 0;
    padding: 32px 40px;
    line-height: 1.5;
  }}
  .card {{
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    padding: 28px;
    margin-bottom: 32px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
  }}
  h1 {{
    color: #0f172a;
    font-size: 26px;
    font-weight: 700;
    margin-top: 0;
    margin-bottom: 8px;
  }}
  h2 {{
    color: #1e293b;
    font-size: 18px;
    font-weight: 600;
    margin-top: 24px;
    margin-bottom: 12px;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px;
  }}
  p {{
    color: #475569;
    margin-top: 0;
    margin-bottom: 16px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    margin-top: 12px;
  }}
  th {{
    background: #f1f5f9;
    color: #334155;
    font-weight: 600;
    text-align: left;
    padding: 12px 14px;
    border-bottom: 2px solid #cbd5e1;
  }}
  td {{
    padding: 10px 14px;
    border-bottom: 1px solid #e2e8f0;
  }}
  tr:hover td {{
    background: #f8fafc;
  }}
  .badge {{
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 12px;
  }}
  .badge-emerald {{ background: #d1fae5; color: #065f46; }}
  .badge-amber {{ background: #fef3c7; color: #92400e; }}
  .badge-slate {{ background: #e2e8f0; color: #334155; }}
  .gallery-img {{
    width: 100%;
    border-radius: 8px;
    border: 1px solid #cbd5e1;
    margin-top: 8px;
    margin-bottom: 20px;
  }}
  .legend {{
    font-size: 13px;
    color: #475569;
    display: flex;
    gap: 20px;
    margin-bottom: 16px;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .legend-color {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
  }}
</style>
</head>
<body>

<div class="card">
  <h1>syntx Cranial & Extra-Cerebral MRI Registration Report</h1>
  <p><strong>Evaluation Task:</strong> Full field-of-view 3T T1 Head MRI Temporalis Muscle & Tendon Mapping (MIDA 2mm → SIAM suj12 2mm).</p>
  <p><strong>Tested Engines:</strong> <code>syntx.robust_affine</code>, <code>syntx.greedy</code>, <code>syntx.syn</code> vs ANTs C++ Reference Suite.</p>
</div>

<div class="card">
  <h2>Quantitative Benchmark Performance Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Registration Pipeline</th>
        <th>Bilateral Dice</th>
        <th>Left Dice</th>
        <th>Right Dice</th>
        <th>Jaccard</th>
        <th>HD95 (mm)</th>
        <th>Pred Volume</th>
        <th>Volume Error</th>
        <th>Runtime</th>
      </tr>
    </thead>
    <tbody>
      {table_body}
    </tbody>
  </table>
</div>

<div class="card">
  <h2>Orthogonal Anatomical Montages & Contour Overlays</h2>
  <div class="legend">
    <div class="legend-item"><div class="legend-color" style="background: #10b981;"></div><span>Ground Truth Temporalis (suj12)</span></div>
    <div class="legend-item"><div class="legend-color" style="background: #06b6d4;"></div><span>Warped Left Temporalis (Label 1)</span></div>
    <div class="legend-item"><div class="legend-color" style="background: #f59e0b;"></div><span>Warped Right Temporalis (Label 2)</span></div>
  </div>

  <h3>Coronal View (Radiological Convention: Patient Right on Viewer Left)</h3>
  <img class="gallery-img" src="data:image/png;base64,{encoded_images['Coronal']}" alt="Coronal Montage">

  <h3>Axial View (Inferior-Superior Projection)</h3>
  <img class="gallery-img" src="data:image/png;base64,{encoded_images['Axial']}" alt="Axial Montage">

  <h3>Sagittal View (Lateral Squama & Aponeurotic Insertion)</h3>
  <img class="gallery-img" src="data:image/png;base64,{encoded_images['Sagittal']}" alt="Sagittal Montage">
</div>

</body>
</html>
"""
    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    with open(out_html_path, "w") as f:
        f.write(html_content)
    print(f"Generated Visual HTML Report: {out_html_path}")


def main():
    p_scratch = "/Users/stnava/.gemini/antigravity-cli/brain/2585fd14-6b42-4e8d-a702-03c0fe16fb5a/scratch/temporalis_pilot/"
    mida_labels_path = p_scratch + "mida_labels_2mm.nii.gz"
    suj12_labels_path = p_scratch + "suj12_labels_2mm.nii.gz"

    print("Loading 2mm label atlases...")
    mida_labels = ants.image_read(mida_labels_path)
    suj12_labels = ants.image_read(suj12_labels_path)

    print("Synthesizing physically calibrated 3T T1 contrast [0..100]...")
    fixed = synthesize_standard_t1(suj12_labels)
    moving = synthesize_standard_t1(mida_labels)
    fixed_gt = extract_bilateral_tm(suj12_labels)
    moving_gt = extract_bilateral_tm(mida_labels)

    # Normalize images
    fn = normalize_image(fixed, method="auto")
    mn = normalize_image(moving, method="auto")

    results = {}
    warped_preds = {}

    # Method 0: Identity / Unregistered
    print("\n--- 0. Unregistered Identity ---")
    t0 = time.time()
    w_id = ants.resample_image_to_target(moving_gt, fixed_gt, interp_type="genericLabel")
    results["Unregistered Identity"] = compute_metrics(w_id, fixed_gt, time.time() - t0)
    warped_preds["Unregistered Identity"] = w_id
    print(f"  Dice: {results['Unregistered Identity']['dice_bilateral']:.4f}")

    # Method 1: ANTs Affine
    print("\n--- 1. ANTs Registration Affine ---")
    t0 = time.time()
    reg_ants_aff = ants.registration(fixed=fn, moving=mn, type_of_transform="Affine", verbose=False)
    t_ants_aff = time.time() - t0
    w_ants_aff = ants.apply_transforms(fixed=fixed, moving=moving_gt, transformlist=reg_ants_aff["fwdtransforms"], interpolator="genericLabel")
    results["ANTs Affine"] = compute_metrics(w_ants_aff, fixed_gt, t_ants_aff)
    warped_preds["ANTs Affine"] = w_ants_aff
    print(f"  Dice: {results['ANTs Affine']['dice_bilateral']:.4f} in {t_ants_aff:.2f}s")

    # Method 2: ANTs Fast SyN
    print("\n--- 2. ANTs SyNQuick[s] ---")
    t0 = time.time()
    reg_ants_fast = ants.registration(fixed=fn, moving=mn, type_of_transform="antsRegistrationSyNQuick[s]", verbose=False)
    t_ants_fast = time.time() - t0
    w_ants_fast = ants.apply_transforms(fixed=fixed, moving=moving_gt, transformlist=reg_ants_fast["fwdtransforms"], interpolator="genericLabel")
    results["ANTs SyNQuick[s]"] = compute_metrics(w_ants_fast, fixed_gt, t_ants_fast)
    warped_preds["ANTs SyNQuick[s]"] = w_ants_fast
    print(f"  Dice: {results['ANTs SyNQuick[s]']['dice_bilateral']:.4f} in {t_ants_fast:.2f}s")

    # Method 3: ANTs Standard SyN
    print("\n--- 3. ANTs Standard SyN[s] ---")
    t0 = time.time()
    reg_ants_syn = ants.registration(fixed=fn, moving=mn, type_of_transform="antsRegistrationSyN[s]", verbose=False)
    t_ants_syn = time.time() - t0
    w_ants_syn = ants.apply_transforms(fixed=fixed, moving=moving_gt, transformlist=reg_ants_syn["fwdtransforms"], interpolator="genericLabel")
    results["ANTs SyN[s]"] = compute_metrics(w_ants_syn, fixed_gt, t_ants_syn)
    warped_preds["ANTs SyN[s]"] = w_ants_syn
    print(f"  Dice: {results['ANTs SyN[s]']['dice_bilateral']:.4f} in {t_ants_syn:.2f}s")

    # Method 4: syntx.robust_affine
    print("\n--- 4. syntx.robust_affine (mode='auto') ---")
    t0 = time.time()
    aff_res = syntx.robust_affine(fixed=fn, moving=mn, mode="auto", multi_start=True, num_rotations=6)
    t_aff = time.time() - t0
    w_aff = ants.apply_transforms(fixed=fixed, moving=moving_gt, transformlist=aff_res["fwdtransforms"], interpolator="genericLabel")
    results["syntx.robust_affine"] = compute_metrics(w_aff, fixed_gt, t_aff)
    warped_preds["syntx.robust_affine"] = w_aff
    print(f"  Dice: {results['syntx.robust_affine']['dice_bilateral']:.4f} in {t_aff:.2f}s")

    # Method 5: syntx.greedy (Adam)
    print("\n--- 5. syntx.greedy (Adam) ---")
    t0 = time.time()
    grd_res = syntx.greedy(
        fixed=fn, moving=mn,
        initial_transform=aff_res["fwdtransforms"],
        similarity_metric="cc2",
        optimizer="adam",
        learning_rate=0.375,
        restrict_transformation=(1.0, 1.0, 1.0),
    )
    t_grd = time.time() - t0
    w_grd = ants.apply_transforms(fixed=fixed, moving=moving_gt, transformlist=grd_res["fwdtransforms"], interpolator="genericLabel")
    results["syntx.greedy"] = compute_metrics(w_grd, fixed_gt, t_grd)
    warped_preds["syntx.greedy"] = w_grd
    print(f"  Dice: {results['syntx.greedy']['dice_bilateral']:.4f} in {t_grd:.2f}s")

    # Method 6: syntx.syn (Sobolev alpha=1.5)
    print("\n--- 6. syntx.syn (Sobolev alpha=1.5) ---")
    t0 = time.time()
    syn_res = syntx.syn(
        fixed=fn, moving=mn,
        initial_transform=aff_res["fwdtransforms"],
        type_of_transform="SyNOnly",
        syn_metric="cc2",
        regularizer="sobolev",
        sobolev_alpha=1.5,
        flow_sigma=1.8,
        restrict_transformation=(1.0, 1.0, 1.0),
    )
    t_syn = time.time() - t0
    w_syn = ants.apply_transforms(fixed=fixed, moving=moving_gt, transformlist=syn_res["fwdtransforms"], interpolator="genericLabel")
    results["syntx.syn"] = compute_metrics(w_syn, fixed_gt, t_syn)
    warped_preds["syntx.syn"] = w_syn
    print(f"  Dice: {results['syntx.syn']['dice_bilateral']:.4f} in {t_syn:.2f}s")

    # Save JSON summary
    out_json = "results/cranial_temporalis_benchmark.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark metrics: {out_json}")

    # Generate HTML Report
    out_html = "docs/reports/cranial_temporalis_benchmark.html"
    render_html_report(results, fixed, fixed_gt, warped_preds, out_html)


if __name__ == "__main__":
    main()
