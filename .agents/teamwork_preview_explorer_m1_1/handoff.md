# Milestone 1 Exploit Baseline Execution Specification Report (`run_m1_baseline.py`)

**Agent**: Explorer M1  
**Working Directory**: `/Users/stnava/code/syntx/.agents/teamwork_preview_explorer_m1_1`  
**Target Milestone**: Milestone 1 (Exploit Baseline at commit `01d74b0` on 3D Native Pair 0 `NKI-TRT-20-3` -> `NKI-RS-22-22`)  
**Date**: 2026-08-10  

---

## 1. Observation

### 1.1 Verified Dataset Files and Metadata for 3D Native Pair 0
Native Pair 0 is defined in `examples/pairs.csv` line 73: `inter,NKI-TRT-20,NKI-TRT-20-3,NKI-RS-22,NKI-RS-22-22`.
Direct filesystem verification via `ants.image_read` confirms the existence and exact properties of all 4 volume files in `/Users/stnava/data/mindboggle/volumes/`:

| Role | Dataset Path | Dimensions | Spacing (mm) | Origin (mm) |
|---|---|---|---|---|
| **Fixed Image** | `NKI-TRT-20_volumes/NKI-TRT-20-3/t1weighted_brain.nii.gz` | `(192, 256, 256)` | `(1.0, 1.0, 1.0)` | `(-95.5, 102.0, -152.0)` |
| **Fixed Label** | `NKI-TRT-20_volumes/NKI-TRT-20-3/labels.DKT31.manual.nii.gz` | `(192, 256, 256)` | `(1.0, 1.0, 1.0)` | `(-95.5, 102.0, -152.0)` |
| **Moving Image** | `NKI-RS-22_volumes/NKI-RS-22-22/t1weighted_brain.nii.gz` | `(192, 256, 256)` | `(1.0, 1.0, 1.0)` | `(-93.17456, 102.0, -143.24036)` |
| **Moving Label** | `NKI-RS-22_volumes/NKI-RS-22-22/labels.DKT31.manual.nii.gz` | `(192, 256, 256)` | `(1.0, 1.0, 1.0)` | `(-93.17456, 102.0, -143.24036)` |

### 1.2 Baseline Exploit Configuration at Commit `01d74b0`
Direct inspection of `01d74b0` historical code in `src/syntx/syn.py` establishes the baseline parameter configuration:
1. **LNCC Metric Padding Mode**: `padding_mode='border'` in `grid_sample_nd` (lines 616, 656, 870, 884). Replicates edge voxel intensities out-of-bounds rather than zero-padding, avoiding boundary penalties in box-filter variance.
2. **Elastic Smoothing**: `fast_smooth=True` in `SyNTo.fit()` (lines 2954–2984). Applies 1D separable/FFT spectral Sobolev Green's operator filtering instead of exact 3D spatial Gaussian convolution (`separable_gaussian_filter`).
3. **In-Loop Inverse Steps**: `in_loop_inv_steps=6` (lines 2998, 3327). Hard-caps in-loop fixed-point inverse diffeomorphism updates at 6 iterations.
4. **Pyramid Schedule & Regularization**: `reg_iterations=[100, 100, 20]`, `fluid_sigma=3.0`, `total_sigma=0.0`.
5. **Initial Affine Alignment**: `syntx.robust_affine(mode='pytorch')` pre-aligning scanner physical spaces.

### 1.3 Interface & Reporting Infrastructure Constraints
1. **Bidirectional Label Evaluation (`compute_bidirectional_dice`)**:
   - `fl_warped`: Warps moving label to fixed space using `fwdtransforms` with `interpolator='nearestNeighbor'`.
   - `ml_warped`: Warps fixed label to moving space using `invtransforms` with `interpolator='nearestNeighbor'`.
   - `dice_sym = 0.5 * (dice_fixed + dice_moving)`. Baseline expectation: `dice_sym ~ 0.65`.
2. **Manifold Regularity Metrics**:
   - `ants.create_jacobian_determinant_image(fi, reg['fwdtransforms'][0], do_log=False)` enforces physical $\det(J)$ calculation (GEMINI.md Rule 3).
   - `folding_pct = np.mean(jac[mask] <= 0.0) * 100.0` over brain mask `mask = ants.get_mask(fi).numpy() > 0`.
3. **Interactive HTML Report (`syntx.viz.create_registration_report`)**:
   - Target HTML path: `docs/reports/baseline_report.html`.
   - Must render the **Standard 5-Figure Visual Suite** (Figure 1: Input Pair, Figure 2: Standard 4-Panel Diagnostic, Figure 3: Keyframe Flow Grid, Figure 4: Multi-Res Loss Curves, Figure 5: Cortical Dice Curves).

---

## 2. Logic Chain

1. **Parameter Isolation**: To replicate the exact `01d74b0` historic baseline performance on Native Pair 0, `run_m1_baseline.py` must explicitly pass `padding_mode='border'`, `fast_smooth=True`, and `in_loop_inv_steps=6` to `syntx.syn(...)`.
2. **Affine Initialization**: Native space images have different physical origins (`-95.5` vs `-93.17`). Per GEMINI.md Rule 16, running non-linear SyN requires pre-alignment via `syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch')` to avoid initial coordinate misalignment.
3. **Execution Isolation & Device Selection**: Apple Silicon MPS or CUDA GPU acceleration is selected dynamically (`device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')`). Process-level garbage collection (`gc.collect()` and `torch.mps.empty_cache()`) guarantees memory cleanup.
4. **Metric Integrity**:
   - Discrete segmentation warping MUST use `nearestNeighbor` interpolation (GEMINI.md Rule 4).
   - `ants.create_jacobian_determinant_image` MUST use `do_log=False` so raw physical determinant values are evaluated for folding percentage ($\det(J) \le 0$).
5. **Interactive Visualization Suite**: Calling `syntx.viz.create_registration_report()` packages the completed registration dictionary `reg`, inputs `fi, mi`, labels `fl, ml`, and outputs into a standalone interactive HTML artifact at `docs/reports/baseline_report.html`.

---

## 3. Script Specification Design (`run_m1_baseline.py`)

Below is the complete, self-contained implementation blueprint for `run_m1_baseline.py`:

```python
#!/usr/bin/env python3
"""
run_m1_baseline.py
==================
Milestone 1 Benchmark Script: Exploit Baseline at Commit 01d74b0.

Executes 3D SyN registration on Native Pair 0 (NKI-TRT-20-3 -> NKI-RS-22-22) using
the historical exploit configuration (padding_mode='border', fast_smooth=True, in_loop_inv_steps=6).

Outputs:
- Console summary reporting Sym Dice (~0.65), Grid Folding %, min det(J), and runtime.
- Interactive HTML report: docs/reports/baseline_report.html with Standard 5-Figure Visual Suite.
- JSON metrics record: docs/reports/baseline_metrics.json.
"""

import os
import sys
import time
import json
import gc
import numpy as np
import torch
import ants

# Ensure syntx package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import syntx
from syntx import syn
from syntx.viz import create_registration_report


DATASET_PATHS = {
    "fixed_image": "/Users/stnava/data/mindboggle/volumes/NKI-TRT-20_volumes/NKI-TRT-20-3/t1weighted_brain.nii.gz",
    "fixed_label": "/Users/stnava/data/mindboggle/volumes/NKI-TRT-20_volumes/NKI-TRT-20-3/labels.DKT31.manual.nii.gz",
    "moving_image": "/Users/stnava/data/mindboggle/volumes/NKI-RS-22_volumes/NKI-RS-22-22/t1weighted_brain.nii.gz",
    "moving_label": "/Users/stnava/data/mindboggle/volumes/NKI-RS-22_volumes/NKI-RS-22-22/labels.DKT31.manual.nii.gz"
}


def compute_bidirectional_dice(fl, ml, fi, mi, fwdtransforms, invtransforms, whichtoinvert_inv=None):
    """Computes bidirectional fixed, moving, and symmetric mean DKT31 Dice scores."""
    if whichtoinvert_inv is None:
        whichtoinvert_inv = [True, False]

    # 1. Fixed Space Dice
    ml_warped = ants.apply_transforms(
        fixed=fi, moving=ml,
        transformlist=fwdtransforms,
        interpolator='nearestNeighbor'
    )
    ov_fixed = ants.label_overlap_measures(fl, ml_warped)
    df_fixed = ov_fixed[~ov_fixed['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_fixed = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_fixed.columns else 'TargetOverlap'
    dice_fixed = float(df_fixed[col_fixed].mean()) if len(df_fixed) > 0 else 0.0

    # 2. Moving Space Dice
    fl_warped = ants.apply_transforms(
        fixed=mi, moving=fl,
        transformlist=invtransforms,
        whichtoinvert=whichtoinvert_inv,
        interpolator='nearestNeighbor'
    )
    ov_moving = ants.label_overlap_measures(ml, fl_warped)
    df_moving = ov_moving[~ov_moving['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_moving = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_moving.columns else 'TargetOverlap'
    dice_moving = float(df_moving[col_moving].mean()) if len(df_moving) > 0 else 0.0

    dice_sym = 0.5 * (dice_fixed + dice_moving)
    return dice_fixed, dice_moving, dice_sym


def run_m1_baseline(
    output_html="docs/reports/baseline_report.html",
    output_json="docs/reports/baseline_metrics.json"
):
    """Executes Milestone 1 Baseline Registration and Report Generation."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_html = os.path.join(project_root, output_html) if not os.path.isabs(output_html) else output_html
    output_json = os.path.join(project_root, output_json) if not os.path.isabs(output_json) else output_json

    print("=====================================================================")
    print(" Milestone 1: Exploit Baseline Benchmark (Commit 01d74b0)")
    print(" Pair 0: NKI-TRT-20-3 (Fixed) -> NKI-RS-22-22 (Moving)")
    print(" Configuration: padding_mode='border', fast_smooth=True, in_loop_inv_steps=6")
    print("=====================================================================")

    # 1. Load Dataset Volumes
    print("\n[1/4] Loading 3D Native Pair 0 Volumes...", flush=True)
    fi = ants.image_read(DATASET_PATHS["fixed_image"])
    fl = ants.image_read(DATASET_PATHS["fixed_label"])
    mi = ants.image_read(DATASET_PATHS["moving_image"])
    ml = ants.image_read(DATASET_PATHS["moving_label"])
    print(f"  Fixed Image:  {fi.shape}, Spacing: {fi.spacing}, Origin: {fi.origin}")
    print(f"  Moving Image: {mi.shape}, Spacing: {mi.spacing}, Origin: {mi.origin}")

    # Device selection
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"  Execution Device: {device}")

    # 2. Compute Robust Affine Pre-Alignment
    print("\n[2/4] Computing Robust Affine Initialization...", flush=True)
    t_aff0 = time.time()
    reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)
    aff_tx = reg_aff['fwdtransforms'][0]
    t_aff = time.time() - t_aff0
    print(f"  Robust Affine completed in {t_aff:.2f} s")

    # 3. Perform Deformable SyN Registration (Commit 01d74b0 Exploit Baseline)
    print("\n[3/4] Running SyN Registration (01d74b0 Exploit Baseline)...", flush=True)
    t_syn0 = time.time()
    reg = syntx.syn(
        fixed=fi,
        moving=mi,
        initial_transform=aff_tx,
        backend='pytorch',
        device=device,
        reg_iterations=[100, 100, 20],
        affine_iterations=[0, 0, 0],
        similarity_metric='lncc',
        syn_sampling=2,
        flow_sigma=3.0,
        total_sigma=0.0,
        in_loop_inv_steps=6,
        fast_smooth=True,
        padding_mode='border',
        verbose=True
    )
    t_syn = time.time() - t_syn0
    print(f"  SyN Registration completed in {t_syn:.2f} s")

    # 4. Compute Baseline Quantitative Metrics
    print("\n[4/4] Computing Quantitative Metrics & Generating HTML Report...", flush=True)
    dice_fixed, dice_moving, dice_sym = compute_bidirectional_dice(
        fl, ml, fi, mi, reg['fwdtransforms'], reg['invtransforms'], reg.get('whichtoinvert_inv')
    )

    # Jacobian Determinant & Grid Folding %
    fwd_warp_file = reg['fwdtransforms'][0]
    jac_ants = ants.create_jacobian_determinant_image(fi, fwd_warp_file, do_log=False)
    jac_np = jac_ants.numpy()
    mask = ants.get_mask(fi).numpy() > 0

    folding_pct = float(np.mean(jac_np[mask] <= 0.0) * 100.0)
    min_jac = float(jac_np[mask].min())

    print("\n=====================================================================")
    print(" MILESTONE 1 BASELINE RESULTS")
    print("=====================================================================")
    print(f"  Fixed Space Cortical Dice:  {dice_fixed:.4f}")
    print(f"  Moving Space Cortical Dice: {dice_moving:.4f}")
    print(f"  Symmetric Mean Cortical Dice: {dice_sym:.4f}  (Target: ~0.65)")
    print(f"  Grid Folding Percentage:     {folding_pct:.4f} %")
    print(f"  Minimum Jacobian Det:        {min_jac:.4f}")
    print(f"  Execution Runtime:           {t_syn:.2f} s")
    print("=====================================================================")

    # 5. Export Interactive HTML Report
    provenance = {
        "milestone": "M1 Exploit Baseline",
        "commit": "01d74b0",
        "pair": "Pair 0 (NKI-TRT-20-3 -> NKI-RS-22-22)",
        "padding_mode": "border",
        "fast_smooth": True,
        "in_loop_inv_steps": 6,
        "reg_iterations": "[100, 100, 20]",
        "dice_fixed": f"{dice_fixed:.4f}",
        "dice_moving": f"{dice_moving:.4f}",
        "dice_sym": f"{dice_sym:.4f}",
        "folding_pct": f"{folding_pct:.4f}%",
        "min_jacobian": f"{min_jac:.4f}",
        "runtime_seconds": f"{t_syn:.2f}",
        "device": device
    }

    report_summary = create_registration_report(
        fixed=fi,
        moving=mi,
        warped=reg['warpedmovout'],
        warp=fwd_warp_file,
        detJ=jac_ants,
        output_html=output_html,
        fixed_name="NKI-TRT-20-3 (Fixed Target)",
        moving_name="NKI-RS-22-22 (Moving Source)",
        provenance=provenance,
        fixed_label=fl,
        moving_label=ml,
        title="M1 Baseline Exploit Registration Report (Commit 01d74b0)",
        show_report=False
    )

    # Save metrics JSON
    metrics_record = {
        "milestone": "M1",
        "commit": "01d74b0",
        "dataset_pair": "Native Pair 0 (NKI-TRT-20-3 -> NKI-RS-22-22)",
        "config": {
            "padding_mode": "border",
            "fast_smooth": True,
            "in_loop_inv_steps": 6,
            "reg_iterations": [100, 100, 20],
            "fluid_sigma": 3.0,
            "total_sigma": 0.0
        },
        "metrics": {
            "dice_fixed": dice_fixed,
            "dice_moving": dice_moving,
            "dice_sym": dice_sym,
            "folding_pct": folding_pct,
            "min_jacobian": min_jac,
            "runtime_seconds": t_syn
        },
        "report_html": output_html
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metrics_record, f, indent=2)

    print(f"\nHTML Report saved to: {output_html}")
    print(f"Metrics JSON saved to: {output_json}")

    # Memory Cleanup
    del reg, jac_ants
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return metrics_record


if __name__ == '__main__':
    run_m1_baseline()
```

---

## 4. Caveats

- **Read-Only Scope**: This report defines the exact design specification for `run_m1_baseline.py`. Explorer M1 performs read-only investigation and specification creation; script execution will be performed by implementer agents/harness.
- **Hardware Acceleration**: Apple Silicon MPS GPU memory constraints require `os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"` (set natively in `syntx.__init__`) and explicit cache flushing.
- **DICE Score Variance**: Minor floating point variations across hardware backends (MPS vs CPU vs CUDA) may yield Sym Dice within $[0.645, 0.655]$.

---

## 5. Conclusion

1. **Dataset Integrity**: Native Pair 0 (`NKI-TRT-20-3` -> `NKI-RS-22-22`) is verified and present at `/Users/stnava/data/mindboggle/volumes/`.
2. **Exploit Baseline Configuration**: The historical `01d74b0` baseline configuration is fully defined: `padding_mode='border'`, `fast_smooth=True`, `in_loop_inv_steps=6`, `reg_iterations=[100, 100, 20]`, `fluid_sigma=3.0`, `total_sigma=0.0`.
3. **Verification Artifacts**: `run_m1_baseline.py` outputs both the interactive HTML report at `docs/reports/baseline_report.html` (containing the Standard 5-Figure Visual Suite) and structured JSON metrics at `docs/reports/baseline_metrics.json`.

---

## 6. Verification Method

1. **Verify Dataset Existence**:
   ```bash
   python3 -c "
   import ants
   fi = ants.image_read('/Users/stnava/data/mindboggle/volumes/NKI-TRT-20_volumes/NKI-TRT-20-3/t1weighted_brain.nii.gz')
   mi = ants.image_read('/Users/stnava/data/mindboggle/volumes/NKI-RS-22_volumes/NKI-RS-22-22/t1weighted_brain.nii.gz')
   print('Fixed:', fi.shape, 'Moving:', mi.shape)
   "
   ```
2. **Script Dry Run / Syntax Check**:
   ```bash
   python3 -m py_compile run_m1_baseline.py
   ```
3. **Execution & Metric Verification**:
   - Run `run_m1_baseline.py`.
   - Inspect console output and `docs/reports/baseline_metrics.json` to verify `dice_sym` is approximately `0.65`.
   - Open `docs/reports/baseline_report.html` in browser to confirm figures 1–5 are present and rendered correctly.
