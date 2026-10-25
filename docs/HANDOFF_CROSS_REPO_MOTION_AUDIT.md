# Cross-Repository Handoff & Technical Audit: Overlapping-Window Motion Estimation and Downstream Pipelines

**Date**: 2026-10-24  
**Impacted Repositories (Dependency Order)**:
1. `antsxqc` (`/Users/stnava/data/repos/antsxqc`, commit `6343cb1` / tag `v0.1.2`)
2. `syntx` (`/Users/stnava/data/repos/syntx`, commit `7a354a5` / tag `v6.19.0` on `origin/main`)
3. `antsxdwi` (`/Users/stnava/data/repos/antsxdwi`, commit `88e94f9` / tag `v0.27.0`)
4. `antsxfunctional` (`/Users/stnava/data/repos/antsxfunctional`, commit `bc04520` / tag `v0.15.0`)
5. `antsxslowflow` (`/Users/stnava/data/repos/antsxslowflow`, commit `1362237` / tag `v2.7.0` on `origin/main` + uncommitted local pipeline fixes)
6. `antsxmm` (`/Users/stnava/Documents/code/antsxmm`, symlinked at `/Users/stnava/data/repos/antsxmm`, version `v4.3.2`)

---

## Executive Summary

A wave of commits pushed on `2026-10-24 ~20:14 EDT` (`antsxqc v0.1.2`, `syntx v6.19.0`, `antsxdwi v0.27.0`, `antsxfunctional v0.15.0`, `antsxslowflow v2.7.0`, `antsxmm v4.3.2`) attempted to implement the generalized overlapping-window encoding-bias and head-motion solver across DWI, ASL, SlowFlow, and BOLD EPI.

An audit of the commits revealed:
1. **In `syntx` (`7a354a5`, `v6.19.0`), the in-optimizer overlapping-window solver is dead code during optimization**: `B_win`, `E_psi_t`, `_project_encoding_gauge`, and `reg_loss_fn` are allocated/defined in `_batched_rigid_register_pass_core`, but **never referenced inside `compute_loss`, `opt_params`, or the multi-stage optimization loop**. The registration still runs 100% unconstrained and only calls `_project_encoding_surround_tukey_np` once post-hoc after convergence.
2. **Immediate test suite regression in `syntx`**: `src/syntx/motion_presets.py` registers `{"dwi", "bold", "asl", "slowflow"}`, but `tests/test_motion_dwi_preset.py:60` asserts `set(MOTION_PRESETS) == {"dwi", "bold", "slowflow"}`, causing `pytest tests/test_motion_dwi_preset.py` to fail immediately.
3. **Unconditional BOLD sagittal nodding erasure (`syntx` + `antsxfunctional`)**: `separate_bold_pe_pseudomotion` runs automatically on every `preset="bold"` call with `T >= 20` and shrinks all three sagittal EPI-plane DOFs (`t_pe`, `t_sl`, and pitch rotation `r_ro`) down to the LR/roll/yaw noise floor `sig_ref`. Because physical head nodding (`t_sl` + `t_pe` + pitch `r_ro`) is the dominant mode of real human head motion in an MRI coil, this systematically erases real sagittal head motion from `fd` and leaves `motion_parameters` inconsistent with `fwdtransforms`.
4. **DWI resampling corruption (`syntx` + `antsxdwi`)**: `motion_correct_grouped(..., preset="dwi")` automatically injects `pe_affine_covariates` into `encoding_covariates`, which subtracts 6-DOF gradient-correlated poses `delta_Y_np` from `t_final` and `R_final` *before* writing the `.mat` files that resample the DWI volumes, while also fitting `B0` to `t_tot_for_pe` in `pe_affine`.
5. **Incomplete SlowFlow remediation in `antsxslowflow` (`1362237`, `v2.7.0`)**: `1362237` only touched 30 lines of `src/antsxslowflow/motion/api.py` (keeping the 4-group split, polynomial OLS, and the destructive `param_smooth_sigma=2.0` Gaussian filter that damps real motion spikes by 80%), while omitting the fixes for stale `unwrap_phase.nii.gz` caching, cross-state 4D BPC bleeding, $\pm\pi$ wrap cliffs in `estimate_flow`, gradient-cycling alias selection in `analyze_spectrum`/`fit_oscillation`, and broken `render_midpoint_montage` visualization.

---

## Repository-by-Repository Audit & Action Plan

---

### Repo 1: `antsxqc` (`/Users/stnava/data/repos/antsxqc`, `v0.1.2` / `6343cb1`)

#### What Changed in `6343cb1`
- Updated `dti_dvars_b0_mean` and `dti_dvars_dwi_mean` units in `src/antsxqc/specs.py` (`lines 224-225`) from `"intensity"` to `"%"` to match standardized DVARS.
- Added `_A` / `_B` series-pair aliases in `src/antsxqc/specs.py` (`lines 324-339`):
  - `dti_FD_max_A`, `dti_FD_max_B`
  - `dti_FD_raw_mean_A`, `dti_FD_raw_mean_B`
  - `dti_eddy_pe_shift_mm_abs_mean_A`, `dti_eddy_pe_shift_mm_abs_mean_B`
  - `dti_eddy_pe_shift_mm_abs_max_A`, `dti_eddy_pe_shift_mm_abs_max_B`
  - `dti_motion_direction_bias_removed_rms_mm_A`, `dti_motion_direction_bias_removed_rms_mm_B`
  - `dti_motion_unconverged_frames_A`, `dti_motion_unconverged_frames_B`
  - `dti_motion_cross_shell_unconverged_A`, `dti_motion_cross_shell_unconverged_B`

#### Assessment & Required Actions
- **Status**: **Clean and backward-compatible**, with one verification check:
  - Ensure every canonical metric name referenced on the right-hand side of `ALIASES` in `src/antsxqc/specs.py:324-339` exists as a key in `SPECS`.

---

### Repo 2: `syntx` (`/Users/stnava/data/repos/syntx`, `v6.19.0` / `7a354a5` on `origin/main`)

#### Bugs Found in `7a354a5`

1. **Dead In-Optimizer Overlapping-Window Variables (`src/syntx/motion_batched.py:1187-1215, 1504-1520, 1564-1669`)**:
   - `B_win` (`shape: (M_win, K_eff, 6)`, `requires_grad=True`) and `E_psi_t` (`shape: (B, M_win, K_eff)`) are allocated at lines 1212–1213, and `_project_encoding_gauge` is defined at line 1504.
   - Neither `B_win` nor `E_psi_t` is used inside `compute_loss` (`line 1564`), `stage_points` (`line 1621`), or `opt_params` (`line 1614`).
   - `reg_loss_fn` was added to `_run_converging_stage` (`line 613`) but never passed at `line 1665`.
   - `_project_encoding_gauge` is never called anywhere.
   - Instead, `_project_encoding_surround_tukey_np` is only called once post-hoc at `line 1688`.

2. **Failing Unit Test (`tests/test_motion_dwi_preset.py:60`)**:
   - `src/syntx/motion_presets.py:1186` sets `MOTION_PRESETS = {"dwi": ..., "bold": ..., "asl": ..., "slowflow": ...}`, whereas `tests/test_motion_dwi_preset.py:60` asserts `assert set(MOTION_PRESETS) == {"dwi", "bold", "slowflow"}`.

3. **Unconditional `separate_bold_pe_pseudomotion` Corrupts Real Sagittal Head Motion (`src/syntx/motion.py:1207-1221` & `src/syntx/motion_batched.py:602-807`)**:
   - Triggered unconditionally for `preset == "bold"` when `num_frames >= 20`.
   - Shrinks all 3 sagittal EPI-plane DOFs (`t_pe`, `t_sl`, and pitch rotation `r_ro`) toward a Hann-window smooth curve whenever their step variance exceeds `sig_ref = median(mad(diff(t_ro)), mad(diff(r_pe)), mad(diff(r_sl)))`.
   - Because true human head motion in an MRI head coil is predominantly sagittal nodding (`t_pe`, `t_sl`, `r_ro`), this suppresses real head motion FD and desynchronizes `result["motion_parameters"]` from `result["fwdtransforms"]`.
   - Contains a silent `except Exception: return moco, None` at `line 643`, violating `docs/antsx_implementation_standards.md` Rule 2.

4. **Corrupted DWI Resampling & Double-Attribution in `motion_correct_grouped` (`src/syntx/motion_reference.py:499-516` & `src/syntx/motion_batched.py:1717-1725`)**:
   - In `motion_correct_grouped(..., preset="dwi")`, lines 502–514 automatically copy `pe_affine_covariates` into `kw["encoding_covariates"]`.
   - Inside `_batched_rigid_register_pass_core`, `delta_Y_np` is subtracted from `t_final` and `R_final` (lines 1695–1697) before writing the `.mat` rigid transforms, while `C_np[0] += B0` is fit to `t_tot_for_pe = t_final + delta_Y_np[:, :3]`.
   - Furthermore, lines 625–680 of `src/syntx/motion_reference.py` add a second post-hoc polynomial + non-PE regression (`X_pe = [1, t, t^2, t^3, T_perp, rv]`) that transfers `t_par` residuals into `pe_affine["shift_mm"]` and overwrites `fd`.

5. **CPU Memory Duplication in `build_bilateral_window_references` (`src/syntx/motion_reference.py:247-305`)**:
   - Allocates two full 4D float32 arrays on CPU (`stack` and `ref_stack`) and returns $T$ CPU `ANTsImage` objects, forcing per-frame CPU-to-GPU copies and a sequential `for b in range(B): gaussian_smooth_tensor` loop in `_batched_rigid_register_pass_core`.

#### Required Fixes in `syntx`
1. **Actually wire `B_win`, `E_psi_t`, `reg_loss_fn`, and `_project_encoding_gauge` into `_batched_rigid_register_pass_core`**:
   - Inside `compute_loss` and `stage_points`:
     ```python
     if enc_on:
         delta_Y = torch.einsum("bmk,mkd->bd", E_psi_t, B_win)
         t_eff = t_param + delta_Y[:, :3]
         om_eff = omega + delta_Y[:, 3:]
     else:
         t_eff = t_param
         om_eff = omega
     ```
   - Pass `reg_loss_fn` into `_run_converging_stage` when `surr_w > 0.0 and B >= 3`:
     ```python
     def stage_reg_loss():
         u_t = t_param[1:-1] - 0.5 * (t_param[:-2] + t_param[2:])
         u_w = 50.0 * (omega[1:-1] - 0.5 * (omega[:-2] + omega[2:]))
         u_sq = (u_t * u_t).sum(dim=-1) + (u_w * u_w).sum(dim=-1)
         return float(B) * surr_w * (surr_delta ** 2) * (torch.sqrt(1.0 + u_sq / (surr_delta ** 2)) - 1.0).mean()
     ```
   - Call `_project_encoding_gauge(final_stage=(stage_idx == len(schedule) - 1))` at the end of each stage so `t_param` and `omega` carry pure head motion $\mathbf{Y}_{\text{head}, t}$ while `B_win` carries the encoding-locked offset.
2. **Fix `tests/test_motion_dwi_preset.py:60`**:
   - Update assertion to `assert set(MOTION_PRESETS) == {"dwi", "bold", "asl", "slowflow"}`.
3. **Gate `separate_bold_pe_pseudomotion`**:
   - Do **not** run `separate_bold_pe_pseudomotion` unconditionally on `preset="bold"` unless explicitly requested (`debias_pe_pseudomotion: bool = False`), and never strip pitch rotation `r_ro` or slice translation `t_sl` as "PE pseudomotion" (respiratory $\Delta B_0$ shifts strictly along $\mathbf{p}_{\text{PE}}$ in 2D EPI). Remove the bare `except Exception: return moco, None`.
4. **Do not auto-inject `pe_affine_covariates` into `encoding_covariates` in `motion_correct_grouped(..., preset="dwi")`**:
   - `pe_affine_covariates` already has its own dedicated 4-DOF eddy-current solver (`Ccoef` along `pe_axis` with Jacobian modulation) in `_batched_rigid_register_pass_core`.

---

### Repo 3: `antsxdwi` (`/Users/stnava/data/repos/antsxdwi`, `v0.27.0` / `88e94f9`)

#### What Changed in `88e94f9`
1. `src/antsxdwi/diagnostics.py`:
   - Changed standardized DVARS scaling factor at `line 77` from `1000.0` to `100.0` (in `%`, matching `antsxqc v0.1.2`).
   - Updated `compute_qc_wide_row` (`lines 85-135`) to accept precomputed `dvars_trace_arr` from `mc_result.dvars`.
2. `src/antsxdwi/io.py`, `src/antsxdwi/motion.py`, `src/antsxdwi/pipeline.py`, `src/antsxdwi/series_pair.py`, `src/antsxdwi/figures.py`:
   - Added `translations_mm` and `rotations_deg` columns (`tx_mm`, `ty_mm`, `tz_mm`, `rx_deg`, `ry_deg`, `rz_deg`) to `motion_params.csv`.
   - Added `series_boundary` dashed divider to `motion_trace` and `render_motion_parameters_figure`.

#### Issues & Required Actions in `antsxdwi`
1. **DVARS Rescaling Side-Effects (`syntx/src/syntx/motion_reference.py:575-610` -> `antsxdwi/src/antsxdwi/pipeline.py:1483`)**:
   - Because `syntx 7a354a5` overwrote `results[g]["dvars"]` by multiplying every non-$b_0$ shell's DVARS by `med0 / medg`, passing `dvars_trace_arr=mc_result.dvars` into `compute_qc_wide_row` makes `dti_dvars_dwi_mean` artificially equal to `dti_dvars_b0_mean`.
   - **Fix**: Compute `dti_dvars_b0_mean` and `dti_dvars_dwi_mean` in `compute_qc_wide_row` from the unscaled per-shell DVARS (`dvars_shell_raw` or `diagnostics.dvars` per subset), using the harmonized trace only for continuous plotting.

---

### Repo 4: `antsxfunctional` (`/Users/stnava/data/repos/antsxfunctional`, `v0.15.0` / `bc04520`)

#### What Changed in `bc04520`
1. `src/antsxfunctional/perfusion/pipeline.py` (`lines 1107-1199`):
   - Passes `encoding_covariates=s_tc` into `syntx.motion_correction(...)` and then calls `_debias_asl_tag_control_motion(asl_image, moco, volume_types)`.
2. `src/antsxfunctional/rsfmri/pipeline.py` (`lines 200-225`):
   - Resolves `pe_axis_vec` and passes `pe_axis=pe_axis_vec` into `syntx_motion_correction`.

#### Issues & Required Actions in `antsxfunctional`
1. **Silent Exception Violation (`src/antsxfunctional/rsfmri/pipeline.py:217`)**:
   - Line 217 has a bare `except Exception: pe_axis_vec = None`, which violates Rule 2 of `docs/antsx_implementation_standards.md` (`tests/test_implementation_standards.py` / `antsxqc.lint`). Log or warn explicitly instead of silently swallowing.
2. **ASL Double-Resampling (`src/antsxfunctional/perfusion/pipeline.py:1148-1156`)**:
   - Passing `encoding_covariates=s_tc` to `syntx.motion_correction` already debiases the transforms and resamples all $T$ 3D volumes once inside `syntx`. Calling `_debias_asl_tag_control_motion` immediately afterward resamples all $T$ volumes a second time. Avoid the redundant resampling pass when `encoding_pose_bias_mm_rad` is already returned by `syntx`.

---

### Repo 5: `antsxslowflow` (`/Users/stnava/data/repos/antsxslowflow`, `v2.7.0` / `1362237` on `origin/main` vs. Local Working Tree)

#### State Comparison: `origin/main` (`1362237`) vs. Local Working Tree (`/Users/stnava/data/repos/antsxslowflow`)

| Pipeline Stage / Bug on `FPA_test_05` | `origin/main` (`1362237` / `v2.7.0`) | Local Working Tree (`/Users/stnava/data/repos/antsxslowflow`) |
|---|---|---|
| **0. Motion Correction (`src/antsxslowflow/motion/api.py`)** | Keeps 4-group split + polynomial OLS + `_project_encoding_surround_tukey_np` on Cardan angles + **`param_smooth_sigma=2.0` Gaussian filter** (which damps real motion spikes by 80%) | Implements `_debias_slowflow_multistate_se3` + `_write_se3_tangent_transform` around shared center $\mathbf{c}_0$ in $\mathfrak{se}(3)$ with `param_smooth_sigma=0.0` |
| **1. Stale Cache Poisoning (`unwrap.py`, `background.py`)** | **Unfixed** — reuses stale raw-phase `unwrap_phase.nii.gz` (`[-16.25, +13.73] rad`) | **Fixed** — SHA-256 input + parameter provenance fingerprinting (`unwrap_phase.cache.json`, `correct_background.cache.json`) |
| **2. Cross-State 4D BPC Bleeding (`background.py`, `study/api.py`)** | **Unfixed** — `_fit_background_pytorch_4d` couples $b_0 \leftrightarrow +X \leftrightarrow +Y \leftrightarrow +Z$; `correct_unwrapped_background` fits to `icv_canonical` | **Fixed** — `encoding_state_labels` restricts temporal & slice-timing TV pairs strictly within each encoding state; `correct_unwrapped_background` anchors strictly to the static parenchymal control mask |
| **3. $\pm\pi$ Wrap Cliff in `estimate_flow` (`flow/api.py`)** | **Unfixed** — linear OLS polynomial fit to wrapped `[-pi, pi]` phase creates `280 um/s` oblique artifact band (`leakage = 1.158`) | **Fixed** — `_fit_wrap_safe_dc_field` ($\text{SO}(2)$ circular-mean centering + 3-step circular Huber IRLS polynomial subtraction) + Huber-weighted circular frame averaging |
| **4. Gradient-Switching Alias in Spectrum/Oscillation (`spectrum/api.py`, `oscillation/api.py`)** | **Unfixed** — selects `0.0155 Hz` ($1/(4\cdot\text{TR})$ cycling alias) or `0.00179 Hz` (1.2-cycle linear drift) | **Fixed** — $\text{SO}(2)$ per-state demeaning before FFT, exclusion of $1/(K\cdot\text{TR})$ alias, polynomial drift detrending, and $\ge 3$-cycle requirement |
| **5. Midpoint Montage & Validity Report (`visualization_helpers.py`, `scireport.py`)** | **Unfixed** — `render_midpoint_montage` slices spatial axis 1 twice (`1D line` stretched to 2D barcode) | **Fixed in `visualization_helpers.py`** (`axis=3` temporal slice + `"RPI"` `quantplot` montage); `scireport.py` update ready to finish |

---

### Repo 6: `antsxmm` (`/Users/stnava/Documents/code/antsxmm`, `v4.3.2`)

- Orchestrates `antsxdwi`, `antsxfunctional`, and `antsxslowflow` via `antsxmm/modalities/bridges/{dwi,functional,slowflow}.py` and grades wide-table outputs via `antsxqc`.
- Once Repos 2–5 are finalized and unit-tested, re-run `FPA_test_05` (`SlowFlow`, `DTI`, `rsfMRI`, `perf`) to regenerate session reports and `mmwide.csv`.
