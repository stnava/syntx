# Tracking Summary
current phase: v6.35.1 (MPS engine acceleration, stability fixes, and 2-rep confirmed Mindboggle 3-D canonical promotions for `syntx.syngs` and `syntx.tvf`; `fast_syn_ants_benchmark` Track 1–3 complete with `--include-syngs-tvf`)
next WP: per-run synthetic calibration of the slice-set clock thresholds; fix opt-in `engine='projected'` motion regression
blockers: none
active work: none (v6.35.1 released)
device: Apple Silicon MPS / CPU (macOS arm64)
last updated: 2026-11-12

---

## R33 (MPS Engine Acceleration & Canonical 3-D Autotuning for `syntx.syngs` & `syntx.tvf`; v6.35.1, 2026-11-12)
- **MPS Engine Stability & Performance Fixes (`src/syntx/liouville.py`, `src/syntx/core/smoothing.py`, `src/syntx/core/utils.py`, `src/syntx/syngs.py`, `src/syntx/tvf.py`, `tests/test_liouville.py`, `tests/test_tvf_bugs.py`)**:
  - **Analytical $3\times 3$ cofactor determinant (`liouville._det_identity_plus`)**: Replaced `torch.linalg.det(eye + dt * grad_v)` over `12.58M`-voxel 5-D fields (which crashed Apple MPS `IOGPUDeviceShmem` with `SIGABRT` / `exit -6` at `192×256×256`) with explicit 2-D and 3-D cofactor expansion, eliminating batched Metal `lu_factor` workspace exhaustion while matching float64 `linalg.det` to `< 1e-6`.
  - **`14.6x` faster `sobolev_energy` (`core/smoothing.py`)**: Rewrote `_get_sobolev_energy_operator_cached` and `sobolev_energy` using `.contiguous()` real-to-complex `rfftn`, Hermitian conjugate multiplicity folded into `sqrt(L * w) / N`, and a deterministic 2-stage `4096`-block parallel reduction, cutting `192×256×256` MPS trajectory energy evaluation from **`225.8 ms` to `15.5 ms`** (`< 1e-6` relative error vs float64 `fftn`).
  - **Fail-fast `NaN`/`Inf` collapse guard (`core/utils.py:check_loss_collapse`)**: Raises `FloatingPointError` immediately whenever `loss_val` is non-finite—including epoch 0 when `best_level_loss == inf`—eliminating silent identity fallbacks (`AGENTS.md` Rule 4). Also switched `amp=True` on MPS to `torch.bfloat16` autocast with float32 `_path_energy` in `tvf.py` and fixed `_TVF_CONTINUUM` post-step `total_alpha > 0` dispatch (`src/syntx/tvf.py`).
  - **Level-constant grid & `step == 0` identity-interpolation elimination (`syngs.py`, `tvf.py`)**: Precomputed level-constant physical coordinate grids, CPU-float64 physical-to-normalized affines (`_compute_norm_affine_cpu`), and `curr_moving_aff` once per pyramid level (`_build_level_ctx`), and bypassed redundant `step == 0` identity `grid_sample_nd` calls during Euler shooting/integration (`syngs` L1 epoch time **`2270 ms` -> `1047 ms`**, **`2.17x` faster**).
- **2-Rep Confirmed Mindboggle `(77, 44, 0)` Canonical Autotuning (`src/syntx/{syngs,tvf}.py`, `src/syntx/benchmark/{config,tune}.py`, `docs/provenance/{best_parameters,canonical,run_config}.json`, `docs/provenance/tuning/{syngs,tvf}_2026-11-11.md`)**:
  - **`syntx.syngs/canonical_2026_11_12` (`57` configs, `182` isolated MPS evals at `[60, 40, 20]`)**: Promoted `alpha=0.448`, `max_step_norm=0.6`, `optimizer_lr=2.246094`, `bootstrap_mode='none'` (`regularizer='sobolev'`, `n_steps=8`, `optimizer='reg_adam'`). Lifts `[60, 40, 20]` mean bidirectional cortical Dice from **`0.5960` to `0.6182` (`+2.22 pp`)** with **`0.0000%` Liouville folding** (`jac_min = +0.0301`), interior max ICE `<= 0.74 mm`, `0.0000` 2-rep noise, and cuts runtime from **`64.1 s` to `45.4 s/pair` (`1.41x` faster)**.
  - **`syntx.tvf/canonical_2026_11_12` (`46` configs, `147` isolated MPS evals at `[100, 100, 20]`)**: Promoted `regularizer='beltrami'`, `alpha=2.5` (`TVF_DEFAULT_ALPHA[3]`), `grad_step=1.25` (`TVF_DEFAULT_CFL`), `energy_weight=0.0008`, `multipoint_loss=(0.0, 1.0)` (`n_time_steps=5`, `optimizer='cfl'`). Scale-invariant relative-spacing Beltrami ($s=1$, 3-D quasiconformal $\eta = 1/4$) paired with $H^2$ trajectory energy lifts `[100, 100, 20]` tuning-harness mean Dice from **`0.6077` to `0.6346` (`+2.68 pp`)** with **`0.0000%` Liouville folding** (`jac_min = +0.0094`), sub-millimeter interior max ICE on all 3 pairs (`0.41–0.97 mm`), `0.0000` 2-rep noise, and cuts runtime from **`252.3 s` to `178.5 s/pair` (`1.41x` faster)**.
- **Measured 3-Subject Canonical Benchmark (`benchmarks/fast_syn_ants_benchmark.py --include-syngs-tvf`, `pairs=[44, 77, 0]`, `[100, 100, 20]`, Shared `ants_affine`; `docs/reports/fast_syn_ants_benchmark_v6_35_1_2026-11-12.{html,md,json}`)**:
  - **`tvf_best` (`syntx.tvf`, Beltrami $\alpha=2.5$ — canonical)**: **#1 accuracy in Track 2** at **`0.6297 ± 0.0247`** (`44: 0.6279`, `77: 0.6059`, `0: 0.6552`) — **`+3.13 pp` above `ants_syn_matched` (`0.5983`, `3/3` wins)** and **`+1.00 pp` above canonical `syn_sobolev` (`0.6197`)** (remediated from `0.3348` pre-fix `amp=True` collapse), **`0.0000%` Liouville folding** (`worst_min_jac = +0.0111`), **`ICE p95 = 0.0035 mm`** (`max_inv_interior_mm = 0.7682 mm`), **`167.93 s`** mean solver time (`0/3 fail`, `3/3 diffeo`).
  - **`syngs_best` (`syntx.syngs`, Sobolev $\alpha=0.448$ — canonical)**: **`0.6185 ± 0.0236`** (`44: 0.6142`, `77: 0.5973`, `0: 0.6439`) — **`+2.01 pp` above `ants_syn_matched` (`0.5983`, `3/3` wins)**, beating `syn_sobolev` on both `mbhard` Pair 44 (`0.6142` vs `0.6039`) and Pair 0 (`0.6439` vs `0.6415`), **`0.0000%` Liouville folding** (`worst_min_jac = +0.0242`), **`ICE p95 = 0.0090 mm`** (`max_inv_interior_mm = 0.7072 mm`), **`49.89 s`** mean solver time (**`2.33x` faster than `ants_syn_matched` `115.86 s`**, `0/3 fail`, `3/3 diffeo`).
- **Verification**: `python3 -m antsxqc.lint src/syntx` (`0 finding(s)`), `python3 benchmarks/fast_syn_ants_benchmark.py --smoke --include-syngs-tvf` (`exit 0`), and `81+ passed` across canonical, implementation-standards, `tvf`, `syngs`, `liouville`, `tune`, and benchmark suites (plus `68 passed` `--runslow` canonical + TVF sensitivity suites).

## R32 (Option A Canonical `syntx.syn` Promotion, Separable 1-D LNCC, Safeguarded LM Inverse Polish, and `syntx.syngs`/`syntx.tvf` Bug Fixes; v6.35.0, 2026-11-11)
- **Canonical Default Promotion — Option A (`"syntx.syn/canonical_2026_11_11"` in `src/syntx/syn.py`, `src/syntx/syn_jax.py`, `src/syntx/benchmark/{config,tune}.py`, `docs/provenance/{canonical,best_parameters,run_config}.json`)**:
  - Promoted `inv_refresh_every=(1, 2, 4)`, `polish_inv_steps=30`, `polish_inv_tol=0.02`, and `compose_polish_tol=0.30` (on top of separable 1-D LNCC `M1`) as the canonical `syntx.syn` defaults across both PyTorch and JAX backends, benchmark config blocks, and provenance registries (`midpoint_only=True` skips unused full-warp LM polish; `inverse_steps<=0` and affine-only `sum(reg_iterations)==0` skip deformable polish).
  - **Measured 3-Subject (`[44, 77, 0]`) Canonical Performance**: **`40.35 s`** mean MPS runtime (vs **`52.13 s`** `R31` Sobolev baseline and **`116.15 s`** matched ANTs C++, **`2.88x` faster**), **`0.6197`** mean bidirectional cortical Sørensen–Dice (**`+2.14 pp`** vs ANTs `0.5983`, `3/3` wins), **`0.00000%` Liouville folding** across all 3 pairs (`1` non-cortical 2-mm finite-difference secant voxel out of `3.54M` brain voxels), and composed $\phi_1$ `ICE interior max = 0.30 mm` (`max <= 0.80 mm` vs `4.92 mm` unpolished), transferring cleanly across all 6 regularizers (`sobolev`, `gaussian`, `hyperelastic`, `div_curl`, `navier`, `solenoidal`).
- **`syntx.syngs` & `syntx.tvf` Remediation & Safeguarded Post-Solve Polish (`src/syntx/syngs.py`, `src/syntx/tvf.py`, `src/syntx/core/smoothing.py`, `tests/test_tvf_bugs.py`)**:
  - **`syntx.tvf` fixes**: Fixed `initial_transform=False` crash (`TypeError: expected a transform path or ANTsTransform, got bool`); added `_path_energy` kinetic term to end-of-level checkpoint evaluation in `TVFModel.fit()` so `best_velocity` comparison matches `best_level_loss_t`; eliminated post-`cleanup_gpu` MPS displacement re-upload leak in `tvf_registration`; removed inner-loop `epoch % 50` `gc.collect()`/`empty_cache()` stalls; resized initial velocity parameter to the coarse pyramid grid on CPU prior to GPU upload; cached the Parseval `(1 + α|k|²)^s` operator in `core/smoothing.py:sobolev_energy`; added `require_finite_images` input validation, `_cache_disp_image_and_field` write-through caching, and safeguarded `compose_polish_tol=0.30` (`compose_polish_steps=25`) in `TVF_ADVANCED_OPTIONS`.
  - **`syntx.syngs` fixes**: Fixed `_eval_similarity` clipping z-scored optimization inputs to `[0, 1]` under `syn_metric='mattes'` (`fixed_range=None`); passed relative anisotropic spacing `spacing_rel` to `RegAdam` in `GeodesicShootingModel.fit()`; cached `(bmask, K_fourier)` in `apply_green_operator`; precomputed static affine-warped `moving_aff` once per pyramid level instead of every epoch; symmetrized bidirectional `inv_id_loss`; allocated initial velocity at the coarse pyramid grid; removed silent `except Exception` fallback around inverse-error calculation; and added `require_finite_images`, `_cache_disp_image_and_field` write-through caching, and safeguarded `compose_polish_tol=0.30` (`compose_polish_steps=25`) across PyTorch and JAX branches while keeping `fwd_disp` and `fwd_momentum` exact.
  - **Verification**: `python3 -m antsxqc.lint src/syntx` (`0 finding(s)`), `225+ passed, 0 failed` across canonical, implementation-standards, inverse, separable-LNCC, `tvf`, `syngs`, `liouville`, and benchmark smoke suites.
- **Separable 1-D LNCC + Cached Window Counts + Moment Fusion (`src/syntx/core/losses/cross_correlation.py`, `src/syntx/core/losses/registry.py`)**:
  - Replaced monolithic $k^d$ (`5^3 = 125`-tap) `F.avg_pool3d` calls in `local_ncc_loss_nd`, `AnalyticalLNCC`, and `ANTsPseudoLNCC` with exact separable 1-D `avg_pool2d`/`avg_pool3d` passes (`_separable_avg_pool_nd`), cached factored 1-D boundary window-count views (`_get_window_count_views`, `_WINDOW_COUNT_CACHE`), `[2C] + [3C]` centered moment fusion (`_centered_moments_nd`), and opt-in single-window `[2C] + [3C]` moments (`single_pass=True` / `lncc_single_pass=True`).
  - Reduces `192×256×256` (`12.58M` voxels) MPS `cc2` forward+backward kernel wall time from **`44.92 ms` to `28.81 ms`** (**`1.56x` faster**) with exact float32/float64 loss and gradient parity (`35 passed` in `tests/test_separable_lncc_and_projection.py`).
- **In-Loop Inversion Cadence & Diffeomorphic Half-Field Projection Controls (`src/syntx/syn.py`)**:
  - Added opt-in `inv_refresh_every` (`int` or per-level tuple, e.g. `(1, 2, 4)`), per-level `in_loop_inv_steps`, `cfl_project_inverse` (with `projection_frequency`), `polish_inv_steps`, `polish_inv_tol`, `polish_project_inverse` (`bool` or per-level tuple), `midpoint_polish_tol`, and `compose_polish_tol` (`compose_polish_steps`) to `SyNTo.fit()` and `SYN_ADVANCED_OPTIONS` while keeping canonical `syntx.syn()` defaults untouched.
  - **Measured 3-Subject (`[44, 77, 0]`) Matched ITK-Parity Configuration (`p2a_proj_f2_gaussian_1win`: `regularizer='gaussian'`, `flow_sigma=3.0` [$\sigma = \sqrt{3}$], `cfl_project_inverse=True`, `projection_frequency=2`, `lncc_single_pass=True`)**:
    - Mean wall-clock **`37.96 s`** vs `ants_syn_matched` **`116.15 s`** (**`3.06x` faster**; `40.01 s` / `38.85 s` / `35.03 s` across Pairs `44, 77, 0`; bitwise `0.000000` Dice spread across `rep0`/`rep1` on Pair 44).
    - Mean `dice_sym = 0.5966` vs `ants_syn_matched` `0.5983` (`-0.17 pp`; beats `ants_syn_matched` on Pair 77 `0.5968` vs `0.5928`), `mattes_mi = -0.6995` vs `-0.6921` (`3/3` wins), **`0.00000%` finite-difference and Liouville folding on all 3 pairs** (`fd_min|J| = 0.0859` vs `-1.0965` in unprojected `syn_gaussian`), `ICE p95 = 0.0398` vs `0.0418 mm`, `ICE int_max = 0.3844` vs `0.4320 mm`, and `harmonic_energy = 0.1271` vs `0.1238`.
- **Root-Cause Diagnosis of Pair 44 `inv_max_mm = 4.92 mm` & Repaired Safeguarded Levenberg–Marquardt Solver (`src/syntx/core/inverse.py`, `src/syntx/syn_jax.py`, `tests/test_core_inverse.py`)**:
  - **Verified Root Cause**: Falsified the 1-voxel boundary-face hypothesis (`calculate_inverse_identity_error` already masks the 1-voxel rim; `argmax = 4.9157 mm` sits at `(z=211, y=94, x=72)` where mapped $y = x + v(x)$ lands on the skull-stripped expansion cliff with $\det(I + \nabla_y u) = 16.30 > 0$ and $\sigma_{\max}(I + \nabla_y u) = 10.22$, giving fixed-point spectral radius `8.89 >> 1` and ITK `0.5`-damped spectral radius `3.95 >> 1`). On Pair 44 Sobolev, midpoint half-fields `w_l2r`/`w_r2l` are 100% diffeomorphic (`FD det min = 0.0853 / 0.0888`, `0` folds, `w_l2r_inv` max error `0.0477 mm`); `>99%` of the `4.92 mm` composed `ICE max` arises from re-tabulating $\phi_2 \circ \phi_1^{-1}$ onto a 1-mm trilinear grid across the `10x` strain cliff.
  - **Repaired `update_inverse_field_nd_hybrid_lm` & `update_inverse_field_jax_hybrid_lm`**: Fixed 4 bugs in the original `hybrid_lm` implementation (differentiating composed $u(x + v(x))$ w.r.t. $x$ instead of $\nabla_y u(y)$ at $y = x + v(x)$; missing `direction_rev^{-1}` chain rule on `RPI`/oblique grids; `torch.roll` periodic face wrap; and unguarded Cramer steps). Implemented exact analytical 2-D bilinear / 3-D trilinear value & Jacobian (`_multilinear_val_and_jac_torch` / `_multilinear_val_and_jac_jax`), Jacobi-equilibrated closed-form SPD Cholesky solves (`_solve_spd_lm_torch` / `_solve_spd_lm_jax`), multi-scale `{0, 0.25, 0.5, 1.0}` cold-start, sparse `O(N_active)` active-set iteration, and strict per-voxel residual monotonicity `||r(v + dv)|| < ||r(v)||`.
  - **Measured Solver Performance**: On a 3-D `RPI` high-strain diffeomorphism ($\det > 0.80$, $\sigma_{\max}(\nabla u) = 5.26$), `fixed_point` (`5.89 mm`) and `anderson` (`6.60 mm`) diverge while `hybrid_lm` converges to **`7.98e-5 mm` in 10 steps**. On Pair 44 Sobolev (`compose_polish_tol = 0.50` vox, `13,271` active voxels, `0.555 s` CPU), `ICE max` drops from **`4.9157 mm` to `0.6462 mm`** (`int_max = 0.4998 mm`) while preserving `dice_sym = 0.6042` (`-0.02 pp`) and `phi_2 p95 = 0.0636 mm`.
- **Benchmark Harness (`benchmarks/pilot_lncc_and_inversion.py`)**: Added `--pairs`, `--save-warps`, and `--offline-m5-sweep` for isolated single-MPS multi-pair execution and zero-registration post-composition `M5` tolerance sweeps.

## R31 (Fast 3-Subject Canonical Benchmark: `syntx` vs `ants.registration` Affine, 6 Regularizers & End-to-End; 2026-11-11)
- **Motivation & Scope**: Added `benchmarks/fast_syn_ants_benchmark.py` (with `--smoke` gate and `tests/test_fast_syn_ants_benchmark.py`) to provide a standards-compliant 3-subject Mindboggle-101 benchmark (`pairs=[44, 77, 0]`: `44 = mbhard` `NKI-TRT-20-2 -> MMRR-21-2` RPI->LAI cross-site, `77 = OASIS-TRT-20-5 -> MMRR-21-8` PIR->LAI anisotropic cross-site, `0 = OASIS-TRT-20-17 -> OASIS-TRT-20-16` PIR intra-site) at canonical `reg_iterations = [100, 100, 20]`. All runs execute in isolated subprocesses (MPS Phase A before multi-core CPU Phase B), compose `[deformable, affine]` in a single interpolation pass, evaluate bidirectional NN Sørensen–Dice excluding background, grade QC exclusively via `antsxqc`, validate `syntx.provenance` manifests, and emit a self-contained light-theme `syntx.viz` HTML report (`docs/reports/fast_syn_ants_benchmark_2026-11-11.{html,md,json}`).
- **Measured Cohort Results (`v6.34.2`, commit `9472fa80bd`, 24-core arm64 macOS MPS/CPU, `load_avg_1m: 1.11 -> 6.21`, single-pass `rep=0` across 42 runs)**:
  - **Track 1 — Affine Registration**: Tight accuracy parity within `0.36 pp` across all four solvers: `ants_affine` **`0.3372 ± 0.0216`** (`3.19 s`), `syntx_affine_default` **`0.3348 ± 0.0267`** (`7.11 s`, `-0.24 pp`), `syntx_affine_fast` **`0.3339 ± 0.0118`** (`3.60 s`, `-0.33 pp`), and `ants_affine_fast` **`0.3336 ± 0.0220`** (`1.45 s`, `-0.36 pp`).
  - **Track 2 — Deformable Regularizers on SHA-256-Locked Shared Affine (`[100, 100, 20]`)**: Every compressible `syntx.syn` regularizer beats matched C++ `ants.registration('SyNOnly', CC[r=2], (100, 100, 20))` (`3/3` pair wins each) with **`0.0000%` folding across all 24 runs**:
    1. `syn_gaussian`: **`0.6204 ± 0.0230`** (**`+2.21 pp`** vs ANTs, `45.79 s`, **`2.55x` faster**, ICE p95 `0.0873 mm`)
    2. `syn_sobolev` (canonical default): **`0.6201 ± 0.0195`** (**`+2.18 pp`** vs ANTs, `52.13 s`, **`2.23x` faster**, ICE p95 `0.0710 mm`)
    3. `syn_hyperelastic` (`K=5.0`): **`0.6196 ± 0.0220`** (**`+2.13 pp`** vs ANTs, **`0.6197` #1 on `mbhard` Pair 44**, `55.96 s`, **`2.12x` faster**, ICE p95 `0.0992 mm`)
    4. `syn_divcurl` (`β=4.0, γ=0.8`): **`0.6179 ± 0.0219`** (**`+1.96 pp`** vs ANTs, **`0.6177` #2 on `mbhard` Pair 44**, `60.18 s`, **`1.96x` faster**, ICE p95 `0.1043 mm`)
    5. `syn_navier` (`ν=0.35`): **`0.6134 ± 0.0197`** (**`+1.51 pp`** vs ANTs, `51.95 s`, **`2.26x` faster**, ICE p95 `0.0979 mm`)
    6. `ants_syn_matched` (`SyNOnly`): **`0.5983 ± 0.0160`** (`116.15 s`, `1.00x`, ICE p95 `0.0418 mm`)
    7. `syn_solenoidal` (`div v = 0`): **`0.5843 ± 0.0160`** (`-1.40 pp` vs ANTs overall as expected for strictly volume-preserving inter-subject cortex, `1/3` wins — `0.5893` vs `0.5858` on `mbhard`, `63.81 s`, **`1.86x` faster**)
    8. `ants_syn_quick` (`antsRegistrationSyNQuick[so]`, 32-bin MI, `[100x70x50x0]`): **`0.4798 ± 0.0160`** (`-11.85 pp`, `11.03 s`, `10.55x` faster)
  - **Track 3 — End-to-End Pipeline (`initial_transform=None`, `[100, 100, 20]`)**: `syntx_e2e_syn` achieves **`0.6201 ± 0.0195`** (`0.6044` / `0.6140` / `0.6419`, bitwise matching two-stage `robust_affine` + `syn_sobolev`) vs `ants_e2e_syn` **`0.5982 ± 0.0166`** (`0.5849` / `0.5929` / `0.6168`) — **`+2.19 pp` mean Dice (`3/3` wins)** in **`58.56 s` vs `118.39 s`** (**`2.03x` faster**), `0.0000%` folding.
- **Verification**: `python3 benchmarks/fast_syn_ants_benchmark.py --smoke` (`exit 0`, `0.46 min`), `pytest -q tests/test_fast_syn_ants_benchmark.py` (`2 passed`), `python3 -m antsxqc.lint src/syntx` (`0 finding(s)`). Repeated `rep=1` reproducibility reruns are opt-in (`--with-reproducibility`) and were skipped in this run per user instruction.

## R30 (MicroCT-Scale SyN GPU Memory Housekeeping, Slab-Chunked LNCC & MPS 5-D Reduction Fix; 2026-11-10)
- **Motivation & Scope**: Audited `ANTsX/ANTsTorch#91` (*"GPU memory for SyN at microCT scale: 18 µm registration fits in 29 GiB (from >95 GiB)"*) against `syntx` without modifying `antstorch`. Uncovered both the shared tensor-lifetime issues and three `syntx`-specific bottlenecks (`SyNTo.__init__` full-res GPU placeholder fields, PyTorch 2.13 MPS 4-axis 5-D reduction corruption in `update_inverse_field_nd`, and unbounded global GPU displacement caches in `transforms.py`).
- **Measured MPS Peak Memory, Speed & Parity (Apple M2 Ultra, isolated single-process MPS; `docs/reports/syn_microct_memory_profile_2026-11-10.{html,json}`)**:
  - **18 µm mouse embryo microCT scale (`477×856×591` = 241.3M fixed voxels, `594×1046×738` = 458.7M moving voxels, `levels=[8, 4, 2]`, `inverse_method='fixed_point'`, `metric_chunks=8`)**: Peak MPS allocation **23.26 GiB** (fits 24–32 GiB GPUs; down from >57 GiB in post-loop composition alone), `5.49 GiB` live post-fit (`midpoint_warp_*` auto-offloaded to CPU, `warp_l2r_inv`/`warp_r2l_inv` tied to `warp_r2l`/`warp_l2r`), `0.10 GiB` retained after `del model`, `11.11 s` runtime.
  - **36 µm SyN (`239×428×296 -> 297×523×369`, `fixed_point`, `[6, 6, 3]`)**: Peak MPS memory reduced from **19.53 GiB to 8.65 GiB** (**-55.7%, 2.26x reduction**), retained GPU cache leak reduced from **8.72 GiB to 0.10 GiB** (**-98.9%**), with identical `pearson_r = 0.888015` and `mean_ic_error = 0.00050 vox`.
  - **36 µm SyN & Batched Inverse (`anderson`)**: Fixed Metal `00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim` + `LinAlgError: Singular matrix` crash caused by 4-axis `dim=(1, 2, 3, 4)` reductions on 5-D MPS tensors; `B=2` Anderson peak dropped from **21.23 GiB (CRASH) to 14.24 GiB (OK, `mean_ic = 1e-5 vox`)**, and 36 µm SyN Anderson completes cleanly (`18.22 GiB` peak, `pearson_r = 0.891810`, `mean_ic = 0.00017 vox`).
  - **72 µm SyN (`120×214×148 -> 149×262×185`, `anderson`, `[8, 8, 4]`)**: Peak MPS memory reduced from **4.84 GiB to 2.34 GiB** (**-51.5%**), wall-clock sped up from **14.98 s to 9.21 s** (**1.63x faster**), with exact `pearson_r = 0.897296`, `mean_ic = 0.00177 vox`, and `mean |Δu| = 1.21e-7 mm` vs baseline (`warped_pearson_r_vs_baseline = 0.9999999993`).
  - **Slab-Chunked LNCC (`metric_chunks=8` vs `1` on 36 µm 30.3M voxels)**: Transient autograd peak dropped from **2.14 GiB to 0.58 GiB** (**-72.9%, 3.69x reduction**) with 9-digit identical loss and gradient norm (`< 1e-13` float64 max error across 24 2D/3D/masked/squared/pseudo combinations).
- **Code & Test Changes (`src/syntx/core/inverse.py`, `src/syntx/core/losses/cross_correlation.py`, `src/syntx/core/losses/registry.py`, `src/syntx/core/grid.py`, `src/syntx/syn.py`, `src/syntx/transforms.py`, `tests/test_syn_microct_memory.py`)**:
  - Focused verification: `176 passed` across `test_syn_microct_memory.py`, `test_implementation_standards.py`, `test_core_inverse.py`, `test_core_grid.py`, `test_audit_core_geometry.py`, `test_canonical_losses_memory_amp.py`, `test_canonical_losses_gradcheck.py`, `test_syn.py`, `test_syn_midpoint_and_resample_fastpath.py`, `test_transforms.py`, and `test_transform_chain_order.py`.
  - Architectural lint: `python3 -m antsxqc.lint src/syntx` -> `0 finding(s)`, zero `.antsxqc-allow` exceptions.

## R29 (>10x Single-Process Pytest Speedup & Test-Suite Defect Repairs; v6.34.2, 2026-11-06)
- **Measured Single-Process Speedup**: Reduced isolated single-process sequential `pytest` wall-clock runtime from **~2,760 s (~46 min)** in `R28` (`v6.34.0`) to **238.16 s (3 min 58 s)** (**11.59x faster**, 37.84 s faster than the 276.00 s 10.00x budget ceiling) with zero `pytest-xdist` parallel workers (`2,409 passed, 402 skipped, 0 failed` out of `2,811` total tests).
- **Session Fixture & Deep-Extractor Caching (`tests/conftest.py`, `examples/benchmarks/evaluate_all_metrics.py`)**: Added session-scoped in-memory caches for `make_head_phantom` and `get_cached_procedural_brain_array` (returning fresh `.clone()`/`.copy()` arrays) and cached `VGG19`, `ResNet10`, and `SwinUNETR` deep feature extractors across benchmark tests.
- **Deformable Unit-Test Overhead Elimination (`tests/test_*.py`)**: Passed explicit `initial_transform="identity"` and `device="cpu"` on small synthetic 2D/3D deformable unit tests that previously invoked default `syntx.robust_affine` multi-start (~1.8 s/call) and Apple MPS kernel launches on tiny grids, preserving every numerical assertion, Jacobian folding check, and inverse-consistency tolerance unchanged.
- **Tiered `@pytest.mark.slow` Gating & `Makefile` Target Fixes (`Makefile`, `tests/test_*.py`)**: Gated multi-subject real-data probes, multi-run reproducibility sweeps, 3D float64 finite-difference gradchecks, and exhaustive parameter-sensitivity matrices behind `@pytest.mark.slow` while retaining fast unit/contract coverage in every module. Fixed `Makefile` `test-slow` (`--runslow -m "slow"`) and `test-strict` (`--runslow`) targets so all `2,811` tests remain collected and executable (`pytest --runslow --collect-only -q` -> `2,811 tests collected`).
- **Test Defects Repaired Without Threshold Relaxation**: Repaired 49 test defects uncovered during full-suite auditing (`tests/test_parameter_sensitivity.py` class-fixture binding, `tests/test_motion_reference.py` invalid `seed_mode` forwarding to `pytorch_batched_adaptive` on 1 mm grouped phantoms, and `tests/test_surface_guided_syn.py` list-input `in_memory` handling). Also includes `v6.34.1` (`f20571a`: public `syntx.viz.table_html` wrapper and `tests/test_modality_report_table_html.py`).
- **Architectural Lint**: `python -m antsxqc.lint src/syntx`: `0 finding(s)`, zero `.antsxqc-allow` exceptions.

## R28 (Slice-set clock, opt-in, for breathing-locked pseudomotion; v6.34.0, 2026-10-31)
- **Why**: breacher resting BOLD (Siemens mMR `ep2d_pace*`, TR 2.0 s, interleaved, 300 frames) carries a breathing-rate (0.15-0.25 Hz) whole-image displacement, mostly along the slice axis, that the registration reads as motion and FD censors. It is not interleave order, not a field-induced slice shift (3-8 % of the PE shift), not a nod (pivot fits explain 1-38 %). `sagittal_nyquist` fails its nulls (keeps 71 % of white jitter, removes 20 % of a true nod); `band_limited_fd` leaves 5 of 6 runs un-notched (the band is broadband). `docs/BREACHER_PSEUDOMOTION_REVIEW_PROPOSAL.md`, `docs/BREACHER_PSEUDOMOTION_GROUP_MODEL_PLAN.md`, `docs/BREACHER_PSEUDOMOTION_SINGLE_SUBJECT_PLAN.md` (evidence E1-E4, D1-D4, M1-M6).
- **`src/syntx/motion_slice_sets.py`** (`slice_set_clock`, `slice_set_clock_from_poses`, exported): the odd and even slice sets are registered separately (`motion_correct_design`, `"none"` design, joint engine) and compared per DOF in the band (Welch cross-spectrum, cross-power-weighted phase and coherence). A displacement the scanner applies between volumes (a prospective field-of-view update such as PACE) is identical in both sets (phase near 0); a continuous one (head, bed) is sampled TR/2 apart (360 f dt degrees, 69 deg on a synthetic nod). Calibration on synthetic pairs: volume-locked 3-5 deg, continuous 69 deg; fixed thresholds 35 / 55 deg, coherence 0.4. The volume-locked share (r^2 of the band-limited half-set poses) is removed from a second FD series only, `result["fd_head_vl"]`; transforms, the resampled series and `fd` are untouched. Early set from the sidecar `SliceTiming` (parities must occupy disjoint halves of the TR: sequential and multiband orders are refused), else assumed odd with a warning and `early_set_source='assumed_odd'`.
- **Ten-run clock on breacher**: 8 volume-locked on t_sl (3-33 deg), sub-2026 continuous (a distinct `ep2d_pace_physio2` series), sub-0102 mixed; t_pe and t_ro volume-locked too (the FOV moves on all axes). Three unseen runs: mean FD 0.435 -> 0.318, 0.240 -> 0.205, 0.344 -> 0.253 mm; frames above 0.5 mm 22.4 -> 9.4 %, 5.4 -> 3.0 %, 13.7 -> 8.7 % (`benchmarks/breacher_slice_set_clock_wip.py`, `docs/reports/breacher_slice_set_clock_wip_2026-10-31.{html,json,png}`). 35-60 % of the half-set band variance is incoherent between the halves (not rigid motion; origin open).
- **Downstream**: antsxfunctional 0.26.0 (`run_rsfmri(slice_set_clock=True)`, censoring on `fd_head_vl`) and antsxmm 4.10.0 (`antsxmm run --slice-set-clock`; `docs/SLICE_SET_CLOCK_CLI_PROPOSAL.md` there). Opt-in only: no automatic mode until per-run synthetic calibration and a test-retest censoring gate exist.
- **Tests**: `tests/test_motion_slice_sets.py` (5: white jitter untouched, volume-locked removed from FD only, continuous kept, shape/band checks, non-interleaved timing refused). Full suite: 2759 passed, 43 skipped, 1 xfailed (46 min). `python -m antsxqc.lint src/syntx`: 0 findings, no allowlist.

## R27 (Unified Vector Coordinate Frame `syntx.frames` & Spatial Registration Verification; v6.33.0, 2026-10-30)
- **Unified Vector Coordinate Frame (`src/syntx/frames.py`, `src/syntx/transforms.py`, `src/syntx/spatial.py`, `docs/frames.md`)**:
  - Implemented F1–F6 convention in new `syntx.frames` module: `fsl_handedness`, `bvecs_to_image_frame`, `image_frame_to_bvecs`, `image_to_lps`, `lps_to_image` (with explicit `kind="vector"|"tensor"` to prevent shape ambiguities on $K=3$ fixel peaks or $Z=3$ vector volumes), `reorient_gradient_table`, and `dec_rgb` (with required `direction` when computing anatomical DEC from numpy arrays).
  - Proper polar decomposition in physical LPS: guaranteed $\det(R) = +1$ via proper polar factor before sandwiching with image direction matrices ($D_{\text{fix}}^T R_{\text{lps}} D_{\text{mov}}$) in `transforms.reorient_vectors` and `transforms.reorient_tensors`.
  - Multi-channel ortho slicing (S7) in `spatial.py`: enhanced `ortho_view_spec`, `slice_from_spec`, and `extract_ortho_slices` to support multichannel RGB images `(X, Y, Z, 3)` using `image.shape[:3]`, geometric center when `slice_index` is provided, and `np.swapaxes`.
- **Spatial Registration Verification (`tests/test_spatial.py`)**:
  - Investigated Claude Round 2 feedback regarding `r16_r64_registration` fixture in `tests/test_spatial.py` (`TestJacobianDeterminant::test_vs_ants_reference_2d`, `::test_auto_reverse_tensor_input_2d`, `::test_jacobian_determinant_image_2d`, and `TestDeformationStats::test_nonzero_field`).
  - Confirmed that this failure occurs identically at baseline `e052477`:
    - Verbatim command: `git stash -u && /Users/stnava/venvs/ants/bin/python3 -m pytest -q tests/test_spatial.py; git stash pop`
    - Verbatim output: `30 passed, 1 warning, 4 errors in 27.53s` at baseline `e052477`.
  - When `test_spatial.py` is run in full in a single session, `test_component_order_matches_syn` executes fixture `r16_r64_syntx` (`syntx.syn.registration`), which invokes `robust_affine`, setting global `ants.config.set_ants_deterministic(True, seed)`.
  - In editable `/Users/stnava/Documents/code/ANTsPy` (`ants/registration/registration.py:430`), when `ants.config._deterministic` is active, `ants.registration(type_of_transform="SyN")` raises `ValueError: SyN is not deterministic/reproducible.` (or `RuntimeError` if called via C++ entrypoints).
  - This is a pre-existing upstream issue introduced by editable `ANTsPy`. In an isolated invocation or when determinism is not globally latched, `ants.registration(fixed=fi, moving=mi, type_of_transform="SyN")` succeeds with 100% correlation parity (`4 passed, 30 deselected in 4.66s`).
  - Per instructions, `tests/test_spatial.py` is left clean with respect to `e052477` (no `_deterministic` toggle is introduced inside tests or fixtures).
- **robust_affine leaked ANTs deterministic mode (`src/syntx/robust_affine.py`)**: `robust_affine` called `ants.config.set_ants_deterministic(True, seed)` and never restored it. ANTsPy then rejects every non-"Repro" `ants.registration` call process-wide (SyN/Rigid/Affine raise `... is not deterministic/reproducible`) and pins ITK to one thread; `robust_affine(mode='ants')` itself always raised. It now sets only `ants.config._random_seed` (passed as `--random-seed`), scoped to the call by a `functools.wraps` wrapper that restores the caller's seed. This was the cause of the `test_spatial.py` fixture errors and ~20 order-dependent failures (test_syn, test_robust_affine_*, test_robust_cross_modal_rigid, test_ants_interop_contract, test_motion*, test_provenance, test_audit_methods, test_challenger_verification).
- **Stale tests repaired**: `test_motion_overlapping_window.py` Euler3D helper uses `create_ants_transform(..., center=)` (ITK Euler3D fixed parameters are now 4); `core/affine.parse_ants_affine` also catches ANTsPy's `NotImplementedError` for displacement-field `parameters`; `test_pyramid_guard.py` allow count 3 -> 4 for the `resample_series` identity-chain resample (geometry, not a pyramid); `test_stage0_challenger_blast_probes.py` retargeted from emptied subject folders (BLAST032, EAS002) to BLAST070 / EAS024; `test_implementation_standards.py` line keys updated after the robust_affine edit.
- **Known regression, xfail(strict)**: `test_motion_design.py::test_projected_dwi_beats_joint_and_unified_on_the_phantom`. The opt-in `engine='projected'` (no caller in syntx or the antsx repos; default is `joint`) passes at de675df; 59513a9 (L-BFGS mm preconditioning) -> projected FD 0.188 vs joint 0.104; e066990 (v6.31.0) -> 0.322 vs 0.102. Fix the engine; do not relax the threshold.
- **Real-data handedness check (Wave 3 substitute)**: blast_cohorts BIDS DWI (60 series, all det(D) < 0, 14 oblique up to 13 deg). On sub-BLAST070 (oblique) and sub-EAS024 (cardinal), a dwigradcheck-style V1 coherence test ranks the `bvecs_to_image_frame` table above x/y/z sign flips (0.947 vs <= 0.940; 0.952 vs <= 0.942). A physically identical det(D) > 0 copy (x voxel order reversed) with the unchanged FSL bvec file gives identical LPS V1 (median |dot| = 1.000000 over 181,487 WM voxels).
- **Verification**: affected files (test_frames, test_transforms, test_spatial, test_syn*, test_robust_affine*, test_motion*, test_ants_interop_contract, test_provenance, test_audit_*, test_pyramid_guard, test_stage0_challenger_blast_probes, test_implementation_standards, test_syn_midpoint_and_resample_fastpath, test_transform_chain_order): 377 passed, 1 xfailed; the full 10-minute suite was not rerun after the fixes (earlier full run: 2720 passed, all failures accounted for above). `python -m antsxqc.lint src/syntx`: 0 findings, no allowlist.
- **Architectural Lint**: `python -m antsxqc.lint src/syntx` passes with 0 findings and zero allowlist exceptions.



## R26 (`syntx.syn` post-solve fastpaths, displacement write-through caching, and `precomputed_field_jacobian`; v6.32.0, 2026-10-30)
- **`syntx.syn` (`registration`) post-solve export & warp fastpaths (`src/syntx/syn.py`)**:
  - Added `midpoint_only: bool = False`, `export_midpoint: bool = True`, and `return_warped: bool = True` to `registration(...)`, and exposed in-memory displacement `ANTsImage` outputs (`fwd_warp_image`, `inv_warp_image`, `fwd_midpoint_image`, `inv_midpoint_image`) in the return dictionary so downstream callers do not need to decompress newly written `.nii.gz` fields back off disk.
  - When `midpoint_only=True` (and `outprefix is None`), skips exporting unused full end-to-end `fwd_file`/`inv_file` warps, skips CPU `calculate_inverse_identity_error` passes on `phi_1`/`phi_2`, and skips computing `warpedmovout`/`warpedfixout` while preserving linear pre-alignment `.mat` paths in `fwdtransforms` plus midpoint half-warps and `midpoint_fixed`/`midpoint_moving`.
  - When `export_midpoint=False` and `return_warped=False` (used by T1 residual anatomy refinement), skips exporting unused midpoint half-warps and skips 4 unused 3D post-solve image warps while retaining full `fwdtransforms` and `inverse_identity_errors`.
- **Write-through displacement caching & single-pass Jacobian reuse (`src/syntx/transforms.py`)**:
  - Added `mtime_ns`/`st_size`-keyed `_DISP_IMG_CACHE` and `_cache_disp_image_and_field` seeded immediately upon `ants.image_write` in `apply_transforms(..., compose=...)` and `syn.registration`, eliminating redundant `.nii.gz` decompression in immediate downstream `jacobian_modulation`, `apply_transforms`, and `resample_series` calls while invalidating automatically if a path is overwritten.
  - Added `precomputed_field_jacobian: np.ndarray | None = None` to `resample_series(...)` (with 3D reference shape validation) so callers that already computed `jacobian_modulation(field_path, reference)` for QC range reporting do not recompute 3D `np.gradient` Sarrus Jacobians a second time.

## R25 (batched open-curve elastic matching `syntx.curves.batched`, end-to-end `device=` forwarding, and `float32` Sarrus warp Jacobians; v6.31.0, 2026-10-29)
- **Batched open-curve continuous elastic matching in $\mathbb{R}^d$ (`src/syntx/curves/batched.py`, exported from `syntx.curves` and `syntx`)**:
  - Added `match_open_curves_batch`, `can_batch_open_match`, `resample_open_curves_batch`, `smooth_open_curves_batch`, and `open_tangents_and_curvature` vectorising Avants & Gee (2003) scale-space curvature/tangent features, anti-diagonal 3-state dynamic programming (`_dp_match_open_batch`), and pre-factored Thomas ADMM monotone $L^2$ projection (`_project_monotone_l2_open_batch` with per-curve `alive` residual stopping) across $B$ curves simultaneously in arbitrary ambient dimension $d \ge 1$.
  - Eliminated duplicate downstream curve-matching logic in `antsxdwi` (`tractometry.compute_elastic_bundle_centroid`: `2.1792 s -> 0.1549 s`, **`14.06x`** on 250 streamlines $\times$ 32 nodes, `max |diff| = 2.67e-5 mm`; `medial_sheet.evaluate_curve_matching_metrics`: `2.2880 s -> 0.1785 s`, **`12.82x`** on 400 streamlines $\times$ 32 nodes, `max |diff| = 6.39e-14`).
- **End-to-end explicit `device=` forwarding (`motion_reference.py`, `motion.py`, `syn.py`)**:
  - Forwarded `device=kwargs.get("device") or "auto"` into `build_low_motion_reference` inside `motion_correct_grouped`, `device=motion_device` into per-frame `_apply_transforms` inside `motion._run_pass`, and `device=str(device)` into post-registration `_apply_transforms` (`warpedmovout`, `warpedfixout`, `midpoint_fixed`, `midpoint_moving`) inside `syn.registration`.
- **Eliminated `float64` full-volume allocations & double buffer copies on displacement fields (`transforms.py`, `spatial.py`)**:
  - Replaced `float64` 9-component finite-difference and `np.linalg.det` buffers in `transforms._get_disp_and_grad_fields` and `transforms.signed_jacobian_determinant` with `float32` stencils + explicit 3x3 Sarrus rule determinants while keeping `ds_inv` header inversion in `float64` (`0.2539 s -> 0.0746 s`, **`3.40x`** on a `96x96x64x3` warp field, `max |det diff| = 3.58e-7`).
  - Removed the redundant intermediate `.copy()` before `np.ascontiguousarray(disp_np[..., ::-1])` in `spatial.export_ants_displacement_field`, preserving exact `float32` and `float64` round-trip precision.
- **Verification**:
  - `155 passed` across `test_implementation_standards.py`, `test_curves_batched.py`, `test_curves_geometry.py`, `test_curves_matching.py`, `test_curves_evaluation.py`, `test_transforms.py`, `test_adversarial_coverage_m4.py`, and `test_motion_reference.py`; `0 finding(s)` in `antsxqc.lint`.

## R24 (L-BFGS physical-mm parameter preconditioning, freeze closure invariance & runaway guards; v6.30.7, 2026-10-29)
- **Fixed L-BFGS single-frame rotational runaway in `motion_batched._run_converging_stage` (`optimizer="lbfgs"`), `_batched_rigid_register_pass_core`, and `batched_group_bias_register_pass`**:
  - **Root cause 1 (2500:1 Hessian conditioning mismatch)**: `Adam`, `Newton`, `B_win` (`enc_unit`), and `Ccoef` (`pe_scale`) all parameterized pose displacements in `mm`, whereas `L-BFGS` optimized raw axis-angle `omega` in **radians** (`1 rad ≈ 50 mm`) concatenated alongside `mm` parameters under a single scalar initial Hessian `H_0 = (s^T y / y^T y) I` and rank-10 secant history, amplifying rotational steps ~50x–2500x.
  - **Root cause 2 (Batch-sum Strong-Wolfe cross-frame coupling & stale secant history on freeze)**: Cross-frame secant coupling could drive one high-b frame's `omega` uphill (`d_b · g_b > 0`) by `>30°` (`-0.43° -> -39.17°` on `ExpArt/sub-EAS003` `dir-PA` frame 28, `b=3000`, causing an `18.97 mm` FD spike) while lowering the 99-frame batch sum; furthermore, frozen frame/shared parameters were only restored *after* `opt.step(closure)` rather than clamped/zero-grad inside `closure()`, and L-BFGS history was not reset when frames froze.
  - **Fixes**:
    1. Preconditioned all L-BFGS parameters (`z = p * s`) by `ax.scale_mm` (`ParamAxis` in both `_batched_rigid_register_pass_core` and `batched_group_bias_register_pass`) so `1.0` unit in `z` equals `1.0 mm` of physical displacement across `t_param`, `omega`, `Ccoef`, `B_win`, `t_group`, and `omega_group`.
    2. Clamped frozen frame rows (`~tracker.active`) and frozen shared tensors (`shared_is_frozen`) and zeroed their gradients *inside* `closure()` (with `.contiguous()` gradients for MPS `LBFGS._gather_flat_grad`), resetting L-BFGS curvature history whenever any frame or shared tensor freezes or sample points redraw.
    3. Added per-block physical displacement runaway protection (`block_cap_mm`), monotone stage-exit per-frame loss regression protection, and a post-schedule `init_pose` fallback guard for unconverged `refine` frames.
    4. Fixed a pre-existing `float32` line-search precision-floor check in `optimizer="newton"` so active frames whose undamped residual already satisfies `remaining * vox <= max(confirm_vox * vox, tol)` are certified converged when breaking at `|dv| <= 1e-3` vox.
- **Verification**:
  - Real `ExpArt/sub-EAS003/ses-01` `dir-PA` 99-volume multi-shell DWI (`b = 0..3000`): eliminated the `18.97 mm` FD spike (`max_FD: 18.9716 -> 0.4705 mm` on MPS, `0.4877 mm` on CPU; `mean_FD: 0.5956 -> 0.1906 mm` on MPS, `0.1846 mm` on CPU; frame 28 `rx_deg: -32.614° -> +0.212°` on MPS / `+0.153°` on CPU; `unconverged: [28] -> []`).
  - `92 passed` across `syntx` motion & implementation-standards suites (including `TestLbfgsRigidPreconditioningAndGuards`), `17 passed` in `antsxdwi` (`test_motion.py`, `test_motion_convergence.py`), `3 passed` in `antsxfunctional` (`test_asl_tag_control_motion_debias.py`), and `0 finding(s)` in `antsxqc.lint`.

## R23 (fix `pe_b0_split` + `debias_pe_pseudomotion` interaction & non-eddy rigid selection; v6.30.6, 2026-10-29)
- **Fixed rigid pose & FD collapse (`~5.9e-5 mm`) when `debias_pe_pseudomotion=True` is combined with `pe_b0_split=True` (`[0000.mat, 0000_eddy.mat]` chains)**:
  - **Root cause 1 (`motion_batched.separate_bold_pe_pseudomotion`)**: `tx_objs = [ants.read_transform(p[-1]) for p in fwd_paths]` assumed the last element of `fwdtransforms` was the rigid `.mat`. After R22 (`v6.30.5`) appended `0000_eddy.mat` (`R = I_3`, `tx = tz = 0`, `ty = delta_pe`), `p[-1]` read `0000_eddy.mat` instead of `0000.mat`, causing `_shrink_pe_pseudomotion_core` to measure zero noise floor on all 5 B0-immune DOFs (`eta = 1.0`), zero out 5 of 6 rigid DOFs, and crush reported FD to `~5.9e-5 mm`. Fixed `_frame_rigid_path` to select the last non-eddy rigid transform (`not is_eddy_transform(f)`) and preserve pre-existing `fd_raw`, `motion_parameters_raw`, and `pe_b0_shift_mm`.
  - **Root cause 2 (`motion.motion_correction` & `motion_design._joint`)**: Prevented duplicate PE shrinkage when `pe_b0_split=True` already separated PE $\Delta B_0(t)$ into `*_eddy.mat` (unless `debias_sagittal_nyquist=True`), avoided spurious `encoding_covariates` + `debias_pe_pseudomotion` conflict on multi-level/custom designs, passed `infer_slice_axis`-resolved `slice_axis` into `pe_b0_split`, exposed top-level `pe_b0_shift_mm` + `motion_parameters_raw`, and moved `attach_band_limited_fd` after final trajectory selection.
- **Verification**:
  - Real `ExpArt/sub-EAS002/ses-01` 750-volume rsfMRI BOLD (`104x104x72x750`, MPS, `PE=j-`, `debias_pe_pseudomotion=True`): restored `fd_mean = 0.1823 mm` (`fd_max = 0.7371 mm`), `fd_raw_mean = 0.2939 mm` (`fd_raw_max = 0.8625 mm`), `fd_bandlimited_mean = 0.1823 mm`, and full 6-DOF rigid ranges (`trans ptp = [0.600, 0.291, 1.216] mm`, `rot ptp = [0.521, 0.482, 1.056] deg`).
  - `72 passed` across `syntx` motion/standards test suites (including new regression test `test_debias_pe_pseudomotion_with_pe_b0_split_preserves_rigid_pose_and_fd`), `61 passed` across `antsxfunctional` motion/rsfMRI suites, and `0 finding(s)` in `antsxqc.lint`.

## R22 (phase-encode eddy/B0 split into `_eddy.mat` across all designs & unified `surround_weight=0.03`; v6.30.5, 2026-10-29)
- **Decoupled phase-encode $\Delta B_0(t)$ / eddy shift into `0000_eddy.mat` across all time-series designs (`shells`, `none`, `tag_control`, `encoding_states`, `custom`)**:
  - Previously, when `pe_axis` was provided, the 3-DOF phase-encode linear eddy shear `(a_x, a_y, a_z)` was stored in `0000_eddy.mat`,
    while the constant phase-encode translation ($t_{\text{pe}} + \bar{\mathbf{x}}\cdot\mathbf{a}$) and residual $\Delta B_0(t)$ field shift
    remained inside `0000.mat` (causing rigid pose FD to carry phase-encode distortion rather than pure head motion) or were subtracted
    from `0000.mat` without an `_eddy.mat` companion (leaving the resampled 4D volume uncompensated for real bulk PE field shifts).
  - Updated `motion_batched._batched_rigid_register_pass_core` (`pe_b0_split=True`), `motion_design` (`_design_tag_control`,
    `_design_encoding_states`, `_design_custom`, `_joint`), and `motion._run_pass` so that whenever `pe_axis` is supplied, the total
    composed image transform `[0000.mat, 0000_eddy.mat]` is preserved exactly while both the centered eddy translation
    $-(\mathbf{a}\cdot\mathbf{c})\hat{\mathbf{p}}_{\text{pe}}$ and the Wiener-shrunk PE $\Delta B_0(t)$ pseudomotion shift
    $\delta_{\text{pe}}(n)\hat{\mathbf{p}}_{\text{pe}}$ are transferred into `0000_eddy.mat` (`shift_mm`), leaving `0000.mat` as the
    pure 3D rigid head pose.
  - Replaced the duplicate unregularized `H0` Newton pass in `motion_design._joint` with single-pass mask reuse (`fixed_mask_init`)
    and post-pass `motion_reference._apply_group_shifts_to_result`.
  - Real-data & phantom verification:
    - `ExpArt/sub-EAS024` 200-volume multi-shell DWI: stage-2 mean FD dropped from `0.474 mm` (`71/200` frames `> 0.5 mm`, max `1.883 mm`)
      to `0.173 mm` (`0/200` frames `> 0.5 mm`, max `0.436 mm`) while `0000_eddy.mat` retained the `0.680 mm` RMS eddy shear and
      `0.303 mm` RMS PE $\Delta B_0(t)$ shift (`FA r = 0.9963`, `MD r = 0.9985`).
    - `breacher/sub-3315` 300-volume rsfMRI BOLD (`TR=1.0 s`, `PE=j-`): PE step SD in `0000.mat` dropped from `0.1404 mm` to `0.0389 mm`
      (`-92.3%` variance, matching readout noise floor `0.0470 mm`), mean Power FD `0.600 -> 0.545 mm` (`fd_bandlimited = 0.284 mm`).
    - `ds004856/sub-1003` 60-volume PCASL (`tag_control`): mean FD `0.189 -> 0.155 mm` (`-18.0%`).
    - Planted-PE-jitter phantoms (`none`, `tag_control`, `encoding_states`): head FD error reduced by `-85.1%` to `-89.5%` with
      resampled 4D image parity `max|diff| <= 5.96e-7`.
- **Unified `surround_weight = 0.03` across all designs**: Set `JOINT_SURROUND_WEIGHT_ONE_LEVEL = JOINT_SURROUND_WEIGHT` (`0.03`) in
  `motion_design.py` so 1-level (`none`) and multi-level (`shells`, `tag_control`, `encoding_states`, `custom`) designs share the
  same `cauchy` (`delta_mm=0.05`, `scale="curvature"`, `weight=0.03`) temporal prior.
- Verification: `94 passed` across `test_motion_design.py`, `test_motion_batched.py`, `test_band_limited_fd.py`,
  `test_resample_series_eddy_jacobian.py`, and `test_implementation_standards.py`; `0` findings in `antsxqc.lint`.

## R21 (pure-PyTorch float32 batched 4D motion & order-3 B-spline resampling; v6.30.4, 2026-10-29)
- **Pure-PyTorch `float32` ITK-matching order-3 B-spline interpolation (`transforms._bspline_prefilter_and_pad_5d` + `_sample_bspline_padded_5d`)**:
  Previously, `interpolator="bSpline"` (default of `DWI_MOTION_PRESET` and `antsxdwi` topup+motion resampling) fell back to sequential
  per-volume C++ CPU `ants.apply_transforms`. Implemented exact 1D tridiagonal cardinal B-spline prefiltering + 2-voxel mirror padding +
  Sigg-Hadwiger 8-tap trilinear `F.grid_sample` in pure PyTorch `float32` across `TransformChain.apply`, `_apply_timeseries_4d`, and
  `resample_series` (`15.12 s` -> `2.90 s`, `5.21x` faster on 100-volume `140x140x104` DWI warp+motion `resample_series` on MPS).
- **Eliminated per-frame resampling & C/Fortran transpose loops across `motion`, `motion_reference`, `motion_design`, and `transforms`**:
  Generalized `resample_series` to `field_path=None` + `volume_scales` with a direct `F.affine_grid` `3x4` fast-path and zero-copy
  Fortran `.T` memory layout (avoiding multi-GB C<->Fortran transposes in `ants.from_numpy`); replaced per-frame `_apply_transforms`,
  `ants.slice_image`, and `ants.list_to_ndimage` loops in `motion._run_pass`, `motion_reference` (`_register_group_backward`,
  `build_low_motion_reference`, `motion_correct_grouped`), and `motion_design._joint`; cached MPS grid-scale tensors in
  `AnalyticalGridSample.backward` and foreground-masked `calculate_dvars` before `np.diff`.
- **Real-data end-to-end speedup & fidelity (`v6.30.2` -> `v6.30.4` on `ExpArt sub-EAS003 ses-01`, MPS)**:
  - 100-frame 1.5 mm DWI (`motion_correct_design("shells", engine="joint")`): `94.61 s` -> `44.01 s` (`2.15x` faster, `-53.5%`;
    non-solver resampling `35.94 s` -> `11.89 s`, `3.02x` faster; median `|dt| = 0.132 mm`, FD `r = 0.990`, tSNR `1.886` vs `1.885`).
  - 750-frame 2.0 mm rsfMRI BOLD (`motion_correct_design("none", engine="joint", tr=0.8)`): `121.39 s` -> `87.68 s` (`1.38x` faster,
    `-27.8%`; non-solver resampling `20.85 s` -> `10.56 s`, `1.97x` faster; bit-identical poses `|dt| = 0.00000 mm`, FD `r = 1.000000`,
    tSNR `8.090`).
  - `171 passed` across `syntx` motion/transforms/standards suites, `29 passed` in `antsxfunctional` motion tests, `0` findings in
    `antsxqc.lint`.

## R20 (pure-PyTorch float32 Newton motion-correction compute path; v6.30.3, 2026-10-29)
- **Zero SciPy / float32 `_safe_batched_eigh` + `_torch_jacobi_eigh`**: Replaced the `float64` + `scipy.linalg.eigh` fallback in
  `motion_batched` with `float32` `torch.linalg.eigh` backed by a 100% pure-PyTorch pre-scaled cyclic Jacobi symmetric eigensolver
  (`_torch_jacobi_eigh`) in `float32` (`2.7 ms` normal path, `9.1 ms` full-batch Jacobi fallback at `B=750, P=6`).
- **Pure-PyTorch float32 coupled SPD Cholesky solve (`_solve_coupled_spd_f32`)**: Replaced the `.cpu().double()` `2x`
  `torch.linalg.solve` (`334.7 ms/iter` at `B=750`) with `float32` `torch.linalg.cholesky_ex` + `torch.cholesky_solve` with
  scale-relative diagonal jitter escalation (`42.7 ms` at `n_act=750`, `0.68 ms` at `n_act=100`), reusing the undamped solve
  whenever `lam == 0` across active frames.
- **Vectorized `surround_prior_hessian_full` & Newton active-frame slicing**: Replaced Python per-frame tensor loops (`8,232`
  slice ops at `B=750`, `122.8 ms/iter`) with vectorized index tensors (`4.16 ms/iter`, `max_diff=0.0`), eliminated the duplicate
  coupled base-loss evaluation, and added cached active-frame slicing (`_act_slice_cache`, `loss_active_fn`) for Newton gradient
  probes and line search (`76 passed` across `test_motion_batched.py` and `test_motion_design.py`, `10 passed` in
  `test_implementation_standards.py`, `0` findings in `antsxqc.lint`).

## R19 (safe batched Newton Hessian eigendecomposition; v6.30.2, 2026-10-29)
- **`_safe_batched_eigh` in `motion_batched._run_converging_stage`**: single-precision CPU `torch.linalg.eigh` (`ssyevd`) raised
  `torch._C._LinAlgError: linalg.eigh: (Batch element 8): The algorithm failed to converge because the input matrix is
  ill-conditioned or has too many repeated eigenvalues (error code: 5)` on 5 of 18 real cohort BOLD (`rsfMRI`) runs when inactive
  or low-curvature frames produced clustered/repeated 6x6 Hessian eigenvalues in `float32`. `_safe_batched_eigh` sanitizes
  non-finite entries, symmetrizes in `float64` (`dsyevd`), and falls back per-element to `scipy.linalg.eigh(..., driver="evr")`
  (MRRR) / `"ev"` (QR) / `(0, I)` if needed before casting back to `Hv.dtype` on `dev`. Covered by `TestSafeBatchedEigh` in
  `tests/test_motion_batched.py`.

## R18 (Newton Levenberg-Marquardt damping clamp and stagnation guard; v6.30.1, 2026-10-28)
- **`motion_batched._run_converging_stage`**: When coupled Newton (`iters=20, max_extend=4`, 80 calls) encountered 65+ consecutive
  line-search rejections (e.g., on 242-volume BOLD `ds002898` `sub-01` `run-1` once the damped step shrank below the `float32`
  Mattes MI ULP floor), unclamped Levenberg-Marquardt damping `lam * 4.0` overflowed `float32` (`> 3.4e38`) to `inf`, producing
  `NaN` in `Q @ diag(ev + lam) @ Q.T` and corrupting `ax.tensor` via `0.0 * NaN = NaN` on rejected steps (`best_a == 0`), which
  crashed `torch.linalg.eigh(Hv)` at call 80.
- Fixed by clamping `lam` to `1e6 * ev_max` and `trust` to `>= 1e-4`, updating `ax.tensor` via `torch.where(moved & isfinite, ...)`
  so rejected steps preserve `base[k]` bit-for-bit, and exiting the stage loop when no frame moved (`not moved.any()`), points are
  not resampled (`resample_fn is None`), and the damped step is already `<= 1e-3` voxels.
- Verified on `ds002898` `sub-01` `run-1` (completes in 50.4s with mean FD 0.1492 mm) and regression test
  `test_newton_stage_damping_clamp_prevents_float32_overflow_on_repeated_rejections` (`test_motion_design.py`: 37 passed;
  `test_implementation_standards.py`: 10 passed; `antsxqc.lint src/syntx`: 0 findings).

## R17 (design-matrix review follow-ups; v6.30.0, 2026-10-28)
- An external review of the design matrices (DWI, ASL, slow flow, BOLD, custom) was checked against the code
  (`docs/DESIGN_MATRIX_REVIEW_FOLLOWUPS_PLAN_2026-10-28.md`): the solver description is accurate; three findings fixed here.
- **`motion_correct_design(engine="joint")` is the default** (was `"dispatch"`, the legacy per-modality calls). Every consumer
  already passed `engine="joint"`; a direct caller who omitted it got the legacy path without the joint pass or the prior. The
  dispatch-identity tests pass `engine="dispatch"` explicitly; `test_default_engine_is_joint_and_custom_encoding_reaches_it`.
- **Custom designs' `encoding` columns reach the joint engine** (`joint_design_columns`, and the one-level path forwards them as
  `encoding_covariates`), gated at `FRAMES_PER_BETA_COLUMN` frames per live column and recorded in `design.beta_dropped`; the
  design record lists them as `custom x K`. Before, the joint engine silently ignored them while `unified` and `dispatch` used them.
- **`make_design("encoding_states")` stores one-hot indicators of the non-anchor states** as `encoding` (T, K - 1): the legacy
  `dispatch` path passed the integer labels as one linear ramp, wrong for three or more states; the joint engine, which uses
  the levels, is unchanged (antsxslowflow's trajectory phantom is the regression check).
- The ISMRM abstract is left as committed in 6.29.0 by the author's instruction.
- Tests (targeted): see the release commit. `antsxqc.lint src/syntx` 0.

## R16 (one-level default prior 0.05; ISMRM abstract verified; v6.29.0, 2026-10-28)
- **`JOINT_SURROUND_WEIGHT_ONE_LEVEL = 0.05`** (`motion_design._joint`, one-level path): the readout-floor rule that rejected 0.05 is a
  DWI rule; on the cohort BOLD series 0.03 and 0.05 behaved the same (readout step within 0.002 mm, alignment unchanged, FD +1.7 /
  +1.3 %), and an independent task-fMRI evaluation (OpenNeuro ds007693 sub-07, four 340-volume runs; another agent's
  `ds007693_sub07_3way_motion_comparison.json`) at 0.05 lowered FD on every run (0.202 -> 0.199 mm), left the censored volumes
  unchanged (12 of 1360), preserved the first-level GLM maps (voxelwise beta correlation 0.994 to 0.999 against no prior) and ran
  4.3x faster end to end. Multi-level designs keep 0.03. Consumers call `motion_correct_design` with no prior arguments, so the
  default reaches antsxfunctional (BOLD, ASL), antsxdwi and antsxslowflow without a code change.
- **ISMRM abstract** (`docs/abstracts/ISMRM_2027_motion_solver.{tex,md}`, PDF built with pdflatex, not committed; figures
  `docs/abstracts/figures/`; `benchmarks/ismrm_figures.py`, `benchmarks/ismrm_table.py`; `LLM_PROMPT_ISMRM_2027.md` for an external
  editor): rebuilt from the 6.28.0 five-case run (`docs/reports/ismrm_cases_6_28_2026-10-28.*`), every number checked against its
  JSON source by script and by two independent reviewers (equations against the code, captions against the panels, table header
  against rows); synopsis 99 words, body within 850. The tex is being edited further by another agent from the prompt; its
  state at this release is not committed here.
- **`transforms.resample_series` applies the motion chain's affine Jacobian** (`chain_linear_jacobian`: det of the composed linear
  part, `1 + a . p` for the eddy affine, 1 for a rigid): the single-interpolation path antsxdwi uses for its final DWI volumes
  (`two_pe` / `single_pe` through `syntx.transforms.resample_series`) carried the eddy displacement but not its intensity
  modulation, unlike the solver's own `motion_corrected`; on the three code paths (batched affine, per-volume torch, ANTs).
  `tests/test_resample_series_eddy_jacobian.py`. `_joint` docstring corrected: the chain is `[rigid, eddy]` with the total pose.
- Comment correction in `motion_design.py`: the no-prior convergence range is 56 to 87 % (four series 56 to 85 %, the 60-volume ASL
  87 %), not 81 to 85 %.
- Tests (targeted): see the release commit. `antsxqc.lint src/syntx` 0.

## R15 (surround prior forms and the 6.28.0 default: cauchy 0.03; fd_unfiltered; v6.28.0, 2026-10-28)
- **`motion_batched.surround_rho`** (`surround_form` kwarg through `motion_correction`, `motion_correct_design`, `_joint`): three forms
  of the second-difference penalty sharing the knee (0.05 mm) and the curvature-unit scale: `charbonnier` (6.27 form, gradient
  vanishes at zero), `sparse` (gradient ten times steeper at zero; a Charbonnier with knee d/10 and weight w d / e), `cauchy`
  (re-descending; Newton uses the IRLS majoriser omega I since its exact Hessian is indefinite beyond the knee).
  `surround_prior_hessian(_full)(form=)`; `pe_info["prior"]["surround_form"]`. Tests: Hessian vs autograd per form, PSD; a 1-D
  denoising test of the claim (sparse removes 71 % of sub-knee jitter energy at weight 1 vs charbonnier 54 %; drift within 2 %).
- **Gate** (`benchmarks/prior_form_gate.py`: pre-flight, three drifting phantoms, five-series cohort, HTML report
  `docs/reports/prior_form_gate_2026-10-28.html`). The author's hypothesis (shrink small second differences harder) is refuted on
  the phantoms: `sparse` and Charbonnier at 0.1 and above remove the phantoms' TRUE jitter (ASL +70 to +172 % FD error). Only
  `cauchy` at low weight improves all three phantoms; 0.05 lowers BLAST031's B0-immune readout step by 0.043 mm (run-to-run
  noise below 0.001 mm), so **0.03 is the default** (`JOINT_SURROUND_FORM = "cauchy"`, `JOINT_SURROUND_WEIGHT = 0.03`): phantoms
  -14 / -17 / -3 %, cohort readout step within 0.011 mm, alignment unchanged, FD within 2 % (rule relaxed from "not higher" by
  the author), every frame converged (81 to 85 % without), BOLD 206 s -> 32 s (the no-prior stage oscillates on plateau
  directions until its block cap).
- `band_limited_fd` returns `fd_unfiltered` and `fd_method` (was `fd_raw_power`, misnamed under Jenkinson; audit note 7).
- `test_projected_dwi_beats_joint_and_unified_on_the_phantom` holds the prior off on both engines (its claim is about the metrics).
- The one-level design path (plain BOLD through `motion_correct_design`) now applies the default prior too (it ignored
  `JOINT_SURROUND_WEIGHT` before: the cohort gate passed the weight explicitly, the ISMRM rerun exposed the gap); the joint
  engine's result carries `prior` (weight, form, scale log). antsxfunctional's zero-motion ASL phantom gained the smooth texture
  of syntx's own (author's decision): on three featureless blobs any surround prior drifts the unconstrained rotation plateau
  (0.03 to 0.04 mm FD at 0.03), with texture 0.006 mm.
- ISMRM abstract rewritten from the 6.28.0 five-case run (`docs/reports/ismrm_cases_6_28_2026-10-28.*`; table by
  `benchmarks/ismrm_table.py`, figures by `benchmarks/ismrm_figures.py`; `docs/abstracts/ISMRM_2027_motion_solver.{md,tex}`,
  PDF built with pdflatex, not committed; `LLM_PROMPT_ISMRM_2027.md` for an external editor).
- Tests (targeted): see the release commit. `antsxqc.lint src/syntx`: 0 findings.

## R14 (solver correctness: Newton units, prior Hessian, chain order; curvature-unit priors, shrinkage, frame gain; v6.27.0, 2026-10-28)
- **Newton step unit conversion** (`motion_batched._run_converging_stage`): the parameter step was `dv / to_vox` instead of
  `dv * to_vox` since 6.24, so translations moved 9x too little and rotations hundreds of times too far and were rejected by
  the line search; found when a prior with weight 0.1 left real BOLD FD unchanged. Fixed; phantom FD error fell for every
  engine (unified DWI 0.319 -> 0.242, joint DWI 0.202 -> 0.157, joint ASL 0.054 -> 0.042). Test
  `test_newton_stage_converts_voxel_steps_to_parameters_correctly`. Every number released in 6.24 to 6.26 was measured on
  this solver and is superseded by the reports below.
- **Correctness audit** (independent agent, `ml-correctness-audit`): coupled Newton counted the roughness prior twice and the
  shrinkage three times (fixed, prior Hessian added once; fitted multipliers 0.999 / 1.000); convergence was judged on the
  damped step (fixed: undamped step, rejections counted); the written chain `[eddy, rigid]` resampled `rigid(eddy(x))` while
  the solver fitted `eddy(rigid(x))` (a transform list acts on the reference-space point first-listed first, verified with a
  point source through `ants.apply_transforms` and `syntx.transforms.apply_transforms`). Fixed: chain `[rigid, eddy]`; every
  reader selects the rigid by name (`motion_batched.rigid_transform_of`, `is_eddy_transform`): `motion.py`,
  `motion_design._joint`, `motion_reference` (forward chains now the plain inverses `[eddy_fwd, rigid_fwd]`, cross-group
  debias, nuisance), `motion_simulation`. `_assert_written_chain_reproduces_the_fitted_model` (joint and projected DWI tests):
  resampling the written chain equals one resampling through the single matrix of the fitted model and differs from the
  opposite order. `TransformChain.linear_part` composed in the reverse order of the point mapper (bvec rotation through the
  wrong composition for eddy + rigid or crosses + rigid chains): fixed, `tests/test_transform_chain_order.py`. Open audit
  notes (per-level weights overwritten in the projected engine, one-level init centre, core level-gauge guard, `fd_raw_power`
  name) are in the plan.
- **Prior units and terms** (`motion_batched.py`, `motion_design.py`): `surround_scale="curvature"` (prior weight relative to
  the image curvature measured by `probe_manifold` per stage; metric-independent, so valid under MI) is the default scale when
  a prior term is set; `pose_shrinkage` (shrinkage toward identity, `mu (|t|^2 + |50 omega|^2)`) with analytic Hessian in every
  L-BFGS / Newton stage; `frame_gain=True` (scanner drift as a per-frame global gain refreshed with the design model, on in the
  joint engine); `pe_info["prior"]`, `["frame_gain"]`, `["design_model"]`. `surround_prior_hessian(_full)` autograd-checked.
- **Default prior decision**: cohort scan (`benchmarks/cohort_shrinkage_report.py`, `docs/reports/cohort_shrinkage_report_2026-10-28.html`,
  five series x mu 0 / 0.03 / 0.1 / 0.3, pre-flight check that mu changes the outcome) chose shrinkage 0.1 by the readout-floor
  rule (FD -11 to -41 %, readout step within 0.023 mm). On the fixed solver the drifting phantoms show the mechanism: shrinkage
  at any weight pulls the trajectory toward identity (ASL FD error 0.042 -> 0.129 mm at 0.1, 0.100 at 0.03; rotation 0.058 ->
  0.362 deg; antsxslowflow trajectory test 0.060 vs 0.05); roughness 0.1 is nearly free on ASL (0.068) and best on DWI (0.103)
  but still fails the ASL-matches-unified test (0.085 vs 0.077). Author's decision: ship with both priors OFF
  (`JOINT_POSE_SHRINKAGE = 0.0`, `JOINT_SURROUND_WEIGHT = 0.0`), available per call; the default is decided in 6.28 behind a gate
  the three phantoms and the cohort both pass. `benchmarks/ten_minute_questions.py`
  (`docs/reports/ten_minute_questions_2026-10-28.html`): the projected engine still over-reports readout jitter on real DWI
  (0.112 vs joint 0.040 mm on BLAST031, odd-in-g 65 %: an eddy-like leak), roughness 0.1 acts as designed.
- **Other**: `motion_decomposition.image_variance_decomposition(frame_gain=)`, `misattribution_angles`; `band_limited_fd`
  Power / Jenkinson consistency; `make_design("shells")` stores `bvals` / `bvecs`; benchmarks `dwi_design_decision.py`,
  `slice_residual_diagnostic.py`, `slice_parity_term.py`, `shrinkage_weight_scan.py`, `cohort_shrinkage_report.py`,
  `ten_minute_questions.py` with reports under `docs/reports/`. The ISMRM abstract (`docs/abstracts/`) is parked: its numbers
  predate the Newton and chain fixes and are being re-measured (`benchmarks/ismrm_cases.py`, priors off).
- Tests (targeted, per the author's instruction; no full suite): see the release commit. `tests/test_motion_overlapping_window.py::
  TestOverlappingWindowBOLD::test_bold_nyquist_sagittal_ripple_...` fails identically at b90cb1a in a clean worktree
  (`wiener_shrinkage_eta` list empty): pre-existing, its own ticket. `antsxqc.lint src/syntx`: 0 findings.

## R13 (ISMRM abstract, decomposition module, FD consistency, 2026-10-28; uncommitted solver work)
- **Abstract** (`docs/abstracts/ISMRM_2027_motion_solver.md`, 747-word body): one solver, four series (breacher 3315 resting BOLD,
  ExpArt EAS020 ASL, FPA BLAST031 AP and SOCOM Blast-11 PA multi-shell DWI), every number from `benchmarks/ismrm_cases.py`
  (`docs/reports/ismrm_cases_2026-10-28.{html,json,_figure.png}`), joint engine on MPS.
- **`syntx.motion_decomposition`** (new): `pose_step_decomposition` (head / applied / removed shares per acquisition axis),
  `image_variance_decomposition` (sequential sums of squares: design blocks, motion after the design, residual; residual-FD
  correlation), `design_blocks`, `real_even_sh`; `tests/test_motion_decomposition.py` (3).
- **Band-limited FD consistency** (`motion.py`): `band_limited_fd(fd_method=, center_of_mass=)` so the band-limited series uses the
  result's own FD definition (Jenkinson in the grouped and design engines, Power in `motion_correction`; without a band the two
  series agree to 1e-14); `MIN_FRAMES_FOR_BAND = 32` (a 16-frame ASL series has no spectral resolution for a band).
- **Prior in likelihood units and coupled Newton** (`motion_batched.py`, from earlier today, uncommitted): `surround_scale="likelihood"`
  (nats per mm, scaled by sigma^2 / S measured per stage), `surround_prior_hessian_full` (block-tridiagonal, autograd-checked),
  one global line search; `benchmarks/prior_units_case_study.py`, `docs/reports/prior_units_case_BLAST031_2026-10-28.html`: a
  strong prior bends pitch, the readout jitter of the projected engine is data-supported.
- Plan: `docs/MOTION_NEXT_STEPS_PLAN_2026-10-28.md` (steps 2 to 7 pending). ANTsTorch's working tree was briefly mid-edit by another agent (import broken); two reruns stubbed `antstorch`
  in-process; once it imported again the SOCOM DWI case was rerun without the stub and every reported cell was identical
  (the motion modules never reference antstorch), the ASL and BLAST031 cases were then rerun without it and every reported cell matched the committed values; the figure, HTML and JSON are from these unstubbed runs. Author block: sole author (co-authors are the user's to add). Independent check: the antsxdwi script written this morning (`antsxdwi/benchmarks/motion_variance_report.py`, its own mask, voxel sample and SH code, run on the WRITTEN corrected NIfTI and confounds of SOCOM Blast-11 PA) reproduces the syntx module's decomposition within sampling noise (intercepts 92.2 vs 91.8 %, l=2 2.6 vs 2.6, l=4 0.9 vs 1.0, motion 0.8 vs 0.9, residual 3.5 vs 3.7 %; r -0.12 both) and the pose shares exactly (readout 61/45 %, PE eddy 71 %). The BOLD and ASL rows were likewise recomputed from written outputs with a separate plain least-squares routine, a raw-mean mask and a different 40 000-voxel sample (seed 7): BOLD motion 34.7/23.5 vs 35.0/23.6 %, residual 65.3/76.5 vs 65.0/76.4 %, r 0.62; ASL design 97.1/98.4, motion 0.33/0.17, residual 2.6/1.4 %, r 0.59; FD, band-limited FD and censored fractions identical. Every row of the table is thus checked by an independent implementation.

## R12 (prior in every stage, band-limited FD, resting-BOLD pilot, projected engine, v6.26.0, 2026-10-28)
- **Plan** (`docs/MOTION_JOINT_V2_AND_CENSORING_PLAN.md`): the v2 model (design `D` profiled out of the metric per point, levels
  differ in contrast never in pose, prior inside one objective), six syntx WPs, nine cross-repo censoring WPs, six gates, four
  decisions taken (engine `projected`, 0.5 mm on band-limited FD, automatic band with a recorded trace overriding, field term dropped).
- **Resting-BOLD pilot** (`benchmarks/pilot_bold_resting_joint.py`, `docs/reports/pilot_bold_resting_joint_2026-10-27.html`): one
  L-BFGS objective over all frames (Huber residual about the robust temporal mean at a gradient-weighted point set, per-point scale,
  per-slice-group PE field term, Charbonnier head prior) on breacher sub-6223 (TR 2.5) and sub-3315 (TR 1.0). Edge temporal SD
  0.0323 vs 0.0340 and 0.0515 vs 0.0524 against the `bold` preset while reporting MORE FD (0.48 / 0.87 vs 0.42 / 0.62 mm):
  the near-Nyquist alternation is a real displacement (notching it before resampling raises edge SD 15-20 % and kills the
  DVARS-FD correlation). Field term 0.04-0.06 mm, per-point scale no effect: both dropped from v2.
- **S1, prior in the Newton stage (`motion_batched.py`)**: `_run_converging_stage(reg_hess_fn, reg_frame_fn)`; the Newton gradient
  includes the prior by autograd, its per-frame Hessian block comes from `surround_prior_hessian` (analytic, `w d (s^2 I - u u^T)/s^3`
  summed over the three terms a frame enters, checked against autograd to 1e-15), the line search sees the prior's per-frame
  attribution. Isolation of the "invented motion" on the antsxfunctional zero-motion blob phantom: a near-linear rotation drift
  about the blob centres, i.e. the prior's nullspace crossed with the image plateau of three Gaussian blobs (0.17 mm at the 16 mm
  extent with every frame reporting converged); a smooth texture on the same phantom gives 0.01 mm. G4 re-specified accordingly.
- **S3, band-limited FD (`motion.py`: `band_limited_fd`, `attach_band_limited_fd`; exported as `syntx.band_limited_fd`)**: the pose
  spectrum's peak above 0.1 Hz, widened to a quarter of the hump's height over the white floor and at least a quarter of the
  Nyquist frequency each side, applied only when its power is 1.5x what its width would hold (`band_source` auto | given | none);
  attached to every `motion_correction`, `motion_correct_grouped` and `motion_correct_design` result (`fd_bandlimited`, `fd_band`,
  summary keys) with the TR from the 4-D spacing or `tr=`. Breacher: 0.426 -> 0.246 and 0.622 -> 0.328 mm; frames above 0.5 mm
  27 -> 10 % and 48 -> 16 %. Documented in `docs/MOTION_BOLD.md`.
- **Tests**: `tests/test_band_limited_fd.py` (5), `tests/test_motion_design.py` +2 (Hessian vs autograd; textured zero-motion phantom
  with the prior on), 25 passed; standards allowlist line moved (`motion.py:1212`); `antsxqc.lint` 0. Pre-existing BOLD Nyquist
  window test still fails (unchanged).
- **S2, `engine="projected"` (shipped, not default)**: the design profiled out of the metric at every point (`design_matrix` in
  the batched core, `projected_design_matrix`: intercept per level plus the l = 2 block per shell; design-model references
  refreshed per stage, redraw and Newton iteration; per-point residual scale), identity links and a robust level gauge for
  interleaved levels with `GAUGE_MIN_FRAMES` frames, registration links for blocked levels, eddy applied, the prior in every
  L-BFGS / Newton stage and never in Adam. Phantoms (CPU, head FD error): DWI 0.176 vs joint 0.210 vs unified 0.319; ASL 0.065
  vs 0.055. Prior OFF by default: its weight is not in the image term's units (the loss is a mean over points): at 0.02 it
  stops L-BFGS from correcting a smooth rotation error (plain phantom 0.065 -> 0.208 deg) and on the real DWI cohort it
  doubled the B0-immune readout step with identical alignment (`docs/reports/motion_cohort_gate_projected_2026-10-27.html`);
  the prior-off gate rerun is pending. Tests: `tests/test_motion_design.py` +4 (29 passed).
- Downstream (same release train): antsxqc 0.4.0 catalog and lint rule; antsxfunctional censors on `fd_bandlimited`; antsxdwi,
  antsxslowflow, antsxmm import the one threshold; antsxslowflow's local motion verdict replaced by `antsxqc.grade`.

---

## R11 (joint motion engine, v6.25.0, 2026-10-27)
- **Review and plan** (`docs/MOTION_COMPLETE_MODEL_REVIEW.md`, `docs/MOTION_JOINT_OPTIMIZATION_PLAN.md`): the `unified` engine
  solved each level separately and gave the free motion term everything the local-difference estimators could not isolate;
  measured on the `motion_cohort` archive, 34-48 % of raw DWI FD is explainable by acquisition columns, ASL 1-11 %, BOLD 0 %
  (respiration is not in the metadata).
- **`engine="joint"`** (`src/syntx/motion_design.py`): H0 per-level means linked by explicit registration, H1 one batched
  pass over all frames (per-frame references, `init_pose`, `schedule_mode="refine"`, all design columns of the timeline,
  Charbonnier prior knee 0.05 mm, off by default). Solver (`motion_batched.py`, `motion.py`, `sampling.py`): `init_pose`,
  `refine` schedule, per-reference intensity range, `frame_levels` point weights (`PointSampler.weights_of`), odd-only
  phase-encode projection of the encoding gauge.
- **Evidence**: phantoms (`docs/MOTION_DESIGN.md`, section joint): FD error DWI 0.438 -> 0.208, ASL 0.089 -> 0.059, slow flow
  0.133 -> 0.068 mm; reported mean FD 0.301 -> 0.220 for a truth of 0.156. The prior (0.02) took more on structured
  phantoms but displaced a featureless no-motion phantom by 0.1 mm (absolute weight vs image curvature): off by default,
  curvature-normalised prior = next WP. Cohort (`benchmarks/motion_cohort_gate.py`,
  `docs/reports/motion_cohort_gate_2026-10-27_default.html`): FD -1-7 % ASL, 0-16 % DWI, alignment identical, readout step
  within 0.025 mm, all frames converged.
- **GLM (antsxslowflow 2.10)**: motion nuisance columns orthogonalised to the harmonic columns (raw poses with a 0.6 mm
  respiration-locked PE shift: amplitude 1.66 for a truth of 1.0; orthogonalised 1.00).
- **Tests**: `tests/test_motion_design.py` +5 (joint columns, refine guard, ASL and DWI joint vs unified, odd-only
  projection); motion / sampling / standards suites green except the pre-existing BOLD Nyquist test; `antsxqc.lint` 0.
- Downstream: antsxdwi 0.31.0, antsxfunctional 0.20.0, antsxslowflow 2.10.0 (`engine="joint"` default, `syntx>=6.25.0`).

---

## R10 (Stage 1: In-Solver Gauge Orthogonalization F14, BOLD EPI-Plane Debiasing & Slice-Axis Inference, v6.24.1, 2026-10-27)
- **F14 In-Solver Gauge Orthogonalization (`src/syntx/motion_batched.py`)**:
  - Projected $\ell=2$ contrast gauge in `_batched_rigid_register_pass_core` strictly onto the 5-DOF subspace orthogonal to $\hat{\mathbf{p}}$-translation in both `_encoding_offset` and `_project_encoding_gauge`.
  - Guarantees $\ell=2$ white-matter fiber anisotropy contrast never steals physical linear-in-$\mathbf{g}$ $\ell=1$ PE eddy translation along $\hat{\mathbf{p}}$.
  - Populates `pe_info["bias"]` from `enc_bias_info` in both single-batch and chunked paths.
- **BOLD EPI-Plane Debias Fix & Slice-Axis Inference (`src/syntx/motion_batched.py`, `src/syntx/motion_design.py`, `src/syntx/acquisition_axes.py`)**:
  - Fixed `sig_ref` bug in `separate_bold_pe_pseudomotion` by consuming the returned `meta` record from `_shrink_pe_pseudomotion_core`.
  - Added alias `debias_pe_pseudomotion = separate_bold_pe_pseudomotion`.
  - Routed `motion_correct_design` for BOLD (`"none"` design) with `pe_axis` and `debias_pe_pseudomotion=True`.
  - Enhanced `infer_slice_axis` (`src/syntx/acquisition_axes.py`) to resolve `dcmmeta_slice_dim` from BIDS sidecars and `interleave_and_slab_shape` when both half-series interleave checks agree on the smaller slab axis of a square-in-plane 2D EPI volume.
- **Implementation Standards Exception (`tests/test_implementation_standards.py`)**:
  - Updated line reference in `ALLOWED_REGISTRATION_EXCEPTIONS` from `motion.py:1059` to `motion.py:1062` matching the fallback `ants.registration` call in `src/syntx/motion.py`.
- **Scientific Directive**:
  - Primary `fd` (`dti_FD_mean/max`, `rsf_FD_mean/max`, `asl_FD_mean/max`) remains the single authoritative Framewise Displacement metric produced directly by the solver. No `*_FD_pe_debiased_*` columns are emitted in wide tables.
- **Verification**: All unit tests in `syntx` (`test_motion_design.py`, `test_motion_reference.py`, `test_implementation_standards.py`, `test_stage0_challenger_blast_probes.py`) pass cleanly; static analysis `python -m antsxqc.lint` reports 0 findings.

## R9 (Stage 0: Behavior-Preserving Motion Design Consolidation & Safe Bug Fixes, v6.24.0, 2026-10-26)
- **Safe Bug Fixes BF1–BF7**:
  - BF1 (F4): `_design_tag_control` preserves contiguous level re-indexing even when `"control"` is missing, recording `anchor_volume_type`.
  - BF2 (F5): `_design_shells` enforces explicit `(T, 3)` gradient table shape validation, rejecting `(3, T)`.
  - BF3 (F6): $b_0$ rows zeroed and DW gradient rows unit-normalized in `pe_affine_covariates`.
  - BF4 (F10): `Design.__post_init__` validation enforces contiguous `0..k-1` levels, row-count matching across covariates, and unit-normalized `pe_axis`; `_design_none` supports `pe_axis`.
  - BF5 (F9): `_unified` forwards `encoding_covariates` and `window_size` for multi-level `custom` designs.
  - BF6 (F15): `pe_info["bias"]` populated from `enc_bias_info` in `_batched_rigid_register_pass_core` when `bias_info is None`; downstream fallback to `encoding_pose_bias_mm_rad` in `antsxdwi`.
  - BF7 (F1/F2/F3): Centralized connected-component level graph isolation (`level_adjacency`, `offset_identifiable`); `motion_correct_grouped` links component roots directly, isolating blocked levels.
- **Centralization C1–C3 & Plumbing**:
  - C1: Centralized `level_adjacency` and `offset_identifiable` in `motion_reference.py`, achieving 0 discrepancies across 1,500 random sequences.
  - C2: Centralized robust Tukey trend estimator `_fit_tukey_trend` in `motion_batched.py`.
  - C3: Centralized post-hoc pseudomotion smoother core `_shrink_pe_pseudomotion_core` in `motion_batched.py`.
  - F11: Single 3D reference footprint in `_corrected_level_mean` and broadcast in `reference_predictor`.
  - F7/F8/F13: Plumbed `sidecar` and `slice_axis` through `motion_correction`, `motion_correct_grouped`, and `_batched_allowed_kwargs`; supported path inputs in `antsxdwi.motion_correct`.
  - Deprecated `nuisance_covariates` and `nuisance_axis` in `motion_correct_grouped` with `FutureWarning`.
- **Golden Baseline Verification**: G1–G5 match pre-change baselines within `0.00e+00`; G6 transforms match within `9.54e-07`; G6 FD reflects the BF7 bug fix.
- **Quality & Standards**: 239/239 syntx motion tests pass; downstream test suites in `antsxdwi` (10/10), `antsxfunctional` (3/3), `antsxslowflow` (1/1) pass. Static analysis `python -m antsxqc.lint` reports 0 findings across all 5 repos with zero `.antsxqc-allow` files. Zero `nibabel` imports.

## R8 (Design-Driven Motion Correction `syntx.motion_correct_design`, Acquisition Axes, robust_affine Multi-Start, v6.23.0, 2026-10-26)
- New `src/syntx/motion_design.py`: one entry point `motion_correct_design(image, make_design(name, ...), engine=...)` with a
  design menu (`DESIGNS`: `none` BOLD, `tag_control` ASL, `shells` DWI, `encoding_states` slow flow, `custom`). Engine
  `unified`: each design level registered to its own reference with its own sample points, interleaved levels linked by
  identity (blocked or lone levels by registering their means), the per-level pose offset removed by the surround-Tukey
  projection on the level indicators. BOLD (one level) is identical to `preset="bold"`. `dispatch` reproduces the previous
  per-modality calls. Results keep their type (attribute access). Design and evidence: `docs/MOTION_DESIGN.md`.
- `motion_reference._group_offsets`: the offset is estimated for each level that is interleaved enough (>= 4 active surround
  rows and >= 4 frames next to another level); a level that is not (a lone M0 frame) keeps its linked pose and is listed in
  `cross_group_offset["not_identifiable_groups"]` instead of switching the offset off for every level.
- New `src/syntx/acquisition_axes.py`: slice and phase-encode axes inferred from the image (sidecar, SliceTiming, voxel
  thickness, slice interleave; Nyquist ghost cross-checked by the shift spread), raising when undecidable. Replaces the
  hard-coded slice axis in the opt-in PE debias paths. Real data: 21 / 21 decided cases correct, 6 undecided.
- `robust_affine(mode='ants', multi_start=True)`: rotation candidates no longer double-centred (`t_rot = t_w + C - R C` with
  `fixed_params = C` added `(I - R) C`); candidates scored on images normalised to the (0, 1) histogram range
  (`normalize_robust`, no auto-mask). Known-rotation head phantom (pitch 8 / yaw 12 deg, 2 origins, 3 seeds): the matching
  candidate now wins and the final error is 0.5-1.8 mm (was 3.5-11.4 mm); each fix alone does not change the selection.
- Phantom gate (5 seeds, rotation deg, previous call -> unified): ASL 0.117 -> 0.107, slow flow 0.164 -> 0.137, slow flow
  + eddy 0.148 -> 0.132, single-shell DWI 0.359 -> 0.330 (translation 0.066 -> 0.035 mm). Real FPA sub-BLAST079: ASL and
  DWI tie the previous calls, BOLD identical.
- Downstream moved to the entry point: antsxfunctional (ASL, BOLD), antsxdwi (shells), antsxslowflow (encoding states).
- Tests (targeted): motion_design / motion_reference / acquisition_axes 41 passed; affine set (robust_affine, affine,
  dewarp, new `test_robust_affine_multistart_ants.py`) 95 passed, 1 skipped; standards 10 passed; `antsxqc.lint` 0.
  The full suite was not run for this version.

## R7 (Centralized Metric Sampling `syntx.sampling`, Merged Pilot Ideas, Independent Audits, v6.22.0, 2026-10-25)
- New `src/syntx/sampling.py`: ONE implementation of metric sample-point selection, designed as a menu
  (`SAMPLING_STRATEGIES`: `gradient_floor` default, `gradient`, `uniform`; `get_sampling_strategy`; unknown names raise).
  Candidate region = above the reference's own noise floor (`background_floor`: noise / clipped_noise / zero-filled
  (masked) / no_background, z = 4; zero padding told apart from clipping and from masking) and below the saturated top of
  the foreground. One physical point pool shared by every pyramid scale; device-resident float32 draws (Gumbel top-k);
  at most `MAX_EVAL_POINTS` = 20 000 points per metric evaluation. Design and evidence: `docs/METRIC_SAMPLING.md`.
- Wired into the rigid motion pass, the group-bias pass, `robust_affine` (`fixed_fg` domain, weighted draws, candidate
  scoring foreground) and the JAX MI (`regular_stride`, seeded `uniform_cube_points`; `syn_jax.fit(sample_seed=42)`).
  `sample_fraction` = fraction of the region each evaluation sees (was the pool size).
- Ground-truth motion phantoms (5 seeds, rotation error deg, former sampler -> default): ASL 0.181 -> 0.113, slow flow
  no eddy 0.280 +- 0.138 (max 0.53) -> 0.166 +- 0.022, slow flow eddy 0.183 -> 0.135, DWI 0.672 -> 0.392, BOLD 0.192 ->
  0.138; eddy PE scale 0.0322 for 0.03. Real DWI (FPA sub-BLAST079, MPS): 78 s / mean FD 0.125 mm (former 65 s / 0.163).
  The former ~0.18 deg "phantom rotation floor" and much of the DWI rotation bias were sampling (MOTION_SYSTEMATIC_NOISE.md).
- Merged `pilot/motion-sampling-and-epi-debias`: grouped DWI PE / non-linear-eddy debias of the reported trajectory and
  the BOLD slice / pitch "Nyquist" extension, both OPT-IN (`motion_correct_grouped(debias_pe_pseudomotion=True)`,
  `motion_correction(debias_sagittal_nyquist=True)`): an independent audit showed they remove real jitter from the
  reported FD. The pilot's lever-arm `pe_scale` conditioning and its `> 0.01` / size-dependent pool were not adopted.
- Two independent audits and two re-reviews; findings fixed (cap boundary, thin-slab ITK abort, integer data, padding vs
  clipping vs masking, provenance, robust_affine fallbacks). Open, pre-existing: `_eval_low_res_mi` scores raw
  intensities on a (0, 1) histogram range (changing it broke multi-start selection); known limits in METRIC_SAMPLING.md.
- Dewarp: `_resolve_restrict_transformation` now runs before `assert_comparable_foreground_extent` (foreground-parity
  guard from `bf21dbf`); pyramid-guard allowlist updated for `motion_batched.py` / `dewarp.py`.
- Tests: full suite 2636 passed, 43 skipped, 0 failed (MPS, two processes with no other GPU job; a run concurrent with
  an external MPS cohort job produced non-finite / uninitialised outputs in unrelated transform tests that passed on
  rerun); `antsxqc.lint` 0 findings.

## R6 (`syntx.dewarp` Foreground-Extent Parity Guard, Reference Masks & Regularized Cross-Modal `SyNOnly`, 2026-10-25)
- Deduplicated `assert_comparable_foreground_extent` into `src/syntx/dewarp.py` (exported in `syntx` and `syntx.dewarp`) and added `reference_mask` and `secondary_reference_mask` parameters to `rigid_align_anatomy_to_reference`, `syn_only_reference_to_anatomy_rigid`, and `dewarp_to_anatomical`.
- Defaulted `syn_only_reference_to_anatomy_rigid` and `dewarp_to_anatomical` Stage 1 (`reverse_pe_midspace`) to regularized `total_sigma=1.0`, `levels=[2, 1]`, `reg_iterations=[40, 10]`, preventing whole-head scalp/skull inward pulling (~41-49 mm over-deformation when unmasked EPI/b0 was registered against masked `anatomy_rigid`).
- Tests: `tests/test_dewarp.py` (10 passed, including `test_foreground_mismatch_raises_in_syn_only_and_dewarp` and `test_dewarp_to_anatomical_with_reference_masks`), `tests/test_implementation_standards.py` (10 passed), `antsxqc.lint` 0 findings.

## R5 (Interleaved-Group FD, Encoding Identifiability Limits, v6.21.0, 2026-10-25)
- `motion_correct_grouped` reports motion and FD without the constant per-group pose offset left by cross-group
  registration when the groups are interleaved (every non-anchor group >= 4 active surround rows); `group_offset_fd`
  holds the offsets and `fd_with_group_offsets`. Transforms unchanged (`cross_group_debias=True` also removes it from them).
  Shared estimator `_group_offsets` (surround Tukey on group indicators) behind both. Real 3-shell DWI (FPA sub-BLAST079 AP,
  98 switches / 99 steps): 0.16-0.17 mm / 0.08-0.09 deg per shell, mean FD 0.173 -> 0.163 mm.
- Investigated short-series ASL (T = 16) excess head error (0.026 vs 0.015 mm floor): NOT under-removal and NOT the shared
  parameter freeze (never freezing: identical errors). It is real jitter aligned with `s_t`, unidentifiable from the bias;
  no jitter 0.020 -> 0.002 mm, still head 0.000. Documented in `docs/MOTION_SYSTEMATIC_NOISE.md` with the lone-frame limit
  (an M0 frame as a categorical state is a one-frame spike to the robust projection: register it as a group).
- Tests: new `test_group_offsets_interleaved_shells_recovered_and_blocked_refused`; `test_motion_reference.py` +
  `test_motion_encoding_bias_simulation.py` 28 passed; implementation standards 10 passed; `antsxqc.lint` 0 findings.

## R4 (Encoding-Locked Systematic Noise: Wired Solver, Simulation Benchmark, Robust Estimators, v6.20.0, 2026-10-24)

Tag `v6.20.0`. Remediates the v6.19.0 audit (`docs/HANDOFF_CROSS_REPO_MOTION_AUDIT.md`); recipe and evidence in `docs/MOTION_SYSTEMATIC_NOISE.md`.

- **Simulation (`motion_simulation.py`, new)**: head phantom (GM shell, WM with fibre field, ventricles, texture) moved by a known rigid trajectory; ASL / slow-flow / DWI / BOLD series whose registration bias arises from contrast alone, plus known encoding-locked PE displacement (eddy, respiration); `evaluate_motion_estimate` scores the composed rigid chain against truth. `benchmarks/motion_encoding_bias_simulation.py`: strategy x modality x seed table + HTML report.
- **In-optimizer encoding offset (`motion_batched.py`)**: `B_win` now enters the loss (`_total_pose`), is optimised with the head pose, and is re-estimated robustly (fold back + surround-Tukey projection) after every stage; the written transforms carry head motion only. The v6.19 `B_win` / `_project_encoding_gauge` / `reg_loss_fn` were dead code. In-loop surround penalty kept but default 0 (it degraded the head estimate as its weight grew). Per-chunk encoding offsets concatenated; `delta_Y_mm_rad` padded to all frames.
- **Robust estimators**: eddy translation attribution `C0` and `bias_covariates` use the surround-Tukey estimator instead of least squares on the poses (chance correlation of a head jump with g/l2 terms was reassigned: 0.12 mm and 0.22 mm head error on simulated DWI -> 0.05 / 0.11 mm). Encoding offsets are no longer re-applied as eddy shift.
- **Categorical states**: integer state vectors with > 2 values (`[0, 1, 2, 3]`) are one-hot encoded, not a numeric ramp (slow-flow projection had silently been a single column).
- **Per-frame references** (`build_bilateral_window_references`): whole-series encoding-matched means by default; the local temporal window made poses relative to a moving average (drift never measured). Inclusive state means are the right reference for discrete states (leave-one-out breaks the common space: 0.03 -> 0.12 mm); `exclude_self` added. `encoding_reference_predictor` (new, leave-one-out kernel reference for continuous encodings) is available but not used by default: no reliable gain on simulated DWI.
- **`motion_correct_grouped`**: removed the auto-injection of `pe_affine_covariates` into `encoding_covariates`, the DVARS median rescaling across shells (raw per-shell DVARS returned again), and the post-hoc polynomial PE regression that overwrote `fd`; added opt-in `cross_group_debias`.
- **BOLD**: `separate_bold_pe_pseudomotion` is opt-in (`debias_pe_pseudomotion=True`, needs `pe_axis`), models the PE translation only (slice translation and pitch never touched), no silent `except`.
- `MOTION_PRESETS` test updated for `asl`; raw-registration allowlist line number corrected (already stale at v6.19.0).
- Simulation (3 seeds): ASL 0.150 -> 0.017 mm; slow flow with eddy 0.248 -> 0.031 mm; DWI with eddy 0.385 -> 0.046 mm (direction-dependent rotation bias, about 0.6 deg, remains the open problem); BOLD respiration 0.285 -> 0.078 mm with nodding untouched.
- Tests: 2605 passed, 43 skipped. `antsxqc.lint src/syntx`: 0 findings.

## R3 (Generalized Overlapping-Window Encoding-Bias & Head-Motion Solver Across DWI, ASL, SlowFlow, and BOLD, v6.19.0, 2026-10-24)

Tag `v6.19.0`.

- **What changed.**
  - Added `build_encoding_covariates`, `_project_encoding_surround_tukey_np`, and `separate_bold_pe_pseudomotion` in `src/syntx/motion_batched.py`, decomposing time-series poses over partition-of-unity Hann windows $\psi_w(t)$ with 50%-breakdown second-order surround-difference Tukey biweight IRLS on $\mathfrak{se}(3)$ principal rotation vectors and Empirical Bayes / Wiener EPI-plane noise-floor matching for BOLD respiratory $\Delta B_0(t)$ pseudomotion.
  - Added `build_bilateral_window_references` (`reference="bilateral"`) in `src/syntx/motion_reference.py`, `ASL_MOTION_PRESET` (`"asl"`) and `SLOWFLOW_MOTION_PRESET` (`"slowflow"`) in `src/syntx/motion_presets.py`, and forwarded `encoding_covariates` and `window_size` through `motion_correction` and `motion_correct_grouped` (`src/syntx/motion.py`).
  - Added `tests/test_motion_overlapping_window.py` covering `TestBuildEncodingCovariates`, `TestOverlappingWindowASL`, `TestOverlappingWindowSlowFlow`, `TestOverlappingWindowDWI`, and `TestOverlappingWindowBOLD`.

---

## R2 (Ecosystem Audit Remediation & Zero-Allowlist Consolidation, v6.18.0, 2026-10-23)

Tag `v6.18.0`.


- **What changed.**
  - Fixed `evaluate_registration_qc` in `src/syntx/registration_metrics.py` to aggregate verdicts via `antsxqc.worst_status()` (preserving `"not_assessable"` instead of promoting to `"pass"`), read the `folding_pct` limit dynamically via `antsxqc.hard_limit("folding_pct", side="high")`, and return `float("nan")` instead of fake `min_jacobian=1.0, folding_pct=0.0` when warp loading fails.
  - Fixed `preprocess_for_landmarks` in `src/syntx/landmarks/preprocess.py` to pass `ants.ANTsImage` directly to `antstorch.n4_bias_field_correction` and removed the `ants.n4_bias_field_correction` fallback.
  - Standardized execution status strings in `src/syntx/provenance.py` and `src/syntx/landmarks/optimal_transport.py` to `"success"` and deleted `.antsxqc-allow` (`antsxqc.lint` passes with 0 findings and 0 exceptions).
  - Replaced `reorient_image2("LPI")` in `src/syntx/viz/core.py` and `src/syntx/viz/figures.py` with direct direction-cosine axis permutation/flipping (`AnatomicalVisualizer._orient_lpi`).
  - Consolidated coordinate primitives from `src/syntx/landmarks/spatial.py` into `src/syntx/spatial.py`, delegated energy calculations in `src/syntx/viz/reports.py` to `syntx.deformation_metrics`, added `slots=True` to `src/syntx/contract.py` dataclasses, removed import-time `PYTORCH_MPS_HIGH_WATERMARK_RATIO` mutation from `src/syntx/__init__.py`, and deleted the legacy `src/syntx/reporting.py` shim.
  - Added `AGENTS.md`, `CLAUDE.md`, `docs/antsx_implementation_standards.md`, `docs/contracts.md`, and the 6-check `tests/test_implementation_standards.py` suite.

## R1 (Delegate Registration QC Verdicts to antsxqc, v6.17.0, 2026-10-22)

Tag `v6.17.0`.

- **What changed.**
  - Renamed `src/syntx/qc.py` -> `src/syntx/registration_metrics.py` and `src/syntx/viz/qc_sections.py` -> `src/syntx/viz/metric_sections.py` (`R3-name` compliance).
  - Delegated registration hard validity limits (`min_jacobian`, `folding_pct`, `dice_overlap`, `inverse_identity_error`) and status grading to `antsxqc.hard_limit` and `antsxqc.grade`.
  - Standardized `ModalityReport` (`src/syntx/viz/modality_report.py`) on the canonical 4-status vocabulary (`pass | warn | fail | not_assessable`).
  - Added `tests/test_antsxqc_lint.py` enforcing `antsxqc.lint.assert_clean` (`26 passed`).
