# syntx 6.2.0

Breaking on purpose: the owner's rule for this release was correctness and consistency over backward compatibility. Several earlier behaviours were wrong
and are not preserved. Details, evidence and the measurement protocol: `docs/PYRAMID.md` (Sections 11 to 13) and `docs/PYRAMID_EVIDENCE.md`.
Reference: 6.1.2 (`v6.1.2`).

## Rigid / affine (`robust_affine`, `motion_batched`)
- **No silent fallback.** A failure of the PyTorch solver raises `RobustAffineSolverError` (chaining the cause); it never falls back to ANTs. The ANTs path is reached only with `mode='ants'` / `'ants_fast'`.
- **Post-solve self-check.** The optimised pose must beat both trivial starts (identity, centre-of-mass) on the same coarse objective, otherwise `RobustAffineSolverError`.
- **Robust normalisation is always applied** (`normalize_image(method='auto', force=True)`): an image already in [0, 1] is no longer passed through with an outlier-dominated range.
- **Corrected millimetre pyramid for anisotropic images, with the slab-edge margin, by default.** Both images are leveled at `level x` the fixed image's finest spacing whenever either is anisotropic (an isotropic T1 against a 0.43 x 0.43 x 3.5 mm slab used to be
  pooled by `level` voxels, 2 to 3 times coarser than the slab), the area average is centroid-exact (the old overlap-weighted box was biased by up to 0.0625 working voxel on fractional levels), and the outermost slice at each end of the thick axis of an anisotropic fixed image
  leaves the metric: new parameter **`valid_margin_slices=1`** (0 disables; negative / non-integer raises `ValueError`; no-op for an isotropic fixed image). The width was chosen from a 0 / 1 / 2 sweep.
- The self-check slack is `max(5e-3, 5 percent of the baseline loss)` (the optimiser works on finer sampled levels, the check on the coarse full grid; they disagree by a few percent on a pair whose start is already optimal).
- New opt-in `translation_scan_steps` (default 0): a lattice scan of start translations. Not the default: it fixes the synthetic 6:1 failure but regresses real sub-07 / sub-01 and costs 40 percent more time (see Known limitations).
- Low-resolution inputs of the phase-correlation and landmark candidates are resampled linearly (two calls used nearest neighbour by accident); `cap_ants_image` accepts 2-D.
- `smooth_sigma_per_level > 0` smooths through `core/smoothing` (replicate padding), not a zero-padded kernel that darkened the border. `motion_batched` uses the level's own per-axis factor and offset. One shared prebuilt warp.

Measured (CPU, back to back; 72 synthetic thin-slab cases = ratios 1..10 x poses A, D, E, F x seeds 7, 42, 123; ten real ds001242 neuromelanin-slab / T1 pairs against the ANTs rigid reference):

| | 6.1.2 (70ee29f) | 6.2.0 |
|---|---|---|
| synthetic 1:1, 2:1, 4:1, 8:1 mean / worst TRE (mm) | 0.05 / 0.11, 0.11 / 0.30, 0.54 / 1.05, 0.59 / 1.03 | 0.05 / 0.11, 0.10 / 0.28, 0.52 / 1.02, 0.62 / 1.00 |
| synthetic 6:1 (cases under 1.5 mm) | 1.40 / 4.56 (9 of 12) | 1.38 / 4.45 (9 of 12), see limitation |
| synthetic 10:1 | 0.47 / 0.86 | 0.67 / 1.16 (the clean synthetic slab has no partial-volume edge slice, so the margin only removes data) |
| real sub-07 pose error | 26.2 mm | **1.24 mm** |
| real, the other nine (max) | 0.56 mm | 0.39 mm |
| original-input repro (`nm_registration_debug/repro_syntx.py`, mps) | fails | VERDICT OK, 0.4 mm, 10.1 s |
| isotropic pair run time (median, CPU) | 2.37 s | 2.17 s (identical TRE) |

(The pose error against the pre-Pass-3c code for sub-01 was 4.4 mm and the 4:1, 10:1 synthetic failures of the earlier millimetre pyramid are gone: see `docs/PYRAMID.md` Section 11.2.)

## Deformable models (`syn`, `syngs`, `tvf`, `greedy`, `tvf_adj`, the JAX models): one convention
`build_image_pyramid` is **removed**; every model uses `syntx.pyramid.build_pyramid(..., 'deformable')` (`build_pyramid` and `plan_levels` are exported from `syntx`).
- **Level shapes are `round((N - 1) s / t) + 1`**, not `floor(N / L)`: 128 voxels at level 4 gives 33, not 32 (spacing 3.97 mm, not 4.03). The 16-voxel floor (one constant) binds earlier than before on small images; an axis shorter than the floor keeps its native samples.
- **Default smoothing is 0.5 x the axis's coarsening factor** (voxels, coarsened axes only) for every model, replacing `log2(level)` (syn, tvf, syngs, JAX), a zero-padded `0.5 n/m` (greedy) and none (tvf_adj, scattered). Same at factors 2 and 4, 4 against 3 voxels at factor 8. One smoother (`core/smoothing`, replicate padding); the moving image gets its own factor (greedy used the fixed image's sigma).
- **Anisotropic images** are leveled in millimetres: the thick axis is not shrunk with the thin ones (greedy, syn, syngs, tvf).
- **Spacing bookkeeping**: one `level_spacing` (`s (N-1)/(n-1)`, `Level.spacing`) feeds CFL, regularisation, Jacobian and normalisation; `tvf` no longer uses `sp * level` against an align-corners grid; `syn` / `syn_jax` take the CFL shrink ratio over all axes (it was z only: 1.0, no shrink, on an 11-slice slab where the in-plane axes shrank by 0.26); `tvf_adj` uses the corner-aligned spacing; JAX moving images keep their own level grid; `tvf_jax` keyframes use align-corners interpolation (was half-pixel, about 0.5 voxel off); `syngs_jax` smooths its pyramid by default; `syngs` / `tvf` velocity grid and image level come from one plan; the CoM-init grid of `syn` / `syn_jax` uses `level_spacing`.
- **TVF CFL fix**: `cfl_scale` was clamped at 1, so a small gradient was never scaled up and `TVFModel` returned the identity on every case since a5362ac (2-D r16 -> r64 Dice 0.563). Now 0.796 in 2-D (folding 0) and 0.92 to 0.955 in 3-D.
- **JAX `syngs` gradient fix**: `shoot` used `jnp.linalg.norm`, whose gradient is 0/0 at the initial v = 0: every gradient was NaN, the CFL guard never fired and `velocity_0_fwd` stayed 0 (every earlier JAX `syngs` result was affine-only plus a zero field); also the CFL momentum is resampled across levels. After the fix max |v| = 3.0 on the 32^3 probe.
- **JAX channel-axis fix**: `syngs_jax` and `tvf_jax` sent a `(D, H, W)` tensor to a channels-last filter, so only the y axis was smoothed.
- **Nearest-neighbour fixes**: the low-resolution candidate volumes of `robust_affine` (landmark / multi-start 2-D branch, isotropic phase-correlation branch) are linear; `landmarks.blob._separable_gaussian3d` no longer returns NaN for a zero sigma; greedy no longer crashes on a full-resolution level of an axis shorter than 16 voxels.
- **JAX SyNGS** raises `ValueError` for `box_lncc` (the box-window metrics belong to the JAX SyN / TVF dispatch), as its docstring says.

Measured (known-warp harness, textured volume with six ellipsoid labels warped by a smooth field of about 5 voxels, Dice of the warped moving labels, CPU, back to back): iso 64^3 syn 0.962 -> 0.963, syngs 0.964 -> 0.964, greedy 0.957 -> 0.957, tvf 0.955 -> 0.955; thin slab 64x64x14 (1, 1, 3) mm syngs **0.866 -> 0.910**, tvf 0.908 -> 0.922, syn 0.935 -> 0.938, greedy 0.925 -> 0.923;
aniso 64x64x32 syn 0.957 -> 0.957, syngs 0.954 -> 0.955, greedy 0.952 -> 0.955, tvf 0.949 -> 0.951. Recorded decreases: 3-D syn on 48x60x40 / (1.5, 1, 1) 0.957 -> 0.951, 2-D syn 0.786 -> 0.779. Time rises 10 to 40 percent on anisotropic cases (the thick axis keeps its native sampling and the floor binds earlier); isotropic within noise.
(tvf rows use the CFL fix on both sides: without it TVF does not move.)

## Guard, tests
- `tests/test_pyramid_guard.py` fails when pyramid / downsample code appears outside `src/syntx/pyramid.py` (explicit allow-list with a reason and a count per entry).
- `tests/test_robust_affine_anisotropy.py`: real assertions at 1, 2, 4, 8, 10:1 and for all ten real pairs; strict xfail for 6:1 pose A only.
- Two of the four previously failing tests fixed (JAX SyNGS `box_lncc`, TVF-JAX metric test). The two scattered-SyN tests (`test_scattered_syn_edge_cases_and_robustness`, `test_scattered_syn_point_weights`) still fail: the identity pose beats every start and the optimiser so the self-check raises; an `Identity` start fixes them but moves the passing `test_scattered_syn_recover_nonrigid_sinusoidal_2d` over its inverse-consistency threshold (1.02e-3 -> 1.86e-3 against 1.5e-3), so it is not included (it would also fix `test_deep_metrics_benchmark.py::test_milestone_4_deep_metrics_benchmark_execution`, which fails on 6.1.2 for the same reason). The fast suite on 6.2.0 fails the same 11 tests as 6.1.2 (nothing new): the two scattered-SyN tests, the deep-metrics benchmark, `test_parse_ants_affine_types_order_and_nonlinear` (ANTsPy TypeError), two mattes tests, `test_single_pe_dewarp_to_anatomical`, `test_rotation_invariant_descriptor_survives_60_degrees`, two `test_restrict_transformation` tests and `test_offline_behavior_and_download_failures` (missing local weights).

## Known limitations
- robust_affine raises (self-check) on a pair where the identity pose beats every start candidate and the optimiser (two scattered-SyN tests).
- Synthetic 6:1 pose A (deterministic, 4.4 mm): no start candidate lies within the metric's ~1 coarse-voxel translation capture. `translation_scan_steps=2` solves it on the phantom but regresses real sub-07 (1.2 -> 7.8 mm) and sub-01 (0.4 -> 1.5 mm), so it is opt-in. Open: a lattice objective that does not prefer the spurious deeper coarse minimum on real data.
- `tvf_adj` (deprecated prototype) does not register in either tree (Dice 0.02 to 0.04); only its spacing bookkeeping was fixed.
- JAX `syngs` results change materially (the old ones were affine plus a zero field).
