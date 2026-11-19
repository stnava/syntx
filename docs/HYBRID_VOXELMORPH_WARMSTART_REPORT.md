# Hybrid Deep-Learning + Variational Registration: VoxelMorph SVF Warm-Starts Across `syntx`

## 1. Objective & Architectural Scope

This report documents the design, cross-solver integration, systems optimization, and 3-pair Mindboggle-101 evaluation of **hybrid deep-learning + variational image registration**, seeding multi-resolution variational solvers (`syntx.greedy`, `syntx.syn`, `syntx.tvf`, `syntx.syngs`, and `syntx.auto_reg`) from zero-shot pretrained 3-D `VoxelMorph` (`VxmPairwise`) stationary velocity fields (SVFs).

We tested two specific engineering hypotheses:

1. **Systems efficiency (`syntx.greedy`)**: Eliminating host serialization bottlenecks in `syntx.transforms._write_ants_images_fast` and redundant work in `syntx.core.digital_diffeomorphism.enforce_digital_diffeomorphism_nd` accelerates both cold-start `syntx.greedy` and hybrid solvers without altering numerical outputs.
2. **Hybrid warm-start hypothesis**: Seeding fine-scale `[2, 1]` `syntx.greedy` or symmetric `syntx.syn` from `<0.4 s` `VxmPairwise` SVF inference improves accuracy over zero-shot `VoxelMorph` (`+6–7 pp` DKT Dice) while maintaining `0.0000%` folding and single-interpolation `[1Warp.nii.gz, 0GenericAffine.mat]` export; we explicitly tested whether `[2, 1]` warm-started refinement reaches `~0.58+` DKT Dice faster than cold-start multi-scale `[4, 2, 1]` `syntx.greedy`.

**Summary of findings**:

- **Non-`fit` acceleration (`1.53×` E2E on `mbhard` Pair 44)**: Parallelizing `_write_ants_images_fast` via multi-member RFC-1952 gzip (up to 8 threads for payloads `≥ 8 MB`) reduced Pair 44 `.nii.gz` export time from `3.51 s` to `0.66 s` (`5.29×` faster), dropping `syntx.greedy [100, 100, 20]` end-to-end runtime on Pair 44 from `8.36 s` to `5.47 s` (`2.40 s` solver loop).
- **Universal `initial_warp` / `initial_inverse_warp` / `[warp, affine]` architecture**: Centralized in `src/syntx/core/initial_warp.py` and supported uniformly across all 4 `syntx` solver families (`greedy`, `syn` PyTorch/JAX, `tvf` PyTorch/JAX, `syngs` PyTorch/JAX, and `auto_reg`), including a fix in `src/syntx/syn.py` and `src/syntx/syn_jax.py` where `M_phys, t_phys` previously transformed fixed-space points into `moving`'s coordinate frame even when `initial_grid` was active (causing `cc2 = 0.0` on cross-orientation `RPI -> LAI` pairs).
- **Hybrid accuracy vs. cold-start falsification**: Across 3 Mindboggle-101 pairs (`44`, `77`, `0`) on isolated Apple MPS, `voxelmorph_hybrid_greedy` (`[2, 1] (35, 15)`) achieved **`0.5543` NN / `0.5670` GL DKT Dice** in **`2.34 s` opt (`5.08 s` E2E)** with **`0.0000%` folding** (`+6.69 pp` NN over zero-shot `voxelmorph_pretrained` at `0.4874` and `+1.68 pp` NN over `voxelmorph_polished` while `2.75×` faster in optimization). However, the `~0.58+` in `3–4 s` hypothesis was **falsified on cross-site brains**: cold-start `syntx_greedy_default` (`[4, 2, 1] (100, 100, 20)`, `adam`) reached **`0.5767` NN / `0.5901` GL** in **`2.17 s` opt (`4.41 s` E2E)** with **`0.0000%` folding**, outperforming `[2, 1]` hybrid warm-starts by `+2.24 pp` NN. Controlled experiments traced this deficit to two verified mechanisms: (i) zero-shot `VxmPairwise` produces severe local SVF velocity spikes (`FD minJ = -5.46` to `-8.55`, `0.03–0.06%` folding) that trap Eulerian pullback updates in wrong sulcal basins unless pre-smoothed (`vel_smooth_sigma=1.5` removes all initial folds and recovers `+0.61 pp` at `[2, 1]`), and (ii) omitting coarse `L4` updates prevents global sulcal re-routing on cross-site pairs (`RPI -> LAI`, `PIR -> LAI`).

---

## 2. Systems Profiling & Safe Fast-Paths (`syntx.greedy`)

Profiling `syntx.greedy` (`[100, 100, 20]`, `192×256×256`, Apple MPS) showed `~2.57 s` inside `GreedyRegistrationModel.fit()` and `~5.79 s` outside `fit()`. Instrumenting every non-`fit` stage identified `_write_ants_images_fast` as the dominant bottleneck (`3.5069 s` spent in single-threaded `zlib.compressobj(1)` compressing 144 MB of float32 displacement data), followed by `enforce_digital_diffeomorphism_nd` allocating unused `X_phys` (`~151 MB`) and `boundary_mask` tensors on unidirectional fields that already satisfied `det_f0 >= jac_floor`.

Two safe optimizations were implemented and verified:

1. **Multi-member RFC-1952 parallel gzip (`src/syntx/transforms.py`)**: `_write_ants_images_fast` partitions raw `.nii` byte streams `≥ 8 MB` into independent blocks, compresses blocks concurrently across up to 8 `ThreadPoolExecutor` workers as valid concatenated RFC-1952 gzip members, and streams the result. Verified bitwise identity against `ants.image_read`, `gzip.open`, and `ants.apply_transforms`. Reduced `_write_ants_images_fast` from `3.5069 s` to `0.6630 s` (`5.29×` faster).
2. **Fast floor pre-check & lazy allocations (`src/syntx/core/digital_diffeomorphism.py`)**: `enforce_digital_diffeomorphism_nd` checks `det_f0 >= jac_floor` in unidirectional mode before allocating `boundary_mask` or `dom_b`, and avoids computing unused normalization matrices (`M_norm`, `b_norm`, `A_to_vox`, `b_to_vox`) when `W_inv_disp is None` (`~0.261 s -> ~0.065 s` on clean fields).

---

## 3. Universal `initial_warp` & Single-Interpolation Architecture

All four variational solver families now share one physical-space contract in `src/syntx/core/initial_warp.py`:

- **Accepted inputs**:
  - `initial_warp`: normalized `[-1, 1]` displacement tensor `(1, *spatial, dim)` in `(z, y, x)` spatial / `(x, y, z)` component order, ITK LPS vector `ANTsImage` (`has_components=True`), or `.nii` / `.nii.gz` warp path.
  - `initial_inverse_warp`: optional inverse in the same formats (computed via `update_inverse_field_nd_anderson` when needed and omitted).
  - `initial_transform=[warp_path, affine_path, ...]`: decomposed into `(u_0, M_phys, t_phys)` representing $T_{\text{init}}(x) = M_{\text{phys}}(x + u_0(x)) + t_{\text{phys}}$. Reversed `[affine, warp]` order raises `NotImplementedError`.
- **Single-interpolation endpoint composition (`compose_initial_and_residual_warps`)**:
  $$u_{\text{total}}(x) = w_{\text{fwd}}(x) + u_0(x + w_{\text{fwd}}(x)), \qquad u_{\text{total}}^{-1}(z) = u_0^{-1}(z) + w_{\text{inv}}(z + u_0^{-1}(z))$$
  Followed by `enforce_digital_diffeomorphism_nd` and exported strictly as `[1Warp.nii.gz, 0GenericAffine.mat]` (forward) and `[0GenericAffine.mat, 1InverseWarp.nii.gz]` with `whichtoinvert_inv=[True, False]` (inverse).
- **Cross-orientation `SyN.fit` / `SyNJAX.fit` fix (`src/syntx/syn.py`, `src/syntx/syn_jax.py`)**: When `self.initial_grid is not None`, `initial_grid` already maps fixed normalized coordinates into moving normalized coordinates. Previously, `SyN.fit` passed `self.moving_shape` / `moving_spacing` / `moving_origin` / `moving_direction` to `grid_to_physical_affine_torch`, shifting `phi_r2l_phys` out of `fixed`'s bounding box on cross-site pairs (`RPI -> LAI`) so `in_bounds_mask` became identically zero (`cc2 = 0.0`). Setting `M_phys = I`, `t_phys = 0` during `fit()` and `predict()` whenever `self.initial_grid is not None` resolved this completely (`loss` active from epoch 0, `0.4737 -> 0.5571` NN on Pair 44).
- **Provenance & adapters (`src/syntx/provenance.py`, `src/syntx/benchmark/external.py`, `evaluate.py`)**: All entry points set `model.has_initial_warp`, `model.initial_warp_kind` (`"none" | "tensor" | "ants_image" | "file" | "transform_list"`), and matching `provenance` keys. `run_voxelmorph_hybrid_greedy` and `run_voxelmorph_hybrid_syn` reuse cached `antsxdata` weights (`_VXM_CKPT_CACHE`), align moving volumes directly on GPU via `_Conv1x1AffineGrid` (`_align_moving_on_device`), permute losslessly to `"PIR"`, apply `vel_smooth_sigma=1.5` before SVF integration (`7` squaring steps), and map voxel shifts directly on GPU (`_voxel_disp_nat_to_norm_zyx`).

---

## 4. Cross-Solver Verification (`tests/test_hybrid_voxelmorph.py`)

All unit, guard, and cross-solver parity suites pass after the final edit:

- `python3 -m antsxqc.lint src/syntx`: `0 finding(s)` (zero `.antsxqc-allow` files).
- `tests/test_hybrid_voxelmorph.py` (`12 passed`):
  - Exact `< 1e-5 mm` coordinate round-trip (`_voxel_disp_nat_to_norm_zyx` $\leftrightarrow$ `normalized_to_physical_disp` $\leftrightarrow$ `physical_to_normalized_disp` $\leftrightarrow$ `compute_unclamped_initial_grid_zyx`) on an oblique anisotropic non-cubic grid `(24, 28, 32)`.
  - `GreedyRegistrationModel.fit` and `syntx.greedy` across `tensor`, `ants_image`, `file`, and `transform_list`, plus `NotImplementedError` on reversed `[affine, warp]`.
  - Cross-solver parity across all 8 solver/backend paths (`greedy-pytorch`, `syn-pytorch`, `syn-jax`, `tvf-pytorch`, `tvf-jax`, `syngs-pytorch`, `syngs-jax`, `auto_reg-pytorch`): exact zero-iteration reproduction (`MAE < 5e-3`), single-interpolation `[1Warp.nii.gz, 0GenericAffine.mat]` and `[0GenericAffine.mat, 1InverseWarp.nii.gz]` (`whichtoinvert_inv == [True, False]`), `0.0%` folding after optimization, and cross-orientation `RPI -> LAI` non-zero `cc2` loss regression test (`test_syn_initial_warp_cross_orientation_nonzero_loss`).
  - End-to-end contracts for `run_voxelmorph_hybrid_greedy` and `run_voxelmorph_hybrid_syn`.

---

## 5. 3-Pair Mindboggle-101 Benchmark Results (Isolated Apple Silicon MPS)

Evaluated sequentially via `scripts/benchmark_hybrid_voxelmorph.py --device mps` (`docs/provenance/hybrid_voxelmorph_benchmark.json`) across Pair `44` (`192×256×256` `RPI -> LAI`), Pair `77` (`256×256×160` `PIR -> LAI`), and Pair `0` (`256×256×160` `PIR -> PIR`), sharing the identical cached `syntx.robust_affine` `.mat` across all 10 arms with `cleanup_gpu` + `torch.mps.synchronize()` between arms.

### Mean Across Pairs `[44, 77, 0]` (Apple MPS)

| Arm | Type | Opt (s) | E2E (s) | Fixed NN Dice | Fixed GL Dice | Sym NN Dice | Fold % (Flow / FD) | Min $\|J\|$ (Worst) | Peak / Driver MPS (MB) | `antsxqc` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `syntx_robust_affine` | Affine (`.mat`) | `0.00` | `0.12` | `0.3342` | `0.3373` | — | `0.0000%` | `+1.0000` | `560.3 / 3365.7` | `not_assessable` (3/3) |
| `voxelmorph_pretrained` | DL Zero-shot (`SVF`) | `0.34` | `2.04` | `0.4874` | `0.4967` | — | `0.0470%` | `-8.5536` | `519.9 / 3235.7` | `fail` (3/3) |
| `voxelmorph_polished` | DL + `30` L1 SVF | `6.43` | `8.06` | `0.5375` | `0.5490` | — | `0.0000%` | `+0.0340` | `576.4 / 5793.7` | `not_assessable` (3/3) |
| `voxelmorph_hybrid_greedy_turbo` | DL -> `greedy [2,1] (25,10)` | `1.74` | `4.36` | `0.5461` | `0.5583` | — | `0.0000%` | `+0.0534` | `721.3 / 4159.6` | `not_assessable` (3/3) |
| `voxelmorph_hybrid_greedy` | DL -> `greedy [2,1] (35,15)` | `2.34` | `5.08` | `0.5543` | `0.5670` | — | `0.0000%` | `+0.0521` | `721.3 / 4321.6` | `not_assessable` (3/3) |
| `syntx_greedy_default` | Cold `greedy [4,2,1] (100,100,20)` | `2.17` | `4.41` | `0.5767` | `0.5901` | — | `0.0000%` | `+0.0050` | `728.3 / 4327.6` | `not_assessable` (3/3) |
| `syntx_greedy_matched` | Cold `greedy [4,2,1] (100,100,50)` | `6.52` | `8.76` | `0.5865` | `0.6005` | — | `0.0003%` | `-0.0517` | `728.3 / 4023.6` | `fail` (2/3) |
| `fireants_greedy` | External `[4,2,1] (100,100,50)` | `24.90` | `26.12` | `0.5894` | `0.6038` | — | `0.0015%` | `-0.0262` | `584.3 / 4031.7` | `fail` (3/3) |
| `voxelmorph_hybrid_syn` | Sym DL -> `syn [2,1] (30,15)` | `33.03` | `33.43` | `0.5674` | `0.5814` | `0.5632` | `0.0000%` | `+0.1111` (`FD 0.0580`) | `1016.3 / 4637.6` | `not_assessable` (3/3) |
| `syntx_syn_beltrami` | Sym Cold `syn [4,2,1] (100,100,20)` | `39.60` | `39.89` | **`0.6217`** | **`0.6387`** | **`0.6216`** | `0.0000%` | `+0.0559` (`FD 0.0529`) | `872.3 / 4645.6` | `not_assessable` (3/3) |

*(Note: For symmetric runs, `voxelmorph_hybrid_syn` achieved mean/max ICE of `0.016 / 1.674 mm`, and `syntx_syn_beltrami` achieved `0.012 / 0.776 mm`; `evaluate_registration_qc` reports `fail` whenever `folding_pct > 0` and `not_assessable` when `0.0%` folding is passed without a reference cohort baseline.)*

### Per-Pair Breakdown (`NN` / `GL` / `opt_s` / `fold%`)

- **Pair 44 (`NKI-TRT-20-2 -> MMRR-21-2`, `RPI -> LAI`, cross-site)**:
  - `voxelmorph_pretrained`: `0.4757 / 0.4856`, `0.38 s`, `0.0595%` (`minJ = -5.4632`)
  - `voxelmorph_polished`: `0.5221 / 0.5335`, `7.21 s`, `0.0000%`
  - `voxelmorph_hybrid_greedy_turbo`: `0.5292 / 0.5418`, `1.95 s` (`4.86 s` E2E), `0.0000%`
  - `voxelmorph_hybrid_greedy`: `0.5375 / 0.5500`, `2.61 s` (`5.62 s` E2E), `0.0000%`
  - `syntx_greedy_default`: `0.5639 / 0.5776`, `2.40 s` (`5.47 s` E2E), `0.0000%`
  - `syntx_greedy_matched`: `0.5779 / 0.5924`, `7.29 s` (`9.84 s` E2E), `0.0000%` (`minJ = 0.0035`)
  - `fireants_greedy`: `0.5786 / 0.5932`, `27.30 s`, `0.0010%`
  - `voxelmorph_hybrid_syn`: `0.5571 / 0.5711` (`SymNN = 0.5496`, `ICE = 0.015 / 1.627 mm`), `35.62 s`, `0.0000%`
  - `syntx_syn_beltrami`: `0.6165 / 0.6338` (`SymNN = 0.6120`, `ICE = 0.011 / 0.635 mm`), `44.20 s`, `0.0000%`
- **Pair 77 (`OASIS-TRT-20-5 -> MMRR-21-8`, `PIR -> LAI`, anisotropic cross-site)**:
  - `voxelmorph_pretrained`: `0.4588 / 0.4682`, `0.31 s`, `0.0501%` (`minJ = -6.9357`)
  - `voxelmorph_polished`: `0.5123 / 0.5243`, `6.04 s`, `0.0000%`
  - `voxelmorph_hybrid_greedy_turbo`: `0.5248 / 0.5370`, `1.65 s` (`4.19 s` E2E), `0.0000%`
  - `voxelmorph_hybrid_greedy`: `0.5333 / 0.5462`, `2.21 s` (`4.80 s` E2E), `0.0000%`
  - `syntx_greedy_default`: `0.5614 / 0.5754`, `2.09 s` (`4.05 s` E2E), `0.0000%`
  - `syntx_greedy_matched`: `0.5648 / 0.5784`, `6.19 s`, `0.0006%`
  - `fireants_greedy`: `0.5616 / 0.5753`, `23.81 s`, `0.0005%`
  - `voxelmorph_hybrid_syn`: `0.5375 / 0.5515` (`SymNN = 0.5393`, `ICE = 0.018 / 1.470 mm`), `33.00 s`, `0.0000%`
  - `syntx_syn_beltrami`: `0.6025 / 0.6198` (`SymNN = 0.6073`, `ICE = 0.013 / 0.776 mm`), `39.50 s`, `0.0000%`
- **Pair 0 (`OASIS-TRT-20-17 -> OASIS-TRT-20-16`, `PIR -> PIR`, intra-site)**:
  - `voxelmorph_pretrained`: `0.5278 / 0.5362`, `0.33 s`, `0.0313%` (`minJ = -8.5536`)
  - `voxelmorph_polished`: `0.5783 / 0.5893`, `6.05 s`, `0.0000%`
  - `voxelmorph_hybrid_greedy_turbo`: `0.5841 / 0.5961`, `1.61 s` (`4.05 s` E2E), `0.0000%`
  - `voxelmorph_hybrid_greedy`: `0.5922 / 0.6048`, `2.20 s` (`4.81 s` E2E), `0.0000%`
  - `syntx_greedy_default`: `0.6049 / 0.6175`, `2.03 s` (`3.71 s` E2E), `0.0000%`
  - `syntx_greedy_matched`: `0.6167 / 0.6307`, `6.09 s`, `0.0002%`
  - `fireants_greedy`: `0.6281 / 0.6429`, `23.59 s`, `0.0031%`
  - `voxelmorph_hybrid_syn`: `0.6076 / 0.6216` (`SymNN = 0.6007`, `ICE = 0.017 / 1.674 mm`), `30.47 s`, `0.0000%`
  - `syntx_syn_beltrami`: `0.6460 / 0.6625` (`SymNN = 0.6455`, `ICE = 0.012 / 0.600 mm`), `35.10 s`, `0.0000%`

---

## 6. Scientific Conclusions & Limitations

1. **Where Hybrid Warm-Starts Help**: When starting from a pretrained deep registration (`voxelmorph_pretrained`: `0.4874` NN, `0.0470%` folding), `voxelmorph_hybrid_greedy` repairs all topological folds (`0.0000%`, `minJ ≥ 0.052`) and adds **`+6.69 pp` NN (`+7.03 pp` GL) DKT Dice** in **`2.34 s` optimization (`5.08 s` E2E)**—strictly dominating `voxelmorph_polished` (`+1.68 pp` NN at `2.75×` lower optimization latency). On in-distribution intra-site Pair 0 (`PIR -> PIR`), `voxelmorph_hybrid_greedy` reaches **`0.5922` NN / `0.6048` GL**, and diagnostic `[4, 2, 1] (100, 100, 20)` from unsmoothed VoxelMorph reached `0.6069` NN (`+0.0020` above cold `syntx_greedy_default` `0.6049`).
2. **Why Cold-Start `syntx.greedy` Wins on Cross-Site Brains**: After parallel `.nii.gz` export and DD fast-path optimization, cold-start `syntx_greedy_default` (`[4, 2, 1] (100, 100, 20)`, `adam`) executes in **`2.17 s` opt / `4.41 s` E2E** (already matching `voxelmorph_hybrid_greedy_turbo` E2E speed) while achieving **`+2.24 pp` higher mean NN Dice (`0.5767` vs. `0.5543`)** with `0.0000%` folding. Zero-shot `VxmPairwise` under-performs on cross-site `RPI -> LAI` and `PIR -> LAI` contrast/geometry shifts (`0.4588–0.4757`), locking fine-level `[2, 1]` updates into local `5^3` LNCC minima (`loss` slightly *lower* on `hg` than `cold` at L1 while DKT Dice is lower).
3. **Symmetric Hybrid SyN**: `voxelmorph_hybrid_syn` (`[2, 1] (30, 15)`) preserves `0.0000%` folding and sub-voxel inverse consistency (`ICE mean = 0.016 mm`), improving `+8.0 pp` NN over zero-shot VoxelMorph, but cold `[4, 2, 1] (100, 100, 20)` `syntx_syn_beltrami` remains `+5.4 pp` NN higher (`0.6217` vs. `0.5674`) in comparable runtime (`39.6 s` vs. `33.0 s`).
4. **Recommendation**: Keep cold-start `syntx.greedy` (`[4, 2, 1]`, `2.17 s` opt / `4.41 s` E2E) and `syntx.syn` (`beltrami`, `0.6217` NN) as the default variational solvers, while providing universal `initial_warp` / `initial_inverse_warp` across all 4 `syntx` solver families and `run_voxelmorph_hybrid_greedy` / `run_voxelmorph_hybrid_syn` as fast, `0%`-folding refinement pipelines for deep-learning initializations.
