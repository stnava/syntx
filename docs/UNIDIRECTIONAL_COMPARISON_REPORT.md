# Unidirectional Registration Comparison on Mindboggle `mbhard`: `syntx.greedy`, `FireANTs`, and `VoxelMorph`

On Mindboggle-101 Pair 44 (`mbhard`, $n = 1$ inter-site 3-D brain pair, `0` dropouts, shared `syntx.robust_affine` initialization), matched unidirectional `syntx.greedy` (`[100, 100, 50]`, `regadam`) reaches `0.5716` nearest-neighbor (`0.5857` `genericLabel`) 62-label cortical Sørensen–Dice in `12.35 s` (`7.28 s` optimization loop) with `0.0000%` folding ($\min \det J = +0.0127$), compared with `0.5786` (`0.5932` `genericLabel`) in `27.37 s` (`24.66 s` loop, `0.0010%` folding, $\min \det J = -0.0058$) for repaired `FireANTs` and `0.5221` (`0.5335` `genericLabel`) in `7.40 s` (`0.0000%` folding) for pretrained `VoxelMorph` with 30-step stationary-velocity refinement. Symmetric bidirectional `syntx.syn` (`beltrami`, `[100, 100, 20]`) reaches `0.6165` (`0.6338` `genericLabel`) fixed-space Dice (`0.6121` bidirectional `NN` Dice) in `49.14 s` with `0.0000%` folding.

| Term | Meaning |
| :--- | :--- |
| `mbhard` | Mindboggle-101 Pair 44: fixed `NKI-TRT-20-2` ($192 \times 256 \times 256$, $1.0\text{ mm}$ isotropic, `RPS`) and moving `MMRR-21-2` ($170 \times 256 \times 256$, $1.2 \times 1.0 \times 1.0\text{ mm}$, `LAS`), evaluated across 62 non-background Desikan–Killiany–Tourville (DKT) cortical labels. |
| Unidirectional registration | Forward-only estimation of the fixed-to-moving pullback $\phi: \Omega_F \to \Omega_M$ with `return_inverse=False` (`invtransforms = []`), omitting post-hoc inverse field computation. |
| `NN` / `GL` Fixed Dice | Mean 62-label fixed-space Sørensen–Dice computed in a single interpolation pass (`[1Warp.nii.gz, 0GenericAffine.mat]`) using `nearestNeighbor` (`NN`, `GEMINI.md` §6 primary standard) or multi-label Gaussian `genericLabel` (`GL`). |
| Folding % | Percentage of fixed-domain voxels where the finite-difference physical Jacobian determinant satisfies $\det D\phi_{\text{def}}(x) \le 0$. |
| Loop / Total MPS time | Isolated single-process wall-clock runtime (seconds) on Apple Silicon `mps` (`torch 2.13.0`, `float32`); loop time measures pyramid optimization only, while total time includes preprocessing, digital diffeomorphism enforcement (where enabled), parallel `.nii.gz` export, and `warpedmovout` resampling. |

---

## 1. Formulation and Implementation Differences

All deformable arms share one cached physical affine $T_{\text{aff}}(x) = A(x - c) + c + t$ estimated by `syntx.robust_affine` (`9.14 s`, `0.3190` `NN` / `0.3226` `GL` Fixed Dice) and compose their fixed-grid physical displacement $u: \Omega_F \to \mathbb{R}^3$ with $T_{\text{aff}}$ in a single resampling pass:

$$\phi(x) = T_{\text{aff}}\bigl(x + u(x)\bigr), \qquad x \in \Omega_F.$$

The three unidirectional families parameterize and regularize $u(x)$ differently:

1. **Compositive Eulerian Displacement (`syntx.greedy` and `FireANTs GreedyRegistration`)**:
   At pyramid scale $s \in \{4, 2, 1\}$, an unconstrained gradient step $\delta_k$ of the $5^3$-voxel local normalized cross-correlation (LNCC) objective is smoothed by Gaussian standard deviation $\sigma_{\text{flow}}$, normalized by its spatial maximum, composed with the running warp, and smoothed by $\sigma_{\text{total}}$:
   $$u_{k+1} = G_{\sigma_{\text{total}}} \ast \Bigl(u_k \circ \bigl(\mathrm{Id} + \eta\,\tilde{\delta}_k\bigr) + \eta\,\tilde{\delta}_k\Bigr), \qquad \tilde{\delta}_k = \frac{G_{\sigma_{\text{flow}}} \ast g_k}{\max_{x}\|G_{\sigma_{\text{flow}}} \ast g_k\|_2}.$$
   In `syntx.greedy`, normalized lookup coordinates inside $T_{\text{aff}}$ are evaluated by the exact closed-form linear map $\operatorname{clamp}(\xi + w, -1, 1) M_\theta^\top + t_\theta$ (`_Conv1x1AffineGrid`), static fixed-image LNCC moments $(\mu_F, \sigma_F^2)$ are cached once per pyramid level (`15` separable 1-D taps per moving update instead of `30`), endpoint warps pass through sparse dual-stencil (`2h + 4h`) digital diffeomorphism projection ($\det J \ge 0.05$), and `.nii.gz` fields stream through parallel level-1 `zlib` export (`_write_ants_images_fast`). Repaired `FireANTs` (`1.5.1`, `/Users/stnava/code/FireANTs`) rotates normalized displacements by $A_{\text{norm}}$ inside `torch_grid_sampler_3d` and exports decoupled fixed-grid warps via `save_as_ants_transforms(..., save_inverse=False, include_affine=False)`.

2. **Amortized and Instance-Optimized Stationary Velocity Fields (`VoxelMorph 0.3.3`)**:
   `VxmPairwise` predicts a stationary velocity field $v_\theta(F, M \circ T_{\text{aff}})$ on the fixed grid and integrates $u = \exp(v_\theta) - \mathrm{Id}$ via $K = 7$ scaling-and-squaring steps (`IntegrateVelocityField`):
   $$\phi^{(0)}(x) = x + 2^{-K} v(x), \qquad \phi^{(m)}(x) = \phi^{(m-1)}\bigl(\phi^{(m-1)}(x)\bigr) \quad (m = 1, \dots, K).$$
   Three modes are implemented in `syntx.benchmark.external`:
   - `voxelmorph_pretrained` (`polish_steps=0`): fetches the MIT-licensed `DalcaLab` 3-D brain checkpoint (`syntx_features/voxelmorph_pairwise_3d_brain`, `435,477` bytes, SHA-256 verified via `antsxdata`), maps `(F, M_aff)` into canonical `"PIR"` axis order by a lossless signed permutation $(\pi, s) \in S_3 \times \{\pm 1\}^3$, executes one forward pass, and applies $(\pi, s)^{-1}$ before $D \cdot \operatorname{diag}(\text{spacing})$ physical export (zero extra image interpolation).
   - `voxelmorph_polished` (`polish_steps=30`): seeds $v$ from the pretrained forward pass and runs 30 full-resolution $L_1$ Adam steps minimizing $5^3$ `local_ncc_loss_nd` (`var_floor = 1e-6`, zero host syncs) plus $0.30\,\|\nabla v\|_2^2$ with Gaussian velocity smoothing ($\sigma = 0.40$ vox).
   - `voxelmorph_svf` (`[4, 2, 1]`, `(100, 100, 50)`): builds a `[4, 2, 1]` deformable pyramid via `syntx.pyramid.build_pyramid`, optimizes `VxmPairwise` weights for 100 steps at $L_4$ ($48 \times 64 \times 64$), and refines the upsampled velocity directly for 100 steps at $L_2$ and 50 steps at $L_1$ ($\sigma_{\text{final}} = 0.45$ vox), avoiding the `6.82 s/step` MPS `BasicUNet` 3-D `Conv3d` backward cost at $192 \times 256 \times 256$.

---

## 2. Quantitative Results on Mindboggle `mbhard` (Pair 44, Isolated Apple MPS)

All runs below start from the identical `syntx.robust_affine` transform on N4-corrected, `[2%, 98%]` foreground-normalized inputs (`NKI-TRT-20-2` $\leftarrow$ `MMRR-21-2`) in isolated `mps` processes (`torch.mps.empty_cache()` between runs).

| Method / Arm | Mode | Schedule & Optimizer | Fixed Dice (`NN`) | Fixed Dice (`GL`) | Folding % | $\min \det J$ | Loop MPS Time | Total MPS Time |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `syntx_robust_affine` | Invertible Affine | Multi-start Mattes MI | `0.3190` | `0.3226` | `0.0000%` | `+1.0000` | — | `9.14 s` |
| `voxelmorph_pretrained` | Unidir (`SVF`) | `0` steps (`PIR` zero-shot) | `0.4757` | `0.4856` | `0.0594%` | `-5.4580` | `0.32 s` | `0.51 s` |
| `voxelmorph_svf` *(unopt pilot)* | Unidir (`SVF`) | $L_2$ only, `200` steps (`neurite` $9^3$ NCC) | `0.3813` | `0.3861` | `0.0000%` | `+0.4016` | `181.9 s` | `184.33 s` |
| `voxelmorph_svf` *(remediated)* | Unidir (`SVF`) | `[4, 2, 1]` `(100, 100, 50)` ($5^3$ LNCC) | `0.5186` | `0.5291` | `0.0004%` | `-0.1866` | `32.1 s` | `34.58 s` |
| `voxelmorph_polished` | Unidir (`SVF`) | Pretrained + `30` $L_1$ SVF steps | `0.5221` | `0.5335` | `0.0000%` | `+0.0340` | `5.99 s` | `7.40 s` |
| `syntx_greedy` *(default)* | Unidir (`Comp`) | `[100, 100, 20]`, `adam` ($\sigma = 1.8 / 0.28$) | `0.5639` | `0.5776` | `0.0000%` | `+0.0538` | `2.57 s` | `8.36 s` |
| `syntx_greedy` *(matched)* | Unidir (`Comp`) | `[100, 100, 50]`, `regadam` ($\sigma = 1.8 / 0.28$) | `0.5716` | `0.5857` | `0.0000%` | `+0.0127` | `7.28 s` | `12.35 s` |
| `fireants_greedy` *(repaired)* | Unidir (`Comp`) | `[100, 100, 50]`, `regadam` ($\sigma = 1.0 / 0.25$) | `0.5786` | `0.5932` | `0.0010%` | `-0.0058` | `24.66 s` | `27.37 s` |
| `syntx_syn_beltrami` *(ref)* | Bidir (`SyN`) | `[100, 100, 20]`, `beltrami` ($\text{dil} = 0.1$) | `0.6165` | `0.6338` | `0.0000%` | `+0.0592` | `43.8 s` | `49.14 s` |

Key empirical comparisons on `mbhard`:

1. **`syntx.greedy` vs. Repaired `FireANTs` at Matched `[100, 100, 50]` (`regadam`)**:
   - `FireANTs` (`smooth_grad_sigma=1.0, smooth_warp_sigma=0.25, smooth_step_sigma=0.5, lr=0.5`) scores `+0.70 pp` higher `NN` Fixed Dice (`0.5786` vs. `0.5716`; `+0.75 pp` `GL`) because its lighter regularization ($\sigma = 1.0 / 0.25$ vs. `syntx.greedy`'s $\sigma = 1.8 / 0.28$) permits higher local cortical strain ($\max \det J = 20.69$ vs. `14.62`), leaving `0.0010%` folded voxels ($\min \det J = -0.0058$).
   - `syntx.greedy` enforces `0.0000%` folding ($\min \det J = +0.0127$), runs **`3.39x` faster inside the optimization loop** (`7.28 s` vs. `24.66 s`) and **`2.22x` faster end-to-end** (`12.35 s` vs. `27.37 s`), and uses **`1,779 MB` peak allocated VRAM** (`-924 MB` reduction from removing the `144 MB` 3-channel `affine_grid_cf` trilinear sampler).
   - Under its default `[100, 100, 20]` schedule, `syntx.greedy` completes its deformable loop in `2.57 s` (`8.36 s` end-to-end) at `0.5639` `NN` (`0.5776` `GL`) with `0.0000%` folding.

2. **`VoxelMorph` Trajectory Across Remedies**:
   - Replacing `neurite`'s host-synchronizing $9^3$ `NCC` (`assert torch.all(valid)`) and single-scale $L_2$ loop with `syntx.pyramid` `[4, 2, 1]` scale space and sync-free $5^3$ `local_ncc_loss_nd` raised untrained `voxelmorph_svf` by `+13.73 pp` `NN` Dice (`0.3813` $\to$ `0.5186`; `+14.30 pp` `GL`) while cutting runtime `5.33x` (`184.33 s` $\to$ `34.58 s`).
   - Pretrained `DalcaLab` zero-shot inference (`voxelmorph_pretrained`) executes in `0.51 s` (`0.4757` `NN` / `0.4856` `GL`) and exhibits `0.0594%` folding ($\min \det J = -5.458$) from discrete 7-step scaling-and-squaring discretization on steep sulcal gradients; 30 steps of smoothed $L_1$ SVF refinement (`voxelmorph_polished`, `7.40 s`) removes all folds (`0.0000%`, $\min \det J = +0.0340$) and raises `NN` Dice by `+4.64 pp` to `0.5221` (`0.5335` `GL`), trailing `syntx_greedy` (`8.36 s`) by `4.18 pp` `NN` (`4.41 pp` `GL`) at comparable wall-clock time.

3. **Unidirectional Ceiling vs. Symmetric Midpoint `SyN`**:
   - Symmetric midpoint `syntx_syn_beltrami` (`[100, 100, 20]`, `49.14 s`) exceeds the best unidirectional solver (`fireants_greedy`) by **`+3.79 pp` `NN` (`+4.06 pp` `GL`)** in fixed space while simultaneously providing a diffeomorphic inverse (`0.6076` Moving `NN` Dice, `0.6121` Bidirectional Mean `NN` Dice, `0.0108 mm` mean / `0.0726 mm` p95 inverse consistency error, `0.0000%` Liouville folding).

---

## 3. Limitations and Scope

1. **Single-Pair Sample Size ($n = 1$)**: All numbers above come from a single inter-site cortical pair (`mbhard`, Pair 44) chosen for its cross-scanner orientation mismatch (`RPS` vs. `LAS`) and large sulcal deformation; paired cohort variance, Wilcoxon tests, and failure-rate tails across the 90-pair Mindboggle-101 cohort (`--include-external` in [`benchmarks/fast_syn_ants_benchmark.py`](../benchmarks/fast_syn_ants_benchmark.py)) are not yet measured for `voxelmorph_polished` or `voxelmorph_svf`.
2. **Pretrained Domain Specificity**: `syntx_features/voxelmorph_pairwise_3d_brain` was trained on skull-stripped 1-mm T1-weighted human brain MRI in canonical `PIR` orientation; unlike `syntx.greedy`, `FireANTs`, and `syntx.syn`, `voxelmorph_pretrained` does not generalize zero-shot to calibrated Hounsfield CT or non-cranial MRI without retraining or multi-scale instance optimization.
3. **Regularization Hyperparameter Difference Between `syntx.greedy` and `FireANTs`**: In the `[100, 100, 50]` row, optimizer schedule and `regadam` match, whereas smoothing sigmas retain each package's benchmarked defaults (`flow_sigma=1.8, total_sigma=0.28` in `syntx.greedy` vs. `1.0 / 0.25` in `FireANTs`); lowering `syntx.greedy` sigmas toward `1.0 / 0.25` trades higher fixed-space cortical overlap against heavier reliance on post-loop digital diffeomorphism projection.
4. **Absence of Moving-Space Overlap and Inverse Consistency on Unidirectional Runs**: Setting `return_inverse=False` (`invtransforms = []`) eliminates `5–18 s` of fixed-point inversion overhead, so moving-space label overlap and round-trip inverse identity error are undefined by construction for the unidirectional rows.
