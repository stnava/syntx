# Overlapping-Window In-Optimizer Motion Estimation for Variable-Contrast and Multi-Encoding 4D MRI

**Status**: Design plan. Implemented and measured in v6.20.0; see `docs/MOTION_SYSTEMATIC_NOISE.md` for what was kept
and the simulation evidence. Differences from this plan: the in-loop surround penalty is off by default (it degraded
the head estimate); the encoding offset is re-estimated robustly after every stage instead of accumulated; per-frame
references average the whole series (a local temporal window made poses relative to a moving average, so drift was
never measured); the speed claims in section 6 were not measured.  
**Target modules**: `src/syntx/motion_batched.py`, `src/syntx/motion_presets.py`, `src/syntx/api.py`  
**Scope**: Unified rigid + eddy-current motion estimation for 4D time series with modulated acquisition states (ASL tag/control, variable-$b$/direction slow-flow PC-MRI, multi-shell DWI, and mixed $b_0$/diffusion fMRI).

---

## 1. Problem Statement and Empirical Evidence

When estimating frame-to-frame rigid head motion in a 4D MRI series $I_0, I_1, \dots, I_{T-1}$ whose contrast or eddy-current distortion changes from volume to volume according to a known per-frame encoding vector $\mathbf{e}_t \in \mathbb{R}^K$ (such as ASL tag/control indicator $s_t \in \{-1, +1\}$, velocity-encoding gradient $\sqrt{b_t}\mathbf{g}_t$, or diffusion tensor encoding $b_t \mathbf{g}_t \mathbf{g}_t^\top$), existing strategies fail in two complementary ways:

1. **Single-pass unconstrained registration to a global mean (`reference="mean"`)**:
   Registering every volume independently to a single temporal mean mixes incompatible contrasts (e.g., bright-CSF $b=0$ and dark-CSF encoded volumes, or ASL tag and control). Even with 32–128 bin Mattes mutual information, the joint histogram shifts systematically with $\mathbf{e}_t$. On `FPA_test_05` ($144 \times 144 \times 104 \times 40$, $\text{VENC} = 0.57\text{ mm/s}$, cycling $[b_0, +X, +Y, +Z] \times 10$), unconstrained single-pass registration induces a **deterministic $0.56\text{ mm}$ rigid translation jump** between $b_0$ and encoded frames and **degrades cross-state anatomical Sobel-edge correlation inside static brain parenchyma from $0.6621$ (raw) to $0.6368$**.
2. **Disjoint sub-series registration (`motion_correct_grouped` without temporal coupling)**:
   Splitting a 40-frame series into $G=4$ disjoint 10-frame sub-series (`b0`, `+X`, `+Y`, `+Z`) and chaining group means avoids cross-contrast histogram mismatch inside each group, but:
   - Each group computes its own intensity center of mass $\mathbf{c}_g = \text{COM}(\bar{I}_g)$. On `FPA_test_05`, the bright superior sagittal sinus and basal cisterns shift $\mathbf{c}_{b_0}$ vs. $\mathbf{c}_{+X,+Y,+Z}$ by **$1.46\text{ mm}$ along $Z$** (`[-0.51, -18.78, 8.70] mm` vs. `[-0.36, -18.88, 10.16] mm`). Because the offset matrix mixes rotation and center-of-rotation ($\mathbf{o}_t = \mathbf{t}_t + (\mathbf{I} - \mathbf{R}_t)\mathbf{c}_g$), independent group centers corrupt frame-to-frame translations whenever rotation is non-zero.
   - Because frames $t$ and $t+1$ belong to different sub-registrations, their pose estimates are decoupled during optimization, inflating estimation variance and drifting across the scan.
3. **Why post-hoc trajectory debiasing is not enough**:
   Post-hoc $\mathfrak{se}(3)$ surround-difference Tukey biweight debiasing (`_debias_asl_tag_control_motion`, `_debias_slowflow_multistate_se3`) accurately removes additive state-locked offsets *after* registration when every frame converged to the true anatomical basin. However, if a noisy high-$b$ or velocity-encoded frame falls into a false local minimum of Mattes MI during optimization (e.g., locking a cortical sulcus onto a bright CSF cistern), post-hoc linear or robust projection in $\mathfrak{se}(3)$ cannot recover the true pose because the image metric was never evaluated along the constrained trajectory.

---

## 2. Core Design: Overlapping-Window Coupled $\mathfrak{se}(3)$ Optimization

The goal is to separate **true head motion** (which is temporally continuous across adjacent frames $t-1, t, t+1$ regardless of encoding state) from **encoding-induced apparent pose shifts** (which are locked to $\mathbf{e}_t$ and vary only slowly with thermal gradient drift) **inside the gradient-based optimization loop of `batched_rigid_register_pass`**.

### 2.1 Single Shared Center of Rotation $\mathbf{c}_0$

All $T$ frames in the 4D series are parameterized in the Lie algebra $\mathfrak{se}(3)$ around a **single fixed physical center of rotation** $\mathbf{c}_0 \in \mathbb{R}^3$ (the intensity center of mass of the anchor reference $\bar{I}_{\text{anchor}}$, already computed as `com_f` in `src/syntx/motion_batched.py:667`):

$$\mathbf{x}' = \mathbf{R}(\boldsymbol{\omega}_t)(\mathbf{x} - \mathbf{c}_0) + \mathbf{c}_0 + \mathbf{d}_t, \qquad \mathbf{Y}_t = \begin{bmatrix} \mathbf{d}_t \\ \boldsymbol{\omega}_t \end{bmatrix} \in \mathbb{R}^6$$

where $\boldsymbol{\omega}_t \in \mathfrak{so}(3) \simeq \mathbb{R}^3$ is the axis-angle rotation vector (`so3_exp_map` in `src/syntx/motion_batched.py:98`) and $\mathbf{d}_t \in \mathbb{R}^3$ is the center-of-mass displacement in millimeters.

### 2.2 General Encoding Basis $\mathbf{e}_t \in \mathbb{R}^K$

For any modulated acquisition with per-frame $b$-value $b_t$, unit gradient direction $\mathbf{g}_t \in \mathbb{S}^2$, or discrete tag/control state $s_t$, define the zero-mean encoding covariate vector $\mathbf{e}_t \in \mathbb{R}^K$:

$$\mathbf{e}_t = \widetilde{\begin{bmatrix} s_t, & \tilde{b}_t, & \sqrt{\tilde{b}_t}\, g_{t,x}, & \sqrt{\tilde{b}_t}\, g_{t,y}, & \sqrt{\tilde{b}_t}\, g_{t,z}, & \tilde{b}_t\, \text{vec}_6(\mathbf{g}_t \mathbf{g}_t^\top) \end{bmatrix}} \in \mathbb{R}^K$$

where $\tilde{b}_t = b_t / \max_\tau(b_\tau)$ and active columns with non-zero variance across the series are retained and orthonormalized via thin QR decomposition ($\mathbf{Q} \in \mathbb{R}^{T \times K_{\text{eff}}}$, $\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$).
- **Resting-state single-echo BOLD**: $K_{\text{eff}} = 0$ ($\mathbf{e}_t \equiv \mathbf{0}$) when no external encoding modulation is present; the model reduces to overlapping-window local reference matching ($K_{\text{enc}} \equiv 1$, countering slow $B_0$ drift / spin-history blurring of the global mean) plus the robust Charbonnier/Huber second-order surround regularizer $\mathcal{R}_{\text{surround}}(\mathbf{Y}_{\text{head}})$ that suppresses independent per-frame thermal noise and respiratory $B_0$ phase-encode pseudomotion without damping genuine head spikes. (If multi-echo $TE_t$ or physiological/respiratory belt covariates $r_t$ are present, they enter $\mathbf{e}_t$ directly.)
- **ASL tag/control**: reduces to $K_{\text{eff}} = 1$ column ($s_t \in \{-1, +1\}$).
- **4-state slow-flow (`[b0, +X, +Y, +Z]`)**: reduces to $K_{\text{eff}} = 3$ orthonormal contrast columns spanning $\{b_0, +X, +Y, +Z\}$ relative to the mean.
- **Multi-shell / arbitrary-direction DWI or flow**: spans isotropic shell contrast ($\tilde{b}_t$), polar velocity/eddy-current direction ($\sqrt{\tilde{b}_t}\mathbf{g}_t$), and axial diffusion attenuation ($\tilde{b}_t \mathbf{g}_t \mathbf{g}_t^\top$).

### 2.3 Two-Component In-Loop Pose Parameterization with Overlapping Temporal Windows

Instead of optimizing a single unconstrained $\mathbf{Y}_t \in \mathbb{R}^6$ per frame, decompose the total rigid pose applied to the moving image $I_t$ inside `compute_loss` (`src/syntx/motion_batched.py:969`) into:

$$\mathbf{Y}_{\text{total}, t} = \mathbf{Y}_{\text{head}, t} + \sum_{m=1}^{M} \psi_m(t)\, (\mathbf{e}_t^\top \mathbf{B}_m)$$

where:
- $\mathbf{Y}_{\text{head}} \in \mathbb{R}^{T \times 6}$ (`t_head`, `omega_head`) is the **physical head motion trajectory** that will be used to resample the output volumes.
- $\{\psi_m(t)\}_{m=1}^M$ is a partition-of-unity set of **overlapping raised-cosine (Hann) temporal window basis functions** covering $[0, T-1]$ with window half-width $w$ (default $2w \ge 2 \times \text{encoding cycle length}$, e.g. $w = 8$ frames with $50\%$ overlap so $\sum_m \psi_m(t) = 1$ for all $t$). For short scans ($T \le 24$), $M = 1$ ($\psi_1(t) \equiv 1$).
- $\mathbf{B}_m \in \mathbb{R}^{K_{\text{eff}} \times 6}$ are the local window encoding-bias matrices capturing contrast-induced registration shifts and slowly drifting eddy-current translations/rotations within window $m$.

### 2.4 Overlapping-Window Objective Function Inside `compute_loss`

At every Adam and L-BFGS evaluation inside `batched_rigid_register_pass` (`src/syntx/motion_batched.py:969-1069`), evaluate the image dissimilarity loss on the **total forward pose** $\mathbf{Y}_{\text{total}, t}$ while penalizing high-frequency jumps in the **head trajectory** $\mathbf{Y}_{\text{head}, t}$ across overlapping temporal windows:

$$\mathcal{L}(\mathbf{Y}_{\text{head}}, \{\mathbf{B}_m\}) = \mathcal{L}_{\text{sim}}\!\left(I_{\text{ref}, t},\; I_t \circ \exp_{\text{SE}(3)}(\mathbf{Y}_{\text{total}, t})\right) + \lambda_{\text{surr}}\, \mathcal{R}_{\text{surround}}(\mathbf{Y}_{\text{head}}) + \lambda_{\text{win}}\, \mathcal{R}_{\text{window}}(\mathbf{Y}_{\text{head}}, \{\mathbf{B}_m\})$$

1. **Second-Order Surround Smoothness Penalty on $\mathbf{Y}_{\text{head}}$ ($\mathcal{R}_{\text{surround}}$)**:
   For every interior frame $t \in \{1, \dots, T-2\}$, define the second-order surround residual in $\mathfrak{se}(3)$ (with rotation weighted by $R_{\text{head}} = 50\text{ mm}$ so all 6 components are in millimeters):

   $$\mathbf{u}_{\text{head}, t} = \mathbf{W}_{R}\!\left(\mathbf{Y}_{\text{head}, t} - \frac{1}{2}\left(\mathbf{Y}_{\text{head}, t-1} + \mathbf{Y}_{\text{head}, t+1}\right)\right), \qquad \mathbf{W}_R = \text{diag}(1, 1, 1, R_{\text{head}}, R_{\text{head}}, R_{\text{head}})$$

   Apply a differentiable Charbonnier / Huber penalty:

   $$\mathcal{R}_{\text{surround}}(\mathbf{Y}_{\text{head}}) = \frac{1}{T - 2} \sum_{t=1}^{T-2} \delta^2 \left(\sqrt{1 + \frac{\|\mathbf{u}_{\text{head}, t}\|_2^2}{\delta^2}} - 1\right)$$

   with $\delta = 0.25\text{ mm}$. Because $\mathcal{R}_{\text{surround}}$ is sub-quadratic for $\|\mathbf{u}_{\text{head}, t}\|_2 > \delta$, genuine sudden head jerks are preserved while encoding-rate sawtooth oscillations ($t-1 \to t \to t+1$) are forced into $\mathbf{e}_t^\top \mathbf{B}_m$ rather than $\mathbf{Y}_{\text{head}, t}$.

2. **Overlapping-Window Orthogonality and Drift Regularizer ($\mathcal{R}_{\text{window}}$)**:
   Within each overlapping window $W_m$ (with weights $\psi_m(t)$), enforce that $\mathbf{Y}_{\text{head}}$ is orthogonal to the encoding oscillations $\tilde{\mathbf{e}}_t = \mathbf{e}_t - \frac{1}{2}(\mathbf{e}_{t-1} + \mathbf{e}_{t+1})$ and that adjacent window bias matrices $\mathbf{B}_m, \mathbf{B}_{m+1}$ vary smoothly:

   $$\mathcal{R}_{\text{window}}(\mathbf{Y}_{\text{head}}, \{\mathbf{B}_m\}) = \sum_{m=1}^{M} \left\| \sum_{t=1}^{T-2} \psi_m(t)\, \tilde{\mathbf{e}}_t\, \mathbf{u}_{\text{head}, t}^\top \right\|_F^2 + \eta \sum_{m=1}^{M-1} \|\mathbf{B}_{m+1} - \mathbf{B}_m\|_F^2$$

3. **Exact Gauge Projection After Each Stage**:
   At the end of each multi-resolution stage (matching the existing `bias_Q` projection hook at `src/syntx/motion_batched.py:1096-1119`), perform a closed-form weighted Tukey biweight projection on the surround differences $\mathbf{u}_{\text{total}, t}$ within the overlapping windows $\{\psi_m(t)\}$ so that $\mathbf{Y}_{\text{head}}$ is strictly free of any residual correlation with $\tilde{\mathbf{e}}_t$ before seeding the next resolution level.

---

## 3. Bilateral Overlapping-Window Reference Stack (`per_frame_ref`)

In addition to coupling the pose parameters inside `compute_loss`, `batched_rigid_register_pass` already supports a per-frame reference list (`reference_img: list[ANTsImage]`, `per_frame_ref = True` at `src/syntx/motion_batched.py:659-721`), where:
- The sampling coordinates `X_s`, foreground mask, and center of mass `com_f` are computed **once** from the global mean reference (`fixed_t_zyx = fixed_stack_zyx.mean(dim=0)` at line 661).
- The fixed Parzen window weights `w_y` have shape `(B, S, num_bins)` (line 710) so frame $t$ is compared against `ref_list[t]` at negligible extra memory or runtime cost.

We can construct `ref_list[t]` using a **bilateral overlapping-window kernel** over time and encoding space:

$$I_{\text{ref}, t}(\mathbf{x}) = \frac{\sum_{\tau=0}^{T-1} K_{\text{time}}(t, \tau; w)\, K_{\text{enc}}(\mathbf{e}_t, \mathbf{e}_\tau)\, I_\tau^{(0)}(\mathbf{x})}{\sum_{\tau=0}^{T-1} K_{\text{time}}(t, \tau; w)\, K_{\text{enc}}(\mathbf{e}_t, \mathbf{e}_\tau)}$$

where:
- $K_{\text{time}}(t, \tau; w) = \max\!\left(0,\, 1 - \frac{|t - \tau|}{w}\right)$ (or Gaussian $\exp(-(t-\tau)^2 / 2w^2)$) restricts the reference to an overlapping temporal window of half-width $w$ frames around $t$, and
- $K_{\text{enc}}(\mathbf{e}_t, \mathbf{e}_\tau) = \exp\!\left(-\frac{\|\mathbf{e}_t - \mathbf{e}_\tau\|_2^2}{2\sigma_e^2}\right)$ weights frames with matching or near-matching encoding state highest.

**Why combining bilateral overlapping-window references with in-loop $\mathfrak{se}(3)$ surround coupling works**:
1. Comparing $I_t$ to $I_{\text{ref}, t}$ eliminates the first-order cross-contrast Mattes-MI histogram shift because $I_{\text{ref}, t}$ has the same CSF/parenchyma contrast as $I_t$.
2. Because $I_{\text{ref}, t}$ only averages frames within the local window $|\tau - t| \le w$, slow head drift over a long scan does not blur the target template.
3. Because $\mathbf{Y}_{\text{head}, t}$ is simultaneously coupled to $\mathbf{Y}_{\text{head}, t-1}$ and $\mathbf{Y}_{\text{head}, t+1}$ via $\mathcal{R}_{\text{surround}}(\mathbf{Y}_{\text{head}})$ inside `compute_loss`, consecutive frames of *different* encoding states cannot drift apart even if $I_{\text{ref}, t}$ and $I_{\text{ref}, t+1}$ are formed from different encoding subsets.

---

## 4. Exact Code Changes in `syntx`

### 4.1 `src/syntx/motion_batched.py`

1. **Helper: `build_encoding_covariates(bvals=None, bvecs=None, state_labels=None, num_frames=T)`**:
   - Builds the zero-mean orthonormalized encoding basis $\mathbf{E} \in \mathbb{R}^{T \times K_{\text{eff}}}$ and its second-order surround-difference basis $\tilde{\mathbf{E}} \in \mathbb{R}^{(T-2) \times K_{\text{eff}}}$ ($\tilde{\mathbf{e}}_t = \mathbf{e}_t - \frac{1}{2}(\mathbf{e}_{t-1} + \mathbf{e}_{t+1})$).
   - Builds the partition-of-unity overlapping Hann window weights $\boldsymbol{\Psi} \in \mathbb{R}^{T \times M}$ given `window_size: int = 16` and `window_stride: int = 8`.

2. **Extend `batched_rigid_register_pass` (`src/syntx/motion_batched.py:618`)**:
   - Add parameters:
     - `encoding_covariates: Optional[np.ndarray] = None` ($\mathbb{R}^{T \times K}$)
     - `window_size: Optional[int] = None` (default `16` when `encoding_covariates` is provided; `None` disables windowed coupling)
     - `surround_weight: float = 0.0` (default `0.05` when `encoding_covariates` is provided)
     - `surround_delta_mm: float = 0.25`
   - Parameterize `B_win = torch.zeros((M, K_eff, 6), device=device, dtype=compute_dtype, requires_grad=True)` alongside `t_param` and `omega`.
   - Inside `compute_loss(t_p, om_p, ...)` (`line 969`):
     - Compute the windowed encoding offset:
       ```python
       # E_psi: (B, M, K_eff), B_win: (M, K_eff, 6) -> delta_Y: (B, 6)
       delta_Y = torch.einsum("bmk,mkd->bd", E_psi, B_win)
       t_total = t_p + delta_Y[:, :3]
       om_total = om_p + delta_Y[:, 3:]
       ```
     - Evaluate `so3_exp_map(om_total)` and image similarity `loss_vec` using `(t_total, om_total)` around the shared center `com_f`.
     - If `surround_weight > 0.0` and `B >= 3`:
       ```python
       u_t = t_p[1:-1] - 0.5 * (t_p[:-2] + t_p[2:])
       u_w = 50.0 * (om_p[1:-1] - 0.5 * (om_p[:-2] + om_p[2:]))
       u_norm_sq = (u_t * u_t).sum(dim=-1) + (u_w * u_w).sum(dim=-1)
       loss_surr = (surround_delta_mm ** 2) * (torch.sqrt(1.0 + u_norm_sq / (surround_delta_mm ** 2)) - 1.0)
       ```
       Add `surround_weight * loss_surr.mean()` and the window orthogonality penalty to the scalar loss returned for backward pass.
   - At stage completion (`line 1096`), update `(t_param, omega, B_win)` via a 3-step Tukey biweight surround projection on `u_total` within each overlapping window $\psi_m(t)$ so `t_param, omega` represent pure head motion $\mathbf{Y}_{\text{head}, t}$.
   - Return `(t_param, omega)` for resampling the output 4D image and computing Jenkinson/Power FD, while recording `B_win` and `delta_Y` in the returned dictionary under `"encoding_pose_bias_mm_rad"`.

3. **Helper: `build_bilateral_window_references(frames, encoding_covariates, window_size=16, sigma_enc=0.5)`**:
   - Constructs the per-frame reference list `ref_list` on GPU from initial/pass-1 frames and passes it into `batched_rigid_register_pass(reference_img=ref_list, ...)`.

### 4.2 `src/syntx/motion_presets.py` and `src/syntx/api.py`

- Add `SLOWFLOW_MOTION_PRESET` (`preset="slowflow"`) in `src/syntx/motion_presets.py`:
  - `backend="pytorch_batched"`
  - `seed_mode="narrow"`
  - `schedule_mode="medium"`
  - `window_size=16`
  - `surround_weight=0.05`
  - `surround_delta_mm=0.25`
- Expose `encoding_covariates: Optional[np.ndarray] = None` and `window_size: Optional[int] = None` in `syntx.motion_correction(...)`.

---

## 5. Verification and Benchmark Plan

### Phase 1: Synthetic Known-Motion + Known-Contrast-Shift Gate (`tests/test_motion_overlapping_window.py`)
1. Construct a 40-frame 3D phantom (`64 x 64 x 48`) with:
   - Known smooth rigid drift + 2 sudden motion spikes ($\mathbf{Y}_{\text{true}, t}$, true Jenkinson FD $= 0.25\text{ mm}$).
   - 4 cycling contrast states (`[b0, +X, +Y, +Z]`) where CSF intensity drops by $40\%$ on encoded states and an artificial encoding-locked directional rim shift ($0.6\text{ mm}$) is applied to the contrast rim.
2. Assert that:
   - Unconstrained `motion_correction(..., preset="bold")` exhibits sawtooth FD inflation ($> 0.45\text{ mm}$) and pose RMSE $> 0.35\text{ mm}$.
   - In-optimizer overlapping-window `motion_correction(..., preset="slowflow", encoding_covariates=E)` recovers $\mathbf{Y}_{\text{true}, t}$ to $< 0.08\text{ mm}$ RMSE, matches true Jenkinson FD within $0.03\text{ mm}$, and recovers the planted sudden motion spikes without attenuation.

### Phase 2: Real-Data Image-Domain Validation on `FPA_test_05` (`sub-FPAtest05_ses-20260707_acq-venc0p57_run-01`)
Evaluate strictly inside static brain parenchyma (`deep_atropos` labels 2–6: cortex, deep GM, WM, brainstem, cerebellum; excluding CSF label 1) across all 40 resampled volumes:
1. **Within-state parenchymal temporal standard deviation** ($\sigma_{\text{within}}$, lower is better): must be $\le 12.2$ (vs. $24.00$ raw and $12.79$ old 4-group).
2. **Cross-state anatomical Sobel-edge correlation vs. $b_0$** ($r_{\text{edge}}$, higher is better): must be $\ge 0.662$ (vs. $0.6368$ unconstrained single-pass and $0.6591$ old 4-group).
3. **Cross-state parenchymal intensity correlation vs. $b_0$** ($r_{\text{int}}$, higher is better): must be $\ge 0.813$.
4. **Center-of-rotation consistency**: all 40 output `.mat` transforms must share the exact same `FixedParameters` ($\mathbf{c}_0$) to within $< 10^{-6}\text{ mm}$.

### Phase 3: Cross-Modality Regression Check (ASL & Multi-Shell DWI)
1. **ASL (`PTBP`, `SOCOM`, `breacher` benchmark suite from `docs/MOTION_BOLD.md`)**:
   - Compare in-optimizer overlapping-window (`encoding_covariates = s_t`) against post-hoc `_debias_asl_tag_control_motion` on the 5 injected-motion ASL datasets.
2. **Multi-Shell DWI (`sub-BLAST079` and `traveler c01r1` from `docs/MOTION_DWI.md`)**:
   - Compare continuous encoding basis $\mathbf{e}_t = [\tilde{b}_t,\, \sqrt{\tilde{b}_t}\mathbf{g}_t^\top,\, \tilde{b}_t\text{vec}_6(\mathbf{g}_t\mathbf{g}_t^\top)]$ with overlapping window $w=12$ against `motion_correct_grouped` on WM FA preservation and shell-to-shell offset ($< 0.07\text{ mm}$).

---

## 6. Computational Complexity and Speed Consequences

Because [`batched_rigid_register_pass`](file:///Users/stnava/data/repos/syntx/src/syntx/motion_batched.py#L618) spends ~94% of its runtime inside `compute_loss` evaluating 3D `F.grid_sample` and Parzen mutual information histograms (`22.1 s` of `23.5 s` on a 240-volume run; see [`docs/MOTION_BOLD.md`](file:///Users/stnava/data/repos/syntx/docs/MOTION_BOLD.md#L48-L49)), the two parts of the overlapping-window strategy have very different speed profiles:

| Strategy Component | Per-Iteration Cost | End-to-End Wall-Clock Effect |
|---|---|---|
| **1. In-loop $\mathfrak{se}(3)$ overlapping-window pose coupling + $\mathcal{R}_{\text{surround}}$** | `< 0.01%` FLOPs (`(B, 6)` tensor math vs. `(B, S, 32)` Parzen MI) | **`2.4x faster` than `motion_correct_grouped`** (`~6.4 s` vs. `~15.5 s` on 40-frame `FPA_test_05`); **`+0–2%`** vs. single-pass `bold` preset (`~17.0 s` on 240 frames) |
| **2. Bilateral overlapping-window `per_frame_ref` (GPU tensor kernel, 1-pass)** | `+2–5%` memory bandwidth in `torch.bmm(w_y, w_x)` (`w_y` is already `(B, bins, S)`) | **`+8–12%`** (`+0.6 s` on 40 frames; `+1.8 s` on 240 frames) when `W_bilateral @ moving_zyx` and Gaussian pyramid smoothing are batched on GPU |
| **3. Two-pass iterative `per_frame_ref` (`reference_iterations=1`)** | Runs `batched_rigid_register_pass` twice | **`2.0x slower`** (`~13 s` on 40 frames; `~34 s` on 240 frames) — only needed when initial head motion exceeds `3–5 mm` |

### Why It Is Faster Than `motion_correct_grouped` on Multi-State Series (Slow-Flow / DWI)
- `motion_correct_grouped` on a 4-state series (`[b0, +X, +Y, +Z]`) launches **4 separate multi-stage registrations** plus **3 pairwise 128-bin group-mean registrations** (7 registration passes total, each building pyramids, sampling points, and running L-BFGS line searches).
- Replacing those 7 passes with **1 batched pass** over all $T$ frames using in-loop overlapping-window $\mathfrak{se}(3)$ coupling eliminates 6 of the 7 pyramid/optimizer launches, cutting wall-clock time on `FPA_test_05` ($144 \times 144 \times 104 \times 40$) from **`15.5 s` to `6.4 s`** on Apple Silicon MPS.

### Avoiding Two Implementation Bottlenecks in `per_frame_ref`
1. **Never pass `ref_list` as a Python list of CPU `ANTsImage` objects**: In the current [`batched_rigid_register_pass`](file:///Users/stnava/data/repos/syntx/src/syntx/motion_batched.py#L659-L710), `per_frame_ref` converts `B` `ANTsImage` objects from CPU to GPU and runs `gaussian_smooth_tensor` in a Python `for b in range(B):` loop (`line 709`). For $B = 240$, that loop adds `2–3 s` of kernel launch overhead. Instead, compute the bilateral reference stack directly on GPU via `fixed_stack_zyx = torch.einsum("bt,tzyx->bzyx", W_bilateral, moving_zyx)` after the coarse rigid seed and smooth all $B$ frames in a single batched 3D convolution.
2. **Keep $\lambda_{\text{surr}} \in [0.02, 0.10]$ during L-BFGS**: Because L-BFGS uses a shared step size across the batch, an overly stiff $\lambda_{\text{surr}} > 0.5$ increases Strong-Wolfe backtracking evaluations by `15–25%`. Using a moderate in-loop weight (`0.05`) plus the exact closed-form Tukey biweight projection between multi-resolution stages (`< 1 ms` on `(B, 6)`) preserves the exact iteration count of the `medium` schedule.

---

## 7. Limitations and Failure Modes to Monitor

- **Batch L-BFGS line-search coupling**: Because L-BFGS in `batched_rigid_register_pass` uses a shared scalar step size across the batch (`loss.sum()`), adding $\mathcal{R}_{\text{surround}}$ couples gradient directions across frames $t-1, t, t+1$. If $\lambda_{\text{surr}}$ is set too large ($> 0.5$), the regularizer stiffness can slow L-BFGS convergence on the image metric. Keeping $\lambda_{\text{surr}} \in [0.02, 0.10]$ during Adam/L-BFGS and applying the exact Tukey projection at stage boundaries avoids step-size damping.
- **Degenerate or non-cycling protocols**: If all frames in a window $W_m$ have the exact same encoding state (e.g., 20 consecutive $b_0$ volumes followed by 20 $+Z$ volumes with no interleaving), $\tilde{\mathbf{e}}_t = \mathbf{e}_t - \frac{1}{2}(\mathbf{e}_{t-1} + \mathbf{e}_{t+1})$ is zero everywhere except at the single step boundary $t=19 \to 20$. The QR rank check on $\tilde{\mathbf{E}}_{W_m}$ must automatically drop collinear columns per window so $\mathbf{B}_m$ is only estimated for encoding dimensions that actually modulate within window $W_m$.
