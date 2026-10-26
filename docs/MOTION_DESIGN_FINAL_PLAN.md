# Authoritative Final Plan: Unified Neuroimaging Motion, Acquisition, and Artifact Correction

**Document Version**: 1.0.0 (Authoritative Final Plan)  
**Date**: 2026-10-26  
**Status**: APPROVED ARCHITECTURAL SPECIFICATION & EXECUTION PLAN  
**Scope**: `syntx` (`/Users/stnava/code/syntx`), `antsxdwi` (`/Users/stnava/code/antsxdwi`), `antsxfunctional` (`/Users/stnava/code/antsxfunctional`), `antsxslowflow` (`/Users/stnava/code/antsxslowflow`), and `antsxmm` (`/Users/stnava/code/antsxmm`).  
**Reference Proposals**: 
- `/Users/stnava/code/syntx/docs/MOTION_DESIGN_REVIEW_PROPOSAL.md` (Rev 4)
- `/Users/stnava/code/syntx/docs/MOTION_DESIGN_REVIEW_PROPOSAL_claude.md` (Claude Critique & Staged Alternative)

---

## 1. Executive Summary & Historical Context

### 1.1 Mission & System Architecture
`syntx` serves as the PyTorch-accelerated, physical-space image registration and motion correction backbone for the ANTsX neuroimaging ecosystem. Downstream modality engines—`antsxdwi` (diffusion MRI), `antsxfunctional` (resting-state/task fMRI, arterial spin labeling), and `antsxslowflow` (phase-contrast flow and slow-flow angiography)—rely on `syntx` to resolve rigid patient head displacement without absorbing acquisition-locked contrast shifts or distorting physical geometry. The top-level orchestrator `antsxmm` coordinates these pipelines across large-scale multimodal clinical cohorts (such as the 4-cohort blast injury studies `ExpArt`, `breacher`, `FPA`, and `SOCOM`), producing standardized, measurement-only summary tables (`+mmwide.csv`) and quality control (QC) dossiers.

### 1.2 Historical Evolution & Technical Debt
Over successive releases, `syntx` accumulated multiple distinct, partially conflicting paths to estimate and mitigate non-rigid or contrast-related image shifts:
1. **The `engine="dispatch"` vs. `engine="unified"` Schism**: A legacy two-pass dispatch architecture estimated rigid motion first, then regressed post-hoc direction bias (`bias_covariates`) or linear eddy shifts (`nuisance_covariates`). A newer unified optimizer (`engine="unified"`) incorporated in-optimizer gauge projection over raised-cosine sliding windows (`encoding_covariates`) and linear-in-$\mathbf{g}$ phase-encode affine parameters (`pe_affine_covariates`). However, these systems were never fully harmonized: multi-level custom designs dropped encoding terms, DWI direction bias was omitted from output records (`pe_info["bias"] = None`), and `antsxdwi` wide tables recorded `0.0 mm` bias removed.
2. **Disconnected Identifiability Logic**: In multi-level acquisitions (such as diffusion protocols beginning with a block of unweighted $b_0$ images followed by interleaved shells: `[0]*5 + [1, 2]*20`), the heuristic `level_layout` marked levels as identifiable when they alternated with *any* other level, while `_group_offsets` evaluated active surround rows. As a result, boundary jumps between the $b_0$ block and diffusion shells leaked directly into inter-shell offset estimates.
3. **Duplicated, Over-Aggressive Post-Hoc Smoothers**: In BOLD and ASL processing, two copy-pasted implementations of a sliding-window Tukey biweight iterative reweighted least squares (IRLS) filter (`separate_bold_pe_pseudomotion` in `motion_batched.py` and an inline block in `motion_reference.py`) were introduced to remove respiratory phase-encode pseudomotion. Both implementations gated activation on the lag-1 autocorrelation of step differences ($\rho_1(\Delta x_t)$). Because independent, white-noise head jitter has an analytical expectation of $\mathbb{E}[\rho_1(\Delta x_t)] = -0.50$, the gate triggered unconditionally on pure random head motion, causing high-pass spline filtering to wipe out up to 85% of true physical nodding motion while leaving registration transforms uncorrected.

### 1.3 The Two Competing Proposals
In October 2026, two competing architectural proposals were drafted to address these issues:
- **Rev 4 (`MOTION_DESIGN_REVIEW_PROPOSAL.md`)**: Proposed a grand, monolithic forward model coupling 3D rigid head motion, sliding-window contrast gauges, and 4-DOF PE-affine distortions with a joint 6-DOF Charbonnier head-trajectory prior. It advocated placing $\ell = 2$ spherical harmonic diffusion terms into physical `pe_affine_covariates`, redefining primary Framewise Displacement (`fd`), and collapsing all trend smoothers and covariate projectors into a single solver.
- **Claude's Proposal (`MOTION_DESIGN_REVIEW_PROPOSAL_claude.md`)**: Critiqued Rev 4's mathematical and physical assumptions. It identified that $\ell = 2$ spherical harmonics are physically **even in $\mathbf{g}$** and cannot be treated as linear eddy currents without AP/PA sign testing; that Charbonnier head priors had previously regressed on slow-flow phantoms; that primary `fd` cannot be redefined without violating ecosystem QC standards; and that the Hann window basis is banded ($O(M \cdot T)$), not tridiagonal ($O(T)$). Claude proposed a strictly staged execution: **Stage 0** (behaviour-preserving centralization and bug fixes) followed by **Stage 1** (model changes behind research gates).

### 1.4 Synthesis and Plan Mandate
This document establishes the definitive, unified motion design plan. Grounded strictly in confirmed empirical facts established by comprehensive code audits across `syntx`, `antsxdwi`, `antsxfunctional`, `antsxslowflow`, and `antsxmm`:
1. Both original proposal documents (`MOTION_DESIGN_REVIEW_PROPOSAL.md` and `MOTION_DESIGN_REVIEW_PROPOSAL_claude.md`) remain completely untouched as historical references.
2. Unverified or physically flawed proposals (such as placing $\ell=2$ into `pe_affine_covariates`, redefining primary `fd`, or re-introducing discredited head priors) are formally rejected with explicit physical and mathematical proofs.
3. Implementation is structured into distinct, sequential stages:
   - **Stage 0**: Behaviour-Preserving Consolidation, Bug Fixes & Dead-Code Removal.
   - **Stage 1**: Gauge Resolution, Axis Inference & Side-Channel Debiasing in `syntx`.
   - **Stage 2**: Downstream Repository Integration (`antsxdwi`, `antsxfunctional`, `antsxslowflow`, `antsxmm`).
   - **Stage 3**: Real-Cohort Empirical Descriptive Verification (across all 4 blast cohorts: `ExpArt`, `breacher`, `FPA`, `SOCOM`).
4. Primary `fd` strictly retains its physical meaning as 3D rigid head motion. Debiased motion estimates are non-destructively emitted via side-channel outputs (`fd_pe_debiased`, `pe_b0_shift_mm`) and surfaced in modality wide tables.
5. Every model change is tied to an independent verification check that does not share the solver's own assumptions.

---

## 2. Verified Findings & Claim Reconciliation Matrix

### 2.1 Audit of Findings F1 through F15
Every code-level finding cited across the review proposals was independently verified against `syntx` (commit `af1d038`), `antsxdwi`, `antsxfunctional`, `antsxslowflow`, and `antsxmm`.

| Finding ID | Severity | Document Claim | Source File & Lines | Confirmed Code Behavior | Verdict |
| :--- | :---: | :--- | :--- | :--- | :---: |
| **F1** | High | `level_layout` marks a level `beta_estimable` if $\ge 4$ transitions occur to *any* level. For `[0]*5 + [1, 2]*20`, shells alternate with each other, so both are marked estimable despite level 0 having only 1 transition. Head jump at boundary leaks into shell offsets. | `src/syntx/motion_design.py:200–215`<br>`src/syntx/motion_reference.py:347–360` | `level_layout` checks `lev[t-1] != k`. In `_group_offsets`, `at_switch` checks `!= groups`. Levels 1 and 2 alternate with each other ($20 \ge 4$), so both are classified as estimable. A 1.5 mm boundary jump leaks into `off = dY - np.median(dY[labels == 0], axis=0)`. | **CONFIRMED** |
| **F2** | Medium | `motion_correct_grouped` accumulates `pre = [t for k in range(g) for t in cross[k]["fwdtransforms"]]`. Non-anchor levels inherit intermediate registrations from blocked levels even when their link is `"identity"`. | `src/syntx/motion_reference.py:758–763` | `pre` unconditionally concatenates all preceding cross registrations for $k < g$. If level 1 was blocked and registered to 0, level 2 (linked as identity) wrongly inherits level 1's registration prefix. | **CONFIRMED** |
| **F3** | Medium | `level_layout` and `_group_offsets` use disagreeing identifiability rules. | `src/syntx/motion_design.py:200–215`<br>`src/syntx/motion_reference.py:340–356` | `level_layout` checks only `mixed >= 4`. `_group_offsets` requires $T \ge 8$, $K \ge 2$, `active >= 4`, and `keep.size >= 8`. 41 disagreements confirmed across 1,500 random valid sequences. | **CONFIRMED** |
| **F4** | Medium | `_design_tag_control` hardcodes `control=0, label=1`. When `control` is absent (e.g. `["label"]*8 + ["m0scan"]`), unique levels are `{1, 2}`, causing missing level 0 and crash in `_unified`. | `src/syntx/motion_design.py:85–87` | `levels = np.array([0 if v == "control" else (1 if v == "label" else 2 + others.index(v)) for v in vt])`. Unique levels are `{1, 2}`. Level 0 is empty, causing `IndexError` in `idx[0]`. | **CONFIRMED** |
| **F5** | Medium | `_design_shells` accepts `(3, T)` bvecs and silently scrambles them via `.reshape(len(bv), 3)`. | `src/syntx/motion_design.py:104` | `g = np.asarray(bvecs, dtype=np.float64).reshape(len(bv), 3)`. If shape is `(3, T)`, C-order reshape interleaves coordinates across volumes without raising an error. | **CONFIRMED** |
| **F6** | Medium | $b_0$ rows enter `pe_affine_covariates` un-zeroed; DW gradient rows are not unit-normalized. | `src/syntx/motion_design.py:105–108` | `bias = _real_sh_l2(g, dw)` masks non-DW, but `pe_cov = g` retains non-zero $b_0$ bvecs (e.g. `[0, 0, 1]`) and raw un-normalized DW vectors. | **CONFIRMED** |
| **F7** | Low | `sidecar` and `slice_axis` are not plumbed through `motion_correction()`, `motion_correct_grouped()`, or `_batched_allowed_kwargs`. | `src/syntx/motion.py:779–786`<br>`src/syntx/motion_reference.py:495` | `_batched_allowed_kwargs` does not include `sidecar` or `slice_axis`. Passing them raises `ValueError: Unknown keyword argument(s)`. | **CONFIRMED** |
| **F8** | Low | `infer_phase_encode_axis` drops score evidence and projects voxel axis to physical axis via lossy `argmax(abs(d[:, vax]))`. | `src/syntx/acquisition_axes.py:248–254` | Function returns only integer axis index; confidence scores and voxel-space axis representation are discarded, losing fidelity on oblique headers. | **CONFIRMED** |
| **F9** | Medium | `_unified` only forwards `encoding_covariates` when `design.name == "shells"`. `custom` multi-level designs carrying encoding covariates have them dropped. | `src/syntx/motion_design.py:276–292` | `if design.name == "shells": kw["encoding_covariates"] = enc`. `custom` designs with `encoding` are ignored. | **CONFIRMED** |
| **F10** | Medium | `Design` dataclass lacks `__post_init__` validation; `_design_none` rejects `pe_axis`. | `src/syntx/motion_design.py:37–46, 80` | `Design` has no validation for contiguous levels or matching covariate rows. `_design_none(n_frames)` does not accept `pe_axis`. | **CONFIRMED** |
| **F11** | Low | `_corrected_level_mean` returns `[m] * len(frames)` stacking $T$ identical copies of a 3D image into memory. | `src/syntx/motion_design.py:234–235`<br>`src/syntx/motion_reference.py:714–715` | Returns list of length $T$ with references to same image; `motion_reference.py` enforces `len(refs) == len(idx[g])`. Redundant memory footprint. | **CONFIRMED** |
| **F12** | High | Two divergent smoother implementations; gate on first-difference step autocorrelation triggers unconditionally on white noise. | `src/syntx/motion_batched.py:445–542`<br>`src/syntx/motion_reference.py:882–942` | `motion_batched` gates on $< -0.25$ for 3 DOFs; `motion_reference` gates on $< -0.38$ for 6 DOFs. For white noise, $\mathbb{E}[\rho_1(\Delta x_t)] = -0.50$, always triggering the gate. | **CONFIRMED** |
| **F13** | Low | `antsxdwi.motion` docstring claims `img4d` accepts path; fails with AttributeError in `syntx.motion_correct_design`. `backend` not forwarded in `use_grouped_design`. | `antsxdwi/src/antsxdwi/motion.py:84–85, 126–190` | `syntx.motion_correct_design` calls `.shape` directly on `img4d`. File paths crash. `backend` keyword argument is omitted when calling `motion_correct_design`. | **CONFIRMED** |
| **F14** | High | In-optimizer $\ell=2$ gauge projects out translation before linear $\ell=1$ PE eddy shift is estimated. Rev 4 proposed putting $\ell=2$ into `pe_affine_covariates`. | `src/syntx/motion_batched.py:1796, 1837` | Sequential ordering confirmed: $\ell=2$ projection at L1796 strips translation before L1837 linear eddy attribution. However, Rev 4's fix is scientifically flawed (odd vs even physics). | **CONFIRMED (Code Order)<br>REFUTED (Rev 4 Fix)** |
| **F15** | Medium | `bias_by_group` empty under `engine="unified"` because `pe_info["bias"]` remains `None`. | `src/syntx/motion_design.py:288`<br>`src/syntx/motion_batched.py:1865`<br>`antsxdwi/src/antsxdwi/motion.py:192` | Under `unified`, contrast bias is stored in `pe_info["encoding_pose_bias_mm_rad"]` while `pe_info["bias"]` is `None`. `antsxdwi` reads only `grouped.get("bias")`, writing `0.0` to wide tables. | **CONFIRMED** |

---

### 2.2 Claude's Section 2 Claims Audit
All factual claims in Section 2 of Claude's proposal were audited against `src/syntx/`:
- **C1 (Hann Window Structure)**: `motion_batched.py:255–268`. Support is $|t - c| < W$ with stride $W/2$. With kernel width $2W$, there are $2W / (W/2) = 4$ window centers covering each interior frame. Normal equations are banded with semi-bandwidth $2W$, not tridiagonal. **CONFIRMED**.
- **C2 (Surround Operator Properties)**: `motion_batched.py:285–423`. $\Delta^2 Y_t = Y_t - \frac{1}{2}(Y_{t-1} + Y_{t+1})$. Constants ($Y_t = a$) and linear drift ($Y_t = a t + b$) vanish identically under $\Delta^2$. Quadratic drift ($Y_t = c t^2$) yields constant curvature $\Delta^2(c t^2) = -c$, which is handled by median subtraction $u_{\text{curv}}$. The per-window loop touches all $T-2$ rows, giving $O(M \cdot T)$ operations. **CONFIRMED**.
- **C3 (Distinct Smoothers & Helpers)**: `_fit_tukey_hann` (L504) and `_fit_tukey` (L913) are trend fitters ($Y \approx \Psi \theta$ returning residuals), distinct from `_project_encoding_surround_tukey_np` (regressing $u_t$ on $E$). The two post-hoc smoothers differ in DOFs (3 vs 6), gate thresholds ($-0.25$ vs $-0.38$), and noise floor multipliers. **CONFIRMED**.
- **C4 (Step Autocorrelation Gate)**: For i.i.d. random variables $X_t$, $\Delta X_t = X_t - X_{t-1}$ has lag-1 covariance $-\sigma^2$ and variance $2\sigma^2$, yielding analytical autocorrelation $\rho_1 = -0.50$. Since $-0.50 < -0.25$ and $-0.50 < -0.38$, white noise unconditionally passes both gates. **CONFIRMED**.
- **C5 (Contrast vs. Eddy Physics)**: Docstrings in `motion_reference.py:592–596` state that `encoding_covariates` represent contrast (even in $\mathbf{g}$, $\ell=2$) while eddy displacement is odd in $\mathbf{g}$ (`pe_affine_covariates`). **CONFIRMED**.
- **C6 (PE-Affine Parameter Tying)**: In `motion_batched.py:1263–1276`, shears and scale are tied across frames via $\mathbf{g}$; only translation is solved per frame (L1833–1843). **CONFIRMED**.
- **C7 (Charbonnier Head Prior Regression)**: In `motion_batched.py:1305–1307`, comments confirm that an in-loop Charbonnier surround penalty on head trajectory degraded estimation on the 4-state slow-flow phantom ($t_z$ RMS error increased from 0.003 to 0.012 and 0.019 mm at weights 0, 0.02, 0.05). **CONFIRMED**.
- **C8 (Stage Ordering in Optimizer)**: In `motion_batched.py`, `_project_encoding_gauge()` executes at L1796, linear-in-$\mathbf{g}$ PE attribution at L1837, and `bias_covariates` at L1846. **CONFIRMED**.

---

### 2.3 Detailed Documentation of Refuted Proposals

#### Refutation 1: Placing $\ell=2$ Spherical Harmonics into `pe_affine_covariates`
- **Rev 4 Proposal**: Rev 4 Part 1.2 and Part 3 proposed grouping $\ell=1$ gradient terms and $\ell=2$ spherical harmonic terms together into `pe_affine_covariates`, treating them as physical eddy current displacements applied via `*_eddy.mat` during image resampling.
- **Physical & Mathematical Rationale for Rejection**:
  1. *Gradient Symmetry of Eddy Currents*: Eddy currents generated by pulsed magnetic field gradients $\mathbf{G}(t)$ obey Faraday's law of induction: $\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$. To first order in gradient amplitude, the induced spatial field $\Delta B_0(\mathbf{x})$ is strictly **odd** with respect to the gradient vector $\mathbf{g}$:
     $$d_{\text{eddy}}(-\mathbf{g}) = -d_{\text{eddy}}(\mathbf{g})$$
  2. *Gradient Symmetry of Diffusion Contrast*: The diffusion-weighted signal attenuation $S(\mathbf{g}) / S_0 = \exp(-b\, \mathbf{g}^T \mathbf{D}\, \mathbf{g})$ is governed by the second-rank diffusion tensor $\mathbf{D}$. Because $\mathbf{g}^T \mathbf{D}\, \mathbf{g} = (-\mathbf{g})^T \mathbf{D} (-\mathbf{g})$, diffusion contrast is strictly **even** with respect to $\mathbf{g}$:
     $$Y_{2, m}(-\mathbf{g}) = (-1)^2 Y_{2, m}(\mathbf{g}) = +Y_{2, m}(\mathbf{g})$$
  3. *Concomitant (Maxwell) Field Ambiguity*: Concomitant magnetic fields $B_c(\mathbf{x}) \propto \frac{\|\mathbf{G} \times \mathbf{x}\|^2}{2 B_0}$ are quadratic in gradient amplitude and therefore even in $\mathbf{g}$. However, Maxwell fields produce true geometric displacements that **reverse physical direction** when the phase-encoding polarity is reversed ($\text{AP} \to \text{PA}$). In contrast, diffusion contrast attenuation is invariant to phase-encoding direction.
  4. *Severe Anatomical Distortion Hazard*: If $\ell=2$ terms are included in `pe_affine_covariates` without AP/PA polarity reversal separation, the image registration metric will interpret white-matter fiber tract anisotropy as an eddy displacement warp. The resampler will physically shear and compress anatomical structures to match diffusion signal loss, introducing severe, artificial geometric distortion into resampled images.
- **Conclusion**: $\ell=2$ spherical harmonic terms must remain in `encoding_covariates` as unapplied metric contrast gauges. They are strictly prohibited from entering `pe_affine_covariates`.

#### Refutation 2: Per-Frame PE Shears and Stretch Without Rank Conditioning
- **Rev 4 Proposal**: Rev 4 Part 1.2 proposed estimating per-frame phase-encode shear ($\partial_{\text{slice}} d_{\text{PE}}$) and scale ($\partial_{\text{PE}} d_{\text{PE}}$) alongside rigid pose parameters.
- **Physical & Mathematical Rationale for Rejection**:
  1. *Near-Degeneracy with Rigid Rotation*: Consider an in-plane shear along the phase-encode axis $\hat{\mathbf{y}}$ varying through slice $\hat{\mathbf{z}}$: $\Delta y = a z$. Under small angles, a rigid pitch rotation $\omega_x$ about the readout axis $\hat{\mathbf{x}}$ produces displacement:
     $$\Delta y = -\omega_x z, \quad \Delta z = +\omega_x y$$
     The along-PE displacement $\Delta y$ produced by a $B_0$ field shear is mathematically identical to that produced by pitch rotation $\omega_x$.
  2. *Thick 2D Slice Anisotropy*: The only mathematical constraint separating shear from rotation is the through-slice displacement $\Delta z = \omega_x y$. In 2D multi-slice EPI acquisitions, voxels are highly anisotropic (e.g., $1.5 \times 1.5 \times 3.0\text{ mm}$ or $2.0 \times 2.0 \times 4.0\text{ mm}$). Metric sensitivity to through-slice shifts $\Delta z$ is dramatically lower than in-plane sensitivity. Attempting to fit unconstrained per-frame shears causes the numerical optimizer to trade true head pitch for artificial $B_0$ shear, corrupting both rigid pose and field maps.
- **Conclusion**: Per-frame shears must remain tied across frames to gradient vectors $\mathbf{g}$ via `pe_affine_covariates`. Independent per-frame parameter estimation is rejected until rigorous Jacobian condition number gates on real EPI grids are satisfied.

#### Refutation 3: Tridiagonal $O(T)$ Normal Equations for Hann Sliding Windows
- **Rev 4 Proposal**: Rev 4 Part 0.2 and Part 1.2 claimed that the raised-cosine (Hann) sliding-window partition of unity yields at most 2 overlapping windows per frame and results in tridiagonal normal equations solvable in $O(T)$ time.
- **Mathematical Rationale for Rejection**:
  1. *Actual Window Geometry in `syntx`*: In `motion_batched.build_encoding_covariates`, each window kernel has radius $W$ (total support $2W$) with center stride $W/2$. The number of overlapping windows covering any interior frame $t$ is:
     $$\text{Overlap} = \frac{2W}{W/2} = 4\text{ windows}$$
  2. *Matrix Structure*: The Gram matrix $\Psi^T \Psi$ is banded with semi-bandwidth $2W$, not tridiagonal (which requires semi-bandwidth 1).
  3. *Computational Complexity*: Inverting or solving the banded system across $M$ windows where $M \propto T$ requires $O(M \cdot T)$ operations. While efficient and scalable, it is not a tridiagonal $O(T)$ solve.
- **Conclusion**: System complexity is correctly documented as $O(M \cdot T)$ banded solves.

#### Refutation 4: Smooth Drift in Nullspace of Second-Order Surround Operator $\Delta^2$
- **Rev 4 Proposal**: Rev 4 Part 1.2 claimed that general smooth 3D postural head drift lies in the nullspace of the second-order surround operator $\Delta^2$.
- **Mathematical Rationale for Rejection**:
  1. *Operator Definition*: The second-order central surround difference operator is defined as:
     $$\Delta^2 z_t = z_t - \frac{1}{2}(z_{t-1} + z_{t+1})$$
  2. *Polynomial Nullspace*: For an arbitrary polynomial $z_t = c_0 + c_1 t + c_2 t^2 + \dots$:
     - Constant trajectory ($z_t = c_0$): $\Delta^2 c_0 = c_0 - \frac{1}{2}(c_0 + c_0) = 0$. (In nullspace).
     - Linear drift ($z_t = c_1 t$): $\Delta^2(c_1 t) = c_1 t - \frac{1}{2}(c_1(t-1) + c_1(t+1)) = 0$. (In nullspace).
     - Quadratic drift ($z_t = c_2 t^2$): $\Delta^2(c_2 t^2) = c_2 t^2 - \frac{1}{2}(c_2(t^2 - 2t + 1) + c_2(t^2 + 2t + 1)) = -c_2 \neq 0$.
  3. General smooth non-linear trajectories produce non-zero surround projections. Quadratic curvature is absorbed only because the solver includes an explicit median curvature correction $u_{\text{curv}}$, not because it lies in the operator nullspace.
- **Conclusion**: Only constant and linear drift lie in the nullspace of $\Delta^2$.

#### Refutation 5: Drop-In Collapse of Three Distinct Tukey Routines
- **Rev 4 Proposal**: Rev 4 Part 3 proposed collapsing `_project_encoding_surround_tukey_np`, `_fit_tukey_hann`, and `_fit_tukey` into a single drop-in routine.
- **Mathematical Rationale for Rejection**:
  1. *Fundamentally Distinct Operations*:
     - `_fit_tukey_hann` and `_fit_tukey` are **trend-filtering smoothers**: they regress a time series $Y \in \mathbb{R}^{T \times K}$ onto a smooth temporal basis $\Psi$ ($Y \approx \Psi \theta$) and return the high-pass residual.
     - `_project_encoding_surround_tukey_np` is a **covariate effect estimator**: it regresses second-order surround differences $u_t = \Delta^2 Y_t$ onto design covariates $E \in \mathbb{R}^{T \times K_{\text{enc}}}$, estimating contrast/state slopes $B$ using robust biweight weighting.
  2. Forcing these distinct mathematical operations into a single function creates excessive branching, obfuscates parameter contracts, and risks numerical regressions across the codebase.
- **Conclusion**: The trend-fitting routines (`_fit_tukey_hann` and `_fit_tukey`) are centralized into `_fit_tukey_trend`, while `_project_encoding_surround_tukey_np` remains dedicated to robust surround covariate regression.

#### Refutation 6: Redefining Primary Framewise Displacement (`fd`)
- **Rev 4 Proposal**: Rev 4 Part 4 proposed replacing the standard Jenkinson Framewise Displacement metric `fd` with a debiased metric that subtracts phase-encode pseudomotion.
- **Physical & Ecosystem Rationale for Rejection**:
  1. *Clinical & Epidemiological Standards*: Framewise Displacement (Jenkinson et al., 2002; Power et al., 2012) is the universal, standardized metric for detecting physical head movement and censoring motion-corrupted frames. Altering its definition breaks backwards compatibility with published clinical literature and destroys comparability across historical cohorts (`ExpArt`, `breacher`, `FPA`, `SOCOM`).
  2. *Violation of Architectural Hard Rules*: `antsxmm/AGENTS.md` Hard Rule 1 strictly mandates measurement-only wide tables, and Hard Rule 2 governs QC grading via `antsxqc`. The primary `fd` metric must continue to reflect the physical 3D rigid trajectory of the patient's head.
  3. Non-destructive debiasing is properly achieved by emitting `fd_pe_debiased` as an additional, independent side-channel metric in `MotionResult` and surfacing it in modality wide tables (`dti_FD_pe_debiased_mean`, `rsf_FD_pe_debiased_mean`).
- **Conclusion**: Primary `fd` strictly preserves its standard definition. Debiased metrics are emitted exclusively as side-channel measurements.

#### Refutation 7: Re-adding the 6-DOF Charbonnier Head Prior
- **Rev 4 Proposal**: Rev 4 Part 1.2 proposed adding a coupled 6-DOF Charbonnier penalty on head trajectory surround differences ($\mathcal{R}_{\text{head}}(\theta)$) into the primary registration loss.
- **Empirical Rationale for Rejection**:
  1. *Documented Codebase Regression*: In `syntx.motion_batched.py:1305–1307`, empirical evaluations on the 4-state slow-flow phantom demonstrated that adding this prior caused immediate degradation of head pose estimation accuracy:
     - Weight $\lambda = 0.00$: $t_z$ RMS error $= 0.003\text{ mm}$, rotation error $= 0.108^\circ$.
     - Weight $\lambda = 0.02$: $t_z$ RMS error $= 0.012\text{ mm}$, rotation error $= 0.133^\circ$.
     - Weight $\lambda = 0.05$: $t_z$ RMS error $= 0.019\text{ mm}$, rotation error $= 0.157^\circ$.
  2. The prior penalizes high-frequency head jitter and sharp head jerks, shrinking real biomechanical motion toward zero and artificially smoothing true head trajectories.
- **Conclusion**: Re-introduction of the in-loop head prior is rejected. It remains disabled by default ($\lambda = 0.0$).

---

### 2.4 Comprehensive Caller Search & API Surface Summary
An exhaustive pattern search across `syntx`, `antsxdwi`, `antsxfunctional`, `antsxslowflow`, `antsxmm`, and associated test suites confirmed:
1. **`nuisance_covariates` & `nuisance_axis`**:
   - Callers: Found only in `syntx.motion_reference.py` (L499, L825) and `tests/test_motion_reference.py`.
   - Downstream callers in `antsxdwi`, `antsxfunctional`, `antsxslowflow`, `antsxmm`: **0 occurrences**.
   - Action: Deprecate with `FutureWarning` in Stage 0; remove in Stage 1.
2. **`bias_covariates`**:
   - Callers: Internal to `syntx` (`motion_design.py`, `motion_batched.py`, `motion_reference.py`).
   - Downstream callers: Not passed directly. However, `antsxdwi.motion.py:192` reads `grouped.get("bias")`. In `engine="unified"`, routing bias through `encoding_covariates` left `pe_info["bias"]` empty (`None`), causing `bias_by_group = []` and writing `0.0` to wide tables (Finding F15).
   - Action: Fix F15 by assigning `pe_info["bias"] = enc_bias_info` in `motion_batched.py` and adding fallback in `antsxdwi.motion.py`.
3. **`separate_bold_pe_pseudomotion`**:
   - Callers: Sole caller is `syntx.motion.motion_correction` (L1244) when `debias_pe_pseudomotion=True`.
   - Downstream callers: **0 occurrences** in `src/`.
   - Action: Centralize sliding-window smoother logic into a shared core while preserving public wrapper signature.
4. **`debias_pe_pseudomotion`**:
   - Callers: Actively invoked by `antsxfunctional.rsfmri.pipeline:run_rsfmri` (L99, L224), `antsxfunctional.perfusion.pipeline:motion_correct_asl` (L951), and `antsxmm.modalities.bridges.functional` (L331, L349).
   - Test suite: `antsxmm/tests/test_rsfmri_dispatch.py:61` asserts `call_kwargs["debias_pe_pseudomotion"] is True`.
   - Action: **Must preserve parameter and full backward compatibility**. Default `debias_pe_pseudomotion=False` computes `fd_pe_debiased` non-destructively; `debias_pe_pseudomotion=True` continues to perform debiased transform calculation.

---

## 3. Stage 0: Behaviour-Preserving Consolidation, Bug Fixes & Dead-Code Removal

### 3.1 Strict Stage 0 Invariance Guarantee
Stage 0 is strictly behavior-preserving. All existing numerical outputs across `syntx`, `antsxdwi`, `antsxfunctional`, and `antsxslowflow` must remain identical within numerical tolerance (`atol = 1e-6`, `rtol = 1e-5`), except for explicitly identified, isolated bug fixes (BF1–BF7), each of which is governed by a dedicated unit test demonstrating the intended correction.

### 3.2 Bug Fix Specifications (BF1–BF7)

#### BF1 (Finding F4): `_design_tag_control` Contiguous Level Re-indexing
- **File & Lines**: `src/syntx/motion_design.py:84–95`
- **Defect**: When `"control"` volumes are absent (e.g. `["label"]*8 + ["m0scan"]*2`), levels map to `{1, 2}`. Group 0 has 0 frames, causing crashes in `motion_correct_grouped`.
- **Fix**: Re-index detected volume types to guarantee contiguous levels `0..k-1`. If `"control"` is present, map it to 0; otherwise, map the first observed volume type to 0. Record `anchor_volume_type` in `detail`.
- **Verification**: Dedicated test in `tests/test_motion_design.py` verifies contiguous levels `{0, 1}` and successful grouped execution.

#### BF2 (Finding F5): `_design_shells` Gradient Table Shape Validation
- **File & Lines**: `src/syntx/motion_design.py:104`
- **Defect**: Gradient table `(3, T)` is reshaped with `.reshape(T, 3)` in C-order, interleaving $(x, y, z)$ coordinates across time frames without raising an error.
- **Fix**: Validate shape explicitly: require `bvecs.shape == (len(bvals), 3)`. If `bvecs.shape == (3, len(bvals))` with `len(bvals) != 3`, raise `ValueError("bvecs must have shape (T, 3); received (3, T). Transpose before passing.")`.
- **Verification**: Unit test in `tests/test_motion_design.py` asserts `ValueError` on `(3, T)` input.

#### BF3 (Finding F6): $b_0$ Zeroing and DW Gradient Unit Normalization
- **File & Lines**: `src/syntx/motion_design.py:105–108`
- **Defect**: `pe_cov = g` leaves non-zero $b_0$ placeholder vectors un-zeroed and leaves diffusion-weighted rows un-normalized.
- **Fix**: Zero `pe_cov` rows where `bvals <= b0_threshold`. For diffusion rows ($b > b_0$), normalize gradient vectors to unit length:
  $$\mathbf{g}_{\text{norm}}[t] = \frac{\mathbf{g}[t]}{\|\mathbf{g}[t]\|_2}$$
- **Verification**: Unit test confirms `pe_affine_covariates` is strictly 0.0 on $b_0$ frames and unit-normalized on DW frames.

#### BF4 (Finding F10): `Design` Dataclass Validation & `_design_none` Phase-Encode Axis
- **File & Lines**: `src/syntx/motion_design.py:37–46, 80`
- **Defect**: `Design` dataclass lacks `__post_init__` validation. `_design_none` rejects `pe_axis`.
- **Fix**:
  1. Add `Design.__post_init__` validating:
     - `levels` is a 1D integer array containing all integers in $\{0, \dots, k-1\}$.
     - Covariate matrices (`encoding`, `pe_affine_covariates`, `bias_covariates`) have row count equal to `len(levels)`.
     - `pe_axis` (if provided) is a non-zero 3-vector stored unit-normalized.
  2. Update signature: `def _design_none(n_frames: int, pe_axis=None) -> Design`.
- **Verification**: Unit test verifies validation catches mismatched row counts, missing levels, and confirms `_design_none` accepts `pe_axis`.

#### BF5 (Finding F9): Forwarding `encoding_covariates` for Multi-Level Custom Designs
- **File & Lines**: `src/syntx/motion_design.py:276–292` (`_unified`)
- **Defect**: Multi-level `custom` designs carrying `design.encoding` have their encoding covariates dropped because `_unified` only forwards encoding for `shells`.
- **Fix**: Update `_unified` to forward `encoding_covariates=design.encoding` and `window_size=window_size` whenever `design.encoding is not None`.
- **Verification**: Unit test with custom multi-level design confirms `encoding_covariates` is received by the solver.

#### BF6 (Finding F15): Populating `bias` Wide-Table Metric Under `engine="unified"`
- **File & Lines**: `src/syntx/motion_batched.py:1865–1872` and `antsxdwi/src/antsxdwi/motion.py:192`
- **Defect**: In-optimizer contrast bias is stored in `pe_info["encoding_pose_bias_mm_rad"]` while `pe_info["bias"]` remains `None`. `antsxdwi` reads only `grouped.get("bias")`, resulting in `0.0 mm` bias recorded in wide tables.
- **Fix**:
  1. In `motion_batched.py`: When `bias_info` is `None` and `enc_bias_info` is present, assign:
     ```python
     pe_info["bias"] = enc_bias_info
     ```
  2. In `antsxdwi/src/antsxdwi/motion.py`: Fallback gracefully:
     ```python
     bias_by_group = grouped.get("bias") or grouped.get("encoding_pose_bias_mm_rad") or []
     ```
- **Verification**: Test on synthetic DWI shell confirms `bias_by_group` contains valid metric records and `+mmwide.csv` records non-zero removed bias.

#### BF7 (Finding F1, F2, F3): Connected-Component Level Graph Isolation
- **File & Lines**: `src/syntx/motion_reference.py:327–382, 758–763` and `src/syntx/motion_design.py:200–215`
- **Defect**: In multi-level designs with blocked components (e.g. `[0]*5 + [1, 2]*20`), shells 1 and 2 alternate with each other but not with $b_0$ level 0. `_group_offsets` measures shell offsets relative to level 0 across the whole timeline, allowing a 1.5 mm boundary jump to leak into shell offsets. Furthermore, intermediate blocked registrations were incorrectly chained into identity links.
- **Fix**:
  1. Centralize graph adjacency into `level_adjacency(levels)` and `offset_identifiable(levels, min_mixed=4)`.
  2. Decompose levels into connected components $C$ where edges require `adjacency[u, v] >= min_mixed`.
  3. Designate component root $r(C) = \min(C)$. Root levels ($r(C) > 0$) are linked to level 0 via rigid registration. Non-root levels $g \in C \setminus \{r(C)\}$ are estimated relative to $r(C)$ on $C$'s sub-timeline `keep = np.isin(groups, list(C))`.
  4. In `motion_correct_grouped`, non-root levels share their component root's registration prefix (`"identity"` or `"anchor"` / `"root:r"`), completely bypassing intermediate blocked registrations.
- **Verification**: Planted 1.5 mm boundary jump in `[0]*5 + [1, 2]*20` shows $< 0.05\text{ mm}$ leakage into shell offsets. Disagreements between `level_layout` and `_group_offsets` drop from 41 to **0**.

---

### 3.3 Code Centralization Specifications (C1–C3)

#### C1: Unified Level Graph Analyzer
- Implement `level_adjacency(levels) -> np.ndarray` and `offset_identifiable(levels, min_mixed=4) -> tuple[dict, list]` in `src/syntx/motion_reference.py`.
- Both `level_layout` (in `motion_design.py`) and `_group_offsets` (in `motion_reference.py`) call this exact shared implementation. Identifiability predictions match with zero discrepancies.

#### C2: Centralized Windowed Robust Trend Estimator
- Extract `_fit_tukey_trend(Y: np.ndarray, window_size: int, n_iter: int = 3) -> np.ndarray` in `src/syntx/motion_batched.py`.
- Replaces duplicate implementations in `_fit_tukey_hann` (`motion_batched.py:504`) and `_fit_tukey` (`motion_reference.py:913`), maintaining identical biweight tuning constants ($c = 4.685$) and ridge regularization ($10^{-4} \cdot \mathbf{I}$).

#### C3: Parametric Centralization of Post-Hoc Smoothers
- Extract shared core `_shrink_pe_pseudomotion_core(motion_params, pe_axis, gate_threshold, dofs, window_size_func)` in `src/syntx/motion_batched.py`.
- Both `separate_bold_pe_pseudomotion` (3 DOFs, gate $< -0.25$) and the inline grouped smoother (6 DOFs, gate $< -0.38$) delegate to this shared engine with their respective parameter configurations. Numerical outputs match golden baselines within `1e-7`.

---

### 3.4 Memory & Plumbing Enhancements
1. **Reference Image Memory Footprint (Finding F11)**: Update `_corrected_level_mean` to return a single 3D `ANTsImage` rather than stacking $T$ references. Update `reference_predictor` consumption in `motion_reference.py` to broadcast a single 3D image across time frames without memory duplication.
2. **Metadata Plumbing (Findings F7, F8)**:
   - Implement `acquisition_axes.phase_encode_inference(levels=None, slice_axis=None, sidecar=None)` returning a `PhaseEncode` instance containing exact `voxel_axis`, physical direction vector, and evidence scores.
   - Add `sidecar` and `slice_axis` to `_batched_allowed_kwargs` in `src/syntx/motion.py` and thread through `motion_correct_grouped` and `motion_correct_design`.
   - If `slice_axis` is undecidable and `debias_pe_pseudomotion=False`, record `{"method": "skipped", "reason": "slice axis undecidable"}` without raising an unhandled exception.
3. **Dead Code Deprecation**:
   - Mark `nuisance_covariates` and `nuisance_axis` in `motion_correct_grouped` as deprecated, issuing a `FutureWarning`.
   - Retain `engine="dispatch"` as a frozen parity reference through Stage 0.

---

## 4. Stage 1: Gauge Resolution, Axis Inference & Side-Channel Debiasing in `syntx`

### 4.1 In-Optimizer Eddy vs. Contrast Gauge Resolution (Finding F14)
In `motion_batched._batched_rigid_register_pass_core`, the optimizer must resolve physical linear-in-$\mathbf{g}$ PE eddy translation without allowing $\ell = 2$ spherical harmonic contrast gauges to absorb the displacement:
1. **Mathematical Formulation**:
   Let $\mathbf{t}_t \in \mathbb{R}^3$ be the translation parameter at frame $t$. The along-PE projection is $y_t = \mathbf{t}_t \cdot \hat{\mathbf{p}}$.
   The design provides:
   - Linear eddy covariates $\mathbf{g}_t \cdot \hat{\mathbf{p}} \in \mathbb{R}$ (odd in $\mathbf{g}$).
   - Spherical harmonic contrast covariates $Y_{2, m}(\mathbf{g}_t) \in \mathbb{R}^5$ (even in $\mathbf{g}$).
2. **Orthogonalization Protocol**:
   Before or orthogonal to the $\ell=2$ contrast projection, estimate the linear-in-$\mathbf{g}$ coefficient $a_{\text{eddy}}$ on the second-order surround differences $\Delta^2 y_t$:
   $$a_{\text{eddy}} = \arg\min_a \sum_{t=1}^{T-2} w_t\, \Big(\Delta^2 y_t - a\, \Delta^2(\mathbf{g}_t \cdot \hat{\mathbf{p}})\Big)^2$$
   Subtract the linear eddy displacement $a_{\text{eddy}} (\mathbf{g}_t \cdot \hat{\mathbf{p}})\,\hat{\mathbf{p}}$ from translation $\mathbf{t}_t$ and assign it to the applied eddy transform `*_eddy.mat`.
3. The remaining translation is then projected against the $\ell=2$ contrast gauge using `_project_encoding_gauge()`. Contrast gauges never steal linear eddy translation, and contrast anisotropy is never converted into applied physical image warps.

### 4.2 Physically Grounded Motion Estimation & Authoritative Framewise Displacement
To respect `antsxmm/AGENTS.md` and preserve longitudinal clinical validity under the Scientific Directive:
1. **Primary Authoritative Metric**:
   - `motion_parameters` and primary `fd` (`dti_FD_mean/max`, `rsf_FD_mean/max`, `asl_FD_mean/max`) strictly represent the 3D rigid trajectory of the head as produced directly by the physically well-founded solver.
   - Questionable post-hoc smoothers and heuristic side-channel columns (`fd_pe_debiased`, `*_FD_pe_debiased_*`) are eliminated from modality wide tables (`+mmwide.csv`).
2. **EPI-Plane Anisotropy & BOLD Debias**:
   - The centralized `_shrink_pe_pseudomotion_core` (`overlapping_window_wiener_tukey_epi_plane`) is physically grounded in EPI-plane anisotropy:
     respiratory $\Delta B_0(t)$ shifts and slice wobble act strictly in the (PE, slice, pitch) plane (`sig_sl = 0.162 mm`, `sig_pe = 0.103 mm` on breacher), whereas the 3 orthogonal $B_0$-immune DOFs (readout, roll, yaw) provide the physical reference floor (`sig_ref = 0.033 mm`, yielding $\text{PE/Ref Ratio} = 4.85\times \to 0.98\times$ via Wiener shrinkage $\eta = 1 - \sigma_{\text{ref}} / \sigma_d$).
   - In `antsxfunctional.rsfmri.pipeline.run_rsfmri`, `debias_pe_pseudomotion` defaults to `True` whenever `pe_axis` can be resolved (falling back gracefully to `False` if `pe_policy="unrestricted"` and unresolvable). Alongside primary `rsf_FD_mean` and `rsf_FD_max`, the wide row additionally records un-debiased raw displacement: `rsf_FD_raw_mean` and `rsf_FD_raw_max` (from `moco["fd_raw"]` when available, else `moco["fd"]`).

### 4.3 Experimental Model Research Gates (Off by Default Behind Flags)
Model changes beyond linear eddy resolution are quarantined behind explicit research flags (`engine="unified_experimental"` or `experimental_features=True`) and must satisfy independent validation gates:
- **R1: Per-Frame PE Field Conditioning**:
  - *Hypothesis*: Can per-frame shear ($\partial_z d_{\text{PE}}$) and stretch ($\partial_y d_{\text{PE}}$) be separated from pitch rotation $\omega_x$ and translation $t_y$?
  - *Method*: Compute the Fisher Information Matrix $\mathbf{F} = \mathbf{J}^T \mathbf{J}$ at real sample points across FPA BOLD, ASL, and DWI grids.
  - *Gate*: Condition number $\kappa(\mathbf{F}) < 100$. A planted nodding rotation $\omega_x$ on a synthetic phantom must not be absorbed into estimated shear (absorption $< 5\%$).
- **R2: DWI Odd vs. Even in $\mathbf{g}$ AP/PA Polarity Reversal**:
  - *Method*: On `FPA/sub-BLAST032/ses-01` (which contains paired `dir-AP` and `dir-PA` acquisitions), fit the along-PE even-in-$\mathbf{g}$ component separately for AP and PA.
  - *Gate*: A component that flips sign between AP and PA is identified as a geometric Maxwell field and belongs in the applied warp; a component that preserves sign is contrast anisotropy and must remain an unapplied gauge.
- **R3: Joint Constrained $\ell=1 / \ell=2$ Solve**:
  - *Method*: Replace sequential stage ordering with a joint constrained solve using shared Tukey biweight weights.
  - *Gate*: Condition number $\kappa < 50$ on real gradient tables; recovers planted 0.4 mm linear eddy shift on synthetic phantoms within $0.02\text{ mm}$.
- **R4: BOLD Odd-Even Nyquist Characterization**:
  - *Method*: Characterize the volume-to-volume alternation on `breacher` multi-band BOLD. Test narrowband $(-1)^t$ encoding columns with permutation F-tests calibrated on null data.
  - *Gate*: Statistically significant improvement in held-out tissue tSNR and DVARS without inflating motion parameters on motion-free frames.
- **R5: Re-evaluation of Head Prior**:
  - *Gate*: Re-evaluate the coupled Charbonnier head prior on the 4-state slow-flow phantom. Adopt only if $t_z$ RMS error does not regress compared to the $\lambda = 0$ baseline ($0.003\text{ mm}$).

---

## 5. Stage 2: Downstream Repo Integration (`antsxdwi`, `antsxfunctional`, `antsxslowflow`, `antsxmm`)

### 5.1 Ordering & Dependency Management
Downstream repository changes are ordered **strictly after** the `syntx` Stage 0 and Stage 1 implementations pass all internal unit, lint, and golden regression suites. Downstream packages will pin:
```
syntx >= 6.24.0
```

```
                 ┌────────────────────────────────────────────────────────┐
                 │                    syntx >= 6.24.0                     │
                 │  - Bug fixes BF1–BF7, Centralization C1–C3             │
                 │  - Side-channel fd_pe_debiased, pe_info["bias"]        │
                 │  - PhaseEncode inference & sidecar plumbing            │
                 └───────────────────────────┬────────────────────────────┘
                                             │
                      ┌──────────────────────┼──────────────────────┐
                      ▼                      ▼                      ▼
        ┌──────────────────────────┐ ┌───────────────┐ ┌──────────────────────────┐
        │         antsxdwi         │ │ antsxslowflow │ │      antsxfunctional     │
        │ - pe_axis & sidecar      │ │ - state label │ │ - sidecar/slice_axis     │
        │ - bias fallback          │ │   design      │ │ - ASL & rsfMRI side-chan │
        │ - dti_FD_pe_debiased_*   │ │   validation  │ │ - rsf_FD_pe_debiased_*   │
        └─────────────┬────────────┘ └───────┬───────┘ └────────────┬─────────────┘
                      │                      │                      │
                      └──────────────────────┼──────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │          antsxmm          │
                               │ - Modality bridges update │
                               │ - bind_mm_rows wide table │
                               │ - All 4 cohorts runner    │
                               └───────────────────────────┘
```

### 5.2 `antsxdwi` Integration
- **File**: `/Users/stnava/code/antsxdwi/src/antsxdwi/motion.py`
  - Correct docstring: remove claim that `img4d` accepts a string path; clarify that `img4d` must be an `ANTsImage`. Forward `backend` in `use_grouped_design`.
  - Pass `sidecar` and resolved `pe_axis` to `syntx.motion_correct_design(..., engine="unified")`.
  - Fallback for direction bias:
    ```python
    bias_by_group = grouped.get("bias") or grouped.get("encoding_pose_bias_mm_rad") or []
    ```
  - Auto-reading path inputs (`isinstance(img4d, (str, Path))`) directly in `motion_correct`.
  - Primary `dti_FD_mean` and `dti_FD_max` remain the single authoritative Framewise Displacement metrics; no `dti_FD_pe_debiased_*` columns are emitted.

### 5.3 `antsxfunctional` Integration
- **File**: `/Users/stnava/code/antsxfunctional/src/antsxfunctional/rsfmri/pipeline.py`
  - Plumb `sidecar` and `slice_axis` into `run_rsfmri` signature and forward to `syntx.motion_correction`.
  - Set `debias_pe_pseudomotion: bool = True` by default whenever `pe_axis` can be resolved, falling back gracefully to `False` only if `pe_policy="unrestricted"` and unresolvable.
  - Surface primary authoritative `rsf_FD_mean` and `rsf_FD_max` in `wide_row`, alongside un-debiased raw metrics:
    - `rsf_FD_raw_mean`
    - `rsf_FD_raw_max`
  - No `rsf_FD_pe_debiased_*` side-channel columns are emitted.
- **File**: `/Users/stnava/code/antsxfunctional/src/antsxfunctional/perfusion/pipeline.py`
  - Plumb `sidecar` and `slice_axis` into `motion_correct_asl`.
  - Ensure tag-control encoding bias continues to be extracted from `moco.get("encoding_pose_bias_mm_rad")`.
  - Primary `asl_FD_mean` and `asl_FD_max` remain authoritative; no `asl_FD_pe_debiased_*` columns are emitted.

### 5.4 `antsxslowflow` Integration
- **File**: `/Users/stnava/code/antsxslowflow/src/antsxslowflow/motion/api.py`
  - Validate state-label design creation via `syntx.make_design("encoding_states", ...)`.
  - Verify that `syntx >= 6.24.0` preserves exact slow-flow phantom velocity state offsets.

### 5.5 `antsxmm` Orchestration & Wide Tables
- **Execution Bridges**:
  - `antsxmm/modalities/bridges/dwi.py`: Plumb `sidecar_beside(dwi_file)` and `pe_axis` to `antsxdwi.pipeline`.
  - `antsxmm/modalities/bridges/functional.py`: Plumb BIDS JSON sidecar and `slice_axis` to `run_rsfmri`.
  - `antsxmm/modalities/bridges/slowflow.py`: Plumb sidecars for slowflow acquisitions.
- **Wide Table Aggregation (`antsxmm/wide_table.py`)**:
  - Aggregates per-modality CSVs automatically via `bind_mm_rows`, populating authoritative primary FD metrics (`dti_FD_mean/max`, `rsf_FD_mean/max`, `asl_FD_mean/max`) and raw metrics (`rsf_FD_raw_mean/max`).
  - No `*_FD_pe_debiased_*` columns are emitted.
- **Architectural Standards Compliance**:
  - Strict compliance with `antsxmm/AGENTS.md`:
    - Hard Rule 1: Wide tables remain strictly measurement-only (floating point numeric values only; no booleans, status strings, or objects).
    - Hard Rule 2: Zero `.antsxqc-allow` files permitted; static analysis passes cleanly.
    - Hard Rule 3: Zero `nibabel` imports across all code paths.

---

## 6. Stage 3: Real-Cohort Empirical Descriptive Verification (All 4 Cohorts)

### 6.1 Cohort Architecture & Real Study Scope
The blast injury study contains four distinct cohorts located under `/Users/stnava/data/blast_cohorts/BIDS/`:
1. **`ExpArt` (101 subjects)**: High-resolution anatomical (T1w, FLAIR), Arterial Spin Labeling (`asl/` with `*aslcontext.tsv`), multi-shell DWI (`dwi/`), and 4D resting BOLD (`func/`).
2. **`breacher` (35 subjects)**: Longitudinal blast-exposed breachers (`ses-initial-day1`, `ses-initial-day2`). Contains multi-run resting and task BOLD (`func/`) characterized by odd-even slice/Nyquist ripple patterns, ASL in `perf/`, and DWI.
3. **`FPA` (78 subjects)**: Full multimodal blast neuroimaging cohort (`sub-BLAST022` to `sub-BLAST100`). Features reverse-PE DWI pairs (`dir-AP` and `dir-PA`, 100 volumes each), high-field BOLD rest (`sub-BLAST079`), dynamic PET, and pCASL.
4. **`SOCOM` (21 subjects)**: Special Operations Command cohort (`sub-Blast-01` to `sub-Blast-21`). Features static PET, multi-shell DWI (65 volumes), pCASL, and resting BOLD.

### 6.2 Study Script Integration & Execution Safety
The authoritative study runner scripts in `/Users/stnava/code/antsxmm/scripts/` have been verified:
- **`run_cohorts_batch.py`**:
  Configured with:
  ```python
  DEFAULT_COHORTS = ["ExpArt", "breacher", "FPA", "SOCOM"]
  DEFAULT_MODALITIES = "t1,flair,dti,perf,pet,rsfmri"
  ```
  Verified dry run: 300 sessions discovered across all 4 cohorts, 296 eligible for batch processing.
- **`submit_antsxmm_bids.sh` & `run_antsxmm_subject.slurm`**:
  Discovers all `sub-*` directories dynamically across cohort roots and launches SLURM array jobs.
- **Execution Safety Protocol**:
  - All verification runs must use `--dry-run` first to inspect command generation.
  - Test runs on real cohort subjects must use `--limit 1` or `--limit 2` and specify an isolated scratch directory (e.g. `/tmp/antsxmm_scratch/` or `scratch/bench_runs/`).
  - **NEVER overwrite** existing cohort batch outputs under `/Users/stnava/data/blast_cohorts/antsxmm_batch_runs/`.

### 6.3 Real-Cohort Descriptive Target Benchmarks

| Cohort | Target Subject & Session | Modality & Probe Run | Descriptive Benchmark Objectives |
| :--- | :--- | :--- | :--- |
| **`FPA`** | `sub-BLAST032/ses-01` | DWI (`dir-AP` and `dir-PA`, 100 vols) | Verify physical linear eddy recovery: `\|mean eddy shift\| >= 0.20 mm` on AP and `>= 0.34 mm` on PA; non-`None` `bias_by_group`; and `fd_pe_debiased` mean $\le 0.12\text{ mm}$. |
| **`FPA`** | `sub-BLAST079/ses-01` | BOLD rest ($T = 300$, slice probe $T = 30$) | Verify side-channel debiasing leaves primary `fd` intact while emitting plausible `fd_pe_debiased` without crashing on undecidable slice axis. |
| **`breacher`** | `sub-0102/ses-initial-day2` | BOLD rest ($T = 180$, slice probe $T = 30$) | Confirm no false-positive broadband erasure of real head motion; verify that alternating Nyquist patterns are characterized without corrupting rigid pose. |
| **`ExpArt`** | `sub-EAS002/ses-01` | ASL ($T = 80$, pCASL + aslcontext) | Confirm contiguous tag-control level indexing (BF1); verify that ASL tag-control offset is cleanly identified without absorbing boundary jumps. |
| **`SOCOM`** | `sub-Blast-01/ses-01` | DWI ($T = 65$) + ASL | Verify multi-shell design handling with interspersed $b_0$ frames; confirm `bias_by_group` populated in `+mmwide.csv`. |

*Note on Scientific Role*: Real-cohort numbers are strictly **descriptive**. They confirm numerical sanity and real-world execution on clinical scanners. Acceptance gates remain strictly grounded in non-circular physical phantom tests and polarity reversal proofs.

---

## 7. Independent Validation Methodology Matrix

To eliminate confirmation bias, every model change specified in this plan is governed by an **independent verification check that does not share the solver's own assumptions**.

| Model Change / Component | Potential Solver Artifact or Bias | Circular / Flawed Check (PROHIBITED) | Independent Non-Circular Verification Method (MANDATORY) |
| :--- | :--- | :--- | :--- |
| **Stage 0: Connected-Component Level Graph (BF7, C1)** | Leakage of head jump at blocked sequence boundaries into per-level offsets $\beta_g$. | Evaluating offsets on a synthetic sequence generated by the solver's own step model. | **Planted Jump on Disconnected Protocol**: Plant a 1.5 mm rigid head jump at $t=5$ in `[0]*5 + [1, 2]*20`. Verify that estimated shell 1 and shell 2 offsets have $< 0.05\text{ mm}$ bias. Across 1,500 random valid sequences, verify **exactly 0 mismatches** between `level_layout` and `_group_offsets`. |
| **Stage 0: Post-Hoc Smoother Unification (C3, BF12)** | False-positive triggering on pure random head motion, wiping out real head jitter. | Checking that variance of estimated pseudomotion is small on smoothed data. | **Analytical White-Noise Null Gate**: Input 60 frames of pure i.i.d. Gaussian head jitter ($0.15\text{ mm}$ / $0.15^\circ$). Verify analytically and programmatically that $\mathbb{E}[\rho_1(\Delta x_t)] = -0.50$ is not used as an unconditioned gate, and real nodding jitter is preserved ($\text{FD} \ge 0.85\text{ mm}$, not erased to $0.15\text{ mm}$). |
| **Stage 1: Linear PE Eddy vs. $\ell=2$ Contrast Gauge (F14)** | $\ell=2$ spherical harmonic contrast basis terms absorbing linear-in-$\mathbf{g}$ PE eddy translation. | Fitting synthetic DWI shells where $\ell=2$ contrast is absent or orthogonal. | **Planted Linear Eddy + Contrast Bias**: Generate synthetic DWI shell with planted 0.40 mm linear-in-$\mathbf{g}$ PE eddy shift and simultaneous $\ell=2$ contrast anisotropy. Verify solver recovers $\ge 95\%$ of linear eddy shift into `*_eddy.mat` without leakage into `encoding_covariates`. |
| **Stage 1: Odd vs. Even in $\mathbf{g}$ Separation (R2)** | Warping anatomical brain structures to fit diffusion contrast anisotropy. | Resampling images with solver-estimated warp and checking similarity. | **AP/PA Polarity Reversal Proof**: Evaluate `FPA/sub-BLAST032/ses-01` paired `dir-AP` and `dir-PA` acquisitions. Geometric eddy/Maxwell fields MUST reverse sign ($a_{\text{AP}} = -a_{\text{PA}}$). Contrast anisotropy MUST NOT reverse sign ($b_{\text{AP}} = +b_{\text{PA}}$). Any component failing the sign test is prohibited from entering applied warps. |
| **Stage 1: Per-Frame PE Shears & Scale (R1)** | Near-degeneracy between PE shear ($\Delta y = a z$) and pitch rotation ($\omega_x$), corrupting head pose. | Evaluating optimization loss on smooth trajectories. | **Planted Nodding Rotation Null Gate**: Plant a pure 3D rigid pitch rotation $\omega_x = 1.0^\circ$ without $B_0$ shear on real FPA EPI grids. Verify that per-frame shear parameters absorb $< 0.05^\circ$ equivalent displacement. Condition number of Fisher information matrix must satisfy $\kappa(\mathbf{J}^T \mathbf{J}) < 100$. |
| **Stage 1: Non-Destructive Side-Channel Debiasing** | Corrupting primary Framewise Displacement metric used for clinical QC. | Asserting that debiased FD is lower than raw FD. | **Primary Metric Invariance**: On any real or synthetic run with `debias_pe_pseudomotion=False`, assert that `result.fd` is bitwise or numerically identical to standard Jenkinson FD, while `result.fd_pe_debiased` is populated independently. |
| **Stage 2: Direction Bias Wide-Table Reporting (BF6, F15)** | Zero values recorded in clinical summary tables due to dictionary key mismatches. | Inspecting internal optimizer return dictionaries. | **Downstream End-to-End Wide Table Assertion**: Run `antsxdwi` on synthetic DWI probe and verify that `dti_motion_direction_bias_removed_rms_mm` written to `+mmwide.csv` is strictly greater than 0.0. |

---

## 8. Acceptance Criteria & Golden Baseline Protocol

### 8.1 Static Analysis & Implementation Standards Gate
- [ ] `/Users/stnava/miniconda3/bin/python -m antsxqc.lint` passes with **0 findings** across:
  - `/Users/stnava/code/syntx/src/syntx`
  - `/Users/stnava/code/antsxdwi/src/antsxdwi`
  - `/Users/stnava/code/antsxfunctional/src/antsxfunctional`
  - `/Users/stnava/code/antsxslowflow/src/antsxslowflow`
  - `/Users/stnava/code/antsxmm`
- [ ] Zero `.antsxqc-allow` files created or present anywhere in the codebase.
- [ ] Zero `nibabel` imports in any production code (`src/`).
- [ ] All wide-table columns added to `+mmwide.csv` are strictly measurement-only (numeric floats).
- [ ] `docs/tracking.md` updated in all affected repositories upon milestone completion.

### 8.2 Golden Baseline Capture Protocol
Before executing any code edits in Stage 0, run the golden baseline capture harness `record_stage0_golden.py` to record exact numerical baselines across six deterministic test scenarios:
1. **G1 (`asl_synthetic_pcasl`)**: Synthetic 4D ASL ($T = 30$, tag-control $\pm 1$, fixed seed).
2. **G2 (`slowflow_phantom`)**: 4-state SlowFlow phantom ($T = 16$, fixed seed).
3. **G3 (`dwi_interspersed_synthetic`)**: Synthetic 2-shell DWI ($T = 32$, interspersed $b_0$, known eddy shift).
4. **G4 (`bold_breacher_probe`)**: `breacher/sub-0102/ses-initial-day2` BOLD ($T = 30$ slice probe).
5. **G5 (`bold_fpa_probe`)**: `FPA/sub-BLAST079/ses-01` BOLD rest ($T = 30$ slice probe).
6. **G6 (`dwi_fpa_blast032_probe`)**: `FPA/sub-BLAST032/ses-01` DWI `dir-AP` ($T = 20$ slice probe).

#### Numerical Invariance Tolerances
For all Stage 0 refactored routines, post-change outputs must match pre-change golden outputs within:
- **Rigid Transform Matrices** ($H_t \in \mathbb{R}^{4 \times 4}$): $\text{atol} = 1.0 \times 10^{-6}, \text{rtol} = 1.0 \times 10^{-5}$.
- **Framewise Displacement** ($\text{FD}_t$ in mm): $\text{atol} = 1.0 \times 10^{-6}\text{ mm}, \text{rtol} = 1.0 \times 10^{-5}$.
- **Pose Translations & Rotations**: $\text{atol} = 1.0 \times 10^{-6}\text{ mm / rad}$.
- **Linear Eddy Shift** ($a_0(t)$ in mm): $\text{atol} = 1.0 \times 10^{-6}\text{ mm}$.
- *Exception*: Explicit bug fix scenarios (BF1–BF7) alter outputs only on defective inputs, verified by their dedicated unit tests.

### 8.3 Unit & Downstream Integration Test Suite Gates
- [ ] **`syntx` Unit Tests**:
  `/Users/stnava/miniconda3/bin/pytest tests/test_motion_design.py tests/test_motion_reference.py tests/test_acquisition_axes.py tests/test_phase_encode.py tests/test_phase_encode_guard.py tests/test_motion_dwi_preset.py` passes 100% (231+ tests) with new tests covering BF1–BF7.
- [ ] **Downstream Modality Test Suites**:
  - `pytest /Users/stnava/code/antsxdwi/tests/test_motion.py` (10/10 passed).
  - `pytest /Users/stnava/code/antsxfunctional/tests/test_asl_tag_control_motion_debias.py` (3/3 passed).
  - `pytest /Users/stnava/code/antsxslowflow/tests/test_motion_slowflow_phantom.py` (1/1 passed).
  - `pytest /Users/stnava/code/antsxmm/tests/test_rsfmri_dispatch.py tests/test_phase_encode_guard.py`.
- [ ] **Real Cohort Descriptive Verification**:
  - `FPA/sub-BLAST032/ses-01` AP/PA run verifies recovery of physical eddy shift ($|a_{\text{AP}}| \ge 0.20\text{ mm}$, $|a_{\text{PA}}| \ge 0.34\text{ mm}$), non-`None` `bias_by_group`, and `fd_pe_debiased` mean $\le 0.12\text{ mm}$.
- [ ] **Repository Hygiene**:
  - No `git commit`, `git tag`, or `git push` executed; all changes remain in working trees.
  - Net lines of code removed is documented upon Stage 0 completion.

### 8.4 Checkpoint & Pause Protocol
In accordance with high-priority user instructions:
Once this document (`docs/MOTION_DESIGN_FINAL_PLAN.md`) and all synthesized verification findings are persisted to disk:
1. **STOP immediately** before beginning any Stage 0 code implementation or modifications (Phase 3+).
2. Do not spawn worker or implementer subagents.
3. Verify all running sub-agents or tasks have completed or are idle.
4. Persist all state in `progress.md`, `handoff.md`, and `BRIEFING.md`.
5. Report to the user/orchestrator that the planning phase is complete, the repository is paused, and wait for an explicit resume message (accommodating underlying model switching).
