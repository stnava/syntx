# A single motion objective for BOLD, ASL, diffusion and slow-flow MRI time series with a re-descending trajectory prior

Author: Brian B. Avants. Affiliation: University of Virginia.

Numbers: `docs/reports/ismrm_cases_6_28_2026-10-28.json` (five series, syntx 6.28.0 default: cauchy 0.03; table by
`benchmarks/ismrm_table.py <json>`),
`docs/reports/prior_form_gate_2026-10-28_{phantoms,cohort,verdict}.json` (the prior gate). Figures:
`docs/abstracts/figures/ismrm_fig1_model_and_prior.png`, `ismrm_fig2_gate.png`, `docs/reports/ismrm_cases_6_28_2026-10-28_figure.png`.

## Synopsis (93 words)

On multi-shell diffusion MRI, eddy currents hold 73 to 83 % of phase-encode step variance and design-locked contrast offsets hold 39 to 62 % of B0-immune readout step variance. Single-reference registration assigns both to head motion. A batched solver in `syntx` registers every frame of an MRI time series to the mean of its own acquisition level, applies physical displacements (eddy-current affine and Jacobian), removes design-locked pose offsets, and damps the head trajectory with a re-descending Cauchy prior gated on three phantoms and five clinical series across BOLD, ASL, diffusion, and slow-flow MRI.

## Introduction

Single-reference registration assigns every acquisition-locked image change to head motion [1,2]: arterial spin labelling (ASL) tag-control alternation, diffusion contrast and eddy currents, and the physiological alternation of fast blood-oxygen-level-dependent (BOLD) series. Trajectory priors are the usual remedy and carry their own risk: a penalty strong enough to damp jitter also damps drift and real motion steps. On three phantoms and five clinical series, one batched objective and a gated re-descending prior separate these terms.

## Methods

**Forward model (Figure 1A).** One batched optimisation over all $T$ frames maximises Mattes mutual information of every frame against the mean $\bar I_{\ell(k)}$ of its acquisition level $\ell(k)$ (b-shell, ASL label state, or flow-encoding state). Frame $k$ at reference-space point $x \in \mathbb{R}^3$ satisfies
$$I_k\!\big(E_k(P_k(x))\big)\,(1 + a_k^{\top}p)\,\gamma_k \;\approx\; \bar I_{\ell(k)}(x),$$
with rigid map $P_k(x) = R_k(x - c) + c + t_k$ about centre of mass $c$ (rotation $R_k \in \mathrm{SO}(3)$, translation $t_k \in \mathbb{R}^3$), eddy-current affine $E_k(y) = y + p\big(a_{k0} + a_k^{\top}(y - c)\big)$ along unit phase-encode axis $p$ with $[a_{k0}, a_k^{\top}]^{\top} = C g_k$ linear in diffusion gradient $g_k$ via shell matrix $C$ [3], Jacobian $1 + a_k^{\top}p$, and per-frame gain $\gamma_k$ for scanner drift. The design enters the default solver in two places only: the level references $\bar I_{\ell(k)}$ and the six pose parameters $m_k + D_k\beta$, where $m_k \in \mathbb{R}^6$ is the free head pose and $D_k\beta \in \mathbb{R}^6$ is the design-locked offset for design row $D_k$ and coefficients $\beta$ (per level plus $l = 2$ direction terms per shell), removed from the reported pose after L-BFGS and coupled Newton stages and never applied. The chain written per frame is `[rigid, eddy]` using $m_k$ alone; a transform list acts on $x$ first-listed first, reproducing $E_k(P_k(x))$ in one interpolation.

**Prior (Figure 1B, C).** With $u_t = m_t - \tfrac{1}{2}(m_{t-1} + m_{t+1})$ the second difference of the head trajectory in mm (rotations at $50\ \text{mm}$), the objective adds $\lambda\kappa \sum_t \rho(|u_t|)$,
$$\rho(u) = \tfrac{\delta^2}{2}\log\!\big(1 + u^2/\delta^2\big), \qquad \delta = 0.05\ \text{mm},$$
where $\kappa$ is the measured curvature of the image term per $\text{mm}^2$ (median over frames and translation axes at each stage) and $\lambda = 0.03$ in curvature units. The pull $\rho'(u) = u/(1 + u^2/\delta^2)$ peaks at the knee $\delta$ and re-descends, so steps of several knees pass while noise-level wiggles are damped. The Newton stage uses the majoriser $(\rho'(|u_t|)/|u_t|)I_6$ with $6\times 6$ identity $I_6$ (the exact Hessian is indefinite beyond $\delta$). Two alternatives tested the hypothesis that small second differences should be shrunk hardest: Charbonnier and a sparse form with full pull at zero (knee $0.1\delta$).

**Gate, account, and data.** A setting passes if, on three phantoms with known trajectories (ASL drift with jitter, DWI with planted eddy shift, slow-flow encoding states), no phantom loses more than $10\ \%$ framewise displacement (FD) error against no prior and DWI improves; and on five clinical series the B0-immune readout step moves less than $0.02\ \text{mm}$, alignment changes less than $1\ \%$, and FD rises less than $2\ \%$. Repeat runs differ by less than $0.001\ \text{mm}$. Step variance per axis is split into head, applied eddy, and removed design-locked shares; voxel variance is split sequentially into gain, design, motion after design, and residual, raw and corrected. FD uses $m_k$ alone; near-Nyquist alternation is filtered for censoring only [2,4] while the full poses $m_k$ are the correction. Data (four cohorts, laptop GPU): resting BOLD ($\text{TR} = 1.0\ \text{s}$, $300$ vol), two ASL series ($25$ and $60$ vol), two multi-shell DWI series ($b = 1500/3000\ \text{s/mm}^2$, $99$ and $100$ vol). Software: `syntx 6.28.0`, `antsxdwi 0.33.x`, `antsxfunctional 0.23.x`, `antsxslowflow 2.12.x`.

## Results

**Gate (Figure 2).** Every prior that shrinks small second differences hardest failed: at weight $0.1$, the sparse form raised phantom FD error $87\ \%$ on ASL and $55\ \%$ on slow flow (DWI $-30\ \%$), and Charbonnier raised it $70\ \%$ and $51\ \%$ (DWI $-34\ \%$), because the phantoms' sub-knee jitter is true motion. The Cauchy form at $0.03$ passed every gate: phantom FD error changed $-14\ \%$ (ASL), $-17\ \%$ (DWI), and $-3\ \%$ (slow flow); on the five clinical series, the readout step moved at most $0.011\ \text{mm}$, alignment was unchanged, FD changed $-0.9$ to $+1.8\ \%$, and every frame converged ($81$ to $85\ \%$ without the prior, whose stage oscillates on plateau directions until its iteration cap; BOLD $206$ to $32\ \text{s}$). At $0.05$, the Cauchy form changed phantom FD error by $-7\ \%$, $-27\ \%$, and $+3\ \%$, but lowered the high-motion DWI readout step by $0.043\ \text{mm}$ and was rejected.

**Account (Table 1, Figure 3).** On the two DWI series, the eddy term holds $73$ to $83\ \%$ of the phase-encode step variance and the removed design-locked term holds $39$ to $62\ \%$ of the readout step variance. After correction, the design explains $96\ \%$ of voxel variance, motion after the design falls from $1.1$ to $0.7\ \%$ and from $2.0$ to $0.8\ \%$, the residual falls from $4.5$ to $3.3\ \%$ and from $7.4$ to $3.6\ \%$, and the residual is uncorrelated with FD ($r = -0.02, -0.13$). On resting BOLD, band-limited FD lowers the mean from $0.60$ to $0.31\ \text{mm}$ and censored frames from $46$ to $13\ \%$. On the 25-volume ASL series, the levels explain $98\ \%$ of variance after correction and motion after the design falls from $0.34$ to $0.18\ \%$.

## Discussion

On the two DWI series, $39$ to $62\ \%$ of displacement variance along the B0-immune readout axis was design-locked and $73$ to $83\ \%$ along phase encode was eddy: the misattribution of single-reference registration is a measured number. The hypothesis that small second differences should be shrunk hardest was refuted: the form that removes the most sub-knee jitter on a synthetic trajectory is rejected by the phantoms because their jitter is real motion. Limitations: the phantoms plant jitter in the trajectory, not in the images, so the claim that clinical sub-knee jitter is measurement noise has no phantom gate yet; on the 60-volume ASL series, the image-domain account attributes most variance ($69.50\ \%$ raw, $55.42\ \%$ corrected) to motion regressors, which is not interpreted here; censoring thresholds are not tied to a downstream outcome; the design-in-metric engine over-reports readout jitter on clinical DWI and is not default.

## Conclusion

On these three phantoms and five clinical series, only the re-descending Cauchy prior at weight $0.03$ reduced phantom FD error ($-3$ to $-17\ \%$) while keeping clinical B0-immune readout steps within $0.011\ \text{mm}$, with every term's variance share reported alongside the corrected series.

## Table 1

One solver, five series. Censored: frames above the antsxqc scrubbing threshold on raw FD / on the censoring series
(band-limited FD for BOLD, rigid FD otherwise). Pose shares: percent of frame-to-frame step variance on the readout
and phase-encode axes held by head / applied eddy / removed design-locked term. Image: percent of per-voxel temporal
variance explained by the design, by motion after the design, and residual, raw / corrected; r: residual-FD
correlation after correction (syntx 6.28.0 default).

| case | T | FD mean / max (mm) | censored raw / band-limited | readout: head / eddy / design (%) | PE: head / eddy / design (%) | design (%) raw / corr | motion after design (%) raw / corr | residual (%) raw / corr | r(resid, FD) | time |
|---|---|---|---|---|---|---|---|---|---|---|
| resting BOLD, TR 1.0 s | 300 | 0.60 / 2.86 | 46 % / 13 % | 100 / 0 / 0 | 100 / 0 / 0 | 0.0 / 0.0 | 32.94 / 20.78 | 63.7 / 74.9 | 0.60 | 66 s |
| ASL, control/label + M0, 25 vol | 25 | 0.51 / 2.55 | 24 % / 24 % | 97 / 0 / 6 | 102 / 0 / 0 | 97.0 / 98.3 | 0.34 / 0.18 | 2.4 / 1.2 | 0.68 | 95 s |
| ASL, control/label, 60 vol | 60 | 0.19 / 0.55 | 3 % / 3 % | 77 / 0 / 14 | 92 / 0 / 4 | 1.3 / 1.8 | 69.50 / 55.42 | 24.0 / 35.2 | 0.62 | 169 s |
| DWI multi-shell, high motion | 100 | 0.60 / 2.52 | 44 % / 44 % | 72 / 0 / 39 | 19 / 73 / 15 | 94.4 / 96.0 | 1.06 / 0.67 | 4.5 / 3.3 | -0.02 | 218 s |
| DWI multi-shell, low SNR | 99 | 0.48 / 1.76 | 36 % / 36 % | 34 / 0 / 62 | 20 / 83 / 7 | 90.5 / 95.5 | 2.04 / 0.82 | 7.4 / 3.6 | -0.13 | 229 s |

## Figure 1

Figure 1. A re-descending Cauchy prior on trajectory second differences damps sub-knee wiggles while passing slow drift and 1.5 mm jumps. (`docs/abstracts/figures/ismrm_fig1_model_and_prior.png`) (A) Forward model for one frame ($n = 1$ simulated multi-shell DWI pair, $b = 0$ and $b = 1000\ \text{s/mm}^2$): the level reference $\bar I_{\ell(k)}$ carries contrast, the design-locked offset $D_k\beta$ is removed from the reported pose, the free head pose $m_k$ is written to the rigid transform, the eddy-current affine $E_k$ and its Jacobian $1 + a_k^{\top}p$ are applied along phase encode, and the gain $\gamma_k$ absorbs scanner drift; axial slices show $b = 0$, $b = 1000\ \text{s/mm}^2$, and their difference (colorbar in signal units). (B) Penalty $\rho(u)$ (solid, $\text{mm}^2$) and pull $\rho'(u)$ (dashed, mm) on the y-axis against second-difference magnitude $u = |u_t|$ (mm, x-axis) at weight 1 in curvature units and knee threshold $\delta = 0.05\ \text{mm}$ for $n = 3$ forms (Cauchy, Charbonnier, and constant-pull sparse with knee $0.1\delta$): the Cauchy pull peaks at $\delta = 0.05\ \text{mm}$ and fades above it. (C) Recovered position (mm, y-axis) against frame index (x-axis; inset: residual jitter in mm across frames 20 to 50) for $n = 3$ penalty forms against a unit-curvature image term on a synthetic trajectory (drift plus sub-knee jitter plus one $1.5\ \text{mm}$ jump): all three forms preserve the drift and the $1.5\ \text{mm}$ jump, while only the Cauchy pull fades above $\delta$.

## Figure 2

Figure 2. Only the re-descending Cauchy prior at weight 0.03 passes both the three-phantom accuracy gate and the five-series clinical readout-step gate. (`docs/abstracts/figures/ismrm_fig2_gate.png`) (A) Framewise displacement (FD) error change (%, y-axis) relative to no prior across candidate settings on $n = 3$ phantoms with known trajectories (ASL drift with jitter, DWI with planted eddy shift, slow-flow encoding states) at threshold $+10\ \%$ maximum allowed FD error loss: only the Cauchy form at 0.03 and 0.05 loses less than $10\ \%$ on every phantom. (B) Head translation along the B0-immune readout axis (mm, y-axis) against frame index (x-axis, $T = 100$ frames) on $n = 1$ high-motion multi-shell DWI series under no prior, Cauchy 0.03, and Cauchy 0.05 (gate threshold $0.02\ \text{mm}$ step change): Cauchy 0.05 over-smooths real jitter (step median absolute deviation $0.081$ to $0.038\ \text{mm}$, change $-0.043\ \text{mm}$) whereas Cauchy 0.03 preserves it ($0.079\ \text{mm}$). (C) Readout step change ($\mu\text{m}$, y-axis) against mean FD change (%, x-axis) for $n = 5$ clinical series; shaded bands mark the allowed thresholds ($\pm 20\ \mu\text{m}$ readout step change, $+2\ \%$ FD rise): every point for Cauchy 0.03 lies inside both bands.

## Figure 3

Figure 3. Decomposition of frame-to-frame steps and voxel variance separates head motion from eddy-current displacements and design-locked contrast across five clinical series. (`docs/reports/ismrm_cases_6_28_2026-10-28_figure.png`) One row per clinical series ($n = 5$: resting BOLD with $T = 300$, 25-volume ASL, 60-volume ASL, high-motion DWI with $T = 100$, low-SNR DWI with $T = 99$). Left and centre-left panels plot displacement (mm, y-axis) against frame index (x-axis) along the readout and phase-encode axes for free head pose $m_k$ (solid), applied eddy shift (dashed), and removed design-locked offset $D_k\beta$ (dotted). Centre-right panels plot raw FD and band-limited FD (mm, y-axis) against frame index (x-axis) relative to the antsxqc scrubbing threshold. Right panels plot per-voxel temporal variance explained (%, log-scale axis) per term before and after correction: correction lowers motion-after-design variance across all five series and leaves DWI residuals uncorrelated with FD ($r = -0.02$ and $-0.13$).

## References

[1] Freire L, Mangin JF. NeuroImage 2001. [2] Power JD et al. PNAS 2019. [3] Andersson JLR, Sotiropoulos SN.
NeuroImage 2016. [4] Fair DA et al. NeuroImage 2020. Also (to be verified): Orchard J et al. IEEE TMI 2003; Huizinga W
et al. MedIA 2016.
