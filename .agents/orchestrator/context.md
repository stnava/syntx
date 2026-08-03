# Mission Context & Project Rules

## Mission Statement
Systematically investigate, debug, and optimize `syntx.tvf` to achieve peak accuracy parity with `syntx.syn` (>=0.8800 Cortical Label 3 Dice under `ants.label_overlap_measures`) with 100% Diffeomorphic Safety (0.0000% Folding, min det(J) > 0.0), without regressing any pre-existing project utilities or unit tests.

## Key Project Rules & Guardrails (from GEMINI.md)
1. **Single Interpolation Policy**:
   - No pre-warping images or intermediate segmentations prior to optimization.
   - Compose multiple transforms and apply directly to native-space images in a single `ants.apply_transforms` call.
2. **Similarity Metric & Variance Floor**:
   - LNCC Variance Floor: `var_safe = max(var, 10^-6)` in PyTorch and JAX to prevent analytical autograd derivative spikes.
   - Cauchy-Schwarz clamping: `clamp(cc, -1.0, 1.0)`.
3. **Physical Spacing & ITK CFL Multiplier**:
   - ITK `gradientStep` is in voxel units — when normalizing gradient field in physical space, multiply step size by physical spacing.
   - Vector fields: `padding_mode='border'` during ODE trajectory integration, fixed-point inversion, and algebraic composition to avoid zero-clamping boundary velocity vectors.
4. **TVF Model & Optimization Guardrails**:
   - Pyramid-proportional velocity grids: `vel_shape = max(8, max_vel_shape // level)`.
   - LARS optimizer for time-varying velocity fields: scale-invariant trust ratios.
   - Euler ODE solver defaults: $T=4$ keyframes, 1 substep per interval.
   - Antisymmetric velocity projection: $e_0 = \delta_l + \delta_r$, $\delta_l \leftarrow \delta_l - 0.5 e_0$, $\delta_r \leftarrow \delta_r - 0.5 e_0$.
5. **Zero Tolerance for Cheating**:
   - All implementations must be genuine. Forensic auditor will independently verify. No hardcoding or facade implementations.
