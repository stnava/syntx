# Removing encoding-locked systematic noise from motion estimates

Modalities that cycle an encoding (ASL tag/control, slow-flow `[b0, +X, +Y, +Z]`, DWI gradient directions) or that
breathe (BOLD) add pose that is not head motion. It comes in two physically different kinds:

| kind | what happened to the image | correct handling | syntx mechanism |
|---|---|---|---|
| **Registration bias** (contrast-driven pseudomotion) | nothing; the frame's contrast differs from the reference and moves the metric optimum | remove from the written transforms; never apply | contrast-matched reference; surround-Tukey projection on the encoding basis (`encoding_covariates`, `bias_covariates`) |
| **Encoding-locked displacement** (eddy currents, respiratory B0) | really displaced along phase encode | apply to the data; exclude from FD and the reported motion | eddy PE affine (`pe_affine_covariates`, tied to the gradient, applied); `debias_pe_pseudomotion` (BOLD, reported motion only) |

In DWI the two are separated by physics: the image contrast (and so the registration bias) is **even** in the
gradient `g` (l = 2 terms), the eddy displacement is **odd** (linear in `g`).

## The estimator

`motion_batched._project_encoding_surround_tukey_np` regresses the second-order surround difference of the pose,
`u_t = Y_t - (Y_{t-1} + Y_{t+1}) / 2`, on the surround difference of the encoding basis with Tukey weights. The mean
pose and linear drift are in its nullspace (quadratic drift becomes a constant that is removed), and real jumps produce
isolated large residuals that the biweight downweights. Head motion that correlates with the encoding by chance over a
short series is therefore NOT reassigned. That reassignment happened with least squares on the poses: 0.12 mm (eddy
attribution) and 0.22 mm (DWI l = 2 bias regression) of head error on the simulated series. The same estimator now
sits behind:

- `motion_correction(..., encoding_covariates=E)`: in-optimizer encoding offset `E_psi . B_win` (overlapping Hann
  windows). After every optimisation stage the offset is folded back into the total pose and re-estimated robustly, so
  gradient steps cannot leave part of a real jump in `B_win`. Written transforms = head motion only. Integer state
  labels with more than two states (`[0, 1, 2, 3]`) are categorical (one indicator per state), never a numeric ramp.
- `bias_covariates` (one window, the group's mean pose kept).
- the eddy translation attribution `C0` along phase encode (one window, covariates as given).
- `motion_correct_grouped(..., cross_group_debias=True)` (opt-in): per-group offset of the composed chain.

The in-loop Charbonnier surround penalty on the head trajectory (`surround_weight`) is wired but OFF by default: on the
slow-flow phantom it made the head estimate worse as the weight grew (tz error 0.003 / 0.012 / 0.019 mm at weights
0 / 0.02 / 0.05).

## Recipes (measured)

`benchmarks/motion_encoding_bias_simulation.py`, `syntx.motion_simulation` phantoms (36 x 40 x 32 at 2.5 mm, gray matter
shell, white matter with a fibre field, ventricles, texture), true head motion = drift + one 1.2 mm / 0.8 deg jump +
jitter, 3 seeds. Errors are against the true head trajectory after demeaning. `floor` = same motion, no systematic
noise (the registration's own precision on this phantom).

| modality | systematic noise | baseline (mean reference) trans mm / rot deg / FD err mm | recommended | trans / rot / FD err | floor |
|---|---|---|---|---|---|
| ASL | 10 % asymmetric perfusion contrast | 0.150 / 0.28 / 0.76 | `preset="asl", encoding_covariates=s_tc` | 0.017 / 0.19 / 0.19 | 0.016 / 0.17 / 0.13 |
| | | | + state-matched references (`build_bilateral_window_references`) | 0.017 / 0.16 / 0.13 | |
| slow flow | flow-encoding contrast + 0.4 mm state-locked eddy shift | 0.248 / 0.36 / 0.89 | `preset="slowflow", encoding_covariates=state_labels` | 0.031 / 0.21 / 0.22 | 0.014 / 0.22 / 0.12 |
| DWI (b1000, 20 dirs) | direction contrast + 0.6 mm eddy (linear in g) | single ref 0.385 / 0.62 / 1.27 | `motion_correct_grouped(preset="dwi", pe_affine_covariates=g, pe_axis)` | 0.046 / 0.64 / 0.88 | 0.013 / 0.18 / 0.10 |
| | | | + `bias_covariates=l2` (antsxdwi) | 0.104 / 0.63 / 0.83 | |
| BOLD | 0.4 mm respiratory shift along PE (+ real nodding) | 0.287 / 0.18 / 0.61 | `debias_pe_pseudomotion=True, pe_axis=...` (reported motion only) | 0.078 / 0.18 / 0.19; nodding and rotations untouched | 0.020 / 0.18 / 0.13 |

Notes and limits:

- **Contrast-matched references** remove contrast bias at its source and assume nothing about head motion. For
  DISCRETE states they must be the inclusive state mean (one reference per state, shared by its frames: a common
  space); leave-one-out references differ per frame and cost 0.03 -> 0.12 mm (slow flow) and 0.017 -> 0.060 mm (ASL).
- **DWI direction-dependent rotation bias is the open problem.** It is deterministic (static head, no noise: 0.48-0.55
  deg, 50-70 % explained by linear + l = 2 gradient terms) but no tested strategy removes it reliably: a kernel
  contrast-matched reference (`encoding_reference_predictor`) helps only when the frame is included in its own
  reference (0.56 -> 0.40 deg), where it is largely registered to its own previous resampling and some frames stop
  converging; leave-one-out gives 0.56 -> 0.51 deg. A spherical-harmonic signal prediction per direction (fit on the
  other directions) is the next candidate. TODO(human): decide whether to pursue it.
- **DWI `bias_covariates` (l = 2)** trades translation for rotation on 20 directions (+0.06 mm / -0.05 deg): 30
  coefficients from about 22 surround rows. antsxdwi keeps it (validated on the still phantom, 0.27 -> 0.18 mm).
- **Encoding projection alone with no encoding-locked noise** costs variance (slow flow, contrast only: 0.025 ->
  0.046 mm), so pass `encoding_covariates` only where an encoding cycle exists.
- **BOLD**: the separation is opt-in because it relies on the PE axis being the only B0-sensitive direction. Slice
  translation and pitch (real nodding) are never touched.
- **Short series: the identifiability limit, not under-removal.** At T = 16 (ASL) the head error is 0.026 mm vs 0.015 mm
  floor (3 seeds); 0.017 mm at T = 24. The excess is locked to the encoding, but it is the part of the REAL frame-to-frame
  jitter that happens to align with `s_t` (no jitter: 0.020 -> 0.002 mm; still head: 0.000), removed with the bias
  because nothing can tell them apart. Expected size jitter x sqrt(3 / T) (0.017 mm at T = 16, 0.04 mm jitter). Freezing
  the shared encoding parameters once half the frames converge plays no part (never freezing: identical errors); the
  written transforms come from the robust re-projection after every stage.
- **A lone frame cannot be an encoding state.** Its offset is the signature of a one-frame head spike, which the robust
  projection keeps as motion. ASL M0 volumes mixed into a control/label series are a registration GROUP
  (`motion_correct_grouped`, control/label = group 0 with `encoding_covariates=s_tc`), not a categorical state (head
  phantom, M0 at the start or in the middle: 0.16-0.27 mm FD at the M0 step as a group, 0.8-2.0 mm as a state).
- **Interleaved groups and FD.** Multi-shell DWI switches shell on almost every volume; a constant per-group pose offset
  left by the cross-group registration is a step at every switch. `motion_correct_grouped` reports motion and FD without
  it (`group_offset_fd`: offsets, `fd_with_group_offsets`) when every non-anchor group has >= 4 active surround rows;
  blocked layouts are left alone (their group offset is the head motion between blocks). Transforms are unchanged unless
  `cross_group_debias=True`. Real 3-shell DWI (FPA sub-BLAST079 AP, 98 switches / 99 steps): offsets 0.16-0.17 mm /
  0.08-0.09 deg per shell, mean FD 0.173 -> 0.163 mm; the rest of its FD is per-direction variation within a shell.
- The phantom's rotation floor (about 0.18 deg) limits what can be resolved in rotation. Real brains have more edge
  structure; the relative ordering of the strategies is the evidence, not the absolute numbers.
