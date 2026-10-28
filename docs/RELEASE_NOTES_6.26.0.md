# syntx 6.26.0 (2026-10-28)

- Head prior inside every L-BFGS and Newton stage (`surround_prior_hessian`, analytic, autograd-checked), never in Adam.
- `band_limited_fd` / `attach_band_limited_fd`: FD with the near-Nyquist physiological alternation removed from the pose
  series, attached to every motion result (`fd_bandlimited`, `fd_band`); the censoring series for BOLD and ASL
  (`docs/MOTION_BOLD.md`). The full poses stay the correction.
- `motion_correct_design(engine="projected")`: the acquisition design profiled out of the metric (available, not default;
  `docs/MOTION_DESIGN.md`). Prior off by default pending its re-scaling to the image term's units.
- Plan: `docs/MOTION_JOINT_V2_AND_CENSORING_PLAN.md`; evidence: `docs/reports/pilot_bold_resting_joint_2026-10-27.html`,
  `docs/reports/motion_cohort_gate_projected_2026-10-27.html`.
- Requires `antsxqc>=0.4.0`.
