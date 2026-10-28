# Plan: design-matrix review follow-ups and the abstract cut (2026-10-28, late)

## Context

An external review of the design matrices used by `syntx.motion_correct_design` (DWI, ASL, slow flow, BOLD, custom) was
checked against the code. Its description of the solver is accurate: the DWI design splits into contrast-gauge columns
(level indicators, the five l = 2 terms per shell, gated at `FRAMES_PER_BETA_COLUMN = 6` frames per column, so 30 frames per
shell), removed from the pose and never applied, and the per-shell eddy covariates (unit gradients), applied with their
Jacobian and excluded from FD; the odd-in-g projector keeps the two apart along phase encode. ASL links control and label
by identity and drops a lone M0's indicator while registering it explicitly; slow flow uses one-hot level indicators in the
joint engine; BOLD carries no design column, a per-frame gain and the Cauchy prior (0.05 one-level since 6.29.0), with the
physiological band removed only from the censoring FD. The review's evidence citations (design decision, projected-engine
leak, parity column) match the reports.

Three of its findings are real and worth fixing; one is a default worth changing; the rest need no change.

1. **`motion_correct_design` defaults to `engine="dispatch"`** (`motion_design.py:752`). Every consumer passes
   `engine="joint"` explicitly, so pipelines are unaffected, but a direct caller who omits `engine` gets the legacy
   per-level path without the joint pass or the prior, and the "recommended call" the author was given this evening
   assumed `engine="joint"`. Change the default to `"joint"`; the dispatch tests and the stage-0 probes that depend on the
   old default pass `engine="dispatch"` explicitly.
2. **Custom designs lose their `encoding` columns in the joint engine.** `joint_design_columns` and the one-level path
   of `_joint` never read `design.encoding`; `_unified` and `_dispatch` do. Fix: append the caller's `encoding` columns
   (T, K) in `joint_design_columns` for `custom` designs, each kept only with `FRAMES_PER_BETA_COLUMN` frames per
   column (recorded in `beta_dropped`), and pass `encoding_covariates` on the one-level path when present.
3. **Slow-flow state labels enter the legacy dispatch path as one linear regressor.** `_design_encoding_states` stores the
   integer labels as `encoding`; for K >= 3 states `_dispatch` hands that ramp to `motion_correction(preset="slowflow")`.
   The joint engine ignores `encoding` for this design (it uses the levels), so only the legacy path is wrong. Fix:
   store the one-hot indicators of the non-anchor states as `encoding` (T, K - 1); for K = 2 this equals today's column
   up to sign, so antsxslowflow's phantom test is the regression check.
4. **No change:** the eddy covariates are wired only for `shells` (slow-flow eddy is not modelled; a documented limit);
   the 30-frame l = 2 gate is by design and reported in `design.beta_dropped`; the windowed l = 2 fit falls back to the
   global Tukey estimate on short series as intended; the 60-volume ASL image-domain account is a property of the
   decomposition, not of the solver.

Separately, the ISMRM abstract committed in 6.29.0 is over the body limit (about 1540 words against 850; a "Novelty
Assessment" section and expanded Methods and Results were added after the reviewed 846-word build). The reviewed text
is not in git. The abstract must be cut back to the limit, re-verified and re-reviewed before submission.

## Steps (one commit each; targeted tests only, per the author's instruction)

1. **Engine default** (`src/syntx/motion_design.py`): `motion_correct_design(..., engine="joint")`; docstring and the
   module docstring's menu text updated. Tests: `tests/test_motion_design.py` lines 51 to 74 (dispatch identity tests)
   and `tests/test_stage0_challenger_blast_probes.py` calls get `engine="dispatch"` explicitly; a new test asserts the
   default result carries `design["engine"] == "joint"` on a 6-frame BOLD phantom.
2. **Custom encoding in the joint engine** (`joint_design_columns`, `_joint` one-level path): columns appended with the
   per-column frame gate; test with a 24-frame one-level series and a planted encoding-locked pose offset recovered into
   `encoding_pose_bias_mm_rad` and removed from `translations_mm`, and a two-level custom design where the extra column
   appears in `design["columns"]`.
3. **One-hot encoding states** (`_design_encoding_states`): `encoding` = indicators of states 1..K-1; `detail["n_states"]`
   unchanged; test that K = 3 labels give two columns and that `_dispatch` receives them; antsxslowflow's
   `tests/test_motion_slowflow_phantom.py` rerun on the released syntx.
4. **Abstract cut** (DROPPED by the author's instruction on 2026-10-28: the abstract is not to be changed)
   ~~ (`docs/abstracts/ISMRM_2027_motion_solver.tex`): remove "Novelty Assessment"; return Methods to about
   330 words and Results to about 220 by restoring the reviewed wording (the trims recorded in this session: one sentence
   per model term, the gate in one paragraph, the account in one paragraph, the task-fMRI session in one sentence);
   keep the visual-abstract figure if the ISMRM figure cap (five) is respected, else drop it; rebuild with pdflatex,
   regenerate the markdown, re-run the number check against the JSON sources, and send the final build to an independent
   reviewer with the same brief as before. Then commit as a docs change.
5. **Release** syntx 6.30.0 (the engine default is a behaviour change for direct callers: minor) with pins bumped in the
   four consumers as patch releases, per the release protocol.

## Verification

- `pytest tests/test_motion_design.py tests/test_motion_reference.py tests/test_stage0_challenger_blast_probes.py
  tests/test_transforms.py -q`; `python -m antsxqc.lint src/syntx` 0; antsxslowflow phantom test and antsxfunctional
  motion tests on the released syntx.
- Abstract: synopsis <= 100, body <= 850 (equations removed), number check all OK, reviewer reports no discrepancy.
