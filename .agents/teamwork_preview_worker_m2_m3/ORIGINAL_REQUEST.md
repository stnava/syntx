## 2026-07-14T21:23:47Z

Please implement and verify Milestones 2, 3, and 4 of the syntx 2D Parity & Deep Feature Triggering project.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Specifically, perform the following tasks:
1. Baseline 2D Parity Tuning (Milestone 2):
   - Evaluate `ants.registration(..., 'SyN')` and `syntx.syn` under both PyTorch and JAX backends on phantoms ('r16', 'r27'), ('r16', 'r64'), ('r27', 'r64') using raw LNCC similarity.
   - Perform a systematic grid search/tuning over `levels`, `reg_iterations`, `affine_iterations`, `grad_step`, `flow_sigma`, and `total_sigma`.
   - Identify the configuration that achieves mean DICE score parity (within 1%) with `ants.registration` across these phantoms.
   - Update default parameters in the high-level `registration` interface and `SyNTo` methods in both `src/syntx/syn.py` and `src/syntx/syn_jax.py` to match the optimal found parameters.

2. Deep Feature Degeneracy Trigger (Milestone 3):
   - Implement the triggering mechanism in `SyNTo.fit` for both PyTorch (`src/syntx/syn.py`) and JAX (`src/syntx/syn_jax.py`) backends.
   - The trigger should dynamically deactivate deep feature similarity losses (ResNet-10, VGG19) and fall back to raw intensity LNCC similarity if the resolution shape at the current scale level is degenerate (min spatial size < 32).
   - Evaluate the trigger's performance: verify it improves DICE score and reduces topological folding (folding rate) compared to the always-on deep features baseline.

3. Reporting and Visualization (Milestone 4):
   - Generate `docs/parity_report.html` comparing the baseline ANTs, tuned `syntx` (LNCC), always-on deep features, and trigger-activated deep features.
   - Per GEMINI.md, the HTML report MUST include rich spatial images illustrating:
     - Edge and/or region overlap
     - Deformed grids
     - Jacobian determinant maps
     - Side-by-side warped/target images
   - Compile summary tables comparing DICE scores, execution runtimes, and folding rates.

4. Test Suite and Compliance Verification:
   - Run the entire test suite (`pytest --runslow`) and verify all tests (including the 78+ unit tests) pass successfully.
   - Ensure the Single Interpolation Policy is maintained (no pre-warping of images/segmentations prior to optimization, composition of all transforms applied in a single step).
   - Ensure VGG 2D mode is not used for cortical labels; ensure VGG 3D LNCC Layer 4 compliance.

Document all details of your modifications, evaluations, and test results in `/Users/stnava/code/syntx/.agents/teamwork_preview_worker_m2_m3/changes.md` and deliver a handoff.md.

Identity:
- Role: Parity & Feature Implementation Worker
- Working directory: /Users/stnava/code/syntx/.agents/teamwork_preview_worker_m2_m3
