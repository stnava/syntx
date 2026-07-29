## 2026-07-27T13:55:22Z
<USER_REQUEST>
You are teamwork_preview_worker for Milestone 2 of the TVF velocity gradient smoothing fix and PyTorch/JAX parity task.
Your working directory is /Users/stnava/code/syntx/.agents/teamwork_preview_worker_m2_1.

DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Refer to the Explorer handoff report at /Users/stnava/code/syntx/.agents/teamwork_preview_explorer_m1_1/handoff.md and /Users/stnava/code/syntx/GEMINI.md.

Tasks:
1. Fix PyTorch `TVFModel.fit()` in `src/syntx/tvf.py` (lines 386–405):
   Remove the channel-first permutation (`grad.permute(...)`). Pass `self.velocity.grad[t]` directly as channel-last tensor `(1, *velocity_shape, dim)` to `separable_gaussian_filter(self.velocity.grad[t], sigma=sigma_voxel, spacing=None, sigma_mode='voxel')`.
2. Implement JAX `TVFModelJAX` in `src/syntx/tvf_jax.py`:
   - Mirror PyTorch `TVFModel` algorithmically.
   - Use `integrate_time_varying_velocity_field_jax` for time integration.
   - Use `local_ncc_loss_nd_jax` and `jax_grid_sample` for midpoint-symmetric LNCC loss.
   - Use `jax.grad` and `separable_gaussian_filter_jax` for multi-res `fit()` fluid regularization gradient smoothing.
3. Export `TVFModel` and `TVFModelJAX` in `src/syntx/__init__.py`.
4. Expand `tests/test_tvf.py`:
   - Test `fit()` on synthetic 2D/3D images.
   - Test isotropic impulse smoothing (verify $V_z$ impulse does not leak into $V_y, V_x$).
   - Test PyTorch <=> JAX parity (verify forward loss and displacement fields match within tolerance $\le 0.001$).
5. Run `pytest tests/test_tvf.py` and document all test outcomes and commands in your handoff report.
6. Write your handoff report to /Users/stnava/code/syntx/.agents/teamwork_preview_worker_m2_1/handoff.md and send a summary back via send_message.
</USER_REQUEST>

## 2026-07-27T13:59:19Z
**Context**: Compliance check with GEMINI.md Rule 8.
**Content**: Per GEMINI.md Rule 8:
- Intensity Images: Must use `padding_mode='zeros'` so out-of-bounds coordinates sample 0.0 intensity without creating artificial edge-color stripes that pull background grid vectors.
- Displacement Fields: Must use `padding_mode='border'` during fixed-point inversion and algebraic composition.

Currently, `TVFModel.forward()` in PyTorch defaulted to `padding_mode='border'` when warping intensity images, and `TVFModelJAX.forward()` was set to `'border'` to match it.
Both PyTorch (`tvf.py`) and JAX (`tvf_jax.py`) MUST explicitly set `padding_mode='zeros'` when warping intensity images in `forward()`.

**Action**: Please update `TVFModel` in `src/syntx/tvf.py` and `TVFModelJAX` in `src/syntx/tvf_jax.py` to explicitly use `padding_mode='zeros'` for intensity image warping, re-run `pytest tests/test_tvf.py` to confirm PyTorch <=> JAX parity, and update handoff.md.
