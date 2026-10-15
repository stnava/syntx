# syntx 6.11.0

Host-device synchronisation in the SyN epoch loop is reduced from about 16 per epoch to about 2,
full-field clones from about 11 per epoch to 1, with registration output unchanged to float32
rounding. Measured on Apple Silicon MPS with the default `SyNOnly` configuration (counting every
`.item()`, `bool(tensor)`, `.tolist()`, `.cpu()` and `.clone()` on a device tensor).

* `syntx.core.inverse.update_inverse_field_nd_anderson`: every batch element is now solved
  independently (per-element clip threshold, least-squares system, stopping test) and an element
  that has met the thresholds is frozen while the others continue. A batch of fields therefore
  returns what separate calls return (float32 rounding), and the result no longer depends on when
  the stopping test runs. New `check_after` defers the first stopping test to a given iteration and
  new `stats` (dict) reports `'steps'` and `'converged'`; both pass through
  `update_inverse_field_nd`. The stopping test transfers its flags in one `.tolist()`. The small
  Anderson system is solved by a Jacobi-equilibrated elementwise Gauss-Jordan (`_solve_small_spd`)
  instead of `torch.linalg.solve`, which pays a fixed per-call cost of about 1.4 ms on MPS for a
  4x4 system; the Gram matrix keeps the reference `torch.dot` reduction order (3.8e-7 against the
  reference on the adversarial ripple field).
* `syntx.syn.SyNTo.fit`: the two half-warp inverses (and the two `project_inverse` projections)
  run as one batched call each, with the iteration count the previous epoch needed passed as
  `check_after`, so a converged epoch pays one synchronisation per batched call. The two
  `if tensor > 1e-12` branches of the CFL step are `torch.where`. The loss, the best loss and the
  best-warp snapshot stay on the device (masked copies into preallocated buffers); the host reads
  the loss history every `loss_sync_every` epochs (new fit kwarg, default 5) and on the last epoch
  in one transfer, so the convergence and collapse tests lag by at most `loss_sync_every - 1`
  epochs. `syn_losses` is converted with one transfer at the end of the fit.
* Parity: `SyNOnly` on MNI at 3 mm (30 epochs, 2 levels) against v6.10.0, same inputs: warp
  max |diff| 9.5e-6 mm on CPU and 3.8e-6 mm on MPS (field max 11.1 mm), loss history identical to
  6 decimals, reruns bitwise reproducible.
* Tests: `test_anderson_check_interval_behavior` now asserts that the result is independent of
  the check cadence and that `stats` reports the steps; new `test_anderson_batch_matches_single_calls`;
  the sync-count test counts `.tolist()` as a synchronisation.
* Timing is not reported in this release: the measurement host had other GPU load during the
  final runs. `docs/plans/GPU_SYNC_OPTIMIZATION_PLAN.md` holds the pre-change profile and the
  measurement protocol.
