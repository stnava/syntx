# syntx 6.8.2

Refactor: the batched registration losses now live in `syntx.core.losses`, the one place for similarity losses (rule: no local
one-off metrics).

- **Moved** from `motion_batched` into `core/losses`: `batched_mattes_mi_from_weights` (`mutual_information.py`, per-frame negative
  MI for (B, S, nbins) frames against a shared or per-frame reference) and `pearson_correlation_loss` (`cross_correlation.py`,
  batched negative global Pearson). Both are exported from `syntx.core.losses`.
- **Removed** the local dense `_batched_parzen_weights`: `motion_batched` now calls the central `parzen_weights` (same cubic B-spline
  weights, four taps per sample instead of all bins).
- **Numerically equivalent, not bit-identical**: migrated registrations differ from the previous code by up to about 1e-3 in the
  transform parameters (rounding differences in the Parzen evaluation amplified by the optimizer), about 50x below the
  accuracy scale of interest. Real-data check (SRPBS BOLD, `preset="bold"`): injected-motion error 0.045 mm before and after;
  quiet-run Power FD 0.084 -> 0.087 mm.
- **Faster as a side effect**: the central Parzen weights are cheaper; the same quiet 240-volume BOLD run takes 6.7 s instead of
  16.9 s. DWI timing not re-measured.
- Tests: `tests/test_core_losses_batched.py` (batched MI equals the single-pair MI per frame, Pearson equals numpy, central
  Parzen equals the dense cubic B-spline); 156 passed with the existing central-loss, parity/accuracy and motion suites.
