# syntx 6.5.1

Device pass-down fixes found while migrating the sibling packages to `syntx.device` (see `docs/DEVICE_POLICY.md`).

## Fixed
* `motion_correct_grouped`: the one cross-contrast `robust_affine` between the group means ignored a `device=` given to the call (only the per-group `motion_correction` honoured it); it is now forwarded.
* `syntx.cli` and `greedy`: the initial `robust_affine` ignored the resolved device (it re-resolved on its own); the CLI's / greedy's resolved device is now passed.
