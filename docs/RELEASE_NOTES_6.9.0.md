# syntx 6.9.0

DWI motion correction redesigned (breaking).
* `motion_correct_grouped(image, groups, ...)` takes k group labels (`shell_groups(bvals)` gives b0 + one group per
  shell; a boolean anchor mask still works for two groups). Chained adjacent-shell cross registration, one resample
  from the original through the composed chain, series Jenkinson FD. Result keys: motion_corrected, fd, fwdtransforms
  (per-frame chains), group_results, group_indices, group_means, cross_registrations, preset.
* Removed: `bvals`/`bvecs` parameters, `dwi_direction_targets`, per-frame reference lists, `max_batch_frames`.
* `syntx.motion_presets`: `DWI_MOTION_PRESET`, `BOLD_MOTION_PRESET`, `get_motion_preset`.
* `motion_batched` uses the central losses (`syntx.core.losses`); `medium` schedule added.
* New export `syntx.shell_groups`.
Evidence: docs/MOTION_DWI.md, docs/MOTION_BOLD.md. Old results from the biased estimator are stale.
