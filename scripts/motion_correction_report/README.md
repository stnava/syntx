# Reproducing docs/ADAPTIVE_MOTION_CORRECTION_AND_RECOVERY.html

Real-data scripts behind that report (2026-10-04). Not part of the test suite -- these
read real local paths (SOCOM, PPMI, ds002898 FDG-PET) that only exist on the machine
this was built on; kept here for reproducibility/reference, not CI.

Run order:
1. `run_bold.py` -> `bold_results.json`
2. `run_dwi.py` -> `dwi_results.json` + `dwi_ref_{blended,grouped}.nii.gz` (note: this
   script's own `motion_correct_grouped` call predates the two-mean fix -- re-run the
   "TWO-MEAN approach" snippet described in the module docstring of
   `syntx.motion_reference` for the current, correct numbers, or see the inline snippet
   used to produce `dwi_results.json`'s `two_mean_*` keys)
3. `run_pet.py` -> `pet_results.json` + `pet_ref_{naive,subset}.nii.gz`
4. `make_figures.py` -> `figs_partial.json` (BOLD + DWI summary figures)
5. `make_pet_figures.py` -> `pet_figs.json`
6. `make_checkerboard_evidence.py` + `make_checkerboard_zoom.py` -> real before/after
   frame-level evidence (the "does this actually look different" figures -- the ones
   that matter most for judging whether the method really works, not just the FD
   summary numbers)
7. `make_edge_overlays.py` -> `dwi_cross_reg_edge_overlay.png` + `pet_edge_overlay.png`
   (edge overlays, not checkerboards, for the two CROSS-CONTRAST comparisons --
   checkerboarding two different contrasts confounds "looks different" with
   "misaligned"; same-contrast BOLD/ASL checkerboards from step 6 remain valid)
8. `make_sift3d_figure.py` -> `pet_sift3d_comparison.png` (SIFT3D vs Mattes-MI on raw
   frames, native PET resolution)
9. `make_temporal_smoothing_figures.py` -> the Gaussian-temporal-averaging comparison
   data (`pet_sift3d_smoothed_comparison.json`, `pet_mattes_fd_on_smoothed.json`) and
   the raw-vs-smoothed visual; the three comparison plots
   (`pet_sift3d_smoothing_matches.png`, `pet_sift3d_smoothing_stability.png`,
   `pet_smoothing_fixes_both_methods.png`) are built from that JSON with the plotting
   code inline in this script's git history (kept short here; re-derive the three
   `matplotlib` panels from the JSON keys `raw`/`smoothed` x `n_matches`/`n_inliers`/
   `translation_mm` per frame).
10. `build_report_v2.py` -> `report_v2.html` (the current report; `build_report.py`, an
    earlier draft with a session-narrative framing instead of general documentation, was
    removed)

All scripts write into (and some read from) a flat working directory -- run them from
one, e.g. `/tmp/motion_report/`, with all files copied there, or adjust the hardcoded
paths at the top of each.
