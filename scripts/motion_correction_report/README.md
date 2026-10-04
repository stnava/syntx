# Reproducing docs/ADAPTIVE_MOTION_CORRECTION_AND_RECOVERY.html

Real-data scripts behind that report (2026-10-04). Not part of the test suite -- these
read real local paths (SOCOM, PPMI, ds002898 FDG-PET) that only exist on the machine
this was built on; kept here for reproducibility/reference, not CI.

Run order:
1. `run_bold.py` -> `bold_results.json`
2. `run_dwi.py` -> `dwi_results.json` + `dwi_ref_{blended,grouped}.nii.gz`
3. `run_pet.py` -> `pet_results.json` + `pet_ref_{naive,subset}.nii.gz`
4. `make_figures.py` -> `figs_partial.json` (BOLD + DWI figures)
5. `make_pet_figures.py` -> `pet_figs.json`
6. `build_report.py` -> `report.html`

All scripts write into (and some read from) a flat working directory -- run them from
one, e.g. `/tmp/motion_report/`, with all 6 files copied there, or adjust the hardcoded
paths at the top of each.
