# ANTsX Implementation Standards (Canonical 5-Rule Specification — 2026-10-05)

Every repository in the ANTsX Python ecosystem (`antsxqc`, `syntx`, `antsxstructural`, `antsxdwi`, `antsxfunctional`, `antsxslowflow`, `antsxmm`, `antsxquality`) is audited against these five rules via `tests/test_implementation_standards.py`:

1. **Rule 1 — Canonical Data & Weight Fetching (`antsxdata` only)**
   All templates, atlases, priors, and pretrained weights must be fetched through `antsxdata`. Direct `urllib.request.urlretrieve`, `requests.get`, `huggingface_hub`, `antspynet.get_antsxnet_data`, or `antspyt1w.get_data` calls are forbidden.
2. **Rule 2 — Registration & Motion Correction (`syntx` only)**
   All pairwise registration (`syntx.robust_affine`, `syntx.syn`, `syntx.tvf`, `syntx.syngs`) and 4-D time-series motion correction (`syntx.motion_correction`) must route through `syntx`. Raw `ants.registration` and `ants.motion_correction` calls are forbidden outside documented benchmark baselines in `syntx`.
3. **Rule 3 — Deep Learning & Bias/Denoising Preprocessing (`antstorch` only)**
   All deep network inference, N4 bias field correction (`antstorch.n4_bias_field_correction`), and Rician NLM denoising (`antstorch.denoise_image`) must route through `antstorch`. Legacy `antspynet`, `antspyt1w`, `tensorflow`, `keras`, `SimpleITK`, `nilearn`, and `nibabel` imports are forbidden.
4. **Rule 4 — Physical-Space Invariance**
   Every spatial computation, derivative, Jacobian determinant, and orthogonal slice visualization must respect the image `origin`, `spacing`, and `direction` cosine matrix. Unjustified `reorient_image2` calls and raw unoriented NumPy slice plots are forbidden.
5. **Rule 5 — Single Quality-Control Judge (`antsxqc`)**
   All QC metric specifications, hard physical/execution validity limits, cohort-relative MAD grading, and status vocabulary (`"pass" | "warn" | "fail" | "not_assessable"`) belong exclusively to `antsxqc`. Producer packages emit raw numeric measurements only and must pass `python -m antsxqc.lint` with zero `.antsxqc-allow` exceptions.
