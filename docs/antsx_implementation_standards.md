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
   
   **Unified Vector Coordinate Frame Sub-Rules (F1–F6):**
   - **F1 Storage.** Every saved vector or tensor pixel (velocity, V1, fixel peaks, FOD SH, harmonic axes) has its components along the image axes of the image that holds it. Physical direction is `v_lps = D · v_img`. This is ANTs/ANTsPyMM behaviour, so `ants.apply_transforms(imagetype=1/2)` and `syntx.apply_transforms` reorient these files correctly with no flags.
     - **ITK exception:** displacement fields and warps (`*Warp.nii.gz`, `*_field.nii.gz`) keep their components in physical LPS mm, never in `image_axes`.
   - **F2 Gradient tables.**
     - On disk, bvecs follow FSL/BIDS.
     - On read, convert once to the image-axis frame: `g_img = F(D) · g_bvec`, where `F = diag(-1,1,1)` if `det(D) > 0`, else `I`.
     - On write, apply `g_bvec = F(D_out) · g_img`.
     - Nothing else touches handedness.
   - **F3 Reorientation.** Vectors, tensors and gradient tables move between spaces only through syntx, using `g_fix = D_fixᵀ · R⁻¹ · D_mov · g_mov`. `R` is the proper polar rotation (det = +1) in physical LPS of the composed linear part, and nonzero gradients are renormalized to unit length.
   - **F4 Physical directions.** Physical LPS directions are produced only at the boundaries that need them, through `syntx.frames.image_to_lps` / `lps_to_image`:
     - tractography stepping;
     - DEC display;
     - reported anatomical angles or labels.
   - **F5 DEC.**
     - **Saved DEC files** (`dtirgb.nii.gz`, any saved velocity RGB) use the FSL/DIPY/ANTsPyMM standard `|v_img| · w` in the image frame. ANTsPyMM uses `dipy.color_fa` on image-frame eigenvectors, and antsxdwi's DIPY parity test depends on this.
     - **Report figures** need channels with anatomical meaning (R = L–R, G = A–P, B = I–S) whatever the acquisition orientation. For example, a sagittal PIR scan stores L–R along image axis k.
       - Computation: `syntx.frames.dec_rgb(frame="anatomical")` computes `|D · v_img| · w` on the native grid, with no resampling.
       - Display: per-channel slicing through syntx's existing, tested display helper `syntx.spatial.ortho_view_spec` / `extract_ortho_slices` (radiological convention, rendered with `origin="lower"`).
       - Display does **not** use `ants.reorient_image2`. Rule 4 forbids unjustified reorient calls, and `reorient_image2` moves the grid without touching channel values.
     - **For axial acquisitions** with D = diag(±1), image-frame and anatomical DEC are identical except for sign, which `|·|` removes.
   - **F6 Metadata.** Every vector/tensor output records `vector_frame: "image_axes"` in its outputs dict or space contract and in its sidecar JSON.
5. **Rule 5 — Single Quality-Control Judge (`antsxqc`)**
   All QC metric specifications, hard physical/execution validity limits, cohort-relative MAD grading, and status vocabulary (`"pass" | "warn" | "fail" | "not_assessable"`) belong exclusively to `antsxqc`. Producer packages emit raw numeric measurements only and must pass `python -m antsxqc.lint` with zero `.antsxqc-allow` exceptions.
