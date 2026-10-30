# Coordinate Frame Conventions (`syntx.frames`, F1–F6)

This document specifies the canonical vector and tensor coordinate frame conventions shared across the ANTsX ecosystem (`syntx`, `antsxdwi`, `antsxslowflow`).

## The Convention (F1–F6)

- **F1 Storage.** Every saved vector or tensor pixel (velocity, V1, fixel peaks, FOD SH, harmonic axes) has its components along the image axes of the image that holds it. Physical direction is $v_{\text{lps}} = D \cdot v_{\text{img}}$ ($v_{\text{img}} = D^T \cdot v_{\text{lps}}$). This is native ANTs/ANTsPyMM behaviour, so `ants.apply_transforms(imagetype=1/2)` and `syntx.apply_transforms` reorient these files correctly with no extra flags.
  - **ITK exception:** displacement fields and warps (`*Warp.nii.gz`, `*_field.nii.gz`) keep their components in physical LPS mm, never in `image_axes`.
- **F2 Gradient tables.**
  - On disk, bvecs follow FSL/BIDS (radiological voxel frame).
  - On read, convert once to the image-axis frame: $g_{\text{img}} = F(D) \cdot g_{\text{bvec}}$, where $F(D) = \text{diag}(-1, 1, 1)$ if $\det(D) > 0$, else $I_3$.
  - On write, apply $g_{\text{bvec}} = F(D_{\text{out}}) \cdot g_{\text{img}}$.
  - Nothing else touches handedness.
- **F3 Reorientation.** Vectors, tensors and gradient tables move between spaces only through syntx, using $g_{\text{fix}} = D_{\text{fix}}^T \cdot R_{\text{lps}} \cdot D_{\text{mov}} \cdot g_{\text{mov}}$, where $R_{\text{lps}} = \text{polar}_+(M^{-1})$ is the proper rotation ($\det = +1$) in physical LPS of the composed linear part, and nonzero gradients are renormalized to unit length.
- **F4 Physical directions.** Physical LPS directions are produced only at the boundaries that need them, through `syntx.frames.image_to_lps(v, direction=D, kind="vector"|"tensor")` / `lps_to_image(v, direction=D, kind="vector"|"tensor")`:
  - tractography stepping;
  - DEC display;
  - reported anatomical angles or labels.
  - `kind` explicitly distinguishes vector (`(..., 3)`) from tensor (`(..., 3, 3)` or `(..., 6)`); kind is never inferred from shape.
- **F5 DEC.**
  - **Saved DEC files** (`dtirgb.nii.gz`, any saved velocity RGB) use the FSL/DIPY/ANTsPyMM standard $|v_{\text{img}}| \cdot w$ in the image frame. ANTsPyMM uses `dipy.color_fa` on image-frame eigenvectors, and antsxdwi's DIPY parity test depends on this.
  - **Report figures** need channels with anatomical meaning (Red = Left–Right, Green = Anterior–Posterior, Blue = Inferior–Superior) whatever the acquisition orientation (e.g., a sagittal PIR scan stores L–R along image axis $k$).
    - Computation: `syntx.frames.dec_rgb(vector_image, frame="anatomical", direction=D)` computes $|D \cdot v_{\text{img}}| \cdot w$ on the native grid, with no resampling. When `vector_image` is a NumPy array, `direction` is required (no silent fallback to identity), and optional `origin` and `spacing` are preserved on the returned ANTsImage.
    - Display: per-channel slicing through syntx's display helper `syntx.spatial.ortho_view_spec` / `extract_ortho_slices` (radiological convention, rendered with `origin="lower"`).
    - Display does **not** use `ants.reorient_image2`. Rule 4 forbids unjustified reorient calls, and `reorient_image2` moves the grid without touching channel values.
  - **For axial acquisitions** with $D = \text{diag}(\pm 1)$, image-frame and anatomical DEC are identical except for sign, which $|\cdot|$ removes.
- **F6 Metadata.** Every vector/tensor output records `vector_frame: "image_axes"` in its outputs dict or space contract and in its sidecar JSON.
