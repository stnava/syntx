# Forensic Audit Report — Image Comparison Metrics Suite

**Work Product**: Image Comparison Metrics Suite
**Profile**: General Project + Syntx Registration Guardrails
**Verdict**: CLEAN

## 1. Observation
* **Source Code Verification**:
  * `src/syntx/image_compare.py`: The similarity metrics (MSE, MAE, RMSE, PSNR, NCC, NMI, Joint Entropy, SSIM, NGF, MS-SSIM) are computed dynamically using authentic math operations. No hardcoded or dummy returns were detected.
  * For deep features, standard extractors (VGG19, DINOv2, ResNet10, SwinUNETR) are loaded and evaluated under `l1`, `l2`, `lncc`, or `cos` mode. Triplanar projection is genuinely implemented for 3D inputs on 2D models:
    ```python
    def compute_reconstructed_loss(extractor, a_3d, b_3d, loss_type='l1'):
        """Calculates triplanar reconstructed features for 2D networks applied to 3D."""
        D, H, W = a_3d.shape[2:]
        B = a_3d.shape[0]
        # Extracts slices in axial, coronal, and sagittal planes and computes the loss
        ...
    ```
  * `src/syntx/generators.py`: The cross-product simulation space is constructed with genuine transformations (intensity: noise, bias, inhomogeneity, modality mapping, step quantization, missing masks; shape: translation, rotation, affine, deformation). The ground truth L2 norm of the physical displacement field is mathematically calculated:
    ```python
    def compute_physical_l2_norm(self, u_norm):
        ...
        spacing_t = torch.tensor(self.spacing, dtype=dtype, device=device)
        direction_t = torch.tensor(self.direction, dtype=dtype, device=device)
        u_vox_scaled = u_vox * spacing_t
        u_phys = torch.matmul(u_vox_scaled, direction_t.t())
        delta_V = float(np.prod(self.spacing))
        sum_sq = torch.sum(u_phys ** 2)
        norm = torch.sqrt(delta_V * sum_sq)
        return norm.item()
    ```
* **Guardrail Compliance**:
  * **Single Interpolation Policy**: Affine and deformable warps are composed onto coordinate grids (`composed_grid = compose_grids(grid_affine, phi_l2r)`) and applied to the image in a single step using PyTorch's `F.grid_sample` in `examples/evaluate_metrics_generative.py` (lines 239-245). No intermediate image pre-warping occurs.
  * **VGG 3D Mode Requirement**: In `src/syntx/image_compare.py` (lines 298-304), deep feature similarity automatically defaults to 3D LNCC with Layer 4 when the input dimensionality is 3D:
    ```python
    if mtype == 'lncc':
        if dim == 3:
            loss_fn = FeatureSpaceLoss(extractor=extractor, mode='lncc_3d')
        else:
            loss_fn = FeatureSpaceLoss(extractor=extractor, mode='lncc')
    ```
* **Dynamic Execution Outputs**:
  * Ran the project unit tests using `pytest` successfully: `117 passed, 6 skipped, 6 warnings in 137.63s` with a total statements coverage of `91%` (specifically `87%` coverage for `src/syntx/image_compare.py` and `90%` for `src/syntx/generators.py`).
  * Ran `examples/evaluate_metrics_generative.py` which dynamically generated the CSV results and `docs/registration_report.html` (size: 435KB), containing base64 encoded plots (edge overlay contours, warp coordinate grids, Jacobian determinant maps, and side-by-side target vs warped visual diagnostics).
  * Ran `examples/compare_metrics_tutorial.py` which executes successfully and prints accurate, naive-user-friendly descriptions of the metric score behavior.
* **Layout Compliance**: All codebase source files are in `src/syntx`, test files in `tests/`, examples in `examples/`, docs in `docs/`, and agent folders contain only metadata (`.agents/auditor_m5`).

## 2. Logic Chain
1. Analysis of `src/syntx/image_compare.py` and `src/syntx/generators.py` shows they calculate metrics and transformations dynamically using PyTorch, JAX, NumPy, and SciPy. This confirms there are no faked outputs or facade implementations.
2. The dynamic generation of `docs/registration_report.html` and `outputs_comparison/generative_evaluation_results.csv` on-the-fly verifies that the visual diagnostic maps and metric tables are based on real registration performance.
3. Code observation shows that coordinate transformations are composed in parameter/grid space prior to a single image interpolation call (`F.grid_sample`), satisfying the **Single Interpolation Policy**.
4. The auto-resolution to `lncc_3d` in 3D feature metrics ensures compliance with the **VGG Feature Space Guidelines**.
5. The successful execution of 117 tests confirms functional correctness and robustness of the implementation.

## 3. Caveats
No caveats.

## 4. Conclusion
The Image Comparison Metrics Suite implementation is authentic and respects the registration guardrails. There are no faked logs, hardcoded results, or dummy facade logic. The final verdict is **CLEAN**.

## 5. Verification Method
1. **Run Tests**: Execute `pytest` in the project root to verify all 117 unit tests pass cleanly.
2. **Review Code**:
   * Inspect `src/syntx/image_compare.py` to confirm the metrics logic.
   * Inspect `src/syntx/generators.py` to confirm the simulation logic.
3. **Inspect Output Files**:
   * Open `docs/registration_report.html` to visually check the overlap contours, grid warps, Jacobian determinants, and side-by-side registration figures.
   * Run the evaluation script `python examples/evaluate_metrics_generative.py` and the tutorial script `python examples/compare_metrics_tutorial.py` to verify command correctness.
