## 2026-07-15T13:15:40Z

Implement the comprehensive image comparison metrics suite in `src/syntx/image_compare.py` and expose it.
Specifically:
1. Create `src/syntx/image_compare.py` containing the callable:
   `def image_compare(a, b, metricname: str, **kwargs) -> float:`
   This function must:
   - Accept inputs `a` and `b`, which can be `ants.ANTsImage` or PyTorch tensors, JAX arrays, or NumPy arrays. Convert them to numpy/torch tensors internally as appropriate.
   - Support both 2D and 3D images for all metrics.
   - Standardize the return score such that a lower score indicates better similarity:
     - For MSE, MAE, RMSE, L1, L2: smaller is better, return value directly.
     - For NCC, global correlation: return `1 - value`.
     - For PSNR: return `-value`.
     - For SSIM: return `1 - value`.
     - For Mutual Information / NMI: return `-value`.
     - For feature space losses, use VGG, DINOv2, ResNet10, SwinUNETR features.
2. Support at least 64 unique metric names (configurations):
   - Classical (18): `mse`, `mae`, `rmse`, `psnr`, `ncc`, `nmi`, `joint_entropy`, local NCC metrics `lncc_w3`, `lncc_w5`, `lncc_w7`, `lncc_w9`, `lncc_w11`, Mattes MI metrics `mmi_b16`, `mmi_b32`, `mmi_b64`, `mmi_b128`, `mmi_b256`, and `ssim`.
   - Gradient/Spatial (6): `gradient_mse`, `gradient_correlation`, `ngf_e01`, `ngf_e1`, `ngf_e10` (Normalized Gradient Fields), `ms_ssim`.
   - Deep Features VGG19 (12): `vgg_l_l1`, `vgg_l_l2`, `vgg_l_lncc` for layers l in {2, 4, 8, 12}. (Ensure `vgg_4_lncc` uses VGG 3D LNCC per GEMINI.md).
   - Deep Features DINOv2 (12): `dino_l_l1`, `dino_l_l2`, `dino_l_lncc` for layers l in {1, 2, 6, 11}.
   - Deep Features ResNet10 (12): `resnet_l_l1`, `resnet_l_l2`, `resnet_l_lncc` for layers l in {1, 2, 3, 4}.
   - Deep Features SwinUNETR (4): `swin_l_lncc` for layers l in {1, 2, 3, 4}.
   - Plus any other metrics/variants to exceed 64 distinct configurations (e.g. cosine distance variant on features).
3. Import `image_compare` in `src/syntx/__init__.py` and add it to `__all__`.
4. Ensure all GEMINI.md constraints are respected:
   - Single Interpolation Policy: do not use intermediate file-based pre-warped images during optimization.
   - VGG 3D LNCC Layer 4 Requirement: Only VGG 3D LNCC with Layer 4 meets performance targets; do not default to coarser layers or 2D.
5. Write your plans, progress, and handoffs in `/Users/stnava/code/syntx/.agents/worker_m2/`.
6. Run existing tests to verify that the implementation does not break any existing functionality.

## 2026-07-15T13:16:40Z

**Context**: Fixing JAX test suite compilation error in tests/test_syn_jax.py.
**Content**: There is a JAX test suite compilation failure: `tests/test_syn_jax.py::test_new_jax_helpers` fails with `TypeError: unhashable type: 'jaxlib._jax.ArrayImpl'` at line 239. The call to `prepare_mid_images_and_gradients_jax` is missing the `affine_grid_level` argument (which should be `identity`), shifting the `identity` array into the static `spacing` slot.
Please modify line 239-242 in `tests/test_syn_jax.py` from:
    I_mid, J_mid, grad_I, grad_J = prepare_mid_images_and_gradients_jax(
        warp_l2r, warp_r2l, I_curr, J_curr,
        True, spacing_arg, identity
    )
to:
    I_mid, J_mid, grad_I, grad_J = prepare_mid_images_and_gradients_jax(
        warp_l2r, warp_r2l, I_curr, J_curr, identity,
        True, spacing_arg, identity
    )
**Action**: Apply this fix to `tests/test_syn_jax.py`, ensure all tests pass (by running `pytest`), and continue with the metric suite implementation in `src/syntx/image_compare.py`.
