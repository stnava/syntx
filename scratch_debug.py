import ants
import numpy as np
from syntx.syn import registration

# Load 2D phantoms
fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r27'))

# Otsu threshold overlap helper
def compute_tissue_overlap(fixed_img, warped_img):
    fixed_seg = ants.threshold_image(fixed_img, 'Otsu', 3)
    warped_seg = ants.threshold_image(warped_img, 'Otsu', 3)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    if 'MeanOverlap' in overlap.columns:
        return float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
    return 0.0

res_py = registration(
    fixed=fi,
    moving=mi,
    type_of_transform='SyNTo',
    backend='pytorch',
    levels=[4, 2, 1],
    affine_iterations=[100, 100, 50],
    reg_iterations=[100, 100, 50],
    grad_step=0.75,
    flow_sigma=1.732,
    use_analytical_gradients=False,
    verbose=True
)
dice_py = compute_tissue_overlap(fi, res_py['warpedmovout'])
mse_py = np.mean((fi.numpy() - res_py['warpedmovout'].numpy())**2)

res_jax = registration(
    fixed=fi,
    moving=mi,
    type_of_transform='SyNTo',
    backend='jax',
    levels=[4, 2, 1],
    affine_iterations=[100, 100, 50],
    reg_iterations=[100, 100, 50],
    grad_step=0.75,
    flow_sigma=1.732,
    use_analytical_gradients=False,
    verbose=True
)
dice_jax = compute_tissue_overlap(fi, res_jax['warpedmovout'])
moving_warped_jax = res_jax['warpedmovout']
mse_jax = np.mean((fi.numpy() - moving_warped_jax.numpy())**2)

print(f"PyTorch (Autograd): Dice={dice_py:.4f}, MSE={mse_py:.4f}")
print(f"JAX (Autograd):     Dice={dice_jax:.4f}, MSE={mse_jax:.4f}")
