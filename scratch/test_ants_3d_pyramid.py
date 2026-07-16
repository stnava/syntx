import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import syntx

# Create synthetic 3D images
shape = (64, 64, 64)
fi_np = np.zeros(shape, dtype=np.float32)
mi_np = np.zeros(shape, dtype=np.float32)
x, y, z = np.ogrid[:64, :64, :64]
mask_fi = (x - 32)**2 + (y - 32)**2 + (z - 32)**2 <= 15**2
fi_np[mask_fi] = 1.0
mask_mi = (x - 35)**2 + (y - 30)**2 + (z - 33)**2 <= 15**2
mi_np[mask_mi] = 1.0

fi = ants.from_numpy(fi_np, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
mi = ants.from_numpy(mi_np, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))

fixed_seg = ants.threshold_image(fi, 0.5)

def get_metrics(warped_image):
    warped_seg = ants.threshold_image(warped_image, 0.5)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    dice = float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
    mi_val = ants.image_mutual_information(fi, warped_image)
    return dice, mi_val

# 1. Run ANTs SyN (CC, 3 levels [40, 20, 0] which is 3 levels: downsampled 4x, 2x, 1x)
# Note: reg_iterations=[40, 20, 0] in ants.registration runs 3 levels:
# Level 0 (4x downsampled): 40 iterations
# Level 1 (2x downsampled): 20 iterations
# Level 2 (1x full res): 0 iterations (does not run full res to save time, or 10 iterations)
# Let's run [40, 20, 10] iterations
levels_ants = [4, 2, 1]
iters_ants = [40, 20, 10]

print("Running ANTs SyN...")
reg_ants = ants.registration(
    fixed=fi, moving=mi, type_of_transform='SyNOnly',
    reg_iterations=iters_ants,
    syn_metric='cc', syn_sampling=4,
    flow_sigma=3.0, total_sigma=0.0,
    grad_step=0.1
)
dice_ants, mi_ants = get_metrics(reg_ants['warpedmovout'])
print(f"ANTs SyN | Dice: {dice_ants:.6f} | MI: {mi_ants:.6f}")

# 2. Run Syntx PyTorch SyN
print("\nRunning Syntx PyTorch...")
reg_syn_py = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyNOnly', backend='pytorch',
    levels=levels_ants, reg_iterations=iters_ants, grad_step=0.1, flow_sigma=3.0,
    syn_metric='lncc', lncc_radius=4, initial_transform=[]
)
dice_py, mi_py = get_metrics(reg_syn_py['warpedmovout'])
print(f"Syntx PyTorch | Dice: {dice_py:.6f} | MI: {mi_py:.6f}")

# 3. Run Syntx JAX SyN
print("\nRunning Syntx JAX...")
reg_syn_jax = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyNOnly', backend='jax',
    levels=levels_ants, reg_iterations=iters_ants, grad_step=0.1, flow_sigma=3.0,
    syn_metric='lncc', lncc_radius=4, initial_transform=[]
)
dice_jax, mi_jax = get_metrics(reg_syn_jax['warpedmovout'])
print(f"Syntx JAX | Dice: {dice_jax:.6f} | MI: {mi_jax:.6f}")

# Clean up
for reg in [reg_ants, reg_syn_py, reg_syn_jax]:
    for path in reg['fwdtransforms'] + reg['invtransforms']:
        if os.path.exists(path):
            os.remove(path)
