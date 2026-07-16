import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import syntx

fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r64'))

fixed_seg = ants.threshold_image(fi, 'Otsu', 3)

def get_metrics(warped_image):
    warped_seg = ants.threshold_image(warped_image, 'Otsu', 3)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    dice = float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
    mi_val = ants.image_mutual_information(fi, warped_image)
    return dice, mi_val

# 1. Run ANTs Affine
reg_ants_affine = ants.registration(fi, mi, 'Affine')

# 2. Run ANTs SyN (Single Level: full resolution only)
print("Running ANTs SyN (Single Level)...")
reg_ants_syn = ants.registration(
    fixed=fi, moving=mi, type_of_transform='SyNOnly',
    initial_transform=reg_ants_affine['fwdtransforms'],
    reg_iterations=[100],
    syn_metric='cc', syn_sampling=4,
    flow_sigma=3.0, total_sigma=0.0,
    grad_step=0.1
)
dice_ants, mi_ants = get_metrics(reg_ants_syn['warpedmovout'])
print(f"ANTs SyN | Dice: {dice_ants:.6f} | MI: {mi_ants:.6f}")

# 3. Run Syntx PyTorch SyN (Single Level)
print("\nRunning Syntx PyTorch SyN (Single Level)...")
reg_syn_py = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyNOnly', backend='pytorch',
    initial_transform=reg_ants_affine['fwdtransforms'],
    levels=[1],
    reg_iterations=[100],
    grad_step=0.1, flow_sigma=3.0, elastic_sigma=0.0,
    syn_metric='lncc', lncc_radius=4, inverse_steps=5
)
dice_py, mi_py = get_metrics(reg_syn_py['warpedmovout'])
print(f"Syntx PyTorch SyN | Dice: {dice_py:.6f} | MI: {mi_py:.6f}")

# Clean up
for reg in [reg_ants_affine, reg_ants_syn, reg_syn_py]:
    for path in reg['fwdtransforms'] + reg['invtransforms']:
        if os.path.exists(path):
            os.remove(path)
