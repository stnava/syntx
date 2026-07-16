import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import syntx

fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r64'))

fixed_seg = ants.threshold_image(fi, 'Otsu', 3)

def get_dice(warped_image):
    warped_seg = ants.threshold_image(warped_image, 'Otsu', 3)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    return float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])

# 1. Syntx Affine-only
print("Running Syntx Affine PyTorch...")
reg_affine = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='Affine', backend='pytorch',
    affine_iterations=[100, 50, 50, 20]
)
print("Syntx Affine Dice:", get_dice(reg_affine['warpedmovout']))

# 2. Syntx SyN initialized with ANTs Affine
print("\nRunning Syntx SyN initialized with ANTs Affine...")
reg_ants_affine = ants.registration(fi, mi, 'Affine')
reg_syn_init = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyN', backend='pytorch',
    initial_transform=reg_ants_affine['fwdtransforms'],
    affine_iterations=[0], reg_iterations=[100, 100, 100, 50],
    grad_step=0.75, flow_sigma=1.732, inverse_steps=5
)
warped_syn_init = ants.apply_transforms(fi, mi, reg_syn_init['fwdtransforms'])
print("Syntx SyN (init with ANTs Affine) Dice:", get_dice(warped_syn_init))

# 3. Composed Affine + SyN (Full SyNTo)
print("\nRunning Syntx Composed SyNTo...")
reg_composed = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyN', backend='pytorch',
    affine_iterations=[100, 50, 50, 20], reg_iterations=[100, 100, 100, 50],
    grad_step=0.75, flow_sigma=1.732, inverse_steps=5
)
warped_composed = ants.apply_transforms(fi, mi, reg_composed['fwdtransforms'])
print("Syntx Composed SyNTo Dice:", get_dice(warped_composed))
