import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import syntx

fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r64'))

fixed_seg = ants.threshold_image(fi, 'Otsu', 3)
reg_ants_affine = ants.registration(fi, mi, 'Affine')

def get_dice(warped_image):
    warped_seg = ants.threshold_image(warped_image, 'Otsu', 3)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    return float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])

# 1. Syntx SyN with mattes_mi
print("Running Syntx SyN with mattes_mi...")
reg_mi = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyN', backend='pytorch',
    initial_transform=reg_ants_affine['fwdtransforms'],
    affine_iterations=[0], reg_iterations=[100, 100, 100, 50],
    grad_step=0.25, flow_sigma=3.0, elastic_sigma=0.0,
    syn_metric='mattes_mi', inverse_steps=5
)
warped_mi = ants.apply_transforms(fi, mi, reg_mi['fwdtransforms'])
print(f"Mattes MI Dice: {get_dice(warped_mi):.4f}")

# 2. Syntx SyN with lncc
print("\nRunning Syntx SyN with lncc...")
reg_lncc = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyN', backend='pytorch',
    initial_transform=reg_ants_affine['fwdtransforms'],
    affine_iterations=[0], reg_iterations=[100, 100, 100, 50],
    grad_step=0.25, flow_sigma=3.0, elastic_sigma=0.0,
    syn_metric='lncc', lncc_radius=2, inverse_steps=5
)
warped_lncc = ants.apply_transforms(fi, mi, reg_lncc['fwdtransforms'])
print(f"LNCC Dice: {get_dice(warped_lncc):.4f}")
