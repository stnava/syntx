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

# 1. ANTs Affine registration
print("Running ANTs Affine registration...")
reg_ants_affine = ants.registration(fi, mi, 'Affine')
mi_affine = ants.apply_transforms(fi, mi, reg_ants_affine['fwdtransforms'])
print(f"Affine-only Dice: {get_dice(mi_affine):.4f}")

# 2. Run Syntx SyN on pre-aligned images
print("\nRunning Syntx SyN on pre-aligned images...")
reg_syn = syntx.syn(
    fixed=fi, moving=mi_affine, type_of_transform='SyN', backend='pytorch',
    affine_iterations=[0], reg_iterations=[100, 100, 100, 50],
    grad_step=0.25, flow_sigma=3.0, elastic_sigma=0.0,
    syn_metric='lncc', lncc_radius=2, inverse_steps=5
)

# Apply the resulting displacement field
warped_syn = ants.apply_transforms(fi, mi_affine, reg_syn['fwdtransforms'])
print(f"Deformable-only SyN Dice: {get_dice(warped_syn):.4f}")

# Clean up
for path in reg_ants_affine['fwdtransforms'] + reg_syn['fwdtransforms'] + reg_syn['invtransforms']:
    if os.path.exists(path):
        os.remove(path)
