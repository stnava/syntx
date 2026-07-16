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

# Try different Syntx SyN settings initialized with ANTs Affine
settings = [
    # (lncc_radius, grad_step, flow_sigma, elastic_sigma)
    (2, 0.25, 3.0, 0.0),
    (2, 0.1, 3.0, 0.0),
    (2, 0.5, 3.0, 0.0),
    (4, 0.25, 3.0, 0.0),
    (2, 0.25, 3.0, 0.5), # with elastic smoothing
]

for r, step, flow, elastic in settings:
    print(f"\nRunning Syntx SyN: lncc_radius={r}, grad_step={step}, flow_sigma={flow}, elastic_sigma={elastic}...")
    reg = syntx.syn(
        fixed=fi, moving=mi, type_of_transform='SyN', backend='pytorch',
        initial_transform=reg_ants_affine['fwdtransforms'],
        affine_iterations=[0], reg_iterations=[100, 100, 100, 50],
        grad_step=step, flow_sigma=flow, elastic_sigma=elastic,
        syn_metric='lncc', lncc_radius=r, inverse_steps=5
    )
    warped = ants.apply_transforms(fi, mi, reg['fwdtransforms'])
    print(f"Dice score: {get_dice(warped):.4f}")
