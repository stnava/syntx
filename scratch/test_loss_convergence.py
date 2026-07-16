import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch
import syntx

fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r64'))

# ANTs Affine registration
reg_ants_affine = ants.registration(fi, mi, 'Affine')
mi_affine = ants.apply_transforms(fi, mi, reg_ants_affine['fwdtransforms'])

# Run Syntx SyN and print loss at each step
print("Running Syntx SyN deformable-only...")
reg_syn = syntx.syn(
    fixed=fi, moving=mi_affine, type_of_transform='SyN', backend='pytorch',
    affine_iterations=[0], reg_iterations=[30, 0, 0], # Run 30 steps at level 0
    grad_step=0.25, flow_sigma=3.0, elastic_sigma=0.0,
    syn_metric='lncc', lncc_radius=2, inverse_steps=5,
    verbose=True
)

# Clean up
for path in reg_ants_affine['fwdtransforms'] + reg_syn['fwdtransforms'] + reg_syn['invtransforms']:
    if os.path.exists(path):
        os.remove(path)
