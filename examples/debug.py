import ants
import base64
import os
import tempfile
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

from syntx.syn import SyNTo
from syntx.syn_jax import SyNTo as SyNToJax
import syntx
# ff='examples/debug.py'
# exec(open(ff).read())

if True:
    print("Loading 3D Mindboggle images...")
    base_path = '/Users/stnava/data/mindboggle/volumes'
#    base_path = '/Users/stnava/Downloads/mindboggle_data/'
    fi_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
    mi_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')
    fl_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 'labels.DKT31.manual.nii.gz')
    ml_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 'labels.DKT31.manual.nii.gz')
    
    fi_full = ants.image_read(fi_path)
    mi_full = ants.image_read(mi_path)
    fl_full = ants.image_read(fl_path)
    ml_full = ants.image_read(ml_path)
    
    print("Cropping fixed image to bounding box to save time...")
    mask = ants.get_mask(fi_full)
    mask_dilated = ants.iMath(mask, "MD", 12)
    fi = ants.crop_image(fi_full, mask_dilated)
    fl = ants.crop_image(fl_full, mask_dilated)
    
    # Keep moving image at full native size to stress-test physical coordinate tracking
    mi = mi_full
    ml = ml_full
    
    # Set moderate deformation iterations since we cropped
    syn_iters = [40, 20, 0]
    
    # --- 2. ANTs Registration ---
    # Use ANTs affine initializer (center of mass)
    init_tx = ants.affine_initializer(fi, mi, search_factor=20, radian_fraction=0.1, use_principal_axis=False, local_search_iterations=10)
    
    print("Running ANTs SyN registration...")
    t0 = time.time()    
    reg_ants = ants.registration(
        fixed=fi, moving=mi, type_of_transform='SyNOnly', 
        initial_transform=init_tx,
        reg_iterations=syn_iters, 
        syn_metric='cc', syn_sampling=2
    )
    t_ants = time.time() - t0
    
    ants_fwd_warp = ants.image_read(reg_ants['fwdtransforms'][0])
    ants_max_norm = np.sqrt(np.sum(ants_fwd_warp.numpy()**2, axis=-1)).max()
    print(f"ANTS warp_l2r max norm: {ants_max_norm}")
    
    ants_mi = ants.image_mutual_information(fi, reg_ants['warpedmovout'])
    print( "ants mi :" + str(ants_mi)  )

    print("Running Syntx SyNOnly JAX...")
    t0_jax = time.time()
    reg_jax = syntx.syn(
            fixed=fi, moving=mi, type_of_transform='SyNOnly', backend='jax',
            initial_transform=init_tx,
            reg_iterations=syn_iters,
            grad_step=0.2,
            syn_metric='lncc', lncc_radius=2,
            inverse_steps=50, optimizer='cfl'
        )
    t_jax = time.time() - t0_jax
        
    print("Running Syntx SyNOnly PyTorch...")
    t0_py = time.time()
    reg_pytorch = syntx.syn(
            fixed=fi, moving=mi, type_of_transform='SyNOnly', backend='pytorch',
            initial_transform=init_tx,
            reg_iterations=syn_iters,
            grad_step=0.2,
            syn_metric='lncc', lncc_radius=2,
            inverse_steps=50, optimizer='cfl',
            verbose=0
        )
    t_py = time.time() - t0_py
    
    print("\n--- Execution Time ---")
    print(f"ANTs: {t_ants:.2f} seconds")
    print(f"JAX: {t_jax:.2f} seconds")
    print(f"PyTorch: {t_py:.2f} seconds\n")
        
    print("Saving fields for analysis...")
    fwd = reg_pytorch['fwdtransforms'][0] # This should be the composed fwd field
    print(f"FWD transforms list: {reg_pytorch['fwdtransforms']}")
    # The first element is fwd_file
    import nibabel as nib
    fwd_data = nib.load(reg_pytorch['fwdtransforms'][0]).get_fdata()
    print(f"FWD FIELD MIN/MAX/MEAN: {fwd_data.min()}, {fwd_data.max()}, {fwd_data.mean()}")
    
    # Also I want to extract the l2r and r2l_inv directly, but I can't easily return them from syn without modifying it.
    # But wait, we can just run ants.apply_transforms to see if composed_fwd works!
        
    ants.image_write( fi, '/tmp/tempf.nii.gz')
    ants.image_write( reg_jax['warpedmovout'], '/tmp/tempm_jax.nii.gz')
    ants.image_write( reg_pytorch['warpedmovout'], '/tmp/tempm_py.nii.gz')
    
    mi_jax_mov = ants.image_mutual_information(fi, reg_jax['warpedmovout'])
    mi_py_mov = ants.image_mutual_information(fi, reg_pytorch['warpedmovout'])
    
    print( "ants mi :" + str(ants_mi)  )
    print( "jax mi :" + str(mi_jax_mov)  )
    print( "pytorch mi :" + str(mi_py_mov)  )
    
    # Evaluate label overlap (Dice)
    ants_warped_labels = ants.apply_transforms(fi, ml, reg_ants['fwdtransforms'], interpolator='nearestNeighbor')
    jax_warped_labels = ants.apply_transforms(fi, ml, reg_jax['fwdtransforms'], interpolator='nearestNeighbor')
    py_warped_labels = ants.apply_transforms(fi, ml, reg_pytorch['fwdtransforms'], interpolator='nearestNeighbor')
    
    def get_dice(fixed_labels, warped_labels):
        overlap = ants.label_overlap_measures(fixed_labels, warped_labels)
        if overlap.shape[0] > 0 and 'TotalOrTargetOverlap' in overlap.columns:
            return overlap['TotalOrTargetOverlap'].mean()
        return 0.0

    init_warped_labels = ants.apply_transforms(fi, ml, init_tx, interpolator='nearestNeighbor')
    print("init affine dice :", get_dice(fl, init_warped_labels))
        
    print("ants dice :", get_dice(fl, ants_warped_labels))
    print("jax dice :", get_dice(fl, jax_warped_labels))
    print("pytorch dice :", get_dice(fl, py_warped_labels))
    
    print( "JAX errors:", reg_jax['inverse_identity_errors']   )
    print( "PyTorch errors:", reg_pytorch['inverse_identity_errors']   )
