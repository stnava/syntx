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
    base_path = '/Users/stnava/Downloads/mindboggle_data/'
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
    syn_iters = [50, 20, 0]
    
    # --- 2. ANTs Registration ---
    
    print("Running ANTs SyN registration...")
    t0 = time.time()
    reg_ants = ants.registration(
        fi, mi, 'SyN', 
        reg_iterations=syn_iters, 
        syn_metric='mattes', syn_sampling=32
    )
    t1 = time.time()
    ants_time = t1 - t0
    
    warped_ants_mov = reg_ants['warpedmovout']
    mi_ants_mov = ants.image_mutual_information(fi, warped_ants_mov)
    print( "ants mi :" + str(mi_ants_mov)  )

    reg_py2 = syntx.syn(
            fixed=fi, moving=mi, type_of_transform='SyN', backend='jax',
            affine_iterations=[200, 100, 5], reg_iterations=[100,50,5],
            syn_metric='mattes_mi', sampling_percentage=0.2, verbose=2
        )
    ants.image_write( fi, '/tmp/tempf.nii.gz')
    ants.image_write( reg_py2['warpedmovout'], '/tmp/tempm.nii.gz')
    mi_py_mov = ants.image_mutual_information(fi, reg_py2['warpedmovout'])
    print( mi_py_mov )
    print( "jax mi :" + str(mi_py_mov)  )


