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
    
    # Use extended iterations with fine-level refinement for best quality
    syn_iters = [80, 40, 10]
    
    # --- 2. ANTs Registration ---
    init_tx = ants.affine_initializer(fi, mi, search_factor=20, radian_fraction=0.1, use_principal_axis=False, local_search_iterations=10)
    
    print("Running ANTs SyN registration...")
    t0 = time.time()    
    reg_ants = ants.registration(
        fixed=fi, moving=mi, type_of_transform='SyN', 
        initial_transform=init_tx,
        reg_iterations=syn_iters, 
        syn_metric='cc', syn_sampling=2
    )
    t_ants = time.time() - t0
    
    print("Running Syntx SyN JAX...")
    t0_jax = time.time()
    reg_jax = syntx.syn(
            fixed=fi, moving=mi, type_of_transform='SyN', backend='jax',
            initial_transform=init_tx,
            reg_iterations=syn_iters,
            grad_step=0.25,
            syn_metric='lncc', lncc_radius=2,
            inverse_steps=50, optimizer='cfl',
            verbose=False
        )
    t_jax = time.time() - t0_jax
        
    print("Running Syntx SyN PyTorch...")
    t0_py = time.time()
    reg_pytorch = syntx.syn(
            fixed=fi, moving=mi, type_of_transform='SyN', backend='pytorch',
            initial_transform=init_tx,
            reg_iterations=syn_iters,
            grad_step=0.25,
            syn_metric='lncc', lncc_radius=2,
            inverse_steps=50, optimizer='cfl',
            verbose=False
        )
    t_py = time.time() - t0_py
    
    print("\n--- Execution Time ---")
    print(f"ANTs: {t_ants:.2f} seconds")
    print(f"JAX: {t_jax:.2f} seconds")
    print(f"PyTorch: {t_py:.2f} seconds\n")
    
    def get_dice(fixed_labels, warped_labels):
        overlap = ants.label_overlap_measures(fixed_labels, warped_labels)
        if overlap.shape[0] > 0 and 'TotalOrTargetOverlap' in overlap.columns:
            return overlap['TotalOrTargetOverlap'].mean()
        return 0.0

    print("Evaluating Dice overlaps...")
    init_warped_labels = ants.apply_transforms(fi, ml, init_tx, interpolator='nearestNeighbor')
    dice_init = get_dice(fl, init_warped_labels)
    
    ants_affine_labels = ants.apply_transforms(fi, ml, reg_ants['fwdtransforms'][1:], interpolator='nearestNeighbor')
    ants_deform_labels = ants.apply_transforms(fi, ml, reg_ants['fwdtransforms'], interpolator='nearestNeighbor')
    dice_ants_aff = get_dice(fl, ants_affine_labels)
    dice_ants_def = get_dice(fl, ants_deform_labels)
    
    jax_affine_labels = ants.apply_transforms(fi, ml, reg_jax['fwdtransforms'][1:], interpolator='nearestNeighbor')
    jax_deform_labels = ants.apply_transforms(fi, ml, reg_jax['fwdtransforms'], interpolator='nearestNeighbor')
    dice_jax_aff = get_dice(fl, jax_affine_labels)
    dice_jax_def = get_dice(fl, jax_deform_labels)
    
    py_affine_labels = ants.apply_transforms(fi, ml, reg_pytorch['fwdtransforms'][1:], interpolator='nearestNeighbor')
    py_deform_labels = ants.apply_transforms(fi, ml, reg_pytorch['fwdtransforms'], interpolator='nearestNeighbor')
    dice_py_aff = get_dice(fl, py_affine_labels)
    dice_py_def = get_dice(fl, py_deform_labels)
    
    err_jax = reg_jax.get('inverse_identity_errors', {})
    err_py = reg_pytorch.get('inverse_identity_errors', {})
    
    # Generate HTML report
    html = f"""
    <html>
    <head><style>
    body {{ font-family: sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 60%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
    th {{ background-color: #f4f4f4; }}
    </style></head>
    <body>
    <h2>3D Registration Parity Report (debug.py)</h2>
    
    <h3>Dice Overlaps</h3>
    <table>
    <tr><th>Stage</th><th>ANTs</th><th>JAX</th><th>PyTorch</th></tr>
    <tr><td>Init Affine</td><td>{dice_init:.4f}</td><td>-</td><td>-</td></tr>
    <tr><td>Optimized Affine</td><td>{dice_ants_aff:.4f}</td><td>{dice_jax_aff:.4f}</td><td>{dice_py_aff:.4f}</td></tr>
    <tr><td>Deformable (SyN)</td><td>{dice_ants_def:.4f}</td><td>{dice_jax_def:.4f}</td><td>{dice_py_def:.4f}</td></tr>
    </table>
    
    <h3>Inverse Identity Errors (Max Error)</h3>
    <table>
    <tr><th>Backend</th><th>phi_1 (F -> Mid)</th><th>phi_2 (M -> Mid)</th></tr>
    <tr><td>JAX</td><td>{err_jax.get('phi_1', {}).get('max_error', 0):.4f}</td><td>{err_jax.get('phi_2', {}).get('max_error', 0):.4f}</td></tr>
    <tr><td>PyTorch</td><td>{err_py.get('phi_1', {}).get('max_error', 0):.4f}</td><td>{err_py.get('phi_2', {}).get('max_error', 0):.4f}</td></tr>
    </table>
    
    <h3>Execution Time</h3>
    <table>
    <tr><th>ANTs</th><th>JAX</th><th>PyTorch</th></tr>
    <tr><td>{t_ants:.2f}s</td><td>{t_jax:.2f}s</td><td>{t_py:.2f}s</td></tr>
    </table>
    
    </body>
    </html>
    """
    with open('report_3d.html', 'w') as f:
        f.write(html)
    print("Report saved to report_3d.html")
