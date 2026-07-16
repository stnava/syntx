import ants
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from syntx.syn import registration

def compute_tissue_overlap(fixed_img, warped_img):
    fixed_seg = ants.threshold_image(fixed_img, 'Otsu', 3)
    warped_seg = ants.threshold_image(warped_img, 'Otsu', 3)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    if 'MeanOverlap' in overlap.columns:
        return float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
    return 0.0

def main():
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))
    
    print("--- ANTs Registrations ---")
    reg_ants_aff = ants.registration(fi, mi, 'Affine')
    dice_ants_aff = compute_tissue_overlap(fi, reg_ants_aff['warpedmovout'])
    print(f"ANTs Affine Dice: {dice_ants_aff:.4f}")
    
    # Save the ANTs affine transform to a persistent file so we can reuse it
    ants_aff_fwd = reg_ants_aff['fwdtransforms'][0]
    import shutil
    persistent_aff_path = "./temp_ants_affine.mat"
    shutil.copy(ants_aff_fwd, persistent_aff_path)
    
    reg_ants_syn = ants.registration(fi, mi, 'SyN')
    dice_ants_syn = compute_tissue_overlap(fi, reg_ants_syn['warpedmovout'])
    print(f"ANTs SyN Dice: {dice_ants_syn:.4f}")
    
    print("\n--- Syntx SyN (Init with ANTs Affine) ---")
    for use_analytical in [True, False]:
        for gs in [0.2, 0.5, 0.75]:
            reg_py_init = registration(
                fi, mi, type_of_transform='SyN', backend='pytorch',
                initial_transform=[persistent_aff_path],
                affine_iterations=[0], reg_iterations=[100, 100, 100, 50],
                grad_step=gs, flow_sigma=1.732,
                use_analytical_gradients=use_analytical
            )
            dice_py_init = compute_tissue_overlap(fi, reg_py_init['warpedmovout'])
            print(f"Syntx SyN (Init, analytical={use_analytical}, grad={gs}) Dice: {dice_py_init:.4f}")
            for path in reg_py_init['fwdtransforms'] + reg_py_init['invtransforms']:
                if os.path.exists(path) and path != persistent_aff_path:
                    os.remove(path)
                    
    # Clean up persistent affine file
    if os.path.exists(persistent_aff_path):
        os.remove(persistent_aff_path)

if __name__ == '__main__':
    main()
