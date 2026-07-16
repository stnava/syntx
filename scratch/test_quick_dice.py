import ants
import numpy as np
import sys
sys.path.insert(0, 'src')
import syntx
import time

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

fi_lbl = ants.threshold_image(fi, "Otsu", 3)
mi_lbl = ants.threshold_image(mi, "Otsu", 3)

reg_iterations = [40, 20, 0]

print("Running ANTs...")
start = time.time()
reg_ants = ants.registration(fi, mi, 'SyN', syn_iterations=reg_iterations)
w_ants_lbl = ants.apply_transforms(fi, mi_lbl, reg_ants['fwdtransforms'], interpolator='nearestNeighbor')
dice_ants = ants.label_overlap_measures(fi_lbl, w_ants_lbl)['TargetOverlap'].mean()
print(f"ANTs DICE: {dice_ants:.4f} (Time: {time.time()-start:.2f}s)")

print("Running SyNTo (PyTorch)...")
start = time.time()
reg_py = syntx.syn(fi, mi, 'SyN', backend='pytorch', reg_iterations=reg_iterations)
w_py_lbl = ants.apply_transforms(fi, mi_lbl, reg_py['fwdtransforms'], interpolator='nearestNeighbor')
dice_py = ants.label_overlap_measures(fi_lbl, w_py_lbl)['TargetOverlap'].mean()
print(f"PyTorch DICE: {dice_py:.4f} (Time: {time.time()-start:.2f}s)")
