import ants
import numpy as np
import sys
sys.path.insert(0, 'src')
import syntx
import time

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

reg_iterations = [40, 20, 0]

print("Running SyNTo (PyTorch)...")
reg_py = syntx.syn(fi, mi, 'SyN', backend='pytorch', reg_iterations=reg_iterations)

# Try swapping components
fwdtransforms = reg_py['fwdtransforms']
print(fwdtransforms)
fwd_img = ants.image_read(fwdtransforms[0])
fwd_arr = fwd_img.numpy()
fwd_arr_swapped = fwd_arr[..., ::-1].copy()
fwd_img_swapped = ants.from_numpy(fwd_arr_swapped, origin=fwd_img.origin, spacing=fwd_img.spacing, direction=fwd_img.direction, has_components=True)
ants.image_write(fwd_img_swapped, 'scratch/fwd_swapped.nii.gz')

w_py = ants.apply_transforms(fi, mi, fwdtransforms)
mi_py = ants.image_mutual_information(fi, w_py)
print(f"PyTorch MI (Unswapped): {mi_py:.4f}")

# Now apply with swapped warp
fwdtransforms_swapped = ['scratch/fwd_swapped.nii.gz'] + fwdtransforms[1:]
w_py_swapped = ants.apply_transforms(fi, mi, fwdtransforms_swapped)
mi_py_swapped = ants.image_mutual_information(fi, w_py_swapped)
print(f"PyTorch MI (Swapped): {mi_py_swapped:.4f}")

