import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np

fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r64'))

# Warp in ANTs using a simple +20 pixel shift in X: t_phys = [20, 0]
import tempfile
affine_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
tx = ants.new_ants_transform(precision='float', dimension=2, transform_type='AffineTransform')
tx.set_parameters(np.array([1.0, 0.0, 0.0, 1.0, 20.0, 0.0]))
tx.set_fixed_parameters(np.zeros(2))
ants.write_transform(tx, affine_file)

warped_ants = ants.apply_transforms(fi, mi, [affine_file])

ants_np = warped_ants.numpy()
mi_np = mi.numpy()

# Find the best shift between mi_np and ants_np along X axis (row 128)
row = 128
sig_mi = mi_np[row, :]
sig_ants = ants_np[row, :]

best_shift = None
min_err = float('inf')
for shift in range(-50, 50):
    if shift < 0:
        err = np.abs(sig_ants[:shift] - sig_mi[-shift:]).mean()
    elif shift > 0:
        err = np.abs(sig_ants[shift:] - sig_mi[:-shift]).mean()
    else:
        err = np.abs(sig_ants - sig_mi).mean()
    
    if err < min_err:
        min_err = err
        best_shift = shift

print(f"For ANTs: best shift along X is {best_shift} pixels with error {min_err:.4f}")

# Let's check: if we shift along Y axis instead?
best_shift_y = None
min_err_y = float('inf')
for shift in range(-50, 50):
    if shift < 0:
        err = np.abs(ants_np[:shift, 128] - mi_np[-shift:, 128]).mean()
    elif shift > 0:
        err = np.abs(ants_np[shift:, 128] - mi_np[:-shift, 128]).mean()
    else:
        err = np.abs(ants_np[:, 128] - mi_np[:, 128]).mean()
    
    if err < min_err_y:
        min_err_y = err
        best_shift_y = shift

print(f"For ANTs: best shift along Y is {best_shift_y} pixels with error {min_err_y:.4f}")

# Clean up
if os.path.exists(affine_file):
    os.remove(affine_file)
