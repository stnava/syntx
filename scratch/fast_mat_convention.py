"""Check ANTs .mat convention."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import numpy as np
import ants

# Create a known affine and check parameter layout
fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
ra = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])

# Read ANTs transform
fwd_aff = ra['fwdtransforms'][0]
tx = ants.read_transform(fwd_aff)
params = tx.parameters
print(f"ANTs params: {params}")

# In ITK, AffineTransform parameters are:
# [M[0,0], M[0,1], M[1,0], M[1,1], t[0], t[1]] for 2D
# BUT actually ITK stores matrix in ROW-MAJOR order
# M.ravel() = [M[0,0], M[0,1], M[1,0], M[1,1]] (row-major)
# So params = [M_ravel..., t_ravel...]

# This matches np.ravel() order (row-major by default)
dim = 2
M = params[:dim*dim].reshape(dim, dim)
t = params[dim*dim:]
print(f"\nM (from params):\n{M}")
print(f"t: {t}")

# Verify: apply the transform manually to the center of the image
# ITK AffineTransform: output = M * input + t
# For the fixed image center:
center = np.array(fi.origin) + np.array(fi.spacing) * (np.array(fi.shape) - 1) / 2
print(f"\nFixed center: {center}")
print(f"M @ center + t = {M @ center + t}")

# Now check: the fwd transform maps FROM reference(fixed) space TO moving space
# So M @ fixed_point + t = corresponding_point_in_moving_space
# apply_transforms uses this to sample the moving image at M @ x + t for each reference point x
