import sys
import os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np

sys.path.insert(0, 'src')
import syntx
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx_affine = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_affine = ants.apply_transforms(fi, mi, tx_affine['fwdtransforms'])

reg_pt = syntx.syn(fi, mi_affine, 'SyNTo', backend='pytorch', reg_iterations=[20, 0, 0], affine_iterations=[0, 0, 0], similarity_metric='mattes_mi', verbose=False, grad_step=0.1, fluid_sigma=np.sqrt(3.0))

import torch
import tempfile
model = reg_pt['model']

# Re-save without inversion
disp_l2r = model.warp_l2r.data.cpu().numpy()[0]
disp_l2r_t = disp_l2r[..., ::-1].copy()  # Just swap channels (z, y, x) -> (x, y, z)

fwd_file = tempfile.NamedTemporaryFile(suffix='_fwd.nii.gz', delete=False).name
ants.image_write(ants.from_numpy(disp_l2r_t, has_components=True, spacing=fi.spacing, origin=fi.origin, direction=fi.direction), fwd_file)

warped_ants = ants.apply_transforms(fi, mi_affine, [fwd_file, reg_pt['fwdtransforms'][1]])
print("MI with NO INVERSION:", ants.image_mutual_information(fi, warped_ants))

