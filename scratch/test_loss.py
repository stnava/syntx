import torch
import sys
sys.path.insert(0, 'src')
import syntx
import ants

fi = ants.image_read(ants.get_ants_data('r16')).resample_image((64, 64), use_voxels=True)
mi = ants.image_read(ants.get_ants_data('r64')).resample_image((64, 64), use_voxels=True)

reg_py = syntx.syn(fi, mi, 'SyNOnly', backend='pytorch', reg_iterations=[10, 0, 0], affine_iterations=[0, 0, 0])
model = reg_py['model']
print("Losses:", model.syn_losses)
