import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import syntx

fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r64'))

# 1. ANTs SyN
reg_ants = ants.registration(fi, mi, 'SyN', reg_iterations=[50, 20, 0], syn_metric='cc', syn_sampling=2)
ants_fwd_img = ants.image_read(reg_ants['fwdtransforms'][0])
ants_fwd_np = ants_fwd_img.numpy()

# 2. Syntx SyN
reg_syn = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyN', backend='pytorch',
    initial_transform=reg_ants['fwdtransforms'][1:], # Init with ANTs Affine
    affine_iterations=[0], reg_iterations=[50, 20, 0],
    syn_metric='lncc', lncc_radius=2
)
syn_fwd_img = ants.image_read(reg_syn['fwdtransforms'][0])
syn_fwd_np = syn_fwd_img.numpy()

print("ANTs fwd shape:", ants_fwd_np.shape)
print("Syntx fwd shape:", syn_fwd_np.shape)

# Print a 5x5 subgrid of displacement values near the center (128, 128)
print("\nANTs center values:")
print(ants_fwd_np[126:131, 126:131, :])

print("\nSyntx center values:")
print(syn_fwd_np[126:131, 126:131, :])

# Clean up
for r in [reg_ants, reg_syn]:
    for path in r['fwdtransforms'] + r['invtransforms']:
        if os.path.exists(path):
            os.remove(path)
