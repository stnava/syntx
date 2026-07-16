import time
import ants
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import syntx

# Load preprocessed images
print("Loading images...")
fixed = ants.image_read('cache/T_template0_brain.nii.gz')
moving = ants.image_read('cache/28497-00000000-T1w-04_brain.nii.gz')

print("Running initial rigid alignment...")
init_tx = ants.registration(fixed=fixed, moving=moving, type_of_transform='Rigid')
tx_path = init_tx['fwdtransforms'][0]

print("Running VGG19 3D registration...")
t0 = time.time()
res = syntx.syn(
    fixed=fixed,
    moving=moving,
    type_of_transform='SyNTo',
    backend='pytorch',
    syn_metric='vgg19',
    syn_sampling=4,
    levels=[8, 4, 2, 1],
    affine_iterations=[10, 5, 3, 2],
    reg_iterations=[10, 5, 3, 2],
    initial_transform=tx_path,
    vgg_mode='lncc_3d',
    vgg_layers=[4]
)
t1 = time.time()
print(f"Registration finished in {t1 - t0:.2f} seconds.")
