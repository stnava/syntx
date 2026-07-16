import ants
import sys

fi = ants.image_read(ants.get_ants_data('r16')).resample_image((64, 64), use_voxels=True)
mi = ants.image_read(ants.get_ants_data('r64')).resample_image((64, 64), use_voxels=True)

# Run with verbose=True to print ANTs internal parameter resolution
print("Running ANTs Registration with verbose=True...")
reg = ants.registration(
    fixed=fi, moving=mi, type_of_transform='SyNOnly',
    reg_iterations=(40, 20, 0), verbose=True
)
