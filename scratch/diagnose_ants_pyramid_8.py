import ants

fi = ants.image_read(ants.get_ants_data('r16')).resample_image((64, 64), use_voxels=True)
mi = ants.image_read(ants.get_ants_data('r64')).resample_image((64, 64), use_voxels=True)

print("Running ANTs Registration with [8, 4, 2, 1]...")
reg = ants.registration(
    fixed=fi, moving=mi, type_of_transform='SyNOnly',
    reg_iterations=(10, 10, 10, 10), verbose=True
)
