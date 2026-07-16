import os
import ants

base_path = '/Users/stnava/data/mindboggle/volumes'

# Fixed Image: OASIS-TRT-20-1
fixed_img_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
moving_img_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')

fi = ants.image_read(fixed_img_path)
mi = ants.image_read(moving_img_path)

print(f"fi shape: {fi.shape}, spacing: {fi.spacing}, origin: {fi.origin}")
print(f"mi shape: {mi.shape}, spacing: {mi.spacing}, origin: {mi.origin}")
