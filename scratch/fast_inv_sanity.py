"""Sanity check: what does MI(mi, warped_fi) even mean for r16 vs r64?"""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Check if MI(mi, warped_fi) vs MI(fi, warped_mi) give same-ish values
ra = ants.registration(fi, mi, 'SyN', reg_iterations=[10,5,0], syn_metric='mattes')

# Forward: warped_mi in fi space
wf = ra['warpedmovout']
mi_fwd = ants.image_mutual_information(fi, wf)

# Check what ANTs returns for warpedmovout vs warpedmoving
# Also try the full inverse
wi = ants.apply_transforms(mi, fi, ra['invtransforms'])
mi_inv = ants.image_mutual_information(mi, wi)

print(f"MI(fi, warped_mi) = {mi_fwd:.4f}")  
print(f"MI(mi, warped_fi) = {mi_inv:.4f}")
print(f"MI(fi, mi) = {ants.image_mutual_information(fi, mi):.4f}")
print(f"MI(mi, fi) = {ants.image_mutual_information(mi, fi):.4f}")
print(f"\nANTs invtransforms: {ra['invtransforms']}")

# Check with warpedmovout inverse
# ANTs SyN stores the inverse warp field
# Let's manually check
# The inverse should map fi→mi: for each mi reference point, find corresponding fi point
# So warped_fi (in mi space) should look like mi
