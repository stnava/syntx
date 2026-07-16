"""Test: does ANTs auto-invert affine in apply_transforms?"""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

ra = ants.registration(fi, mi, 'SyN', reg_iterations=[10,5,0], syn_metric='mattes')
print(f"ANTs invtransforms: {ra['invtransforms']}")

# How does ANTs.registration's result know to invert?
# The key: ANTs SyN returns the SAME GenericAffine.mat file
# apply_transforms should be called WITHOUT whichtoinvert for this to work

wi1 = ants.apply_transforms(mi, fi, ra['invtransforms'])
print(f"No whichtoinvert: MI={ants.image_mutual_information(mi, wi1):.4f}")

wi2 = ants.apply_transforms(mi, fi, ra['invtransforms'], whichtoinvert=[False, False])
print(f"whichtoinvert=[F,F]: MI={ants.image_mutual_information(mi, wi2):.4f}")

wi3 = ants.apply_transforms(mi, fi, ra['invtransforms'], whichtoinvert=[True, False])
print(f"whichtoinvert=[T,F]: MI={ants.image_mutual_information(mi, wi3):.4f}")

# So does apply_transforms look at the transform TYPE and auto-invert?
# Or does the ANTs SyN wrapper handle it?
# Let's check: what about directly listing a forward .mat in invtransforms?
# For SyN, GenericAffine.mat is the FORWARD affine
# When listed in invtransforms, ANTs must auto-invert it somehow
