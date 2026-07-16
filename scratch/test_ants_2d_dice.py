import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import ants
import numpy as np

fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r64'))

# Let's run ANTs Affine only and check Dice
reg_affine = ants.registration(fi, mi, 'Affine')
dice_affine = ants.label_overlap_measures(
    ants.threshold_image(fi, 'Otsu', 3),
    ants.threshold_image(reg_affine['warpedmovout'], 'Otsu', 3)
)
print("ANTs Affine-only Dice:", dice_affine.loc[dice_affine['Label'] == 'All', 'MeanOverlap'].values[0])

# Let's run ANTs SyN and check Dice
reg_syn = ants.registration(fi, mi, 'SyN', reg_iterations=[100, 100, 100, 50], syn_metric='cc', syn_sampling=2)
dice_syn = ants.label_overlap_measures(
    ants.threshold_image(fi, 'Otsu', 3),
    ants.threshold_image(reg_syn['warpedmovout'], 'Otsu', 3)
)
print("ANTs SyN Dice:", dice_syn.loc[dice_syn['Label'] == 'All', 'MeanOverlap'].values[0])
