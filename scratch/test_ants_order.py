import ants
import numpy as np
import torch

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Run ANTs SyN
reg = ants.registration(fi, mi, 'SyN')
print(reg['fwdtransforms'])
