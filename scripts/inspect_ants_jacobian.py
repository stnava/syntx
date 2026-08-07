import ants
import numpy as np

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

reg = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN')
fwd_tx = reg['fwdtransforms'][0]

j1 = ants.create_jacobian_determinant_image(fi, fwd_tx)
j2 = ants.create_jacobian_determinant_image(fi, fwd_tx, do_log=False)
j3 = ants.create_jacobian_determinant_image(fi, fwd_tx, do_log=True)

print("j1 (default) min/max:", j1.min(), j1.max())
print("j2 (do_log=False) min/max:", j2.min(), j2.max())
print("j3 (do_log=True) min/max:", j3.min(), j3.max())
