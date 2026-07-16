import ants
import numpy as np
import sys
sys.path.insert(0, 'src')
import syntx

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

reg_iterations = [20, 0, 0]
reg_py = syntx.syn(fi, mi, 'SyN', backend='pytorch', reg_iterations=reg_iterations)

fwd_img = ants.image_read(reg_py['fwdtransforms'][0])
fwd_arr = fwd_img.numpy()
print("Max warp:", np.max(np.abs(fwd_arr)))
print("Mean warp:", np.mean(np.abs(fwd_arr)))
print("Warp std:", np.std(fwd_arr))
