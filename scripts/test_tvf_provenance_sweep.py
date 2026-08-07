import time
import ants
import numpy as np
import syntx
from syntx.spatial import jacobian_determinant

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

otsu_fi = ants.threshold_image(fi, 'Otsu', 3)
otsu_mi = ants.threshold_image(mi, 'Otsu', 3)
l2_fi = otsu_fi.threshold_image(2, 2)
l2_mi = otsu_mi.threshold_image(2, 2)
l3_fi = otsu_fi.threshold_image(3, 3)
l3_mi = otsu_mi.threshold_image(3, 3)
l23_fi = otsu_fi.threshold_image(2, 3)
l23_mi = otsu_mi.threshold_image(2, 3)

for fsig in [0.5, 1.0, 1.5]:
    for gstep in [0.45, 0.60]:
        for tsig in [0.02, 0.05]:
            reg = syntx.tvf(
                fixed=fi,
                moving=mi,
                type_of_transform='SyNTVF',
                regularizer='dsti',
                flow_sigma=fsig,
                total_sigma=tsig,
                grad_step=gstep,
                cfl_momentum=0.95,
                n_time_steps=3,
                use_analytical_gradients=True,
                antisymmetric=False,
                constant_speed=True,
                constant_speed_relaxation=0.1,
                cfl_max=None,
                solver='euler',
                integration_steps_per_interval=3,
                multipoint_loss=[0.0, 0.5, 1.0],
                reg_iterations=[100, 100, 20],
                tol=1e-9,
                verbose=False
            )
            fwd_tx = reg['fwdtransforms']
            inv_tx = reg['invtransforms']

            warped_l2 = ants.apply_transforms(fixed=fi, moving=l2_mi, transformlist=fwd_tx, interpolator='nearestNeighbor')
            inv_l2 = ants.apply_transforms(fixed=mi, moving=l2_fi, transformlist=inv_tx, interpolator='nearestNeighbor')
            d2_fixed = float(ants.label_overlap_measures(l2_fi, warped_l2)['TotalOrTargetOverlap'][0])
            d2_moving = float(ants.label_overlap_measures(l2_mi, inv_l2)['TotalOrTargetOverlap'][0])
            d2_sym = 0.5 * (d2_fixed + d2_moving)

            warp_img = ants.image_read(fwd_tx[0])
            jac = jacobian_determinant(warp_img, ref_image=fi)
            mask = ants.get_mask(fi).numpy() > 0
            jac_vals = jac[mask]
            folding = float(np.mean(jac_vals <= 0.0) * 100.0)
            min_j = float(jac_vals.min())

            print(f"fsig={fsig:.1f}, gstep={gstep:.2f}, tsig={tsig:.2f} -> Class 2 Sym Dice: {d2_sym:.6f} | Folding: {folding:.4f}% | min(detJ): {min_j:.4f}")
