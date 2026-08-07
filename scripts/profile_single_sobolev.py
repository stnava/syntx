import time
import ants
import numpy as np
import syntx
from syntx.spatial import jacobian_determinant

def main():
    print("================================================================================")
    print("               SINGLE SOBOLEV SPEED & EVALUATION PROFILER                       ")
    print("================================================================================")

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

    print("[1/2] Running syntx.tvf (regularizer='sobolev', fast_smooth=False, optimizer='lars')...")
    t0 = time.time()
    reg = syntx.tvf(
        fixed=fi,
        moving=mi,
        type_of_transform='SyNTVF',
        regularizer='sobolev',
        fast_smooth=False,
        optimizer='lars',
        lr=0.60,
        flow_sigma=0.5,
        total_sigma=0.05,
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
    t1 = time.time()
    runtime = t1 - t0

    print(f"\n[2/2] Registration complete in {runtime:.2f} seconds.")

    fwd_tx = reg['fwdtransforms']
    inv_tx = reg['invtransforms']

    warped_l2 = ants.apply_transforms(fixed=fi, moving=l2_mi, transformlist=fwd_tx, interpolator='nearestNeighbor')
    warped_l3 = ants.apply_transforms(fixed=fi, moving=l3_mi, transformlist=fwd_tx, interpolator='nearestNeighbor')
    warped_l23 = ants.apply_transforms(fixed=fi, moving=l23_mi, transformlist=fwd_tx, interpolator='nearestNeighbor')

    d2_fixed = float(ants.label_overlap_measures(l2_fi, warped_l2)['TotalOrTargetOverlap'][0])
    d3_fixed = float(ants.label_overlap_measures(l3_fi, warped_l3)['TotalOrTargetOverlap'][0])
    d23_fixed = float(ants.label_overlap_measures(l23_fi, warped_l23)['TotalOrTargetOverlap'][0])

    inv_l2 = ants.apply_transforms(fixed=mi, moving=l2_fi, transformlist=inv_tx, interpolator='nearestNeighbor')
    inv_l3 = ants.apply_transforms(fixed=mi, moving=l3_fi, transformlist=inv_tx, interpolator='nearestNeighbor')
    inv_l23 = ants.apply_transforms(fixed=mi, moving=l23_fi, transformlist=inv_tx, interpolator='nearestNeighbor')

    d2_moving = float(ants.label_overlap_measures(l2_mi, inv_l2)['TotalOrTargetOverlap'][0])
    d3_moving = float(ants.label_overlap_measures(l3_mi, inv_l3)['TotalOrTargetOverlap'][0])
    d23_moving = float(ants.label_overlap_measures(l23_mi, inv_l23)['TotalOrTargetOverlap'][0])

    d2_sym = 0.5 * (d2_fixed + d2_moving)
    d3_sym = 0.5 * (d3_fixed + d3_moving)
    d23_sym = 0.5 * (d23_fixed + d23_moving)

    warp_img = ants.image_read(fwd_tx[0])
    jac = jacobian_determinant(warp_img, ref_image=fi)
    mask = ants.get_mask(fi).numpy() > 0
    jac_vals = jac[mask]
    folding = float(np.mean(jac_vals <= 0.0) * 100.0)
    min_j = float(jac_vals.min())

    print("================================================================================")
    print("                SINGLE SOBOLEV EVALUATION RESULTS                              ")
    print("================================================================================")
    print(f"Cortical Gray Matter (Class 2) Sym Dice : {d2_sym:.6f}")
    print(f"Deep Structures (Class 3) Sym Dice     : {d3_sym:.6f}")
    print(f"Parenchymal Brain (Class 2+3) Sym Dice : {d23_sym:.6f}")
    print(f"Grid Folding Percentage                 : {folding:.4f}%")
    print(f"Minimum Jacobian Determinant min(detJ)  : {min_j:.6f}")
    print(f"Total Execution Time                    : {runtime:.2f} seconds  (< 10s target)")
    print("================================================================================")

if __name__ == "__main__":
    main()
