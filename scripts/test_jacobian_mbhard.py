import numpy as np
import ants
import syntx
from syntx.spatial import jacobian_determinant

def main():
    print("================================================================================")
    print("            3D JACOBIAN DETERMINANT VALIDATION (mbhard / Pair 00)               ")
    print("================================================================================")

    data = syntx.benchmark_data('mbhard')
    fi, mi = data['fixed'], data['moving']

    print("\n[1/2] Running fast 3D SyN registration (reg_iterations=[20, 0, 0])...")
    reg = syntx.syn(fixed=fi, moving=mi, reg_iterations=[20, 0, 0], verbose=True)
    fwd_tx = reg['fwdtransforms'][0]

    print("\n[2/2] Comparing 3D Jacobian map against ANTs C++ ITK reference...")
    # ANTs C++ ITK Reference
    jac_ants = ants.create_jacobian_determinant_image(fi, fwd_tx)
    jac_ants_np = jac_ants.numpy()

    # Syntx implementation
    warp_img = ants.image_read(fwd_tx)
    jac_syntx_np = jacobian_determinant(warp_img, ref_image=fi)

    mask = ants.get_mask(fi).numpy() > 0
    ref_vals = jac_ants_np[mask].ravel()
    syn_vals = jac_syntx_np[mask].ravel()

    r = np.corrcoef(ref_vals, syn_vals)[0, 1]
    diff = np.abs(ref_vals - syn_vals)

    print("================================================================================")
    print("                        3D MBHARD JACOBIAN RESULTS                              ")
    print("================================================================================")
    print(f"Pearson Correlation r        : {r:.6f}")
    print(f"Max Absolute Difference      : {diff.max():.6e}")
    print(f"Mean Absolute Difference     : {diff.mean():.6e}")
    print(f"95th %ile Absolute Diff      : {np.percentile(diff, 95):.6e}")
    print(f"ANTs Jacobian range          : [{ref_vals.min():.4f}, {ref_vals.max():.4f}]")
    print(f"Syntx Jacobian range         : [{syn_vals.min():.4f}, {syn_vals.max():.4f}]")
    print("================================================================================")

if __name__ == "__main__":
    main()
