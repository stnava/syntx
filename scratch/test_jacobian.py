import ants
import numpy as np
from syntx import registration
import os

print("Loading phantoms...")
fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r64'))

for backend in ['pytorch', 'jax']:
    print(f"\n--- Running registration with {backend} backend ---")
    try:
        res = registration(
            fixed=fi,
            moving=mi,
            type_of_transform='SyNTo',
            backend=backend,
            levels=[2, 1],
            affine_iterations=[30, 20],
            reg_iterations=[30, 20],
            grad_step=0.5,
            flow_sigma=1.0
        )
        
        fwd_tx_files = res['fwdtransforms']
        fwd_warp_file = next((tx for tx in fwd_tx_files if tx.endswith('.nii.gz')), None)
        
        if fwd_warp_file is not None:
            jac_img = ants.create_jacobian_determinant_image(fi, fwd_warp_file)
            jac_np = jac_img.numpy()
            print(f"[{backend}] Min Jacobian:", jac_np.min())
            print(f"[{backend}] Max Jacobian:", jac_np.max())
            print(f"[{backend}] Mean Jacobian:", jac_np.mean())
            print(f"[{backend}] Folding rate (<=0):", np.mean(jac_np <= 0.0))
            
            # Clean up
            for tx in res['fwdtransforms'] + res['invtransforms']:
                if os.path.exists(tx):
                    os.remove(tx)
        else:
            print(f"[{backend}] Error: fwd_warp_file is None")
    except Exception as e:
        print(f"[{backend}] Exception caught:", e)
        import traceback
        traceback.print_exc()
