import ants
import numpy as np
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from syntx.syn import registration

def main():
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))
    
    reg_py = registration(
        fixed=fi,
        moving=mi,
        type_of_transform='SyNTo',
        backend='pytorch',
        levels=[4, 2, 1],
        affine_iterations=[0],  # No affine pre-alignment, focus on SyN
        reg_iterations=[50, 50, 50],
        grad_step=0.5,
        flow_sigma=1.732,
        total_sigma=0.0,
        syn_metric='lncc',
        syn_sampling=4
    )
    
    print("\n--- SyN Losses per Epoch ---")
    losses = reg_py['syn_losses']
    print(f"Total epochs run: {len(losses)}")
    print(f"First 10 losses: {[f'{l:.4f}' for l in losses[:10]]}")
    print(f"Last 10 losses: {[f'{l:.4f}' for l in losses[-10:]]}")
    
    # Read the generated warp field
    l2r_path = next(tx for tx in reg_py['fwdtransforms'] if tx.endswith('.nii.gz'))
    warp = ants.image_read(l2r_path).numpy()
    warp_magnitude = np.sqrt(np.sum(warp**2, axis=-1))
    print(f"\nWarp Field Max Magnitude (voxels/mm): {warp_magnitude.max():.4f}")
    print(f"Warp Field Mean Magnitude: {warp_magnitude.mean():.4f}")
    
    # Clean up temp files
    for path in reg_py['fwdtransforms'] + reg_py['invtransforms']:
        if os.path.exists(path):
            os.remove(path)

if __name__ == '__main__':
    main()
