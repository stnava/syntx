import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import ants
import pandas as pd

# Add src to sys.path to allow importing syntx locally
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from syntx import registration
from examples.compare_metrics import compute_multiregion_dice

def main():
    print("=== Starting VGG Feature Space Hyperparameter Sweep ===")
    
    # Load cached images
    img1_brain = ants.image_read('cache/img1_brain.nii.gz')
    img2_brain = ants.image_read('cache/28497-00000000-T1w-04_brain.nii.gz')
    dktseg1 = ants.image_read('cache/dktseg1.nii.gz')
    dktseg2 = ants.image_read('cache/28497-00000000-T1w-04_dktseg.nii.gz')

    # Initial pre-alignment to establish common frame
    print("Running initial ANTs Rigid registration for common frame alignment...")
    init_tx = ants.registration(fixed=img1_brain, moving=img2_brain, type_of_transform='Rigid')
    tx_path = init_tx['fwdtransforms'][0]

    # Evaluate Baseline (unregistered but resampled to fixed space)
    dktseg2_resampled = ants.apply_transforms(
        fixed=dktseg1,
        moving=dktseg2,
        transformlist=[],
        interpolator='nearestNeighbor'
    )
    base_dices = compute_multiregion_dice(dktseg1, dktseg2_resampled)
    base_mean = np.mean(list(base_dices.values()))
    print(f"Baseline Mean DICE: {base_mean:.4f}")

    # Set up parameter grid sweep
    sweep_configs = {
        'mattes_mi': {
            'syn_metric': 'mattes_mi',
            'syn_sampling': 32,
            'kwargs': {}
        },
        'lncc_5x5x5': {
            'syn_metric': 'lncc',
            'syn_sampling': 2,
            'kwargs': {}
        },
        'vgg_lncc_layer3': {
            'syn_metric': 'vgg19',
            'syn_sampling': 4,
            'kwargs': {'vgg_mode': 'lncc', 'vgg_layers': [3]}
        },
        'vgg_lncc_layer4': {
            'syn_metric': 'vgg19',
            'syn_sampling': 4,
            'kwargs': {'vgg_mode': 'lncc', 'vgg_layers': [4]}
        },
        'vgg_lncc_layer8': {
            'syn_metric': 'vgg19',
            'syn_sampling': 4,
            'kwargs': {'vgg_mode': 'lncc', 'vgg_layers': [8]}
        },
        'vgg_lncc_multiscale': {
            'syn_metric': 'vgg19',
            'syn_sampling': 4,
            'kwargs': {'vgg_mode': 'lncc', 'vgg_layers': [4, 8]}
        },
        'vgg_lncc3d_layer4': {
            'syn_metric': 'vgg19',
            'syn_sampling': 4,
            'kwargs': {'vgg_mode': 'lncc_3d', 'vgg_layers': [4]}
        },
        'vgg_lncc3d_layer8': {
            'syn_metric': 'vgg19',
            'syn_sampling': 4,
            'kwargs': {'vgg_mode': 'lncc_3d', 'vgg_layers': [8]}
        }
    }

    results = []
    
    for name, conf in sweep_configs.items():
        print(f"\n-----------------------------------------")
        print(f" Sweep Config: {name}")
        print(f"-----------------------------------------")
        
        start_time = time.time()
        res = registration(
            fixed=img1_brain,
            moving=img2_brain,
            type_of_transform='SyNTo',
            backend='pytorch',
            syn_metric=conf['syn_metric'],
            syn_sampling=conf['syn_sampling'],
            levels=[8, 4, 2],
            affine_iterations=[30, 20, 10],
            reg_iterations=[30, 20, 10],
            initial_transform=tx_path,
            verbose=False,
            **conf['kwargs']
        )
        runtime = time.time() - start_time
        
        # Warp DKT and compute DICE
        warped_dkt = ants.apply_transforms(
            fixed=img1_brain,
            moving=dktseg2,
            transformlist=res['fwdtransforms'],
            interpolator='nearestNeighbor'
        )
        dices = compute_multiregion_dice(dktseg1, warped_dkt)
        mean_dice = np.mean(list(dices.values()))
        
        # Compute folding rate from Jacobian
        composite_warp_path = f"outputs_comparison/{name}_CompositeWarp.nii.gz"
        ants.image_write(ants.image_read(res['fwdtransforms'][0]), composite_warp_path)
        jac_img = ants.create_jacobian_determinant_image(img1_brain, composite_warp_path, do_log=False)
        folds = np.sum(jac_img.numpy() <= 0) / jac_img.numpy().size * 100
        
        print(f"Result for {name}: Mean DICE = {mean_dice:.4f}, Folds = {folds:.4f}%, Time = {runtime:.2f}s")
        
        results.append({
            'Config': name,
            'Metric': conf['syn_metric'],
            'Layers': str(conf['kwargs'].get('vgg_layers', 'N/A')),
            'Mode': conf['kwargs'].get('vgg_mode', 'N/A'),
            'Mean DICE': mean_dice,
            'Folds': folds,
            'Time': runtime
        })

    # Output markdown summary
    df = pd.DataFrame(results)
    df = df.sort_values(by='Mean DICE', ascending=False)
    print("\n=== SWEEP RESULTS ===")
    print(df.to_markdown(index=False))

    # Write results to a file
    df.to_csv("outputs_comparison/vgg_sweep_results.csv", index=False)

if __name__ == "__main__":
    main()
