import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import ants

# Add src to sys.path to allow importing syntx locally
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import syntx

def compute_tissue_overlap(fi, warped):
    fixed_seg = ants.threshold_image(fi, 'Otsu', 3)
    warped_seg = ants.threshold_image(warped, 'Otsu', 3)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    dice = float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
    return dice

def main():
    print("Loading phantom images...")
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))
    
    # Exclude JAX warnings and limit outputs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # 8 Layer configs
    layer_configs = [
        [2],          # Conv1_2
        [7],          # Conv2_2
        [12],         # Conv3_2
        [21],         # Conv4_2
        [30],         # Conv5_2
        [2, 7],       # Conv1_2 & Conv2_2
        [7, 12],      # Conv2_2 & Conv3_2
        [12, 21]      # Conv3_2 & Conv4_2
    ]
    
    # 4 VGG metric configurations
    # (vgg_mode, vgg_lncc_window_size)
    metric_configs = [
        ('lncc', 5),
        ('lncc', 9),
        ('l1', None),
        ('mse', None)
    ]
    
    # 2 Flow sigma configs
    flow_sigmas = [1.0, 1.732]
    
    # Generate 64 parameter sets
    configs = []
    config_id = 1
    for layers in layer_configs:
        for mode, win_size in metric_configs:
            for sigma in flow_sigmas:
                configs.append({
                    'id': config_id,
                    'layers': layers,
                    'mode': mode,
                    'window_size': win_size,
                    'flow_sigma': sigma
                })
                config_id += 1
                
    print(f"Generated {len(configs)} configurations for systematic sweep.")
    
    output_dir = "/Users/stnava/data/syntx/reports"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "vgg_sweep_results.csv")
    
    results = []
    
    # Load existing results to allow resume capability if needed
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            results = df_existing.to_dict('records')
            completed_ids = set(df_existing['id'].values)
            print(f"Loaded {len(results)} existing results. Resuming sweep...")
        except Exception as e:
            print(f"Failed to read existing CSV: {e}. Starting fresh.")
            completed_ids = set()
    else:
        completed_ids = set()
        
    aff_its = [100, 50, 50, 10]
    reg_its = [50, 50, 50, 20]
    sampling_percent = 0.1
    cfl_voxels_setting = 0.70
    
    for conf in configs:
        cid = conf['id']
        if cid in completed_ids:
            continue
            
        layers_str = "-".join(map(str, conf['layers']))
        print(f"\n[{cid}/64] Running configuration: Layers={layers_str}, Mode={conf['mode']}, WinSize={conf['window_size']}, FlowSigma={conf['flow_sigma']}...")
        
        t0 = time.time()
        try:
            reg = syntx.syn(
                fixed=fi,
                moving=mi,
                reg_iterations=reg_its,
                affine_iterations=aff_its,
                grad_step=cfl_voxels_setting,
                flow_sigma=conf['flow_sigma'],
                syn_metric='vgg19',
                vgg_mode=conf['mode'],
                vgg_layers=conf['layers'],
                vgg_lncc_window_size=conf['window_size'] if conf['window_size'] is not None else 9,
                mattes_bins=32,
                sampling_percentage=sampling_percent,
                backend='pytorch',
                inverse_steps=5
            )
            t1 = time.time()
            time_taken = t1 - t0
            
            # Extract warped image
            warped = reg['warpedmovout']
            
            # Extract transforms & compute metrics
            l2r_path = next(tx for tx in reg['fwdtransforms'] if tx.endswith('.nii.gz'))
            
            mi_val = float(ants.image_mutual_information(fi, warped))
            dice_val = float(compute_tissue_overlap(fi, warped))
            
            jac_img = ants.create_jacobian_determinant_image(fi, l2r_path)
            jac_np = jac_img.numpy()
            folding_val = float(100.0 * np.mean(jac_np <= 0))
            min_jac_val = float(jac_np.min())
            
            # Clean up temp transform files from this run
            for path in reg['fwdtransforms'] + reg['invtransforms']:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            
            row = {
                'id': cid,
                'layers': str(conf['layers']),
                'mode': conf['mode'],
                'window_size': conf['window_size'] if conf['window_size'] is not None else -1,
                'flow_sigma': conf['flow_sigma'],
                'mi': mi_val,
                'dice': dice_val,
                'folding_pct': folding_val,
                'min_jacobian': min_jac_val,
                'time_seconds': time_taken,
                'status': 'SUCCESS'
            }
            print(f"  Result -> MI: {mi_val:.4f} | Dice: {dice_val:.4f} | Folding: {folding_val:.4f}% | MinJac: {min_jac_val:.4f} | Time: {time_taken:.2f}s")
            
        except Exception as e:
            print(f"  Configuration {cid} FAILED: {e}")
            row = {
                'id': cid,
                'layers': str(conf['layers']),
                'mode': conf['mode'],
                'window_size': conf['window_size'] if conf['window_size'] is not None else -1,
                'flow_sigma': conf['flow_sigma'],
                'mi': np.nan,
                'dice': np.nan,
                'folding_pct': np.nan,
                'min_jacobian': np.nan,
                'time_seconds': np.nan,
                'status': f'FAILED: {str(e)[:100]}'
            }
            
        results.append(row)
        
        # Save progress dynamically
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        
    print(f"\nSweep complete. Results saved to {csv_path}")
    
    # Sort and display top 5 configs by Dice score
    df_success = df[df['status'] == 'SUCCESS']
    if len(df_success) > 0:
        df_sorted = df_success.sort_values(by=['dice', 'folding_pct'], ascending=[False, True])
        print("\n================ TOP 5 CONFIGURATIONS ================ ")
        print(df_sorted.head(5).to_string(index=False))
    else:
        print("\nNo successful configurations found.")

if __name__ == "__main__":
    main()
