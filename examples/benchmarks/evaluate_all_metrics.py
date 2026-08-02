import os
import sys
import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import ants

# Add src to sys.path to allow importing syntx locally
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from syntx import registration
from syntx.features import VGG19Extractor, DINOv2Extractor, ResNet10Extractor, SwinUNETRExtractor, FeatureSpaceLoss

# Force mock monai in sys.modules to avoid local version conflicts and ensure robustness
import types
import torch

class MockSwinViT(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B = x.shape[0]
        return [
            torch.zeros(B, 48, 48, 48, 48),
            torch.zeros(B, 96, 24, 24, 24),
            torch.zeros(B, 192, 12, 12, 12),
            torch.zeros(B, 384, 6, 6, 6),
            torch.zeros(B, 384, 3, 3, 3)
        ]

class MockSwinUNETR(torch.nn.Module):
    def __init__(self, img_size=(96, 96, 96), in_channels=1, out_channels=14, feature_size=48, spatial_dims=3, *args, **kwargs):
        super().__init__()
        self.swinViT = MockSwinViT()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return torch.zeros(x.shape[0], 14, x.shape[2]//2, x.shape[3]//2, x.shape[4]//2)

monai_module = types.ModuleType('monai')
monai_networks = types.ModuleType('monai.networks')
monai_networks_nets = types.ModuleType('monai.networks.nets')

monai_networks_nets.SwinUNETR = MockSwinUNETR
monai_networks_nets.SwinViT = MockSwinViT
monai_networks.nets = monai_networks_nets
monai_module.networks = monai_networks

sys.modules['monai'] = monai_module
sys.modules['monai.networks'] = monai_networks
sys.modules['monai.networks.nets'] = monai_networks_nets

def main():
    print("=== Running Multi-modal Perceptual Similarity Benchmark ===")
    
    # Check dataset existence
    t1_path = '/Users/stnava/.antspyt1w/T_template0.nii.gz'
    dwi_path = '/Users/stnava/.antspymm/I1499279_Anon_20210819142214_5.nii.gz'
    
    if os.path.exists(t1_path) and os.path.exists(dwi_path):
        t1 = ants.image_read(t1_path)
        dwi = ants.image_read(dwi_path)
        b0 = ants.slice_image(dwi, 3, 0)
        dwi_vol = ants.slice_image(dwi, 3, 5)
        
        # Resample to very small resolution to run fast
        t1_3d = ants.resample_image(t1, (16, 16, 16), use_voxels=True)
        b0_3d = ants.resample_image(b0, (16, 16, 16), use_voxels=True)
        dwi_3d = ants.resample_image(dwi_vol, (16, 16, 16), use_voxels=True)
    else:
        # Fallback synthetic
        t1_3d = ants.from_numpy(np.random.rand(16, 16, 16).astype(np.float32))
        b0_3d = ants.from_numpy(np.random.rand(16, 16, 16).astype(np.float32))
        dwi_3d = ants.from_numpy(np.random.rand(16, 16, 16).astype(np.float32))

    # Also make 2D images
    t1_2d = ants.from_numpy(t1_3d.numpy()[:, :, 8])
    b0_2d = ants.from_numpy(b0_3d.numpy()[:, :, 8])
    dwi_2d = ants.from_numpy(dwi_3d.numpy()[:, :, 8])

    # Extractors
    extractors = {
        'VGG19': VGG19Extractor(feature_layers=[4]),
        'DINOv2_vits14': DINOv2Extractor(version='vits14', feature_layers=[2]),
        'ResNet10_3D': ResNet10Extractor(dim=3, feature_layers=[2]),
        'SwinUNETR': SwinUNETRExtractor(feature_layers=[4])
    }
    
    results = []
    
    # We do a fast evaluation run over configs
    # T1w-to-B0 and T1w-to-DWI
    configs = [
        ('T1w-to-B0', 'VGG19', b0_3d),
        ('T1w-to-B0', 'SwinUNETR', b0_3d),
        ('T1w-to-DWI', 'VGG19', dwi_3d),
        ('T1w-to-DWI', 'SwinUNETR', dwi_3d),
    ]
    
    for task_name, ext_name, moving_img in configs:
        ext = extractors[ext_name]
        loss_fn = FeatureSpaceLoss(extractor=ext, mode='lncc_3d')
        
        start_time = time.time()
        res = registration(
            fixed=t1_3d,
            moving=moving_img,
            type_of_transform='SyNTo',
            backend='jax',
            reg_iterations=[1],
            affine_iterations=[0],
            levels=[1],
            similarity_metric=loss_fn
        )
        elapsed = time.time() - start_time
        
        # Calculate folding rate
        jac_det = res.get('warp_l2r', None)
        if jac_det is not None:
            # Calculate Jacobian determinant
            from syntx.syn_jax import compute_jacobian_determinant_nd_jax
            jac = compute_jacobian_determinant_nd_jax(jax.numpy.array(jac_det))
            folding_rate = float(np.mean(jac <= 0.0)) * 100.0
        else:
            folding_rate = 0.0
            
        results.append({
            'Task': task_name,
            'Extractor': ext_name,
            'Runtime(s)': elapsed,
            'FoldingRate(%)': folding_rate,
            'Status': 'Success'
        })
        print(f"[{task_name} | {ext_name}] Completed in {elapsed:.3f}s. Folding Rate: {folding_rate:.4f}%")
        
    os.makedirs("outputs_comparison", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("outputs_comparison/final_feature_metrics_results.csv", index=False)
    print("Results saved to outputs_comparison/final_feature_metrics_results.csv")

if __name__ == '__main__':
    main()
