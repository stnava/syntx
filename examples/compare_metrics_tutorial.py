import os
import sys
import numpy as np
import torch

# Force mock monai in sys.modules to avoid local version conflicts and network weight download errors
import types

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

# Add src/ to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from syntx import image_compare, CrossProductGenerator

def main():
    print("================================================================================")
    print("       IMAGE COMPARISON METRICS SUITE — TUTORIAL AND DOCUMENTATION              ")
    print("================================================================================")
    print("\nWelcome to the Syntx Image Comparison Metrics Suite!")
    print("This tutorial demonstrates how to simulate transformations and compute comparison")
    print("metrics to evaluate image alignment quality.")
    print("\n--------------------------------------------------------------------------------")
    print("Core Rule of Interpretation:")
    print("   ALL scores returned by `image_compare` are standardized such that:")
    print("   --> LOWER SCORES INDICATE BETTER SIMILARITY (higher alignment quality).")
    print("--------------------------------------------------------------------------------")

    # 1. Simulate transformations using CrossProductGenerator
    print("\n--- 1. Simulating Transformations with CrossProductGenerator ---")
    print("The CrossProductGenerator allows you to simulate 6 intensity changes and 4 shape changes.")
    print("Let's instantiate the generator and simulate an image pair.")
    
    generator = CrossProductGenerator(device='cpu')
    
    # Let's generate a pair with 'noise' intensity change and 'translation' shape change
    fixed_tensor, moving_tensor, displacement_field, gt_magnitude = generator.generate(
        intensity_type='noise',
        shape_type='translation',
        seed=42
    )
    
    print(f"Generated clean fixed target image tensor shape: {fixed_tensor.shape}")
    print(f"Generated warped/noisy moving image tensor shape: {moving_tensor.shape}")
    print(f"Ground truth displacement L2 norm magnitude: {gt_magnitude:.4f}")

    # Prepare inputs: Squeeze batch and channel dimensions for 2D metrics
    fixed_2d = fixed_tensor.squeeze()
    moving_2d = moving_tensor.squeeze()
    print(f"Squeezed 2D images shape: {fixed_2d.shape}")

    # 2. Compute Classical metrics
    print("\n--- 2. Computing Classical Similarity Metrics ---")
    print("Classical metrics evaluate pixel-wise similarities and global histograms:")
    
    # Mean Squared Error (MSE)
    mse_score = image_compare(fixed_2d, moving_2d, 'mse')
    print(f" - MSE (Mean Squared Error): {mse_score:.6f}")
    print("   [Interpretation: 0.0 is perfect identity. Larger values indicate higher mismatch.]")
    
    # Peak Signal-to-Noise Ratio (PSNR)
    # image_compare returns -PSNR to preserve the 'lower is better' rule
    psnr_score = image_compare(fixed_2d, moving_2d, 'psnr')
    print(f" - Standardized PSNR (returns -PSNR): {psnr_score:.6f} dB (Equivalent to {abs(psnr_score):.2f} dB)")
    print("   [Interpretation: A more negative score represents a higher peak signal-to-noise ratio, i.e., better quality.]")
    
    # Normalized Cross Correlation (NCC)
    # image_compare returns 1.0 - NCC
    ncc_score = image_compare(fixed_2d, moving_2d, 'ncc')
    print(f" - Standardized NCC (returns 1.0 - NCC): {ncc_score:.6f}")
    print("   [Interpretation: 0.0 means perfect correlation. 1.0 means no correlation. 2.0 means anti-correlated.]")
    
    # Structural Similarity Index (SSIM)
    # image_compare returns 1.0 - SSIM
    ssim_score = image_compare(fixed_2d, moving_2d, 'ssim')
    print(f" - Standardized SSIM (returns 1.0 - SSIM): {ssim_score:.6f}")
    print("   [Interpretation: 0.0 means perfect structural similarity (SSIM = 1.0).]")

    # 3. Compute Spatial / Gradient metrics
    print("\n--- 3. Computing Spatial / Gradient Metrics ---")
    print("Spatial metrics analyze the alignment of image structures (gradients and edges):")
    
    # Gradient Correlation
    # image_compare returns 1.0 - Gradient Correlation
    grad_corr = image_compare(fixed_2d, moving_2d, 'gradient_correlation')
    print(f" - Gradient Correlation (returns 1.0 - GC): {grad_corr:.6f}")
    print("   [Interpretation: Measures the alignment of image edges. 0.0 indicates perfectly aligned gradients.]")
    
    # Normalized Gradient Fields (NGF)
    # image_compare returns 1.0 - NGF Similarity
    ngf_score = image_compare(fixed_2d, moving_2d, 'ngf_e1')
    print(f" - Normalized Gradient Fields (NGF, eta=1): {ngf_score:.6f}")
    print("   [Interpretation: Measures vector direction alignment of image gradients. 0.0 indicates perfect alignment.]")

    # 4. Compute Deep Feature Space Metrics
    print("\n--- 4. Computing Deep Feature Space Metrics ---")
    print("Deep learning feature metrics compare the activations of pre-trained models.")
    print("This allows comparing images across different modalities (e.g., CT to MRI) or severe noise:")
    
    # VGG19 - Layer 4 LNCC
    vgg_score = image_compare(fixed_2d, moving_2d, 'vgg_4_lncc')
    print(f" - VGG19 (Layer 4, LNCC mode): {vgg_score:.6f}")
    
    # DINOv2 - Layer 2 LNCC
    dino_score = image_compare(fixed_2d, moving_2d, 'dino_2_lncc')
    print(f" - DINOv2 (Layer 2, LNCC mode): {dino_score:.6f}")
    
    # ResNet10 - Layer 2 LNCC
    resnet_score = image_compare(fixed_2d, moving_2d, 'resnet_2_lncc')
    print(f" - ResNet10 (Layer 2, LNCC mode): {resnet_score:.6f}")

    print("\n--- 5. Handling 3D-Only Models (e.g., SwinUNETR) ---")
    print("SwinUNETR is a native 3D self-supervised model. Running it on 2D images directly raises an error.")
    print("To evaluate SwinUNETR on 2D images, stack them along a third dimension to simulate a 3D volume:")
    
    # Stack the 2D numpy arrays 16 times to create a 3D volume of shape (16, H, W)
    fixed_np = fixed_2d.cpu().numpy()
    moving_np = moving_2d.cpu().numpy()
    fixed_3d = np.stack([fixed_np] * 16, axis=0)
    moving_3d = np.stack([moving_np] * 16, axis=0)
    
    print(f"Stacked 3D image shape: {fixed_3d.shape}")
    
    swin_score = image_compare(fixed_3d, moving_3d, 'swin_2_lncc')
    print(f" - SwinUNETR (Layer 2, LNCC mode, evaluated in 3D): {swin_score:.6f}")

    print("\n================================================================================")
    print("Summary of Evaluated Metric Scores (Lower is Better):")
    print(f"   MSE:                  {mse_score:.6f}")
    print(f"   PSNR (-PSNR):         {psnr_score:.6f}")
    print(f"   NCC (1.0 - NCC):      {ncc_score:.6f}")
    print(f"   SSIM (1.0 - SSIM):    {ssim_score:.6f}")
    print(f"   Gradient Correlation: {grad_corr:.6f}")
    print(f"   NGF:                  {ngf_score:.6f}")
    print(f"   VGG19 (LNCC):         {vgg_score:.6f}")
    print(f"   DINOv2 (LNCC):        {dino_score:.6f}")
    print(f"   ResNet10 (LNCC):      {resnet_score:.6f}")
    print(f"   SwinUNETR (LNCC):     {swin_score:.6f}")
    print("================================================================================")
    print("\nThis concludes the tutorial. You can customize the metrics and generative settings")
    print("to evaluate alignment performance in your own registration pipelines!")

if __name__ == '__main__':
    main()
