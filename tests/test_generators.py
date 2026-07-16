import pytest
import torch
import torch.nn.functional as F
import numpy as np
import ants
from syntx.generators import CrossProductGenerator, temp_seed


def test_cross_product_combinations():
    """
    Assert that the generative pipeline outputs a cross-product of the specified
    6 intensity and 4 shape changes.
    """
    generator = CrossProductGenerator()
    
    intensities = generator.intensity_types
    shapes = generator.shape_types
    
    assert len(intensities) == 6
    assert len(shapes) == 4
    
    # Check all 6 x 4 = 24 combinations
    count = 0
    for intensity in intensities:
        for shape in shapes:
            fixed_img, moving_img, disp_field, magnitude = generator.generate(
                intensity_type=intensity,
                shape_type=shape,
                seed=42 + count
            )
            
            # Assert correct shapes
            assert fixed_img.ndim == 4
            assert fixed_img.shape[0] == 1
            assert fixed_img.shape[1] == 1
            assert fixed_img.shape[2:] == (64, 64)
            
            assert moving_img.shape == fixed_img.shape
            assert disp_field.shape == (1, 64, 64, 2)
            assert isinstance(magnitude, float)
            
            count += 1
            
    assert count == 24


def test_spatial_overlap_constraint():
    """
    Assert that every generated pair maintains >= 80% spatial overlap.
    We verify this by checking the Dice coefficient of the warped foreground mask.
    """
    # Use custom spacing/direction to test flexibility
    spacing = (0.8, 1.2)
    direction = [[0.96, -0.28], [0.28, 0.96]]
    generator = CrossProductGenerator(spacing=spacing, direction=direction)
    
    base_mask = generator.base_tensor > 0.05
    H, W = generator.base_tensor.shape[-2:]
    
    # We will sample combinations of all shape changes to check that they satisfy >= 80% overlap
    shapes = generator.shape_types
    intensities = generator.intensity_types
    
    # Check shape change only (without intensity changes)
    for shape in shapes:
        fixed_img, moving_img, disp_field, magnitude = generator.generate(
            intensity_type=None,
            shape_type=shape,
            seed=123
        )
        
        # Warp the base mask with the ground-truth displacement field to check spatial overlap
        y = torch.linspace(-1, 1, H, device=disp_field.device, dtype=disp_field.dtype)
        x = torch.linspace(-1, 1, W, device=disp_field.device, dtype=disp_field.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        identity = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        grid = identity + disp_field
        warped_mask = F.grid_sample(
            base_mask.float(),
            grid,
            mode='nearest',
            padding_mode='border',
            align_corners=True
        ) > 0.5
        
        intersection = torch.sum(base_mask & warped_mask).float()
        total = torch.sum(base_mask).float() + torch.sum(warped_mask).float()
        dice = (2.0 * intersection / total).item()
        
        assert dice >= 0.80, f"Shape transformation '{shape}' failed overlap test with Dice = {dice:.4f}"
        
    # Check all 24 combinations with intensity changes to ensure no errors
    for intensity in intensities:
        for shape in shapes:
            fixed_img, moving_img, disp_field, magnitude = generator.generate(
                intensity_type=intensity,
                shape_type=shape,
                seed=999
            )
            
            # Check overlap on the underlying warp
            grid = identity + disp_field
            warped_mask = F.grid_sample(
                base_mask.float(),
                grid,
                mode='nearest',
                padding_mode='border',
                align_corners=True
            ) > 0.5
            
            intersection = torch.sum(base_mask & warped_mask).float()
            total = torch.sum(base_mask).float() + torch.sum(warped_mask).float()
            dice = (2.0 * intersection / total).item()
            
            assert dice >= 0.80, f"Combination ({intensity}, {shape}) failed overlap test with Dice = {dice:.4f}"


def test_ground_truth_magnitudes():
    """
    Assert that ground truth magnitudes (physical L2 norm of the displacement field)
    are explicitly returned and match the expected physical calculations.
    """
    spacing = (1.5, 0.75)
    direction = [[1.0, 0.0], [0.0, 1.0]]
    generator = CrossProductGenerator(spacing=spacing, direction=direction)
    
    # 1. Translation: magnitude should be exactly solvable analytically
    # Let's override the translation generation by applying a known translation
    fixed_img, moving_img, disp_field, magnitude = generator.generate(
        intensity_type=None,
        shape_type='translation',
        seed=100
    )
    
    # Verify magnitude calculations manually
    H, W = disp_field.shape[1:3]
    # Extract translation components
    tx = disp_field[0, 0, 0, 0].item()
    ty = disp_field[0, 0, 0, 1].item()
    
    # Calculate physical displacement:
    # u_vox_x = tx * (W - 1) / 2
    # u_vox_y = ty * (H - 1) / 2
    # u_phys_x = u_vox_x * spacing[0]
    # u_phys_y = u_vox_y * spacing[1]
    # sum(||u_phys||^2) = H * W * (u_phys_x^2 + u_phys_y^2)
    # L2 = sqrt( delta_V * sum(||u_phys||^2) )
    # delta_V = spacing[0] * spacing[1]
    u_vox_x = tx * (W - 1) / 2.0
    u_vox_y = ty * (H - 1) / 2.0
    u_phys_x = u_vox_x * spacing[0]
    u_phys_y = u_vox_y * spacing[1]
    
    expected_sq = H * W * (u_phys_x**2 + u_phys_y**2)
    expected_mag = np.sqrt(spacing[0] * spacing[1] * expected_sq)
    
    assert np.isclose(magnitude, expected_mag, rtol=1e-5)
    
    # 2. Identity (None, None): magnitude should be 0.0
    _, _, disp_none, mag_none = generator.generate(None, None)
    assert mag_none == 0.0
    assert torch.all(disp_none == 0.0)


def test_ants_image_support():
    """
    Test that the generator can receive an ANTsImage as input and
    correctly sets spacing and direction metadata.
    """
    vol = np.zeros((32, 32), dtype=np.float32)
    y, x = np.ogrid[:32, :32]
    vol[(x-16)**2 + (y-16)**2 < 8**2] = 1.0
    ants_img = ants.from_numpy(vol, spacing=(0.5, 0.5), origin=(10.0, -5.0))
    
    generator = CrossProductGenerator(base_image=ants_img)
    
    assert generator.spacing == (0.5, 0.5)
    assert np.allclose(generator.base_origin, (10.0, -5.0))
    
    fixed_img, moving_img, disp_field, magnitude = generator.generate('noise', 'translation', seed=1)
    
    assert fixed_img.shape[-2:] == (32, 32)
    
    # Check converting back to ants image
    ants_fixed = generator.to_ants_image(fixed_img)
    assert isinstance(ants_fixed, ants.ANTsImage)
    assert ants_fixed.spacing == (0.5, 0.5)
    assert np.allclose(ants_fixed.origin, (10.0, -5.0))
