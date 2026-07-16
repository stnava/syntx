# Handoff Report: 3D Registration Parity between `syntx` and `ants.registration`

## 1. Observation

In the read-only investigation of `src/syntx/syn.py` and `src/syntx/syn_jax.py`, the following specifics were observed regarding the SyN optimization loops, coordinate-mapping logic, and transform compositions:

### A. Current Optimization Loop and Midpoint Domain Warping
In `src/syntx/syn.py` (lines 1240–1245):
```python
                    # Single Interpolation: compose affine with SyN displacement
                    # so J_curr (original moving) is sampled only once
                    phi_r2l_composed = compose_grids(affine_grid_level, phi_r2l)
                    
                    # Real SyN: Pull both images to the midpoint domain
                    I_mid = F.grid_sample(I_curr, phi_l2r, padding_mode='border', align_corners=True)
                    J_mid = F.grid_sample(J_curr, phi_r2l_composed, padding_mode='border', align_corners=True)
```
Where `phi_l2r` and `phi_r2l` represent warp fields optimized in normalized `[-1, 1]` coordinates:
```python
            warp_l2r = torch.zeros(1, *curr_spatial, dim, device=device, dtype=dtype)
            warp_r2l = torch.zeros(1, *curr_spatial, dim, device=device, dtype=dtype)
```

### B. Displacement Field Updates, Scaling, and Regularization (CFL)
In the PyTorch CFL optimizer branch (lines 1300–1340):
```python
                        voxel_scale = torch.tensor(
                            [float((s - 1) / 2.0) for s in reversed(curr_spatial)],
                            device=device, dtype=dtype
                        )
                        grad_l_voxel = grad_l * voxel_scale
                        grad_r_voxel = grad_r * voxel_scale
                        
                        max_norm_l = torch.sqrt(torch.sum(grad_l_voxel**2, dim=-1)).max() + 1e-8
                        max_norm_r = torch.sqrt(torch.sum(grad_r_voxel**2, dim=-1)).max() + 1e-8
                        
                        # Uniform scalar learning rate: max voxel displacement = cfl_voxels
                        lr_l = cfl_voxels / max_norm_l
                        lr_r = cfl_voxels / max_norm_r
                        
                        # Scale the normalized-space gradient uniformly (preserves direction)
                        delta_l = lr_l * grad_l
                        delta_r = lr_r * grad_r
                        
                        # Greedy SyN composition: φ_new = φ_old ∘ (Id - ∂loss/∂warp)
                        coords_l = identity - delta_l
                        coords_r = identity - delta_r
                        
                        warp_l2r_sampled = F.grid_sample(torch.movedim(warp_l2r, -1, 1), coords_l, padding_mode='border', align_corners=True)
                        warp_l2r_sampled = torch.movedim(warp_l2r_sampled, 1, -1)
                        warp_r2l_sampled = F.grid_sample(torch.movedim(warp_r2l, -1, 1), coords_r, padding_mode='border', align_corners=True)
                        warp_r2l_sampled = torch.movedim(warp_r2l_sampled, 1, -1)
                        
                        warp_l2r.copy_(warp_l2r_sampled - delta_l)
                        warp_r2l.copy_(warp_r2l_sampled - delta_r)
```
Followed by Dirichlet zero-boundary condition enforcement, Gaussian elastic smoothing, and inverse projections (lines 1464–1477):
```python
                    # ITK-standard Dirichlet zero boundary enforcement after composition.
                    warp_l2r.mul_(b_mask)
                    warp_r2l.mul_(b_mask)
                    
                    if self.elastic_sigma > 0.0:
                        warp_l2r.copy_(separable_gaussian_filter(warp_l2r, self.elastic_sigma, spacing=curr_physical_spacing))
                        warp_r2l.copy_(separable_gaussian_filter(warp_r2l, self.elastic_sigma, spacing=curr_physical_spacing))
                        
                    # ITK-style diffeomorphic projection: double-inversion.
                    warp_l2r_inv = update_inverse_field_nd(warp_l2r, warp_l2r_inv.detach(), steps=self.inverse_steps, method=self.inverse_method)
                    warp_l2r.copy_(update_inverse_field_nd(warp_l2r_inv, warp_l2r.detach(), steps=self.inverse_steps, method=self.inverse_method))
                    
                    warp_r2l_inv = update_inverse_field_nd(warp_r2l, warp_r2l_inv.detach(), steps=self.inverse_steps, method=self.inverse_method)
                    warp_r2l.copy_(update_inverse_field_nd(warp_r2l_inv, warp_r2l.detach(), steps=self.inverse_steps, method=self.inverse_method))
```

### C. End-of-Registration Coordinate Mapping and Transforms Composition
At the end of registration (lines 1498–1506):
```python
            # Compose midpoint fields into full endpoint-to-endpoint fields
            # Forward L2R (fixed→moving, used to resample moving onto fixed):
            #   For each fixed point x: go to midpoint via phi_l2r, then to moving via phi_r2l_inv
            #   = phi_r2l_inv ∘ phi_l2r
            full_l2r = compose_grids(identity_full + w_r2l_inv, identity_full + w_l2r)
            # Inverse R2L (moving→fixed, used to resample fixed onto moving):
            #   For each moving point x: go to midpoint via phi_r2l, then to fixed via phi_l2r_inv
            #   = phi_l2r_inv ∘ phi_r2l
            full_r2l = compose_grids(identity_full + w_l2r_inv, identity_full + w_r2l)
```

In `forward` (lines 1541–1543):
```python
        phi_l2r = identity + warp_resampled
        composed_grid = compose_grids(grid_affine, phi_l2r)
```

### D. Grid-to-Physical Affine Coordinate Transformations
In `grid_to_physical_affine` (lines 1630–1635):
```python
    Kx_inv = np.linalg.inv(Kx)
    Sx_inv = np.linalg.inv(np.diag(Sx))
    Wx = Kx_inv @ Sx_inv @ Dx.T
    bx = - Kx_inv @ Sx_inv @ Dx.T @ Ox - Kx_inv @ Cx
    
    # Vy transforms grid u_y to physical y: y_phys = Vy @ u_y + cy
    Vy = Dy @ np.diag(Sy) @ Ky
    cy = Dy @ np.diag(Sy) @ Cy + Oy
```
And final physical conversion (lines 1646–1647):
```python
    M_phys = Vy @ A_grid @ Wx
    t_phys = Vy @ (A_grid @ bx + t_grid) + cy
```

---

## 2. Logic Chain

The current optimization and mapping design has minor spatial alignment inconsistencies that prevent exact parity with ITK SyN:

### A. Duality of Grid vs Physical Coordinate Spaces
* ITK displacement fields store displacements in physical space (mm) with component vectors in LPS coordinates.
* `F.grid_sample` operates in normalized coordinate grids of `[-1, 1]` where component values are reversed relative to voxel coordinate array indexes (to align innermost dimension to first coordinate x).
* In the current implementation, `warp_l2r` and `warp_r2l` are parameterized and optimized in normalized space `[-1, 1]`.
* In `compose_grids(identity_full + w_r2l_inv, identity_full + w_l2r)`, `w_r2l_inv` is the inverse of `w_r2l`. Since `w_r2l` is the displacement mapping midpoint to moving, `w_r2l_inv` (which is $\phi_2$) maps moving to midpoint.
* Applying $\phi_2 \circ \phi_1^{-1}$ is dimensionally incorrect. The target forward mapping should map fixed to moving, which is $\phi_2^{-1} \circ \phi_1$.
* Therefore, the composition should strictly be `compose_grids(identity + w_r2l, identity + w_l2r_inv)`.

### B. Single Interpolation Policy and Affine Grid Composition
* Let $x$ be a physical coordinate in the fixed image space.
* $\phi_1(x) = x + u_{1\_phys}(x)$ maps it to midpoint physical space.
* $\phi_2^{-1}$ maps midpoint physical space to moving physical space: $w = \phi_2^{-1}(\phi_1(x)) = \phi_1(x) + u_{2\_inv\_phys}(\phi_1(x))$.
* The physical affine transform $A$ maps $w$ to physical moving coordinate $y = M_{phys} @ w + t_{phys}$.
* To preserve the single interpolation policy, this entire chain of physical coordinate calculations must be composed first, then mapped to moving image normalized coordinates `[-1, 1]` for a single `F.grid_sample` call on the original moving image.

---

## 3. Caveats

* The physical grid mapping relies on correct direction matrix (`fixed.direction`), voxel spacing (`fixed.spacing`), and origin coordinates. Any misalignment or sign errors in the ANTs metadata parser will propagate to the PyTorch/JAX coordinate mapping.
* Nearest neighbor interpolation must be strictly applied when resampling labels/segmentations to prevent boundary blurring, as outlined in `GEMINI.md` (Constraint 4).

---

## 4. Conclusion

To achieve complete parity with ITK SyN:
1. **Physical mm Coordinates**: Store and optimize displacement fields natively in physical space (mm) using LPS orientation.
2. **Transform Composition**: Strict composition mapping $y = A(\phi_2^{-1}(\phi_1(x)))$ in physical space, using a single interpolation of the moving image.
3. **Optimizers and regularizers**: Gradients must be smoothed in physical units using physical spacing, and CFL step limits computed relative to spacing.

---

## 5. Verification Method

### Draft Specification: `scratch/test_internal_dice.py`

This script evaluates physical coordinate-mapping accuracy by comparing segmentations warped via PyTorch vs. ANTs.

```python
import numpy as np
import torch
import torch.nn.functional as F
import ants

def get_physical_grid_torch(shape, spacing, origin, direction, device='cpu', dtype=torch.float32):
    dim = len(shape)
    grids = [torch.arange(s, device=device, dtype=dtype) for s in shape]
    # 'ij' indexing yields order (z, y, x)
    meshgrid = torch.meshgrid(*grids, indexing='ij')
    # Reverse to physical order: (x, y, z)
    meshgrid_reversed = list(reversed(meshgrid))
    
    idxs = torch.stack(meshgrid_reversed, dim=-1) # (Nz, Ny, Nx, 3)
    spacing_t = torch.tensor(spacing, device=device, dtype=dtype)
    origin_t = torch.tensor(origin, device=device, dtype=dtype)
    direction_t = torch.tensor(direction, device=device, dtype=dtype)
    
    scaled = idxs * spacing_t
    flat_scaled = scaled.view(-1, dim)
    flat_phys = flat_scaled @ direction_t.t() + origin_t
    return flat_phys.view(*shape, dim).unsqueeze(0)

def physical_to_normalized_torch(phys_coords, target_shape, spacing, origin, direction):
    device = phys_coords.device
    dtype = phys_coords.dtype
    dim = len(target_shape)
    
    spacing_t = torch.tensor(spacing, device=device, dtype=dtype)
    origin_t = torch.tensor(origin, device=device, dtype=dtype)
    direction_t = torch.tensor(direction, device=device, dtype=dtype)
    
    flat_phys = phys_coords.view(-1, dim)
    diff = flat_phys - origin_t
    rotated = diff @ direction_t
    voxel_coords = rotated / spacing_t
    
    shape_t = torch.tensor(list(reversed(target_shape)), device=device, dtype=dtype)
    norm_coords = (voxel_coords / (shape_t - 1)) * 2.0 - 1.0
    return norm_coords.view(phys_coords.shape)

def test_internal_dice():
    # 1. Load sample images (using ants r16/r27 or a dummy template)
    fixed = ants.image_read(ants.get_ants_data('r16'))
    moving = ants.image_read(ants.get_ants_data('r27'))
    
    # 2. Get dummy discrete label segmentation
    fixed_label = ants.threshold_image(fixed, 'Otsu', 3)
    
    # Define physical transforms: 
    # Let's set a simple physical translation and rotation
    tx = ants.new_ants_transform(precision='float', dimension=2)
    tx.set_parameters([1.0, 0.0, 0.0, 1.0, 5.0, -3.0])
    
    # 3. Method A: Warp label using ANTs py
    warped_ants = ants.apply_transforms(
        fixed=fixed, 
        moving=fixed_label, 
        transformlist=[tx], 
        interpolator='nearestNeighbor'
    )
    
    # 4. Method B: Compose in PyTorch using physical grid coordinates
    device = 'cpu'
    dtype = torch.float32
    
    # Convert fixed_label to PyTorch tensor
    label_tensor = torch.tensor(
        fixed_label.numpy(), 
        dtype=dtype, 
        device=device
    ).unsqueeze(0).unsqueeze(0) # (1, 1, Ny, Nx)
    
    # Get M_phys and t_phys from ANTs transform
    params = tx.parameters
    M_phys = params[:4].reshape(2, 2)
    t_phys = params[4:]
    
    # Compute warped coordinates in moving physical space
    identity_phys = get_physical_grid_torch(
        fixed.shape, 
        fixed.spacing, 
        fixed.origin, 
        fixed.direction, 
        device, dtype
    )
    
    flat_phys = identity_phys.view(-1, 2)
    M_phys_t = torch.tensor(M_phys, device=device, dtype=dtype)
    t_phys_t = torch.tensor(t_phys, device=device, dtype=dtype)
    
    # Apply affine transformation in physical mm space
    y_phys = flat_phys @ M_phys_t.t() + t_phys_t
    y_phys = y_phys.view(identity_phys.shape)
    
    # Map to normalized coordinates of the moving image (which is fixed_label in this test)
    norm_grid = physical_to_normalized_torch(
        y_phys, 
        fixed_label.shape, 
        fixed_label.spacing, 
        fixed_label.origin, 
        fixed_label.direction
    )
    
    # Sample using nearest-neighbor to satisfy integer label maps constraints
    warped_torch_tensor = F.grid_sample(
        label_tensor, 
        norm_grid, 
        mode='nearest', 
        padding_mode='border', 
        align_corners=True
    )
    
    # Convert back to ANTs image
    warped_torch = ants.from_numpy(
        warped_torch_tensor.squeeze().numpy(), 
        origin=fixed.origin, 
        spacing=fixed.spacing, 
        direction=fixed.direction
    )
    
    # 5. Compute Dice overlap
    overlap = ants.label_overlap_measures(warped_ants, warped_torch)
    dice = overlap['MeanOverlap'].iloc[0]
    
    print(f"Computed DICE between ANTs and PyTorch physical warped grids: {dice:.6f}")
    assert dice >= 0.999, f"DICE discrepancy detected: {dice:.6f}"
    print("Physical coordinate mapping verification PASSED!")

if __name__ == '__main__':
    test_internal_dice()
```

### Invalidation Conditions
* A Mean DICE score of $< 0.999$ in `test_internal_dice` invalidates the proposed coordinate-mapping conversion math.
* Spatial coordinate flipping under anisotropic spacing or non-orthogonal direction cosines invalidates the LPS orientation stacking method.
