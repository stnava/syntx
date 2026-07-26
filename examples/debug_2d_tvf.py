"""
Test TVF registration on 2D r16/r64 images.
Validates the time-varying velocity field registration model.
"""
import ants
import numpy as np
import time
import torch
import tempfile
import os
import sys

sys.path.insert(0, '/Users/stnava/code/syntx/src')
import syntx
from syntx.tvf import TVFModel

def compute_tissue_overlap(fi, warped):
    fixed_seg = ants.threshold_image(fi, 'Otsu', 3)
    warped_seg = ants.threshold_image(warped, 'Otsu', 3)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    dice = float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
    return dice


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 70)
    print("  TVF Registration 2D Test")
    print("  Images: r16 (fixed) -> r64 (moving)")
    print("=" * 70)
    
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))
    H, W = fi.shape
    
    # Normalize to [0, 1]
    fi_np = fi.numpy().astype(np.float32)
    mi_np = mi.numpy().astype(np.float32)
    fi_np = (fi_np - fi_np.min()) / (fi_np.max() - fi_np.min() + 1e-8)
    mi_np = (mi_np - mi_np.min()) / (mi_np.max() - mi_np.min() + 1e-8)
    fi_t = torch.tensor(fi_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    mi_t = torch.tensor(mi_np).unsqueeze(0).unsqueeze(0)
    
    # =========================================================================
    # Context: Standard SyN
    # =========================================================================
    print("\n[1] Standard SyN (reference)...")
    t0 = time.time()
    reg_syn = syntx.syn(
        fixed=fi, moving=mi,
        reg_iterations=[100, 100, 100, 20],
        affine_iterations=[1000, 1000, 1000, 1000],
        grad_step=0.25, flow_sigma=3.0, sampling_percentage=0.2,
        syn_metric='lncc', lncc_radius=2,
        backend='pytorch', device='cpu', inverse_steps=15
    )
    syn_time = time.time() - t0
    dice_syn = compute_tissue_overlap(fi, reg_syn['warpedmovout'])
    print(f"    SyN Dice={dice_syn:.4f}, Time={syn_time:.1f}s")
    
    # =========================================================================
    # TVF Registration
    # =========================================================================
    print("\n[2] TVF Registration (velocity_shape=64x64, T=4, RK4)...")
    
    model = TVFModel(
        dim=2,
        image_shape=(H, W),
        velocity_shape=(64, 64),  # 4x coarser than image
        n_time_steps=4,
        spacing=list(fi.spacing),
        origin=list(fi.origin),
        direction=fi.direction.tolist(),
        fluid_sigma=1.0,
        transform_type='Affine',
        solver='rk4',
        integration_steps_per_interval=4
    )
    
    t0 = time.time()
    model.fit(
        fi_t, mi_t,
        levels=[1],           # Single level for now  
        epochs_per_level=[200],
        affine_epochs=500,
        lr=5e-3,
        lncc_radius=2,
        similarity_metric='lncc',
        verbose=True,
        fixed_spacing=list(fi.spacing),
        fixed_origin=list(fi.origin),
        fixed_direction=fi.direction.tolist(),
        moving_spacing=list(mi.spacing),
        moving_origin=list(mi.origin),
        moving_direction=mi.direction.tolist(),
    )
    tvf_time = time.time() - t0
    
    # Get warps
    with torch.no_grad():
        phi_fwd = model.get_forward_warp()  # physical displacement
        phi_inv = model.get_inverse_warp()
    
    print(f"\n    TVF Time={tvf_time:.1f}s")
    print(f"    Forward warp max |disp|={float(phi_fwd.abs().max()):.2f} mm")
    print(f"    Inverse warp max |disp|={float(phi_inv.abs().max()):.2f} mm")
    
    # Symmetry check
    sym_err = float((phi_fwd + phi_inv).abs().max())
    print(f"    Symmetry error (max): {sym_err:.4f} mm")
    
    # Apply forward warp via ANTs for fair Dice comparison
    # The warp needs to be in ANTs convention: (x, y) component order
    phi_fwd_np = phi_fwd.squeeze(0).detach().numpy()
    # TVF outputs in (row, col) = (y, x) order; ANTs expects (x, y)
    # The physical grid in get_physical_grid_torch uses reversed spacing/origin,
    # so components are already in (row, col) order matching the iteration order.
    # ANTs displacement expects (x=col, y=row), so we swap components:
    phi_fwd_ants = np.stack([phi_fwd_np[..., 1], phi_fwd_np[..., 0]], axis=-1)
    
    warp_img = ants.from_numpy(
        phi_fwd_ants.astype(np.float32),
        origin=fi.origin, spacing=fi.spacing, direction=fi.direction,
        has_components=True
    )
    with tempfile.NamedTemporaryFile(suffix='_tvf_fwd.nii', delete=False) as f:
        fwd_path = f.name
    ants.image_write(warp_img, fwd_path)
    
    warped_tvf = ants.apply_transforms(fi, mi, [fwd_path])
    dice_tvf_warponly = compute_tissue_overlap(fi, warped_tvf)
    os.unlink(fwd_path)
    
    print(f"    TVF (warp only) Dice={dice_tvf_warponly:.4f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  SyN Dice:            {dice_syn:.4f} ({syn_time:.1f}s)")
    print(f"  TVF (warp only):     {dice_tvf_warponly:.4f} ({tvf_time:.1f}s)")
    print(f"  Symmetry error:      {sym_err:.4f} mm")
    print(f"  Max displacement:    {float(phi_fwd.abs().max()):.2f} mm")
    print("=" * 70)


if __name__ == "__main__":
    main()
