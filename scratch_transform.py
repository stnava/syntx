import torch
import numpy as np
import os
import ants

from syntx.transform import SyNToTransform

device = torch.device('cpu')
shape = (16, 24)
dim = 2

grids = [torch.linspace(-1, 1, size, device=device) for size in shape]
meshgrid = torch.meshgrid(*grids, indexing='ij')
identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)

warp_field = torch.zeros_like(identity)
# Put displacement in component 0
warp_field[..., 0] = 1.0

metadata = {
    'origin': (0.0, 0.0),
    'spacing': (1.0, 1.0),
    'direction': [[1.0, 0.0], [0.0, 1.0]],
    'shape': shape
}

tx = SyNToTransform(
    affine_grid=identity,
    warp_field=warp_field,
    metadata=metadata,
    device=device
)

print("tx.warp_field shape:", tx.warp_field.shape)
print("tx.warp_field component 0 max:", tx.warp_field[..., 0].abs().max().item())
print("tx.warp_field component 1 max:", tx.warp_field[..., 1].abs().max().item())

prefix = "/tmp/tx2d_"
tx.export_classic(prefix)

# Load and check components
f = f"{prefix}1SyNWarp.nii.gz"
img = ants.image_read(f)
img_np = img.numpy()

print("img_np shape:", img_np.shape)
print("img_np component 0 max:", np.abs(img_np[..., 0]).max())
print("img_np component 1 max:", np.abs(img_np[..., 1]).max())
