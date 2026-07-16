"""Check if torch.flip blocks gradients."""
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = torch.flip(x, dims=[-1])
loss = y.sum()
loss.backward()
print(f"x.grad after flip: {x.grad}")
print(f"Gradients flow through flip: {x.grad is not None}")

# Also check: does the cached version produce same output as non-cached?
import sys, os
sys.path.insert(0, 'src')
import numpy as np
from syntx.syn import (physical_to_normalized_torch, physical_to_normalized_torch_cached)

phys = torch.randn(1, 32, 32, 2, dtype=torch.float32) * 100

shape_t = torch.tensor([32, 32], dtype=torch.float32)  # YX order for cached
spacing = (8.23, 8.23)  # XY order (ANTs)
origin = (0.0, 0.0)
direction = np.eye(2)

spacing_rev = tuple(reversed(spacing))
origin_rev = tuple(reversed(origin))
direction_rev = np.asarray(direction)[::-1, ::-1].copy()

spacing_t = torch.tensor(spacing_rev, dtype=torch.float32)
origin_t = torch.tensor(origin_rev, dtype=torch.float32)
direction_t = torch.tensor(direction_rev, dtype=torch.float32)

out_cached = physical_to_normalized_torch_cached(phys, shape_t, spacing_t, origin_t, direction_t)
out_normal = physical_to_normalized_torch(phys, (32, 32), spacing, origin, direction)

print(f"\nCached output shape: {out_cached.shape}")
print(f"Normal output shape: {out_normal.shape}")
print(f"Max diff: {(out_cached - out_normal).abs().max().item():.10f}")
print(f"Are they equal? {torch.allclose(out_cached, out_normal, atol=1e-6)}")

