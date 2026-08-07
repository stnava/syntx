import os, sys, time, torch, ants, syntx
from syntx.tvf import TVFModel

print("Loading 2D benchmark data...")
data = syntx.benchmark_data('2d')
fi, mi = data['fixed'], data['moving']

def print_hook(grad):
    print("Gradient norm:", torch.norm(grad).item())

print("Running TVF init...")
fi_tensor = torch.from_numpy(fi.numpy()).unsqueeze(0).unsqueeze(0).float()
mi_tensor = torch.from_numpy(mi.numpy()).unsqueeze(0).unsqueeze(0).float()

model = TVFModel(dim=2, image_shape=fi.shape, velocity_shape=fi.shape)
model.velocity.requires_grad_(True)
model.velocity.register_hook(print_hook)

print("Forward...")
loss = model.forward(fi_tensor, mi_tensor, multipoint_loss=[0.5], lncc_window_size=5)
print("Loss:", loss.item())

print("Backward...")
loss.backward()

print("Velocity grad norm directly:", torch.norm(model.velocity.grad).item())
