import os, sys, time, torch, ants, syntx

print("Loading 2D benchmark data...")
data = syntx.benchmark_data('2d')
fi, mi = data['fixed'], data['moving']
fl, ml = data['fixed_label'], data['moving_label']

reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)
aff_tx = reg_aff['fwdtransforms'][0]

print("Running TVF...")
reg = syntx.tvf(
    fixed=fi, moving=mi, initial_transform=aff_tx, backend='pytorch', device='cpu' if not torch.backends.mps.is_available() else 'mps',
    reg_iterations=[100, 100, 20], similarity_metric='lncc', flow_sigma=0.4, total_sigma=0.05, grad_step=0.5,
    optimizer='lars', cfl_max=0.0, use_analytical_gradients=True, verbose=True
)
tvf_tx = reg['fwdtransforms'][0]
tvf_field = ants.image_read(tvf_tx)
import numpy as np
print("Max TVF displacement:", np.nanmax(tvf_field.numpy()), "Min:", np.nanmin(tvf_field.numpy()))
print("Contains NaNs?", np.isnan(tvf_field.numpy()).any())
