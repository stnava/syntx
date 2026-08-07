import os, sys, time, torch, ants, syntx

print("Loading 2D benchmark data...")
data = syntx.benchmark_data('2d')
fi, mi = data['fixed'], data['moving']

print("Running TVF...")
reg = syntx.tvf(
    fixed=fi, moving=mi, backend='pytorch', device='cpu' if not torch.backends.mps.is_available() else 'mps',
    reg_iterations=[10], similarity_metric='lncc', flow_sigma=0.4, total_sigma=0.05, grad_step=0.5,
    optimizer='lars', cfl_max=0.0, use_analytical_gradients=True, verbose=True
)
tvf_tx = reg['fwdtransforms'][0]
tvf_field = ants.image_read(tvf_tx)
print("Max TVF displacement:", tvf_field.numpy().max(), "Min:", tvf_field.numpy().min())
