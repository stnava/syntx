import syntx, ants, pandas as pd, time

data = syntx.benchmark_data('mbhard')
fixed, moving = data['fixed'], data['moving']
fi_lbl, mi_lbl = data['fixed_label'], data['moving_label']

reg_iters = [100, 100, 20]
init_tx = syntx.robust_affine(fixed, moving, mode='pytorch')
if isinstance(init_tx, dict):
    init_tx = init_tx['fwdtransforms'][0]

print(f"Initial Affine Dice (evaluated previously): 0.3346")

# 1. Run syntx.syn
t0 = time.time()
syn_ret = syntx.syn(
    fixed=fixed, moving=moving,
    initial_transform=init_tx,
    reg_iterations=reg_iters,
    similarity_metric='lncc',
    flow_sigma=3.0, total_sigma=0.0, grad_step=0.25,
    verbose=False
)
syn_time = time.time() - t0
warped_syn_lbl = ants.apply_transforms(fixed=fi_lbl, moving=mi_lbl, transformlist=syn_ret['fwdtransforms'], interpolator='nearestNeighbor')
ov_syn = ants.label_overlap_measures(fi_lbl, warped_syn_lbl)
syn_dice = float(pd.to_numeric(ov_syn[ov_syn['Label'].astype(str) != 'All']['MeanOverlap']).mean())
print(f"syntx.syn Dice: {syn_dice:.4f} (Time: {syn_time:.1f}s)")

# 2. Run syntx.tvf
t0 = time.time()
tvf_ret = syntx.tvf(
    fixed=fixed, moving=moving,
    initial_transform=init_tx,
    reg_iterations=reg_iters,
    similarity_metric='lncc',
    multipoint_loss=[0.0, 0.5, 1.0],
    flow_sigma=0.4, total_sigma=0.05, grad_step=0.45,
    cfl_momentum=0.95, n_time_steps=3, constant_speed=True,
    use_analytical_gradients=True,
    verbose=False
)
tvf_time = time.time() - t0
warped_tvf_lbl = ants.apply_transforms(fixed=fi_lbl, moving=mi_lbl, transformlist=tvf_ret['fwdtransforms'], interpolator='nearestNeighbor')
ov_tvf = ants.label_overlap_measures(fi_lbl, warped_tvf_lbl)
tvf_dice = float(pd.to_numeric(ov_tvf[ov_tvf['Label'].astype(str) != 'All']['MeanOverlap']).mean())
print(f"syntx.tvf Dice: {tvf_dice:.4f} (Time: {tvf_time:.1f}s)")
