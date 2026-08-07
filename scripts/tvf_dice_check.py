import os, sys, time, torch, ants, syntx

print("Loading 2D benchmark data...")
data = syntx.benchmark_data('2d')
fi, mi = data['fixed'], data['moving']
fl, ml = data['fixed_label'], data['moving_label']

reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)
aff_tx = reg_aff['fwdtransforms'][0]

ml_warped_aff = ants.apply_transforms(fixed=fi, moving=ml, transformlist=[aff_tx], interpolator='nearestNeighbor')
overlap_aff = ants.label_overlap_measures(fl, ml_warped_aff)
df_aff = overlap_aff[~overlap_aff['Label'].astype(str).isin(['All', '0', '0.0'])]
col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_aff.columns else 'TargetOverlap'
dice_aff = float(df_aff[col].mean()) if len(df_aff) > 0 else 0.0
print("Affine Dice:", dice_aff)

print("Running TVF...")
reg = syntx.tvf(
    fixed=fi, moving=mi, initial_transform=aff_tx, backend='pytorch', device='cpu' if not torch.backends.mps.is_available() else 'mps',
    reg_iterations=[10], similarity_metric='lncc', flow_sigma=0.4, total_sigma=0.05, grad_step=0.5,
    optimizer='lars', cfl_max=0.0, use_analytical_gradients=True, verbose=True
)
ml_warped = ants.apply_transforms(fixed=fi, moving=ml, transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor')
overlap = ants.label_overlap_measures(fl, ml_warped)
df = overlap[~overlap['Label'].astype(str).isin(['All', '0', '0.0'])]
col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df.columns else 'TargetOverlap'
dice = float(df[col].mean()) if len(df) > 0 else 0.0
print("TVF Dice:", dice)
