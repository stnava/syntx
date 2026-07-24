import os
import time
import json
import argparse
import ants
import antspyt1w
import syntx

parser = argparse.ArgumentParser(description="Run syntx.auto_reg with optional backend, device, and output directory CLI arguments.")
parser.add_argument('--fixed', type=str, default="~/.antspyt1w/T_template0.nii.gz", help="Path to fixed image template")
parser.add_argument('--moving', type=str, default="~/data/blast_cohorts/BIDS/SOCOM/sub-Blast-05/ses-01/anat/sub-Blast-05_ses-01_run-001_T1w.nii.gz", help="Path to moving subject image")
parser.add_argument('--outdir', '-o', type=str, default="./auto_reg_output", help="Output directory to save extracted brain, warped image, jacobian map, and metrics")
parser.add_argument('--backend', type=str, default=None, choices=['jax', 'pytorch'], help="Compute backend override ('jax' or 'pytorch')")
parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'], help="Hardware device override ('cuda', 'mps', or 'cpu')")
args = parser.parse_args()

print("=== Running syntx.auto_reg with antspyt1w Brain Extraction ===")

# 1. Fixed Template & Moving BIDS Subject File Paths
fixed_path = os.path.expanduser(args.fixed)
moving_path = os.path.expanduser(args.moving)
outdir_path = os.path.expanduser(args.outdir)

print(f"Fixed Path:  {fixed_path}")
print(f"Moving Path: {moving_path}")
print(f"Output Dir:  {outdir_path}")

os.makedirs(outdir_path, exist_ok=True)

fi_raw = ants.image_read(fixed_path)
mi_raw = ants.image_read(moving_path)

# 2. antspyt1w Deep Brain Extraction
print("\n--- Running antspyt1w.brain_extraction ---")
t0_bext = time.time()
fi_brain = fi_raw * antspyt1w.brain_extraction(fi_raw)
mi_brain = mi_raw * antspyt1w.brain_extraction(mi_raw)
print(f"Brain Extraction Complete in {time.time() - t0_bext:.2f}s")

print(f"Fixed Brain Dimensions:  {fi_brain.shape}, Spacing: {fi_brain.spacing}")
print(f"Moving Brain Dimensions: {mi_brain.shape}, Spacing: {mi_brain.spacing}")

# 3. Execute syntx.auto_reg(fi_brain, mi_brain)
print("\n--- Executing syntx.auto_reg(fixed_brain, moving_brain) ---")
reg_kwargs = {}
if args.backend is not None:
    reg_kwargs['backend'] = args.backend
if args.device is not None:
    reg_kwargs['device'] = args.device

res = syntx.auto_reg(fixed=fi_brain, moving=mi_brain, verbose=True, **reg_kwargs)

# 4. Save Outputs to Output Directory
print(f"\n--- Saving Output Files to {outdir_path} ---")

# Save extracted brains
fi_brain_file = os.path.join(outdir_path, "fixed_brain.nii.gz")
mi_brain_file = os.path.join(outdir_path, "moving_brain.nii.gz")
ants.image_write(fi_brain, fi_brain_file)
ants.image_write(mi_brain, mi_brain_file)
print(f"Saved Fixed Brain:  {fi_brain_file}")
print(f"Saved Moving Brain: {mi_brain_file}")

# Save warped moving image
warped_file = os.path.join(outdir_path, "warped_moving.nii.gz")
ants.image_write(res['warpedmovout'], warped_file)
print(f"Saved Warped Image: {warped_file}")

# Save Jacobian Determinant Map (if forward warp exists)
fwd_tx = res.get('fwdtransforms', [])
warp_file = next((tx for tx in fwd_tx if isinstance(tx, str) and tx.endswith(('.nii', '.nii.gz'))), None)
if warp_file is not None:
    try:
        jac_img = ants.create_jacobian_determinant_image(fi_brain, warp_file)
        jac_file = os.path.join(outdir_path, "jacobian_determinant.nii.gz")
        ants.image_write(jac_img, jac_file)
        print(f"Saved Jacobian Map:{jac_file}")
    except Exception as e:
        print(f"Could not write 3D Jacobian map: {e}")

# Save Metrics JSON
metrics_file = os.path.join(outdir_path, "metrics.json")
with open(metrics_file, 'w') as f:
    json.dump(res['metrics'], f, indent=2)
print(f"Saved Metrics JSON: {metrics_file}")

# 5. Display Results & Integrated Metrics
print("\n=======================================================")
print("             syntx.auto_reg OUTPUT SUMMARY              ")
print("=======================================================")
print(f"Warped Moving Image: {res['warpedmovout']}")
print(f"Forward Transforms:  {res['fwdtransforms']}")
print(f"Inverse Transforms:  {res['invtransforms']}")

metrics = res['metrics']
print("\n--- INTEGRATED EVALUATION METRICS ---")
print(f"Execution Runtime:   {metrics['execution_time_seconds']:.2f} seconds")
print(f"Hardware Device:     {metrics['device_used']}")
print(f"Compute Engine:      {metrics['backend_used']}")
print(f"LNCC Score:          {metrics['lncc_score']:.4f}")
print(f"MSE Score:           {metrics['mse_score']:.6f}")
print(f"Mattes MI Score:     {metrics['mattes_mi_score']:.4f}")
if 'jac_mean' in metrics:
    print(f"Mean Jacobian (J):   {metrics['jac_mean']:.4f}")
    print(f"Min Jacobian (J_min):{metrics['jac_min']:.4f}")
    print(f"Max Jacobian (J_max):{metrics['jac_max']:.4f}")
    print(f"Folding Rate:        {metrics['folding_pct']:.4f}%")
if 'smooth_1st' in metrics:
    print(f"1st Deriv Smoothness:{metrics['smooth_1st']:.4f}")
    print(f"2nd Deriv Smoothness:{metrics['smooth_2nd']:.4f}")
print("=======================================================")
