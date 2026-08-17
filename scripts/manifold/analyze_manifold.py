import json
import pandas as pd
import os

manifold_path = '/Users/stnava/code/syntx/docs/provenance/manifold_64.json'
if not os.path.exists(manifold_path):
    print("Manifold not generated yet.")
    exit(1)

with open(manifold_path, 'r') as f:
    data = json.load(f)

rows = []
for d in data:
    row = {
        'idx': d['idx'],
        'engine': d['engine'],
        'grad_step': d['params'].get('grad_step', 0),
        'flow_sigma': d['params'].get('flow_sigma', 0),
        'regularizer': d['params'].get('regularizer', 'None'),
        'inverse_steps': d['params'].get('inverse_steps', 0),
        'dice_fwd': d['metrics']['dice_fwd'],
        'dice_inv': d['metrics']['dice_inv'],
        'dice_sym': d['metrics']['dice_sym'],
        'fold_fwd': d['metrics']['fold_pct_fwd'],
        'fold_inv': d['metrics']['fold_pct_inv'],
        'inv_err': d['metrics']['mean_inv_err'],
        'bnd_fwd': d['metrics']['bnd_fwd'],
        'bnd_inv': d['metrics']['bnd_inv'],
        'jac_min_fwd': d['metrics']['jac_min_fwd'],
        'jac_min_inv': d['metrics']['jac_min_inv']
    }
    rows.append(row)

df = pd.DataFrame(rows)
out_csv = '/Users/stnava/code/syntx/docs/provenance/manifold_64.csv'
df.to_csv(out_csv, index=False)

print("="*60)
print(f"Manifold CSV saved to {out_csv}")
print("="*60)

print("\n1. Overall Folding Rate by Engine")
print(df.groupby('engine')[['fold_fwd', 'fold_inv']].mean())

print("\n2. Folding Rate by Grad Step (PyTorch SyN)")
print(df[df['engine'] == 'syntx'].groupby('grad_step')[['fold_fwd', 'fold_inv', 'dice_sym']].mean())

print("\n3. Folding Rate by Grad Step (ANTs C++ SyN)")
print(df[df['engine'] == 'ants'].groupby('grad_step')[['fold_fwd', 'fold_inv', 'dice_sym']].mean())

print("\n4. Impact of Inverse Steps on Inverse Folding & Dice (PyTorch SyN)")
print(df[df['engine'] == 'syntx'].groupby('inverse_steps')[['fold_inv', 'dice_inv', 'dice_sym', 'inv_err']].mean())

print("\n5. Bending Energy by Regularizer (PyTorch SyN)")
print(df[df['engine'] == 'syntx'].groupby('regularizer')[['bnd_fwd', 'bnd_inv', 'fold_fwd', 'dice_sym']].mean())

print("\n6. Direct Baseline Parity (grad_step=0.15, flow_sigma=3.0, gaussian/None)")
parity = df[(df['grad_step'] == 0.15) & (df['flow_sigma'] == 3.0) & (df['regularizer'].isin(['gaussian', 'None']))]
print(parity[['engine', 'dice_sym', 'fold_fwd', 'fold_inv', 'bnd_fwd', 'jac_min_fwd']])
