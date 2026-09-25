#!/usr/bin/env python
"""
scripts/diagnose_template_convergence.py
=========================================
Diagnostic harness to isolate the root cause of build_template's
dynamical instability on r16 with backend='pytorch'.
"""

import os
import sys
import time
import argparse
import numpy as np
import ants

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_src = os.path.join(_repo, 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

import syntx

def inspect_affine(mat_file):
    """Decompose 2D affine matrix from ITK .mat file into components."""
    tx = ants.read_transform(mat_file)
    params = np.array(tx.parameters, dtype=np.float64)
    fixed = np.array(tx.fixed_parameters, dtype=np.float64)
    A = params[:4].reshape((2, 2))
    t = params[4:6]
    det = np.linalg.det(A)
    # SVD for singular values (scale) and rotation
    U, s, Vt = np.linalg.svd(A)
    rot_det = np.linalg.det(U @ Vt)
    angle_rad = np.arctan2(U[1, 0], U[0, 0])
    angle_deg = np.degrees(angle_rad)
    return {
        'A': A,
        't': t,
        'det': det,
        'singular_values': s,
        'angle_deg': angle_deg,
        'fixed_center': fixed,
    }

def run_diagnostic(name, transform_type='SyN', reg_iters=[30, 10, 0],
                   gradient_step=0.20, blending_weight=0.75,
                   backend='pytorch', iterations=5, device=None, **kwargs):
    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENT: {name}")
    print(f"transform={transform_type}, iters={iterations}, reg_iters={reg_iters}, "
          f"grad_step={gradient_step}, blend={blending_weight}, backend={backend}, device={device}")
    print(f"{'='*70}")

    r16 = ants.image_read(ants.get_ants_data('r16'))
    r16_ref = syntx.reflect_image(r16, axis='x')

    extra_kwargs = dict(kwargs)
    if device is not None:
        extra_kwargs['device'] = device

    t0 = time.time()
    res = syntx.build_template(
        image_list=[r16, r16_ref],
        iterations=iterations,
        gradient_step=gradient_step,
        blending_weight=blending_weight,
        type_of_transform=transform_type,
        reg_iterations=reg_iters,
        backend=backend,
        verbose=True,
        **extra_kwargs
    )
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.2f}s ({elapsed/iterations:.2f}s / iter)")
    print("\n  Iter │  Template MAE  │  L2 (mm)  │  Membrane  │  Bending")
    print("  " + "─" * 62)
    for i, (mae, sr) in enumerate(zip(res['convergence'], res['shape_residuals'])):
        lbl = f"Iter {i}" + (" (init)" if i == 0 else "")
        print(f"  {lbl:>9s} │  {mae:>12.6f}  │  {sr['l2_norm']:>9.4f}  │"
              f"  {sr['membrane_energy']:>10.4f}  │  {sr['bending_energy']:>10.6f}")

    # Inspect work_dir affine transforms
    work_dir = res['work_dir']
    print("\nAffine Transform Inspection:")
    for it in range(len(res['shape_residuals'])):
        aff_file = os.path.join(work_dir, f"avgAffine_{it}.mat")
        if os.path.exists(aff_file):
            info = inspect_affine(aff_file)
            print(f"  Iter {it} avgAffine: det={info['det']:.6f}, "
                  f"s={info['singular_values']}, rot_deg={info['angle_deg']:.3f}, "
                  f"trans={info['t']}")

    tmpl = res['template']
    tmpl_ref = syntx.reflect_image(tmpl, axis='x')
    asym_mae = float((tmpl - tmpl_ref).abs().mean())
    print(f"\nFinal Template L/R Asymmetry MAE: {asym_mae:.4f}")
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='all', choices=['all', 'baseline', 'synonly', 'fine_level', 'anneal', 'cpu_vs_mps'])
    parser.add_argument('--iterations', type=int, default=5)
    args = parser.parse_args()

    if args.exp in ('all', 'baseline'):
        run_diagnostic("1. Baseline PyTorch (reproducing issue)",
                       iterations=args.iterations, reg_iters=[30, 10, 0], gradient_step=0.20)

    if args.exp in ('all', 'synonly'):
        run_diagnostic("2. SyNOnly (Affine frozen at Identity)",
                       transform_type='SyNOnly',
                       iterations=args.iterations, reg_iters=[30, 10, 0], gradient_step=0.20)

    if args.exp in ('all', 'fine_level'):
        run_diagnostic("3. With Fine-Level iterations [30, 15, 5]",
                       iterations=args.iterations, reg_iters=[30, 15, 5], gradient_step=0.20)

    if args.exp in ('all', 'anneal'):
        run_diagnostic("4. Lower gradient step 0.10",
                       iterations=args.iterations, reg_iters=[30, 10, 0], gradient_step=0.10)

    if args.exp in ('all', 'cpu_vs_mps'):
        run_diagnostic("5A. CPU device",
                       iterations=3, reg_iters=[30, 10, 0], device='cpu')
        run_diagnostic("5B. MPS device",
                       iterations=3, reg_iters=[30, 10, 0], device='mps')
