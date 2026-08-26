#!/usr/bin/env python
import argparse, time, os, json, numpy as np, torch, ants, syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.deformation_metrics import compute_bidirectional_dice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair-idx', type=int, required=True)
    parser.add_argument('--method', type=str, required=True, choices=['syngs', 'syn', 'tvf'])
    args = parser.parse_args()
    p_idx = args.pair_idx
    method = args.method
    
    p = load_mindboggle_pair(p_idx, 'examples/pairs.csv')
    fi = p['fixed']
    mi = p['moving']
    fl = p['fixed_label']
    ml = p['moving_label']
    
    # 1. Deterministic Robust Affine
    reg_aff = syntx.robust_affine(fi, mi, mode='auto', verbose=False)
    aff_0 = reg_aff['fwdtransforms'][0]
    
    t0 = time.time()
    if method == 'syngs':
        res = syntx.syngs(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend='pytorch', device='mps',
            flow_sigma=3.0, total_sigma=0.0,
            alpha=0.35, regularizer='sobolev',
            transport_mode='transport',
            optimizer='reg_adam', optimizer_lr=1.2,
            max_step_norm=0.25,
            reg_iterations=[100, 100, 20],
            similarity_metric='cc2',
            bootstrap_mode='antithetic',
            n_steps=8, solver='euler', verbose=False
        )
        which_inv = res.get('whichtoinvert_inv', [True, False])
        dfix, dmov, dice_sym = compute_bidirectional_dice(fl, ml, fi, mi, res['fwdtransforms'], res['invtransforms'], which_inv)
    elif method == 'syn':
        res = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend='pytorch', device='mps',
            grad_step=0.25, flow_sigma=1.0, total_sigma=0.0,
            reg_iterations=[100, 100, 20], similarity_metric='cc2',
            fast_smooth=True, inverse_method='anderson', in_loop_inv_steps=10,
            formulation='eulerian', regularizer='sobolev', sobolev_alpha=1.0,
            bootstrap_mode='antithetic', verbose=False
        )
        dfix, dmov, dice_sym = compute_bidirectional_dice(fl, ml, fi, mi, res['fwdtransforms'], res['invtransforms'])
    elif method == 'tvf':
        res = syntx.tvf(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend='pytorch', device='mps',
            regularizer='sobolev', sobolev_alpha=0.035,
            flow_sigma=1.0, total_sigma=0.035,
            optimizer='reg_adam', optimizer_lr=1.2, max_step_norm=0.50,
            reg_iterations=[100, 100, 20],
            multipoint_loss=[0.0, 0.5, 1.0], solver='euler', verbose=False
        )
        dfix, dmov, dice_sym = compute_bidirectional_dice(fl, ml, fi, mi, res['fwdtransforms'], res['invtransforms'])
    t_elapsed = time.time() - t0
    
    with open(f'results/pair_{p_idx:03d}_ants_syn.json', 'r') as f:
        ants_rec = json.load(f)
    dice_ants = float(ants_rec['dice_sym'])
    
    out = {
        'pair': p_idx,
        'method': method,
        'subject1': p['fixed_id'],
        'subject2': p['moving_id'],
        'dice_ants': dice_ants,
        'dice_method': dice_sym,
        'gain_pct': (dice_sym - dice_ants) * 100.0,
        'time_s': t_elapsed
    }
    print('RESULT_JSON:' + json.dumps(out), flush=True)

if __name__ == '__main__':
    main()
