import sys, json, os, numpy as np, ants, torch, syntx

def compute_3d_bending_energy(warp_np, spacing):
    sx, sy, sz = spacing
    vx = warp_np[..., 0]
    vy = warp_np[..., 1]
    vz = warp_np[..., 2]
    
    def get_bnd(v):
        dx = np.gradient(v, axis=0) / sx
        dy = np.gradient(v, axis=1) / sy
        dz = np.gradient(v, axis=2) / sz
        dx2 = np.gradient(dx, axis=0) / sx
        dy2 = np.gradient(dy, axis=1) / sy
        dz2 = np.gradient(dz, axis=2) / sz
        dxy = np.gradient(dx, axis=1) / sy
        dxz = np.gradient(dx, axis=2) / sz
        dyz = np.gradient(dy, axis=2) / sz
        return dx2**2 + dy2**2 + dz2**2 + 2*(dxy**2 + dxz**2 + dyz**2)
        
    bnd_x = get_bnd(vx)
    bnd_y = get_bnd(vy)
    bnd_z = get_bnd(vz)
    return float(np.mean(bnd_x + bnd_y + bnd_z))

def main():
    with open(sys.argv[1], 'r') as f:
        job = json.load(f)
    
    idx = job['idx']
    engine = job['engine']
    params = job['params']
    aff_tx = job['aff_tx']
    out_file = job['out_file']
    
    d3 = syntx.benchmark_data('3d')
    fi_raw, mi_raw = d3['fixed'], d3['moving']
    fl_raw, ml_raw = d3['fixed_label'], d3['moving_label']
    
    mask_f = ants.iMath(ants.get_mask(fi_raw), "MD", 12)
    mask_m = ants.iMath(ants.get_mask(mi_raw), "MD", 12)
    
    fi = ants.crop_image(fi_raw, mask_f)
    mi = ants.crop_image(mi_raw, mask_m)
    fl = ants.crop_image(fl_raw, mask_f)
    ml = ants.crop_image(ml_raw, mask_m)
    
    import time
    t0 = time.time()
    if engine == 'syntx':
        reg = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_tx,
            backend='pytorch', device='mps',
            reg_iterations=[100, 100, 20], similarity_metric='lncc',
            syn_sampling=2, antisymmetric=True, verbose=False,
            **params
        )
        fwd = reg['fwdtransforms']
        inv = reg['invtransforms']
        
        # bending energy
        model = reg['model']
        w_fwd = model.warp_l2r.data.cpu().numpy()
        w_inv = model.warp_l2r_inv.data.cpu().numpy()
        
        w_fwd = w_fwd[0].transpose(3, 2, 1, 0)
        w_inv = w_inv[0].transpose(3, 2, 1, 0)
        
        bnd_fwd = compute_3d_bending_energy(w_fwd, fi.spacing)
        bnd_inv = compute_3d_bending_energy(w_inv, fi.spacing)
        
        from syntx.syn import calculate_inverse_identity_error
        err_dict = calculate_inverse_identity_error(model.warp_l2r.data.cpu(), model.warp_l2r_inv.data.cpu(), fi.spacing, fi.origin, fi.direction)
        mean_inv_err = float(err_dict['mean_error'])
        
    else:
        reg = ants.registration(
            fixed=fi, moving=mi, type_of_transform='SyN',
            initial_transform=aff_tx,
            **params
        )
        fwd = reg['fwdtransforms']
        inv = reg['invtransforms']
        
        try:
            w_fwd_ants = ants.image_read(fwd[0])
            w_inv_ants = ants.image_read(inv[1])
            w_fwd_np = w_fwd_ants.numpy()
            w_inv_np = w_inv_ants.numpy()
            if w_fwd_np.ndim == 4 and w_fwd_np.shape[-1] == 3:
                bnd_fwd = compute_3d_bending_energy(w_fwd_np, fi.spacing)
                bnd_inv = compute_3d_bending_energy(w_inv_np, fi.spacing)
            else:
                bnd_fwd = bnd_inv = 0.0
        except Exception as e:
            print("ANTs bnd error:", e)
            bnd_fwd = bnd_inv = 0.0
            
        mean_inv_err = 0.0
        
    elapsed = time.time() - t0
    
    # Calculate bidirectional dice
    ml_w = ants.apply_transforms(fixed=fi, moving=ml, transformlist=fwd, interpolator='nearestNeighbor')
    ov_f = ants.label_overlap_measures(fl, ml_w)
    df_f = ov_f[~ov_f['Label'].astype(str).isin(['All', '0', '0.0'])]
    d_fixed = float(df_f['TotalOrTargetOverlap'].mean() if 'TotalOrTargetOverlap' in df_f.columns else df_f['TargetOverlap'].mean())
    
    whichtoinvert_inv = reg.get('whichtoinvert_inv', [True, False]) if isinstance(reg, dict) and engine == 'syntx' else [True, False]
    
    fl_w = ants.apply_transforms(fixed=mi, moving=fl, transformlist=inv, whichtoinvert=whichtoinvert_inv, interpolator='nearestNeighbor')
    ov_m = ants.label_overlap_measures(ml, fl_w)
    df_m = ov_m[~ov_m['Label'].astype(str).isin(['All', '0', '0.0'])]
    d_moving = float(df_m['TotalOrTargetOverlap'].mean() if 'TotalOrTargetOverlap' in df_m.columns else df_m['TargetOverlap'].mean())
    
    dsym = 0.5 * (d_fixed + d_moving)
    
    # Calculate Jacobians
    fwd_tx = fwd[0]
    inv_tx = inv[1] # For BOTH syntx and ants, invtransforms = [affine, warp_inv] so the warp is at index 1
    
    jac_fwd_img = ants.create_jacobian_determinant_image(fi, fwd_tx, do_log=False)
    jac_inv_img = ants.create_jacobian_determinant_image(mi, inv_tx, do_log=False)
    
    jac_fwd = jac_fwd_img.numpy()
    jac_inv = jac_inv_img.numpy()
    
    # Create masks that perfectly match the physical grids of the Jacobian images
    mask_fwd = jac_fwd != 0.0 # simple foreground mask or we can use ants.get_mask on the jac image
    mask_inv = jac_inv != 0.0
    
    # actually, the jacobian determinant is 1.0 in the background. 
    # To get a brain mask that matches the jacobian shape, we can resample the original masks:
    mask_f_resampled = ants.resample_image_to_target(ants.get_mask(fi), jac_fwd_img, interp_type='nearestNeighbor').numpy() > 0
    mask_m_resampled = ants.resample_image_to_target(ants.get_mask(mi) if jac_inv_img.shape == mi.shape else ants.get_mask(fi), jac_inv_img, interp_type='nearestNeighbor').numpy() > 0
    
    fold_pct_fwd = float(np.mean(jac_fwd[mask_f_resampled] <= 0) * 100.0) if mask_f_resampled.any() else 0.0
    fold_pct_inv = float(np.mean(jac_inv[mask_m_resampled] <= 0) * 100.0) if mask_m_resampled.any() else 0.0
    jac_min_fwd = float(np.min(jac_fwd[mask_f_resampled])) if mask_f_resampled.any() else 1.0
    jac_min_inv = float(np.min(jac_inv[mask_m_resampled])) if mask_m_resampled.any() else 1.0
    
    result = {
        'idx': idx,
        'engine': engine,
        'params': params,
        'metrics': {
            'dice_fwd': d_fixed,
            'dice_inv': d_moving,
            'dice_sym': dsym,
            'bnd_fwd': bnd_fwd,
            'bnd_inv': bnd_inv,
            'fold_pct_fwd': fold_pct_fwd,
            'fold_pct_inv': fold_pct_inv,
            'jac_min_fwd': jac_min_fwd,
            'jac_min_inv': jac_min_inv,
            'mean_inv_err': mean_inv_err,
            'time_s': elapsed
        }
    }
    
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    main()
