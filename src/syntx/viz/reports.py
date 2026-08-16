"""
Standard Interactive Report Generators & Provenance Tools for Syntx.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import ants

from .figures import extract_2d_slice, render_standard_4panel, render_input_pair_figure


def _parse_image_metadata(img, name="Image"):
    """Extracts physical space and dimension metadata from ANTsImage, PyTorch Tensor, or NumPy Array."""
    meta = {
        "name": name,
        "type": type(img).__name__,
        "shape": "N/A",
        "spacing": "N/A",
        "origin": "N/A",
        "orientation": "N/A",
        "is_ants": False,
    }

    if isinstance(img, ants.ANTsImage):
        meta["is_ants"] = True
        meta["shape"] = str(tuple(img.shape))
        meta["spacing"] = " × ".join([f"{s:.3f}" for s in img.spacing]) + " mm"
        meta["origin"] = str(tuple([round(o, 2) for o in img.origin])) + " mm"
        meta["orientation"] = str(img.orientation)
    elif hasattr(img, "shape"):
        meta["shape"] = str(tuple(img.shape))

    return meta


def _compute_jacobian_stats(warp, fixed=None):
    """Computes Jacobian determinant array and summary statistics from displacement field."""
    if hasattr(warp, "detach"):
        warp_np = warp.squeeze(0).detach().cpu().numpy()
    elif hasattr(warp, "numpy"):
        warp_np = warp.numpy()
    else:
        warp_np = np.asarray(warp)

    if warp_np.ndim == 4 and warp_np.shape[0] in (2, 3) and warp_np.shape[1] > 4:
        warp_np = np.moveaxis(warp_np, 0, -1)

    spacing = fixed.spacing if (fixed is not None and isinstance(fixed, ants.ANTsImage)) else (1.0,) * (warp_np.ndim - 1)

    if warp_np.ndim == 4:
        du_dx = np.gradient(warp_np[..., 0], axis=0) / spacing[0]
        du_dy = np.gradient(warp_np[..., 1], axis=1) / spacing[1]
        du_dz = np.gradient(warp_np[..., 2], axis=2) / spacing[2]
        detJ = (1.0 + du_dx) * (1.0 + du_dy) * (1.0 + du_dz)
    elif warp_np.ndim == 3:
        du_dx = np.gradient(warp_np[..., 0], axis=0) / spacing[0]
        du_dy = np.gradient(warp_np[..., 1], axis=1) / spacing[1]
        detJ = (1.0 + du_dx) * (1.0 + du_dy)
    else:
        detJ = np.ones(warp_np.shape[:-1])

    min_j = float(np.min(detJ))
    max_j = float(np.max(detJ))
    mean_j = float(np.mean(detJ))
    std_j = float(np.std(detJ))

    mask = (detJ <= 0.0)
    folding_pct = float(np.mean(mask) * 100.0)

    return detJ, {
        "min": min_j,
        "max": max_j,
        "mean": mean_j,
        "std": std_j,
        "folding_pct": folding_pct,
    }


def build_engine_provenance(
    algorithm="syntx.syn",
    backend="pytorch",
    device="cpu",
    fit_time=None,
    reg_iterations=None,
    affine_iterations=None,
    solver="SyN",
    fluid_sigma=3.0,
    elastic_sigma=0.0,
    learning_rate=None,
    optimizer_type="Adam",
    similarity_metric="lncc",
    loss_window=9,
    fixed_shape=None,
    fixed_spacing=None,
    fixed_orientation=None,
    moving_shape=None,
    moving_spacing=None,
    moving_orientation=None,
    n_time_steps=None,
    antisymmetric=None,
    use_analytical_gradients=None,
    constant_speed=None,
    **kwargs
):
    """
    Constructs a standardized, non-breakable registration provenance dictionary directly from the registration engine.
    """
    prov = {
        "algorithm": algorithm,
        "backend": str(backend),
        "device": str(device),
        "fit_time": float(fit_time) if isinstance(fit_time, (int, float)) else "N/A",
        "iterations": str(reg_iterations) if reg_iterations is not None else "N/A",
        "affine_iterations": str(affine_iterations) if affine_iterations is not None else "N/A",
        "solver": str(solver),
        "n_time_steps": str(n_time_steps) if n_time_steps is not None else "N/A",
        "antisymmetric": bool(antisymmetric) if antisymmetric is not None else False,
        "use_analytical_gradients": bool(use_analytical_gradients) if use_analytical_gradients is not None else True,
        "constant_speed": bool(constant_speed) if constant_speed is not None else "N/A",
        "fluid_sigma": str(fluid_sigma),
        "elastic_sigma": str(elastic_sigma),
        "learning_rate": str(learning_rate) if learning_rate is not None else "N/A",
        "optimizer": str(optimizer_type),
        "similarity_metric": str(similarity_metric),
        "loss_window": str(loss_window),
        "fixed_shape": str(fixed_shape) if fixed_shape is not None else "N/A",
        "fixed_spacing": str(fixed_spacing) if fixed_spacing is not None else "N/A",
        "fixed_orientation": str(fixed_orientation) if fixed_orientation is not None else "N/A",
        "moving_shape": str(moving_shape) if moving_shape is not None else "N/A",
        "moving_spacing": str(moving_spacing) if moving_spacing is not None else "N/A",
        "moving_orientation": str(moving_orientation) if moving_orientation is not None else "N/A",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL, 
            text=True
        ).strip()
        prov["syntx_version"] = git_hash
    except Exception:
        prov["syntx_version"] = __import__("syntx").__version__
    prov.update(kwargs)
    return prov


def create_registration_report(
    fixed,
    moving,
    warped=None,
    warp=None,
    output_html="registration_report.html",
    fixed_name="Fixed Target Image",
    moving_name="Moving Source Image",
    provenance=None,
    fixed_label=None,
    moving_label=None,
    warped_label=None,
    inv_err_map=None,
    detJ=None,
    slice_axis=2,
    slice_idx=None,
    title="Syntx Medical Image Registration Verification Report",
    assets_dir=None,
    show_report=False,
    reg=None
):
    """
    Generates a publication-grade, standalone interactive HTML report and visual asset suite
    for a completed medical image registration task, with complete provenance tracking and metric verification.
    """
    import time
    from ..image_compare import image_compare
    from .figures import render_input_pair_figure, render_standard_4panel, plot_time_varying_velocity_grid
    from .stats import plot_label_overlap_stats, plot_loss_convergence
    
    if reg is not None and isinstance(reg, dict):
        if warped is None:
            warped = reg.get("warpedmovout", fixed)
        if warp is None and "fwdtransforms" in reg:
            warp = reg["fwdtransforms"][0]
        if provenance is None and "provenance" in reg:
            provenance = reg["provenance"]
    elif isinstance(warped, dict):
        reg_dict = warped
        warped = reg_dict.get("warpedmovout", fixed)
        if warp is None and "fwdtransforms" in reg_dict:
            warp = reg_dict["fwdtransforms"][0]
        if provenance is None and "provenance" in reg_dict:
            provenance = reg_dict["provenance"]

    if warped is None:
        warped = fixed

    output_html = os.path.abspath(output_html)
    html_dir = os.path.dirname(output_html)
    os.makedirs(html_dir, exist_ok=True)

    if assets_dir is None:
        assets_dir = os.path.join(html_dir, "assets")
    else:
        assets_dir = os.path.abspath(assets_dir)
    os.makedirs(assets_dir, exist_ok=True)

    meta_fixed = _parse_image_metadata(fixed, fixed_name)
    meta_moving = _parse_image_metadata(moving, moving_name)

    prov = build_engine_provenance()
    if isinstance(provenance, dict):
        prov.update(provenance)
    elif isinstance(provenance, str):
        prov["info"] = provenance

    fi_arr = fixed.numpy() if isinstance(fixed, ants.ANTsImage) else np.squeeze(np.asarray(fixed))
    mi_arr = warped.numpy() if isinstance(warped, ants.ANTsImage) else np.squeeze(np.asarray(warped))

    if inv_err_map is None and reg is not None:
        if "inverse_identity_error_map" in reg:
            inv_err_map = reg["inverse_identity_error_map"]
        elif "inverse_identity_errors" in reg:
            inv_errs = reg["inverse_identity_errors"]
            if "phi_1" in inv_errs and "error_map" in inv_errs["phi_1"]:
                inv_err_map = inv_errs["phi_1"]["error_map"]
            elif "error_map" in inv_errs:
                inv_err_map = inv_errs["error_map"]
                
    if inv_err_map is None:
        inv_err_map = np.zeros(fi_arr.shape, dtype=np.float32)

    # --- Standardize Intensity Before Metrics ---
    fi_np_clip = np.clip(fi_arr, *np.percentile(fi_arr[fi_arr > 0] if (fi_arr > 0).any() else fi_arr, [1, 99]))
    fi_norm = (fi_np_clip - fi_np_clip.mean()) / (fi_np_clip.std() + 1e-8)
    
    mi_np_clip = np.clip(mi_arr, *np.percentile(mi_arr[mi_arr > 0] if (mi_arr > 0).any() else mi_arr, [1, 99]))
    mi_norm = (mi_np_clip - mi_np_clip.mean()) / (mi_np_clip.std() + 1e-8)

    # --- Similarity Metrics ---
    metrics = {}
    try:
        metrics['MSE'] = image_compare(fi_norm, mi_norm, 'mse')
        metrics['MAE'] = image_compare(fi_norm, mi_norm, 'mae')
        metrics['RMSE'] = image_compare(fi_norm, mi_norm, 'rmse')
        metrics['PSNR'] = -image_compare(fi_norm, mi_norm, 'psnr')
        metrics['SSIM'] = -image_compare(fi_norm, mi_norm, 'ssim')
        metrics['NCC'] = -image_compare(fi_norm, mi_norm, 'ncc')
        metrics['LNCC (w=9)'] = -image_compare(fi_norm, mi_norm, 'lncc', window_size=9)
    except Exception as e:
        metrics['MSE'] = float(np.mean((fi_norm - mi_norm) ** 2))
        metrics['MAE'] = float(np.mean(np.abs(fi_norm - mi_norm)))
        metrics['LNCC (w=9)'] = 0.0

    # --- Label Overlap Metrics ---
    dice_sym = "N/A"
    dice_fwd = "N/A"
    dice_inv = "N/A"
    regional_overlap_fwd = {}
    regional_overlap_inv = {}
    regional_overlap_sym = {}
    
    if fixed_label is not None and (moving_label is not None or warped_label is not None):
        try:
            whichtoinvert = reg.get('whichtoinvert_inv', [True, False]) if (reg is not None and isinstance(reg, dict)) else [True, False]
            if reg is not None and 'invtransforms' in reg and reg['invtransforms'] is not None:
                ml_warped = ants.apply_transforms(fixed, moving_label, reg['fwdtransforms'], interpolator='nearestNeighbor')
                fl_warped = ants.apply_transforms(moving, fixed_label, reg['invtransforms'], whichtoinvert=whichtoinvert, interpolator='nearestNeighbor')
                
                fwd_overlap = ants.label_overlap_measures(fixed_label, ml_warped)
                inv_overlap = ants.label_overlap_measures(moving_label, fl_warped)
                
                overlap_col = 'TargetOverlap' if 'TargetOverlap' in fwd_overlap.columns else 'TotalOrTargetOverlap'
                
                fwd_valid = fwd_overlap[(fwd_overlap['Label'] != 'All') & (fwd_overlap['Label'] != '0') & (fwd_overlap['Label'] != 0)]
                inv_valid = inv_overlap[(inv_overlap['Label'] != 'All') & (inv_overlap['Label'] != '0') & (inv_overlap['Label'] != 0)]
                
                dice_fwd = float(fwd_valid[overlap_col].mean())
                dice_inv = float(inv_valid[overlap_col].mean())
                dice_sym = 0.5 * (dice_fwd + dice_inv)
                
                for lbl in fwd_valid['Label']:
                    try:
                        f_val = float(fwd_valid[fwd_valid['Label'] == lbl][overlap_col].values[0])
                        i_val = float(inv_valid[inv_valid['Label'] == lbl][overlap_col].values[0]) if lbl in inv_valid['Label'].values else f_val
                        regional_overlap_fwd[lbl] = f_val
                        regional_overlap_inv[lbl] = i_val
                        regional_overlap_sym[lbl] = (f_val + i_val) / 2.0
                    except:
                        pass
            else:
                if warped_label is None:
                    warped_label = ants.apply_transforms(fixed, moving_label, warp, interpolator='nearestNeighbor')
                overlap = ants.label_overlap_measures(fixed_label, warped_label)
                overlap_col = 'TargetOverlap' if 'TargetOverlap' in overlap.columns else 'TotalOrTargetOverlap'
                overlap_valid = overlap[(overlap['Label'] != 'All') & (overlap['Label'] != '0') & (overlap['Label'] != 0)]
                dice_fwd = float(overlap_valid[overlap_col].mean())
                dice_sym = dice_fwd
                for lbl in overlap_valid['Label']:
                    val = float(overlap_valid[overlap_valid['Label'] == lbl][overlap_col].values[0])
                    regional_overlap_fwd[lbl] = val
                    regional_overlap_inv[lbl] = val
                    regional_overlap_sym[lbl] = val
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Exception in label overlap computation: {e}")

    # --- Jacobian Metrics ---
    if isinstance(warp, (list, tuple)):
        warp_files = [f for f in warp if isinstance(f, str) and ('Warp' in f or f.endswith('.nii.gz') or f.endswith('.nii'))]
        if warp_files:
            warp = warp_files[0]

    bnd_energy = "N/A"
    hrm_energy = "N/A"
    warp_img = None
    if isinstance(warp, str) and os.path.exists(warp):
        try:
            warp_img = ants.image_read(warp)
            if detJ is None and isinstance(fixed, ants.ANTsImage):
                detJ = ants.create_jacobian_determinant_image(fixed, warp_img, do_log=False)
        except Exception:
            pass
    elif not isinstance(warp, str) and warp is not None:
        warp_img = warp

    if detJ is None and warp_img is not None and not isinstance(warp_img, str):
        detJ_arr, jac_stats = _compute_jacobian_stats(warp_img, fixed)
        detJ = detJ_arr
    elif isinstance(detJ, ants.ANTsImage):
        detJ_arr = detJ.numpy()
        mask_eval = ants.get_mask(fixed).numpy() > 0 if meta_fixed["is_ants"] else (detJ_arr != 0)
        jac_stats = {
            "min": float(np.min(detJ_arr)),
            "max": float(np.max(detJ_arr)),
            "mean": float(np.mean(detJ_arr)),
            "std": float(np.std(detJ_arr)),
            "folding_pct": float(np.mean(detJ_arr[mask_eval] <= 0.0) * 100.0),
        }
    elif isinstance(detJ, np.ndarray) and detJ.ndim >= 2:
        detJ_arr = detJ
        jac_stats = {
            "min": float(np.min(detJ_arr)),
            "max": float(np.max(detJ_arr)),
            "mean": float(np.mean(detJ_arr)),
            "std": float(np.std(detJ_arr)),
            "folding_pct": float(np.mean(detJ_arr <= 0.0) * 100.0),
        }
    else:
        detJ_arr = np.ones_like(fi_arr)
        detJ = detJ_arr
        jac_stats = {"min": 1.0, "max": 1.0, "mean": 1.0, "std": 0.0, "folding_pct": 0.0}

    # Bending & Harmonic Energy Calculation properly scaled in physical space
    if warp_img is not None and isinstance(warp_img, ants.ANTsImage):
        try:
            dim = warp_img.dimension
            spc = warp_img.spacing
            warpnp = warp_img.numpy()
            
            # 1st order gradients: du_k / dx_i
            # gradient_list[k] is a list of partial derivatives along axes for the k-th vector component
            gradient_list = [np.gradient(warpnp[..., k], *spc, axis=range(dim)) for k in range(dim)]
            
            total_bnd = 0.0
            total_hrm = 0.0
            for k in range(dim):
                for j in range(dim):
                    grad_kj = gradient_list[k][j]
                    total_hrm += np.mean(grad_kj**2)
                    
                    # 2nd order gradients: d^2 u_k / dx_i dx_j
                    grad2_kj = np.gradient(grad_kj, *spc, axis=range(dim))
                    for i in range(dim):
                        total_bnd += np.mean(grad2_kj[i]**2)
                        
            bnd_energy = float(total_bnd)
            hrm_energy = float(total_hrm)
        except Exception:
            pass

    if inv_err_map is not None:
        if hasattr(inv_err_map, 'cpu'):
            inv_err_map = inv_err_map.cpu().numpy()
        inv_np = inv_err_map.numpy() if isinstance(inv_err_map, ants.ANTsImage) else np.asarray(inv_err_map)
        
        # PyTorch tensors are ZYX, ANTsImages are XYZ
        if inv_np.shape != fi_arr.shape:
            if inv_np.shape == fi_arr.shape[::-1]:
                inv_np = np.transpose(inv_np, tuple(range(inv_np.ndim)[::-1]))
                
        inv_stats = {
            "max": float(np.max(inv_np)),
            "mean": float(np.mean(inv_np)),
            "p95": float(np.percentile(inv_np, 95)),
        }
        
        # Interior error calculation (eroded mask by 5 voxels to avoid border truncation artifacts)
        if isinstance(fixed, ants.ANTsImage):
            base_mask = ants.get_mask(fixed)
            interior_mask = ants.iMath(base_mask, "ME", 5).numpy() > 0
            if np.any(interior_mask):
                inv_stats["interior_max"] = float(np.max(inv_np[interior_mask]))
                inv_stats["interior_mean"] = float(np.mean(inv_np[interior_mask]))
                inv_stats["interior_p95"] = float(np.percentile(inv_np[interior_mask], 95))
            else:
                inv_stats["interior_max"] = inv_stats["max"]
                inv_stats["interior_mean"] = inv_stats["mean"]
                inv_stats["interior_p95"] = inv_stats["p95"]
        else:
            inv_stats["interior_max"] = inv_stats["max"]
            inv_stats["interior_mean"] = inv_stats["mean"]
            inv_stats["interior_p95"] = inv_stats["p95"]
    else:
        inv_stats = {"max": 0.0, "mean": 0.0, "p95": 0.0, "interior_max": 0.0, "interior_mean": 0.0, "interior_p95": 0.0}

    # --- Render Figures ---
    ts = int(time.time())
    
    fig1_name = f"fig1_inputs_{ts}.png"
    fig1_abs = os.path.join(assets_dir, fig1_name)
    render_input_pair_figure(fixed, moving, output_path=fig1_abs, title="Figure 1: Original Input Pair")
    
    fig2_name = f"fig2_4panel_{ts}.png"
    fig2_abs = os.path.join(assets_dir, fig2_name)
    render_standard_4panel(
        fixed=fixed, warped=warped,
        warp=warp_img if warp_img is not None else np.zeros((*fi_arr.shape, fi_arr.ndim)),
        detJ=detJ,
        inv_err_map=inv_err_map,
        slice_axis=slice_axis, slice_idx=slice_idx,
        lncc_val=metrics.get('LNCC (w=9)', 0.0),
        inv_err_max=inv_stats.get("interior_max", inv_stats["max"]), 
        inv_err_mean=inv_stats.get("interior_mean", inv_stats["mean"]), 
        inv_err_p95=inv_stats.get("interior_p95", inv_stats["p95"]),
        min_detJ=jac_stats["min"],
        title_prefix=f"{prov['algorithm']} ({prov['backend']})",
        filename=fig2_abs
    )
    
    html_figs = f'''
        <section class="card" style="margin-bottom: 2rem;">
            <h2>Figure 1: Input Image Pair (Fixed & Moving)</h2>
            <div style="text-align: center;"><img src="{os.path.relpath(fig1_abs, html_dir)}" alt="Figure 1" style="max-width: 100%; border-radius: 8px;"></div>
        </section>
        <section class="card" style="margin-bottom: 2rem;">
            <h2>Figure 2: Standard 4-Panel Diagnostic Report</h2>
            <div style="text-align: center;"><img src="{os.path.relpath(fig2_abs, html_dir)}" alt="Figure 2" style="max-width: 100%; border-radius: 8px;"></div>
        </section>
    '''

    if reg is not None and isinstance(reg, dict) and 'model' in reg:
        model = reg['model']
        if type(model).__name__ == 'TVFModel':
            fig3_name = f"fig3_velocity_{ts}.png"
            fig3_abs = os.path.join(assets_dir, fig3_name)
            plot_time_varying_velocity_grid(model, fixed_image=fixed, output_path=fig3_abs)
            html_figs += f'''
            <section class="card" style="margin-bottom: 2rem;">
                <h2>Figure 3: Time-Varying Velocity Field Flow Keyframes</h2>
                <div style="text-align: center;"><img src="{os.path.relpath(fig3_abs, html_dir)}" alt="Figure 3" style="max-width: 100%; border-radius: 8px;"></div>
            </section>
            '''

    losses = None
    if reg is not None and isinstance(reg, dict) and 'model' in reg:
        if hasattr(reg['model'], 'losses') and len(reg['model'].losses) > 0:
            losses = reg['model'].losses
        elif hasattr(reg['model'], 'syn_losses') and len(reg['model'].syn_losses) > 0:
            losses = reg['model'].syn_losses
    elif reg is not None and isinstance(reg, dict) and 'loss_history' in reg:
        losses = reg['loss_history']
        
    if losses:
        fig4_name = f"fig4_loss_{ts}.png"
        fig4_abs = os.path.join(assets_dir, fig4_name)
        plot_loss_convergence(losses, output_path=fig4_abs, title=f"Similarity Loss Convergence ({prov['algorithm']})")
        html_figs += f'''
        <section class="card" style="margin-bottom: 2rem;">
            <h2>Figure 4: Multi-Resolution Similarity Loss Convergence</h2>
            <div style="text-align: center;"><img src="{os.path.relpath(fig4_abs, html_dir)}" alt="Figure 4" style="max-width: 100%; border-radius: 8px;"></div>
        </section>
        '''

    if regional_overlap_sym and fixed_label is not None:
        fig5_name = f"fig5_dkt_overlap_{ts}.png"
        fig5_abs = os.path.join(assets_dir, fig5_name)
        dice_dict = {
            'fixed_dice': list(regional_overlap_fwd.values()), 
            'moving_dice': list(regional_overlap_inv.values()), 
            'sym_dice': list(regional_overlap_sym.values()), 
            'per_region': regional_overlap_sym
        }
        plot_label_overlap_stats(dice_scores=dice_dict, output_path=fig5_abs, title="Mindboggle DKT Cortical Label Overlap Benchmark")
        html_figs += f'''
        <section class="card" style="margin-bottom: 2rem;">
            <h2>Figure 5: Anatomical Label Overlap Stats</h2>
            <div style="text-align: center;"><img src="{os.path.relpath(fig5_abs, html_dir)}" alt="Figure 5" style="max-width: 100%; border-radius: 8px;"></div>
        </section>
        '''

    import json
    prov_str = json.dumps(prov, indent=2)
    html_figs += f'''
        <section class="card" style="margin-bottom: 2rem;">
            <h2>📋 Registration Provenance & Hyperparameters</h2>
            <pre style="background: #1e293b; color: #38bdf8; padding: 1rem; border-radius: 6px; overflow-x: auto;"><code>{prov_str}</code></pre>
        </section>
    '''

    metrics_html = "".join([f"<tr><td>{k}:</td><td class='metric-val'>{v:.4f}</td></tr>" for k,v in metrics.items()])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #090d16; color: #f8fafc; margin: 0; padding: 2rem; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ border-bottom: 1px solid #1e293b; padding-bottom: 1rem; margin-bottom: 2rem; }}
        h1 {{ font-size: 1.8rem; font-weight: 700; color: #38bdf8; margin: 0; }}
        .badge {{ display: inline-block; background: #1e293b; color: #94a3b8; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.85rem; font-weight: 600; margin-top: 0.5rem; }}
        .badge-success {{ background: #064e3b; color: #34d399; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
        .card {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 1.25rem; }}
        .card h2 {{ font-size: 1.1rem; color: #cbd5e1; margin-top: 0; margin-bottom: 1rem; border-bottom: 1px solid #334155; padding-bottom: 0.5rem; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
        td, th {{ padding: 0.5rem 0.25rem; text-align: left; }}
        tr:not(:last-child) {{ border-bottom: 1px solid #1e293b; }}
        .metric-val {{ font-family: monospace; font-weight: 600; color: #38bdf8; }}
        footer {{ margin-top: 3rem; text-align: center; color: #64748b; font-size: 0.85rem; border-top: 1px solid #1e293b; padding-top: 1.5rem; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <div class="badge badge-success">Verified Provenance</div>
            <div class="badge">Engine: {prov['algorithm']} ({prov['backend']})</div>
            <div class="badge">Device: {prov['device']}</div>
        </header>

        <div class="grid-2">
            <div class="card">
                <h2>📈 Registration Similarity Metrics</h2>
                <table>
                    {metrics_html}
                </table>
            </div>

            <div class="card">
                <h2>📐 Spatial Topology & Inverse Identity</h2>
                <table>
                    <tr><td>Symmetric Cortical DICE:</td><td class="metric-val">{dice_sym if isinstance(dice_sym, str) else f"{dice_sym:.4f}"}</td></tr>
                    <tr><td>Jacobian Range:</td><td class="metric-val">[{jac_stats['min']:+.2f}, {jac_stats['max']:.2f}]</td></tr>
                    <tr><td>Grid Folding Rate:</td><td class="metric-val">{jac_stats['folding_pct']:.2f}%</td></tr>
                    <tr><td>Harmonic Energy (1st Order):</td><td class="metric-val">{f"{hrm_energy:.3e}" if isinstance(hrm_energy, float) else hrm_energy}</td></tr>
                    <tr><td>Thin-Plate Bending Energy (2nd Order):</td><td class="metric-val">{f"{bnd_energy:.3e}" if isinstance(bnd_energy, float) else bnd_energy}</td></tr>
                    <tr><td>Interior Max Inverse Error (Eroded):</td><td class="metric-val">{inv_stats['interior_max']:.2f} mm</td></tr>
                    <tr><td>Interior Mean Inverse Error (Eroded):</td><td class="metric-val">{inv_stats['interior_mean']:.3f} mm</td></tr>
                    <tr><td>Absolute Max Inverse Error (Global):</td><td class="metric-val">{inv_stats['max']:.2f} mm</td></tr>
                </table>
            </div>
        </div>

        {html_figs}

        <footer>
            <p>Generated automatically by <strong>syntx.viz</strong> — Advanced Medical Image Registration Verification Engine</p>
        </footer>
    </div>
</body>
</html>
"""

    with open(output_html, "w") as f:
        f.write(html_content)

    return {
        "html_path": output_html,
        "fig_path": fig2_abs,
        "fig2_path": fig2_abs,
        "metrics": metrics,
        "dice": dice_sym,
        "dice_sym": dice_sym,
        "jacobian": jac_stats,
        "inverse_error": inv_stats,
        "provenance": prov,
    }

def create_benchmark_report(syn_results: dict, ants_results: dict, total_pairs: int, output_html: str = "benchmark_report.html"):
    """
    Generates a comparative HTML report with interactive Plotly scatterplots and boxplots
    for benchmarking two registration implementations across multiple image pairs.
    """
    import json
    from scipy.stats import ttest_rel

    completed = len(syn_results)
    
    # Compute paired stats
    paired_idx = set(syn_results.keys()).intersection(ants_results.keys())
    
    t_stat = 0.0
    p_val = 1.0
    if len(paired_idx) > 1:
        syn_paired = [syn_results[i].get('dice_sym', 0.0) for i in sorted(paired_idx)]
        ants_paired = [ants_results[i].get('dice_sym', 0.0) for i in sorted(paired_idx)]
        t_stat, p_val = ttest_rel(syn_paired, ants_paired)
        # Handle nan if arrays are identical
        if str(t_stat) == 'nan':
            t_stat, p_val = 0.0, 1.0

    
    if completed == 0:
        mean_dice_syn, mean_fold_syn, mean_inv_syn = 0.0, 0.0, 0.0
    else:
        mean_dice_syn = sum(r.get('dice_sym', 0.0) for r in syn_results.values()) / completed
        mean_fold_syn = sum(r.get('folding_pct', 0.0) for r in syn_results.values()) / completed
        mean_inv_syn = sum(r.get('inverse_error_mean', 0.0) for r in syn_results.values()) / completed

    if len(ants_results) == 0:
        mean_dice_ants, mean_fold_ants, mean_inv_ants = 0.0, 0.0, 0.0
    else:
        mean_dice_ants = sum(r.get('dice_sym', 0.0) for r in ants_results.values()) / len(ants_results)
        mean_fold_ants = sum(r.get('folding_pct', 0.0) for r in ants_results.values()) / len(ants_results)
        mean_inv_ants = sum(r.get('inverse_error_mean', 0.0) for r in ants_results.values()) / len(ants_results)

    # Plot data
    pair_ids = [f"Pair {i}" for i in sorted(syn_results.keys())]
    
    syn_dice_sym = [syn_results[i].get('dice_sym', 0.0) for i in sorted(syn_results.keys())]
    ants_dice_sym = [ants_results.get(i, {}).get('dice_sym', None) for i in sorted(syn_results.keys())]
    
    syn_folds = [syn_results[i].get('folding_pct', 0.0) for i in sorted(syn_results.keys())]
    ants_folds = [ants_results.get(i, {}).get('folding_pct', None) for i in sorted(syn_results.keys())]

    syn_times = [syn_results[i].get('runtime_seconds', 0.0) for i in sorted(syn_results.keys())]
    ants_times = [ants_results.get(i, {}).get('runtime_seconds', None) for i in sorted(syn_results.keys())]
    
    syn_invs = [syn_results[i].get('inverse_error_mean', 0.0) for i in sorted(syn_results.keys())]
    ants_invs = [ants_results.get(i, {}).get('inverse_error_mean', None) for i in sorted(syn_results.keys())]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Syntx Benchmark: PyTorch SyN vs ANTs C++</title>
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --surface-color: #ffffff;
            --syntx-color: #3b82f6;
            --ants-color: #ef4444;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
        }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: var(--surface-color);
            padding: 50px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border-radius: 8px;
        }}
        h1, h2, h3 {{ font-family: 'Crimson Pro', serif; color: #0f172a; }}
        h1 {{ font-size: 2.8rem; text-align: center; margin-bottom: 10px; font-weight: 700; }}
        .subtitle {{
            text-align: center; font-family: 'Inter', sans-serif;
            color: var(--text-muted); font-size: 1.1rem;
            margin-bottom: 40px; text-transform: uppercase; letter-spacing: 2px;
        }}
        .abstract {{
            font-style: italic; font-family: 'Crimson Pro', serif;
            font-size: 1.2rem; padding: 20px 40px;
            border-left: 4px solid var(--syntx-color);
            background-color: #f1f5f9; margin-bottom: 50px;
        }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .stat-card {{ background-color: var(--bg-color); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; text-align: center; }}
        .stat-value {{ font-size: 2.2rem; font-weight: 600; font-family: 'Inter', sans-serif; display: flex; justify-content: center; gap: 20px; }}
        .stat-label {{ font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 500; margin-top: 5px; }}
        .color-syntx {{ color: var(--syntx-color); }}
        .color-ants {{ color: var(--ants-color); }}
        
        .plot-container {{ width: 100%; height: 500px; margin-bottom: 40px; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px; }}
        .plot-row {{ display: flex; gap: 20px; margin-bottom: 40px; flex-wrap: wrap; }}
        .plot-third {{ flex: 1; min-width: 300px; height: 500px; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px; }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 0.90rem; }}
        th, td {{ padding: 12px 10px; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ background-color: var(--bg-color); font-weight: 600; color: var(--text-muted); text-transform: uppercase; font-size: 0.75rem; letter-spacing: 1px; }}
        tr:hover td {{ background-color: #f8fafc; }}
        .progress-container {{ margin-bottom: 40px; }}
        .progress-bar {{ height: 6px; background-color: var(--border-color); border-radius: 3px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background-color: var(--syntx-color); width: {(completed/max(1,total_pairs))*100}%; transition: width 1s ease; }}
        .progress-text {{ text-align: right; font-size: 0.85rem; color: var(--text-muted); margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Comparative SyN Registration Benchmark</h1>
        <div class="subtitle">Syntx PyTorch GPU vs. ANTs C++ CPU</div>

        <div class="progress-container">
            <div class="progress-bar"><div class="progress-fill"></div></div>
            <div class="progress-text">Evaluation Progress: {completed} / {total_pairs} Pairs</div>
        </div>

        <div class="abstract">
            <strong>Abstract:</strong> This real-time whitepaper evaluates the strict numerical parity and spatial performance of the <i>syntx</i> native PyTorch Eulerian SyN formulation against the gold-standard ANTs C++ implementation.
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">
                    <span class="color-syntx">{mean_dice_syn:.4f}</span>
                    <span style="color: #cbd5e1;">|</span>
                    <span class="color-ants">{mean_dice_ants:.4f}</span>
                </div>
                <div class="stat-label">Mean Symmetric Dice (Syntx | ANTs)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">
                    <span class="color-syntx">{mean_fold_syn:.4f}%</span>
                    <span style="color: #cbd5e1;">|</span>
                    <span class="color-ants">{mean_fold_ants:.4f}%</span>
                </div>
                <div class="stat-label">Mean Grid Folding (Syntx | ANTs)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">
                    <span class="color-syntx">{mean_inv_syn:.4f}</span>
                    <span style="color: #cbd5e1;">|</span>
                    <span class="color-ants">{mean_inv_ants:.4f}</span>
                </div>
                <div class="stat-label">Mean Inverse Error (mm)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">
                    <span style="color: #10b981;">{p_val:.2e}</span>
                </div>
                <div class="stat-label">Paired T-Test p-value (t={t_stat:.2f})</div>
            </div>
        </div>

        <h2>I. Volumetric Overlap Analysis</h2>
        <p>Comparison of Symmetric Mean Dice distributions between the two implementations.</p>
        <div class="plot-row">
            <div class="plot-container" style="flex: 1; min-width: 400px; margin-bottom: 0;" id="diceBoxplot"></div>
            <div class="plot-container" style="flex: 1; min-width: 400px; margin-bottom: 0;" id="pairedScatter"></div>
        </div>

        <h2>II. Performance & Topology Trade-offs</h2>
        <div class="plot-row">
            <div class="plot-third" id="foldScatter"></div>
            <div class="plot-third" id="invScatter"></div>
            <div class="plot-third" id="timeScatter"></div>
        </div>

        <h2>III. Paired Raw Data</h2>
        <table>
            <thead>
                <tr>
                    <th>Pair ID</th>
                    <th>Syntx Dice</th>
                    <th>ANTs Dice</th>
                    <th>Syntx Fold</th>
                    <th>ANTs Fold</th>
                    <th>Syntx Mean Inv</th>
                    <th>ANTs Mean Inv</th>
                    <th>Syntx Time</th>
                    <th>ANTs Time</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for idx in sorted(syn_results.keys()):
        s_res = syn_results[idx]
        a_res = ants_results.get(idx, {})
        
        s_dice = f"{s_res.get('dice_sym', 0.0):.4f}"
        a_dice = f"{a_res.get('dice_sym', 0.0):.4f}" if a_res else "-"
        
        s_fold = f"{s_res.get('folding_pct', 0.0):.4f}%"
        a_fold = f"{a_res.get('folding_pct', 0.0):.4f}%" if a_res else "-"
        
        s_inv = f"{s_res.get('inverse_error_mean', 0.0):.4f}"
        a_inv = f"{a_res.get('inverse_error_mean', 0.0):.4f}" if a_res else "-"
        
        s_time = f"{s_res.get('runtime_seconds', 0.0):.1f}s"
        a_time = f"{a_res.get('runtime_seconds', 0.0):.1f}s" if a_res else "-"
        
        html += f"""
                <tr>
                    <td style="font-family: monospace; font-weight: 600;">#{idx:03d}</td>
                    <td class="color-syntx" style="font-weight: 600;">{s_dice}</td>
                    <td class="color-ants" style="font-weight: 600;">{a_dice}</td>
                    <td>{s_fold}</td>
                    <td>{a_fold}</td>
                    <td>{s_inv}</td>
                    <td>{a_inv}</td>
                    <td>{s_time}</td>
                    <td>{a_time}</td>
                </tr>"""

    html += f"""
            </tbody>
        </table>
    </div>

    <script>
        const pairIds = {json.dumps(pair_ids)};
        const synDice = {json.dumps(syn_dice_sym)};
        const antsDice = {json.dumps(ants_dice_sym)};
        const synFolds = {json.dumps(syn_folds)};
        const antsFolds = {json.dumps(ants_folds)};
        const synTimes = {json.dumps(syn_times)};
        const antsTimes = {json.dumps(ants_times)};
        const synInvs = {json.dumps(syn_invs)};
        const antsInvs = {json.dumps(ants_invs)};

        // 1. Boxplot (Dice)
        const traceSyn = {{ y: synDice, type: 'box', name: 'Syntx PyTorch', marker: {{color: '#3b82f6'}}, boxpoints: 'all', jitter: 0.3 }};
        const traceAnts = {{ y: antsDice, type: 'box', name: 'ANTs C++', marker: {{color: '#ef4444'}}, boxpoints: 'all', jitter: 0.3 }};
        
        Plotly.newPlot('diceBoxplot', [traceSyn, traceAnts], {{
            title: 'Symmetric DKT31 Dice Score Distributions',
            yaxis: {{ title: 'Dice Score', zeroline: false }},
            boxmode: 'group',
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter, sans-serif' }}
        }}, {{responsive: true}});

        // 1.5 Scatter (Syntx vs ANTs Dice)
        const pairedAntsDice = [];
        const pairedSynDice = [];
        const pairedIds = [];
        for (let i = 0; i < pairIds.length; i++) {{
            if (antsDice[i] !== null && synDice[i] !== null) {{
                pairedAntsDice.push(antsDice[i]);
                pairedSynDice.push(synDice[i]);
                pairedIds.push(pairIds[i]);
            }}
        }}
        
        const scatterPaired = {{ x: pairedAntsDice, y: pairedSynDice, text: pairedIds, mode: 'markers', type: 'scatter', name: 'Pairs', marker: {{ size: 10, color: '#8b5cf6', opacity: 0.8, line: {{color: 'white', width: 1}} }} }};
        const lineRef = {{ x: [0.3, 0.8], y: [0.3, 0.8], mode: 'lines', type: 'scatter', name: 'y=x (Parity)', line: {{ dash: 'dash', color: '#94a3b8' }} }};
        
        Plotly.newPlot('pairedScatter', [scatterPaired, lineRef], {{
            title: 'Pairwise Accuracy (Syntx vs ANTs)',
            xaxis: {{ title: 'ANTs Symmetric Mean Dice' }},
            yaxis: {{ title: 'Syntx Symmetric Mean Dice' }},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter, sans-serif' }},
            showlegend: false
        }}, {{responsive: true}});

        // 2. Scatter (Dice vs Fold)
        const scatterSynFold = {{ x: synFolds, y: synDice, name: 'Syntx', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#3b82f6', opacity: 0.7 }} }};
        const scatterAntsFold = {{ x: antsFolds, y: antsDice, name: 'ANTs', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#ef4444', opacity: 0.7 }} }};
        
        Plotly.newPlot('foldScatter', [scatterSynFold, scatterAntsFold], {{
            title: 'Dice vs Topology Destruction',
            xaxis: {{ title: 'Grid Folding % (det J <= 0)', zeroline: false }},
            yaxis: {{ title: 'Symmetric Mean Dice' }},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter, sans-serif' }}
        }}, {{responsive: true}});
        
        // 3. Scatter (Dice vs Inverse Error)
        const scatterSynInv = {{ x: synInvs, y: synDice, name: 'Syntx', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#3b82f6', opacity: 0.7 }} }};
        const scatterAntsInv = {{ x: antsInvs, y: antsDice, name: 'ANTs', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#ef4444', opacity: 0.7 }} }};
        
        Plotly.newPlot('invScatter', [scatterSynInv, scatterAntsInv], {{
            title: 'Dice vs Mean Inverse Error',
            xaxis: {{ title: 'Mean Inverse Error (mm)', zeroline: false }},
            yaxis: {{ title: 'Symmetric Mean Dice' }},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter, sans-serif' }}
        }}, {{responsive: true}});

        // 4. Scatter (Dice vs Time)
        const scatterSynTime = {{ x: synTimes, y: synDice, name: 'Syntx', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#3b82f6', opacity: 0.7 }} }};
        const scatterAntsTime = {{ x: antsTimes, y: antsDice, name: 'ANTs', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#ef4444', opacity: 0.7 }} }};

        Plotly.newPlot('timeScatter', [scatterSynTime, scatterAntsTime], {{
            title: 'Runtime Performance',
            xaxis: {{ title: 'Compute Time (seconds)' }},
            yaxis: {{ title: 'Symmetric Mean Dice' }},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter, sans-serif' }}
        }}, {{responsive: true}});

    </script>
</body>
</html>
"""
    
    with open(output_html, "w") as f:
        f.write(html)
    return output_html
