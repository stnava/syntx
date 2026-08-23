"""
syntx.cli — Unified Command-Line Interface for Syntx
====================================================

Provides command-line entry points for:
- `syntx register`: Zero-configuration 2D/3D symmetric diffeomorphic registration (SyN / TVF) with automated HTML reporting.
- `syntx benchmark`: Cohort benchmarking suite for Mindboggle-101.
- `syntx info`: Environment, device acceleration, and backend introspection.
"""

import os
import sys
import time
import json
import argparse
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import ants

import syntx
from syntx.deformation_metrics import (
    compute_bidirectional_dice,
    compute_harmonic_energy,
    compute_bending_energy,
    compute_jacobian_metrics
)
from syntx.viz import create_registration_report


def normalize_intensity(img: ants.ANTsImage) -> ants.ANTsImage:
    """
    Applies Foreground 2nd–98th Percentile Normalization Policy:
    Clamps and scales non-zero intensities to [0, 1].
    """
    arr = img.numpy()
    pos = arr[arr > 0]
    if len(pos) > 0:
        p02 = float(np.percentile(pos, 2.0))
        p98 = float(np.percentile(pos, 98.0))
        if p98 <= p02 + 1e-4:
            p02 = 0.0
            p98 = float(pos.max())
    else:
        p02 = float(arr.min())
        p98 = float(arr.max())
    norm_arr = np.clip((arr - p02) / (p98 - p02 + 1e-6), 0.0, 1.0).astype(np.float32)
    return img.new_image_like(norm_arr)


def parse_iterations(iter_str: str) -> List[int]:
    """Parses iteration string like '100x100x20' or '100,100,20' or '100 100 20' into integer list."""
    cleaned = iter_str.replace('x', ' ').replace(',', ' ').replace('[', ' ').replace(']', ' ')
    parts = [int(p) for p in cleaned.split() if p.strip()]
    if not parts:
        return [100, 100, 20]
    return parts


def cmd_register(args: argparse.Namespace) -> int:
    """Executes the `syntx register` subcommand."""
    print("=" * 80)
    print("                SYNTX DIFFEOMORPHIC IMAGE REGISTRATION")
    print("=" * 80)

    # 1. Validate inputs
    if not os.path.exists(args.fixed):
        print(f"Error: Fixed target image not found: '{args.fixed}'", file=sys.stderr)
        return 1
    if not os.path.exists(args.moving):
        print(f"Error: Moving source image not found: '{args.moving}'", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = args.prefix
    out_dir = os.path.abspath(args.out_dir)

    print(f"  * Fixed Target : {os.path.abspath(args.fixed)}")
    print(f"  * Moving Source : {os.path.abspath(args.moving)}")
    if args.fixed_label:
        print(f"  * Fixed Label  : {os.path.abspath(args.fixed_label)}")
    if args.moving_label:
        print(f"  * Moving Label : {os.path.abspath(args.moving_label)}")
    print(f"  * Model/Solver : {args.model.upper()} (Regularizer: {args.regularizer})")
    print(f"  * Output Path  : {os.path.join(out_dir, prefix)}*")
    print("=" * 80 + "\n", flush=True)

    # 2. Load images
    t_start = time.time()
    fi_raw = ants.image_read(args.fixed)
    mi_raw = ants.image_read(args.moving)
    fl_raw = ants.image_read(args.fixed_label) if args.fixed_label else None
    ml_raw = ants.image_read(args.moving_label) if args.moving_label else None

    # Intensity Normalization
    fi = normalize_intensity(fi_raw)
    mi = normalize_intensity(mi_raw)

    reg_iterations = parse_iterations(args.iterations)

    # 3. Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    print(f"[syntx] Hardware Acceleration: {device.upper()} (Backend: {args.backend})")

    # 4. Affine Initialization
    aff_0 = None
    aff_inv = None
    if not args.no_affine:
        print(f"[syntx] Computing Deterministic 18-Cone Robust Affine Initialization...", flush=True)
        t_aff_0 = time.time()
        reg_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=args.verbose)
        aff_0 = reg_aff["fwdtransforms"][0]
        aff_inv = reg_aff["invtransforms"][0]
        t_aff = time.time() - t_aff_0
        print(f"[syntx] Affine initialization converged in {t_aff:.2f}s.\n", flush=True)
    else:
        t_aff = 0.0

    if args.model.lower() == "affine":
        fwd_transforms = [aff_0] if aff_0 else []
        inv_transforms = [aff_inv] if aff_inv else []
        t_reg = t_aff
        reg_res = {"fwdtransforms": fwd_transforms, "invtransforms": inv_transforms}
    elif args.model.lower() == "tvf":
        print(f"[syntx] Starting Continuous Time-Varying Velocity Field (TVF) Registration...", flush=True)
        t0 = time.time()
        reg_res = syntx.tvf(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend=args.backend, device=device,
            grad_step=args.grad_step, flow_sigma=args.flow_sigma, total_sigma=args.total_sigma,
            reg_iterations=reg_iterations, similarity_metric=args.similarity_metric,
            regularizer=args.regularizer, optimizer=args.optimizer,
            verbose=args.verbose
        )
        t_reg = time.time() - t0
        fwd_transforms = reg_res["fwdtransforms"]
        inv_transforms = reg_res["invtransforms"]
        print(f"[syntx] TVF registration completed in {t_reg:.2f}s.\n", flush=True)
    else:  # Default: SyN
        print(f"[syntx] Starting Symmetric Diffeomorphic (SyN) Registration...", flush=True)
        t0 = time.time()
        reg_res = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend=args.backend, device=device,
            grad_step=args.grad_step, flow_sigma=args.flow_sigma, total_sigma=args.total_sigma,
            reg_iterations=reg_iterations, similarity_metric=args.similarity_metric,
            use_ants_pseudo_gradient=False, use_analytical_gradients=args.analytical_gradients,
            syn_sampling=args.syn_sampling, fast_smooth=True, inverse_method=args.inverse_method,
            in_loop_inv_steps=args.in_loop_inv_steps, formulation=args.formulation,
            regularizer=args.regularizer, smooth_in_deformed_space=False, antisymmetric=True,
            bootstrap_mode=args.bootstrap_mode,
            bootstrap_orig_weight=args.bootstrap_orig_weight,
            bootstrap_jitter_scale=args.bootstrap_jitter_scale,
            verbose=args.verbose
        )
        t_reg = time.time() - t0
        fwd_transforms = reg_res["fwdtransforms"]
        inv_transforms = reg_res["invtransforms"]
        print(f"[syntx] SyN registration completed in {t_reg:.2f}s.\n", flush=True)

    # 5. Apply Transforms (Single Interpolation Policy)
    print(f"[syntx] Applying composite forward and inverse transforms (Single Interpolation Policy)...", flush=True)
    whichtoinvert_inv = [True] + [False] * (len(inv_transforms) - 1) if len(inv_transforms) > 0 else []

    warped_moving = ants.apply_transforms(
        fixed=fi_raw, moving=mi_raw,
        transformlist=fwd_transforms,
        interpolator=args.interpolator
    )
    inverse_warped_fixed = ants.apply_transforms(
        fixed=mi_raw, moving=fi_raw,
        transformlist=inv_transforms,
        whichtoinvert=whichtoinvert_inv,
        interpolator=args.interpolator
    )

    warped_label = None
    if fl_raw is not None and ml_raw is not None:
        warped_label = ants.apply_transforms(
            fixed=fi_raw, moving=ml_raw,
            transformlist=fwd_transforms,
            interpolator='nearestNeighbor'
        )
        dice_fix, dice_mov, dice_sym = compute_bidirectional_dice(
            fl_raw, ml_raw, fi_raw, mi_raw, fwd_transforms, inv_transforms
        )
    else:
        dice_fix, dice_mov, dice_sym = None, None, None

    # 6. Compute deformation metrics & Jacobian
    warp_file = None
    for tr in fwd_transforms:
        if isinstance(tr, str) and ('Warp' in tr or tr.endswith('.nii.gz') or tr.endswith('.nii')):
            warp_file = tr
            break

    if warp_file:
        harm_energy = compute_harmonic_energy(warp_file, fi_raw.spacing)
        bend_energy = compute_bending_energy(warp_file, fi_raw.spacing)
        jac_img = ants.create_jacobian_determinant_image(fi_raw, warp_file, do_log=False)
        jac_arr = jac_img.numpy()
        mask_fg = ants.get_mask(fi_raw).numpy() > 0
        if not np.any(mask_fg):
            mask_fg = np.ones_like(jac_arr, dtype=bool)
        fold_pct = float(np.mean(jac_arr[mask_fg] <= 0.0) * 100.0)
        min_jac = float(np.min(jac_arr[mask_fg]))
        mean_jac = float(np.mean(jac_arr[mask_fg]))
    else:
        harm_energy, bend_energy, fold_pct, min_jac, mean_jac = 0.0, 0.0, 0.0, 1.0, 1.0
        jac_img = fi_raw.new_image_like(np.ones(fi_raw.shape, dtype=np.float32))

    # 7. Save outputs
    path_warped = os.path.join(out_dir, f"{prefix}Warped.nii.gz")
    path_inv_warped = os.path.join(out_dir, f"{prefix}InverseWarped.nii.gz")
    path_jac = os.path.join(out_dir, f"{prefix}Jacobian.nii.gz")
    path_metrics = os.path.join(out_dir, f"{prefix}metrics.json")

    ants.image_write(warped_moving, path_warped)
    ants.image_write(inverse_warped_fixed, path_inv_warped)
    ants.image_write(jac_img, path_jac)

    # Save copy of transforms if requested or in-place
    saved_transforms = []
    for idx_t, t_path in enumerate(fwd_transforms):
        if isinstance(t_path, str) and os.path.exists(t_path):
            ext = ".mat" if t_path.endswith(".mat") else ".nii.gz"
            dest = os.path.join(out_dir, f"{prefix}{idx_t}Forward{ext}")
            if os.path.abspath(t_path) != os.path.abspath(dest):
                import shutil
                shutil.copy2(t_path, dest)
            saved_transforms.append(dest)

    metrics_dict = {
        "fixed_image": os.path.abspath(args.fixed),
        "moving_image": os.path.abspath(args.moving),
        "model": args.model,
        "regularizer": args.regularizer,
        "flow_sigma": args.flow_sigma,
        "total_sigma": args.total_sigma,
        "grad_step": args.grad_step,
        "iterations": reg_iterations,
        "time_total_s": time.time() - t_start,
        "time_reg_s": t_reg,
        "time_affine_s": t_aff,
        "harmonic_energy": harm_energy,
        "bending_energy": bend_energy,
        "jacobian_min": min_jac,
        "jacobian_mean": mean_jac,
        "jacobian_folding_pct": fold_pct,
        "dice_fixed": dice_fix,
        "dice_moving": dice_mov,
        "dice_symmetric": dice_sym,
        "warped_image": path_warped,
        "inverse_warped_image": path_inv_warped,
        "jacobian_image": path_jac,
        "transforms": saved_transforms
    }

    with open(path_metrics, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # 8. Generate Automated Interactive HTML Report
    if args.report:
        report_path = os.path.join(out_dir, args.report_name)
        print(f"[syntx] Generating Standalone Interactive HTML Registration Report...", flush=True)
        try:
            provenance = {
                "algorithm": f"syntx.{args.model}",
                "backend": args.backend,
                "device": device,
                "fit_time": t_reg + t_aff,
                "reg_iterations": reg_iterations,
                "fluid_sigma": args.flow_sigma,
                "elastic_sigma": args.total_sigma,
                "solver": args.model.upper(),
                "similarity_metric": args.similarity_metric,
            }
            create_registration_report(
                fixed=fi_raw,
                moving=mi_raw,
                warped=warped_moving,
                warp=fwd_transforms,
                detJ=jac_img,
                fixed_label=fl_raw,
                moving_label=ml_raw,
                warped_label=warped_label,
                reg=reg_res,
                output_html=report_path,
                provenance=provenance,
                title=f"Syntx Registration Report — {prefix.rstrip('_')}"
            )
            print(f"[syntx] HTML Report successfully saved to: {report_path}", flush=True)
        except Exception as e:
            print(f"[syntx] Warning: HTML report generation encountered error: {e}", file=sys.stderr)

    print("\n" + "=" * 80)
    print("                     REGISTRATION COMPLETE")
    print("=" * 80)
    print(f"  * Warped Moving Image      : {path_warped}")
    print(f"  * Inverse Warped Target    : {path_inv_warped}")
    print(f"  * Jacobian Determinant Map : {path_jac}")
    print(f"  * Quantitative Metrics JSON: {path_metrics}")
    if args.report:
        print(f"  * Interactive HTML Report  : {os.path.join(out_dir, args.report_name)}")
    if dice_sym is not None:
        print(f"  * Mean Symmetric DICE      : {dice_sym:.4f} (Fixed: {dice_fix:.4f}, Moving: {dice_mov:.4f})")
    print(f"  * Topological Folding Rate : {fold_pct:.5f}% (Min detJ: {min_jac:+.4f})")
    print(f"  * Harmonic Energy          : {harm_energy:.4f} (Bending: {bend_energy:.5f})")
    print(f"  * Total Execution Time     : {time.time() - t_start:.2f}s")
    print("=" * 80 + "\n", flush=True)

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Executes the `syntx info` subcommand."""
    print("=" * 70)
    print("              SYNTX ENVIRONMENT & SYSTEM INTROSPECTION")
    print("=" * 70)
    print(f"  * Syntx Version   : {getattr(syntx, '__version__', '4.0.2')}")
    print(f"  * Python Version  : {sys.version.split()[0]}")
    print(f"  * PyTorch Version : {torch.__version__}")
    print(f"  * ANTsPy Version  : {getattr(ants, '__version__', 'N/A')}")
    try:
        import jax
        print(f"  * JAX Version     : {jax.__version__}")
    except ImportError:
        print(f"  * JAX Version     : Not Installed")

    print("\n  Hardware Acceleration Capabilities:")
    cuda_avail = torch.cuda.is_available()
    mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"  * NVIDIA CUDA     : {'AVAILABLE (' + torch.cuda.get_device_name(0) + ')' if cuda_avail else 'Unavailable'}")
    print(f"  * Apple Silicon MPS: {'AVAILABLE (Metal Performance Shaders)' if mps_avail else 'Unavailable'}")
    print(f"  * Default Device  : {'cuda' if cuda_avail else ('mps' if mps_avail else 'cpu')}")
    print("=" * 70 + "\n")
    return 0


def main():
    """Main CLI entrypoint for syntx."""
    parser = argparse.ArgumentParser(
        prog="syntx",
        description="Syntx: High-Performance Symmetric Diffeomorphic & Riemannian Image Registration in PyTorch & JAX",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # --------------------------------------------------
    # Subcommand: register
    # --------------------------------------------------
    p_reg = subparsers.add_parser(
        "register",
        help="Register two images using symmetric diffeomorphic (SyN / TVF) algorithms."
    )
    p_reg.add_argument("-f", "--fixed", type=str, required=True, help="Fixed target image path.")
    p_reg.add_argument("-m", "--moving", type=str, required=True, help="Moving source image path.")
    p_reg.add_argument("-fl", "--fixed-label", type=str, default=None, help="Fixed target label map path (optional).")
    p_reg.add_argument("-ml", "--moving-label", type=str, default=None, help="Moving source label map path (optional).")
    p_reg.add_argument("-o", "--out-dir", type=str, default="./syntx_output", help="Output directory path.")
    p_reg.add_argument("-p", "--prefix", type=str, default="syntx_", help="Output filename prefix.")
    p_reg.add_argument("--model", type=str, default="syn", choices=["syn", "tvf", "affine"], help="Registration model formulation.")
    p_reg.add_argument("--regularizer", type=str, default="gaussian", choices=["gaussian", "sobolev", "dsti", "dsti1"], help="Spatial velocity regularizer.")
    p_reg.add_argument("--flow-sigma", type=float, default=5.0, help="Fluid velocity smoothing parameter in physical mm (default: 5.0 mm for peak DICE, 7.0 mm for ANTs energy parity).")
    p_reg.add_argument("--total-sigma", type=float, default=0.0, help="Elastic field smoothing parameter in physical mm.")
    p_reg.add_argument("--grad-step", type=float, default=0.25, help="Optimization gradient descent step size.")
    p_reg.add_argument("-i", "--iterations", type=str, default="100x100x20", help="Multi-resolution iterations (e.g. '100x100x20' or '100,100,20').")
    p_reg.add_argument("--similarity-metric", type=str, default="cc2", help="Image similarity loss functional (e.g. 'cc2', 'lncc', 'mattes_mi', 'mi', 'dino_2_lncc', 'vgg_4_lncc').")
    p_reg.add_argument("--optimizer", type=str, default="reg_adam", help="Optimizer type ('adam', 'reg_adam', 'sgd').")
    p_reg.add_argument("--formulation", type=str, default="eulerian", choices=["eulerian", "lagrangian"], help="SyN coordinate formulation.")
    p_reg.add_argument("--inverse-method", type=str, default="anderson", choices=["anderson", "fixed_point"], help="Sub-voxel inverse solver.")
    p_reg.add_argument("--in-loop-inv-steps", type=int, default=10, help="Anderson mixing inverse iterations.")
    p_reg.add_argument("--bootstrap-mode", type=str, default="antithetic", choices=["antithetic", "forward", "none"], help="Coordinate bootstrapping mode.")
    p_reg.add_argument("--bootstrap-orig-weight", type=float, default=0.50, help="Antithetic center-sample weight w0.")
    p_reg.add_argument("--bootstrap-jitter-scale", type=float, default=0.25, help="Antithetic coordinate jitter scale in voxels.")
    p_reg.add_argument("--syn-sampling", type=int, default=2, help="Spatial downsampling factor for metric computation.")
    p_reg.add_argument("--analytical-gradients", action="store_true", help="Use analytical ITK CC² pseudo-derivative instead of autograd.")
    p_reg.add_argument("--no-affine", action="store_true", help="Skip 18-cone robust affine pre-alignment.")
    p_reg.add_argument("--interpolator", type=str, default="linear", choices=["linear", "nearestNeighbor", "bSpline", "gaussian"], help="Warping interpolator.")
    p_reg.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "jax"], help="Computation backend.")
    p_reg.add_argument("--device", type=str, default="auto", help="Hardware device ('auto', 'cuda', 'mps', 'cpu').")
    p_reg.add_argument("--report", action="store_true", default=True, help="Generate comprehensive standalone interactive HTML diagnostic report.")
    p_reg.add_argument("--no-report", dest="report", action="store_false", help="Disable HTML report generation.")
    p_reg.add_argument("--report-name", type=str, default="registration_report.html", help="HTML report output filename.")
    p_reg.add_argument("-v", "--verbose", action="store_true", help="Print verbose step-by-step optimization logs.")

    # --------------------------------------------------
    # Subcommand: benchmark
    # --------------------------------------------------
    p_bench = subparsers.add_parser(
        "benchmark",
        help="Execute Mindboggle-101 cohort benchmark suite."
    )
    p_bench.add_argument("bench_args", nargs=argparse.REMAINDER, help="Arguments passed directly to syntx-benchmark.")

    # --------------------------------------------------
    # Subcommand: info
    # --------------------------------------------------
    subparsers.add_parser(
        "info",
        help="Display system info, PyTorch/JAX/ANTsPy versions, and hardware acceleration devices."
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    if args.subcommand == "register":
        return cmd_register(args)
    elif args.subcommand == "benchmark":
        from syntx.benchmark.cli import main as bench_main
        sys.argv = [sys.argv[0]] + (args.bench_args or [])
        return bench_main()
    elif args.subcommand == "info":
        return cmd_info(args)
    else:
        parser.print_help(sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
