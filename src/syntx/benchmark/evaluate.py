import time
import torch
import numpy as np
import logging
import gc
import syntx
import ants
from typing import Dict, Any

from syntx.benchmark.data import load_mindboggle_pair
from syntx.benchmark.metrics import compute_pair_metrics

logger = logging.getLogger(__name__)

def evaluate_pair(
    pair_idx: int = None, 
    model: str = "syn", 
    device: str = "mps", 
    config: dict = None, 
    pairs_csv: str = None, 
    data_dir: str = None,
    dataset_key: str = None,
    out_dir: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Orchestrates loading, affine init, nonlinear registration, and metric computation for a single pair."""
    t_start_total = time.time()
    
    if config is None:
        config = {}
        
    # Load configuration
    if model == "syn":
        cfg = config.get("syn_config", config)
    elif model == "tvf":
        cfg = config.get("tvf_config", config)
    else:
        cfg = config

    # Apply any direct kwargs as overrides to the config
    cfg.update(kwargs)

    # Load data
    if dataset_key:
        data = syntx.benchmark_data(dataset_key)
        fixed = data["fixed"]
        moving = data["moving"]
        fixed_label = data["fixed_label"]
        moving_label = data["moving_label"]
        fixed_id = "dataset_" + dataset_key
        moving_id = "dataset_" + dataset_key
        pair_type = dataset_key
    else:
        load_kwargs = {}
        if pairs_csv: load_kwargs['pairs_csv'] = pairs_csv
        if data_dir: load_kwargs['data_dir'] = data_dir
        data = load_mindboggle_pair(pair_idx=pair_idx, **load_kwargs)
        fixed = data["fixed"]
        moving = data["moving"]
        fixed_label = data["fixed_label"]
        moving_label = data["moving_label"]
        fixed_id = data["fixed_id"]
        moving_id = data["moving_id"]
        pair_type = data["pair_type"]

    # 1. Deterministic Affine Initialization (always on CPU)
    torch.manual_seed(42)
    np.random.seed(42)

    logger.info("Computing deterministic robust affine (CPU)...")
    t_aff_start = time.time()
    res_aff = syntx.robust_affine(
        fixed=fixed, moving=moving,
        mode="pytorch", device="cpu",
        multi_start=True, verbose=False,
        n_starts=cfg.get("n_starts", 4),
        cone_angles_deg=cfg.get("cone_angles_deg", [-25.0, -15.0, -5.0, 0.0, 5.0, 15.0, 25.0])
    )
    initial_transform = res_aff["fwdtransforms"][0]
    t_aff = time.time() - t_aff_start
    logger.info(f"Affine alignment completed in {t_aff:.1f}s")
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 2. Nonlinear Registration
    t_reg_start = time.time()

    if model == "syn":
        reg = syntx.syn(
            fixed=fixed,
            moving=moving,
            initial_transform=initial_transform,
            backend="pytorch",
            device=device,
            grad_step=cfg.get("grad_step", 0.25),
            flow_sigma=cfg.get("fluid_sigma", cfg.get("flow_sigma", 3.0)),
            total_sigma=cfg.get("elastic_sigma", cfg.get("total_sigma", 0.0)),
            similarity_metric=cfg.get("syn_metric", cfg.get("similarity_metric", "lncc")),
            syn_sampling=cfg.get("lncc_radius", cfg.get("syn_sampling", 2)),
            inverse_steps=cfg.get("inverse_steps", 10),
            regularizer=cfg.get("syn_regularizer", cfg.get("regularizer", "gaussian")),
            fast_smooth=cfg.get("syn_fast_smooth", cfg.get("fast_smooth", False)),
            use_analytical_gradients=cfg.get("syn_use_analytical_gradients", cfg.get("use_analytical_gradients", False)),
            inverse_method=cfg.get("syn_inverse_method", cfg.get("inverse_method", "anderson")),
            formulation=cfg.get("syn_formulation", cfg.get("formulation", "eulerian")),
            reg_iterations=cfg.get("reg_iterations", [100, 100, 20]),
            antisymmetric=True,
            verbose=False,
        )
    elif model == "tvf":
        reg = syntx.tvf(
            fixed=fixed,
            moving=moving,
            initial_transform=initial_transform,
            backend="pytorch",
            device=device,
            grad_step=cfg.get("tvf_grad_step", cfg.get("grad_step", 0.211)),
            flow_sigma=cfg.get("tvf_flow_sigma", cfg.get("flow_sigma", 0.0)),
            total_sigma=cfg.get("tvf_total_sigma", cfg.get("total_sigma", 0.2)),
            regularizer=cfg.get("tvf_regularizer", cfg.get("regularizer", "gaussian")),
            fast_smooth=cfg.get("tvf_fast_smooth", cfg.get("fast_smooth", False)),
            use_analytical_gradients=cfg.get("tvf_use_analytical_gradients", cfg.get("use_analytical_gradients", False)),
            antisymmetric=cfg.get("tvf_antisymmetric", cfg.get("antisymmetric", True)),
            cfl_momentum=cfg.get("tvf_cfl_momentum", cfg.get("cfl_momentum", 0.9)),
            n_time_steps=cfg.get("tvf_n_time_steps", cfg.get("n_time_steps", 3)),
            constant_speed=cfg.get("tvf_constant_speed", cfg.get("constant_speed", True)),
            constant_speed_relaxation=cfg.get("tvf_constant_speed_relaxation", cfg.get("constant_speed_relaxation", 0.10)),
            multipoint_loss=cfg.get("multipoint_loss", [0.0, 0.5, 1.0]),
            reg_iterations=cfg.get("reg_iterations", [80, 80, 20]),
            verbose=False,
        )
    elif model == "ants_syn":
        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            initial_transform=initial_transform,
            type_of_transform="SyN",
            syn_metric="CC",
            grad_step=0.25,
            flow_sigma=3.0,
            total_sigma=0.0,
            syn_sampling=2,
            reg_iterations=(100, 100, 20),
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown model: {model}. Must be 'syn', 'tvf', or 'ants_syn'.")

    t_reg = time.time() - t_reg_start
    t_total = t_aff + t_reg
    
    fwdtransforms = reg["fwdtransforms"]
    invtransforms = reg["invtransforms"]
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 3. Compute Metrics
    metrics = compute_pair_metrics(fixed, moving, fixed_label, moving_label, fwdtransforms, invtransforms, reg=reg)
    
    metrics["runtime_seconds"] = round(t_total, 2)
    metrics["runtime_affine_seconds"] = round(t_aff, 2)
    metrics["runtime_nonlinear_seconds"] = round(t_reg, 2)
    metrics["fixed_id"] = fixed_id
    metrics["moving_id"] = moving_id
    metrics["pair_type"] = pair_type
    if pair_idx is not None:
        metrics["pair_idx"] = pair_idx
    metrics["model"] = model
    metrics["device"] = device
    
    # 3.5 Save Artifacts if requested
    if out_dir is not None:
        import os
        import shutil
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"pair_{pair_idx:03d}_{model}"
        
        # Save Warped Image
        if "warpedmovout" in reg:
            ants.image_write(reg["warpedmovout"], os.path.join(out_dir, f"{prefix}_warped_image.nii.gz"))
            
        # Warp and Save Moving Label
        warped_label = ants.apply_transforms(fixed, moving_label, fwdtransforms, interpolator="nearestNeighbor")
        ants.image_write(warped_label, os.path.join(out_dir, f"{prefix}_warped_label.nii.gz"))
        
        # Save Transforms
        for i, tpath in enumerate(fwdtransforms):
            if os.path.exists(tpath):
                ext = ".nii.gz" if tpath.endswith(".nii.gz") else ".mat"
                shutil.copy2(tpath, os.path.join(out_dir, f"{prefix}_fwd_transform_{i}{ext}"))
        
        for i, tpath in enumerate(invtransforms):
            if os.path.exists(tpath):
                ext = ".nii.gz" if tpath.endswith(".nii.gz") else ".mat"
                shutil.copy2(tpath, os.path.join(out_dir, f"{prefix}_inv_transform_{i}{ext}"))

    # 4. Memory Cleanup
    del reg
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return metrics
