"""
syntx.benchmark.msd — ``syntx.auto_reg`` on Medical Segmentation Decathlon (MSD) pairs
=====================================================================================

For case pairs of an unpacked MSD task (a directory with ``dataset.json``), registers the
moving case to the fixed one with ``syntx.auto_reg`` and records the label Dice before and
after, folding, runtime, and the modality / anatomy diagnosis ``auto_reg`` reports. The
diagnosis is recorded, not checked against the task metadata.
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import ants

import syntx
from syntx.provenance import with_provenance
from syntx.deformation_metrics import compute_bidirectional_dice


def _prepare_image_and_label(img_path: str, lbl_path: str, channel: int = 0, max_dimension: Optional[int] = 256) -> Tuple[ants.ANTsImage, ants.ANTsImage]:
    """Read an image and its label map; a 4-D image keeps volume ``channel`` (a 4-D label map
    volume 0). If the image's largest dimension exceeds ``max_dimension``, both are resampled
    to isotropically scaled spacing (image linear, labels nearest neighbour) so that the
    largest dimension becomes about ``max_dimension``. Returns (image, labels)."""
    img = ants.image_read(img_path)
    lbl = ants.image_read(lbl_path)

    if img.dimension == 4:
        # Extract specific 3D channel
        arr = img.numpy()[..., channel]
        img = ants.from_numpy(arr, origin=img.origin[:3], spacing=img.spacing[:3], direction=img.direction[:3, :3])

    if lbl.dimension == 4:
        arr_l = lbl.numpy()[..., 0]
        lbl = ants.from_numpy(arr_l, origin=lbl.origin[:3], spacing=lbl.spacing[:3], direction=lbl.direction[:3, :3])

    if max_dimension is not None and max(img.shape) > max_dimension:
        scale = float(max_dimension / max(img.shape))
        new_spacing = tuple(float(s / scale) for s in img.spacing)
        img = ants.resample_image(img, resample_params=new_spacing, use_voxels=False, interp_type=0)
        lbl = ants.resample_image(lbl, resample_params=new_spacing, use_voxels=False, interp_type=1)

    return img, lbl


@with_provenance("syntx.benchmark.msd.evaluate_msd_pair")
def evaluate_msd_pair(
    task_dir: str,
    fixed_idx: int,
    moving_idx: int,
    reg_iterations: Optional[List[int]] = None,
    max_dimension: Optional[int] = 256,
    verbose: bool = False
) -> Dict[str, Any]:
    """Register MSD training case ``moving_idx`` to case ``fixed_idx`` with ``syntx.auto_reg``.

    Images and labels are read with ``_prepare_image_and_label`` (channel 0 of 4-D images).
    The initial Dice is ``compute_bidirectional_dice`` with no transforms (labels resampled
    into the other image's grid). ``auto_reg`` is called with both label maps, so its
    reported metrics include Dice.

    Parameters
    ----------
    task_dir : str
        Unpacked MSD task directory containing ``dataset.json``.
    fixed_idx, moving_idx : int
        Indices into ``dataset.json['training']``.
    reg_iterations : list of int, optional
        Deformable iterations; default [40, 20, 10]. (The affine stage is ``auto_reg``'s
        ``robust_affine``; it has no iteration schedule here.)
    max_dimension : int or None, default 256
        Downsample cases whose largest dimension exceeds this; None keeps full resolution.
    verbose : bool, default False
        Passed to ``auto_reg``.

    Returns
    -------
    dict
        'task_name', 'fixed_case', 'moving_case' (image file names), 'initial_dice_sym',
        'final_dice_sym', 'final_dice_fixed', 'final_dice_moving' (NaN if ``auto_reg``
        reported none), 'dice_gain' (final - initial), 'folding_pct', 'min_jac'
        (``metrics['jac_min']``; both NaN if ``auto_reg`` reported none), 'time_seconds',
        'diagnosed_relationship', 'diagnosed_fixed_anatomy', 'diagnosed_fixed_modality'
        ('UNKNOWN' without a diagnosis), 'policy_explanation', 'fwdtransforms',
        'invtransforms', plus 'provenance' added by the ``with_provenance`` decorator.

    Raises
    ------
    FileNotFoundError
        No ``dataset.json`` in ``task_dir``.
    IndexError
        An index is beyond the number of training cases.
    """
    meta_path = os.path.join(task_dir, "dataset.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"dataset.json not found in {task_dir}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    task_name = meta.get("name", os.path.basename(task_dir.rstrip("/")))
    training_cases = meta.get("training", [])
    if fixed_idx >= len(training_cases) or moving_idx >= len(training_cases):
        raise IndexError(f"Case indices ({fixed_idx}, {moving_idx}) exceed dataset size ({len(training_cases)})")

    fix_case = training_cases[fixed_idx]
    mov_case = training_cases[moving_idx]

    fix_img_p = os.path.normpath(os.path.join(task_dir, fix_case["image"]))
    fix_lbl_p = os.path.normpath(os.path.join(task_dir, fix_case["label"]))
    mov_img_p = os.path.normpath(os.path.join(task_dir, mov_case["image"]))
    mov_lbl_p = os.path.normpath(os.path.join(task_dir, mov_case["label"]))

    fi, fl = _prepare_image_and_label(fix_img_p, fix_lbl_p, max_dimension=max_dimension)
    mi, ml = _prepare_image_and_label(mov_img_p, mov_lbl_p, max_dimension=max_dimension)

    # Initial overlap (unaligned baseline)
    d_fix_0, d_mov_0, d_sym_0 = compute_bidirectional_dice(fl, ml, fi, mi, [], [], [])

    # Run auto_reg with autonomous diagnosis & policy synthesis
    t0 = time.time()
    res = syntx.auto_reg(
        fixed=fi,
        moving=mi,
        fixed_label=fl,
        moving_label=ml,
        reg_iterations=reg_iterations or [40, 20, 10],
        verbose=verbose
    )
    t_elapsed = time.time() - t0

    metrics = res.get("metrics", {})
    dice_sym = metrics.get("dice_sym", float("nan"))
    dice_fix = metrics.get("dice_fixed", float("nan"))
    dice_mov = metrics.get("dice_moving", float("nan"))
    folding_pct = metrics.get("folding_pct", float("nan"))
    min_jac = metrics.get("jac_min", float("nan"))
    diag = metrics.get("diagnosis", {})

    return {
        "task_name": task_name,
        "fixed_case": os.path.basename(fix_img_p),
        "moving_case": os.path.basename(mov_img_p),
        "initial_dice_sym": float(d_sym_0),
        "final_dice_sym": float(dice_sym),
        "final_dice_fixed": float(dice_fix),
        "final_dice_moving": float(dice_mov),
        "dice_gain": float(dice_sym - d_sym_0),
        "folding_pct": float(folding_pct),
        "min_jac": float(min_jac),
        "time_seconds": float(t_elapsed),
        "diagnosed_relationship": diag.get("relationship", "UNKNOWN"),
        "diagnosed_fixed_anatomy": diag.get("fixed_anatomy", "UNKNOWN"),
        "diagnosed_fixed_modality": diag.get("fixed_modality", "UNKNOWN"),
        "policy_explanation": metrics.get("policy_explanation", ""),
        "fwdtransforms": res.get("fwdtransforms", []),
        "invtransforms": res.get("invtransforms", [])
    }


def run_msd_task_benchmark(
    task_dir: str,
    pairs: List[Tuple[int, int]],
    reg_iterations: Optional[List[int]] = None,
    max_dimension: Optional[int] = 256,
    output_json: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Run ``evaluate_msd_pair`` on each (fixed_idx, moving_idx) pair and print a summary.

    The summary gives the mean initial / final Dice, mean gain, mean folding %, mean runtime
    (NaN values skipped) and the fraction of pairs with a positive gain.

    Parameters
    ----------
    task_dir : str
        Unpacked MSD task directory.
    pairs : list of (int, int)
        (fixed_idx, moving_idx) tuples.
    reg_iterations : list of int, optional
        Passed to ``evaluate_msd_pair`` (default [40, 20, 10] there).
    max_dimension : int or None, default 256
        Passed to ``evaluate_msd_pair``.
    output_json : str, optional
        If given, the list of records is written there as JSON (directory created).

    Returns
    -------
    list of dict
        The ``evaluate_msd_pair`` records, in input order.
    """
    results = []
    print("=" * 80)
    print(f"AUTOMATED DECATHLON BENCHMARK: {os.path.basename(task_dir.rstrip('/'))}")
    print("=" * 80, flush=True)

    for p_idx, (f_idx, m_idx) in enumerate(pairs):
        print(f"\n[{p_idx + 1}/{len(pairs)}] Registering Case {m_idx:03d} -> Case {f_idx:03d}...", flush=True)
        res = evaluate_msd_pair(
            task_dir=task_dir,
            fixed_idx=f_idx,
            moving_idx=m_idx,
            reg_iterations=reg_iterations,
            max_dimension=max_dimension,
            verbose=False
        )
        results.append(res)
        print(
            f"  Result: Dice {res['initial_dice_sym']:.4f} -> {res['final_dice_sym']:.4f} "
            f"({res['dice_gain']:+.4f}) | Folds={res['folding_pct']:.4f}% | {res['time_seconds']:.1f}s | "
            f"Diagnosed: {res['diagnosed_fixed_anatomy']} ({res['diagnosed_fixed_modality']})",
            flush=True
        )

    # Print Summary Table
    d_initial = [r["initial_dice_sym"] for r in results]
    d_final = [r["final_dice_sym"] for r in results]
    d_gains = [r["dice_gain"] for r in results]
    folds = [r["folding_pct"] for r in results]
    times = [r["time_seconds"] for r in results]

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("-" * 80)
    print(f"Mean Initial DICE : {np.nanmean(d_initial):.4f}")
    print(f"Mean Final DICE   : {np.nanmean(d_final):.4f} (Mean Gain: {np.nanmean(d_gains):+.4f})")
    print(f"Mean Folding %    : {np.nanmean(folds):.4f}%")
    print(f"Mean Runtime      : {np.nanmean(times):.1f}s")
    print(f"Win Rate (>0 gain): {sum(g > 0 for g in d_gains)}/{len(d_gains)} ({np.mean([g > 0 for g in d_gains])*100:.1f}%)")
    print("=" * 80, flush=True)

    if output_json is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved benchmark results to {output_json}")

    return results
