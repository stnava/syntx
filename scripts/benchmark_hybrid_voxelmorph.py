#!/usr/bin/env python3
"""Isolated 3-pair Mindboggle-101 benchmark comparing zero-shot VoxelMorph,
VoxelMorph hybrid warm-started variational solvers (greedy turbo, greedy,
symmetric SyN Beltrami), and cold-start greedy / FireANTs / SyN Beltrami.

Usage:
    python3 scripts/benchmark_hybrid_voxelmorph.py --smoke
    python3 scripts/benchmark_hybrid_voxelmorph.py --device mps
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import ants
import numpy as np
import torch

import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.benchmark.evaluate import (
    get_model_config,
    normalize_intensity,
    syn_config_to_syn_kwargs,
)
from syntx.benchmark.external import (
    run_fireants_greedy,
    run_voxelmorph_hybrid_greedy,
    run_voxelmorph_hybrid_syn,
    run_voxelmorph_pretrained,
)
from syntx.core.pipeline import cleanup_gpu
from syntx.deformation_metrics import (
    compute_bidirectional_dice,
    compute_jacobian_metrics,
    flow_jacobian_metrics,
)
from syntx.registration_metrics import evaluate_registration_qc
from syntx.transforms import apply_transforms as syntx_apply_transforms

DEFAULT_PAIRS: Tuple[int, ...] = (44, 77, 0)
ARM_ORDER: Tuple[str, ...] = (
    "syntx_robust_affine",
    "voxelmorph_pretrained",
    "voxelmorph_polished",
    "voxelmorph_hybrid_greedy_turbo",
    "voxelmorph_hybrid_greedy",
    "syntx_greedy_default",
    "syntx_greedy_matched",
    "fireants_greedy",
    "voxelmorph_hybrid_syn",
    "syntx_syn_beltrami",
)


def _label_dice_generic(fl: ants.ANTsImage, ml: ants.ANTsImage, fwd_tx: Sequence[Any]) -> float:
    warped_ml = syntx_apply_transforms(
        fixed=fl, moving=ml, transformlist=list(fwd_tx), interpolator="genericLabel"
    )
    fl_np = fl.numpy().astype(np.int32)
    wm_np = warped_ml.numpy().astype(np.int32)
    labels = sorted((set(np.unique(fl_np).tolist()) | set(np.unique(wm_np).tolist())) - {0})
    dices = []
    for lab in labels:
        a = fl_np == lab
        b = wm_np == lab
        denom = int(a.sum()) + int(b.sum())
        if denom > 0:
            dices.append(2.0 * int((a & b).sum()) / float(denom))
    return float(np.mean(dices)) if dices else 0.0


def _reset_device_stats(dev: str) -> None:
    cleanup_gpu(dev)
    if dev == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
        if hasattr(torch.mps, "reset_peak_memory_stats"):
            torch.mps.reset_peak_memory_stats()
    elif dev == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _read_device_memory_mb(dev: str) -> Tuple[float, float]:
    if dev == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
        curr_alloc = (
            torch.mps.current_allocated_memory()
            if hasattr(torch.mps, "current_allocated_memory")
            else 0
        )
        driver_mem = (
            torch.mps.driver_allocated_memory()
            if hasattr(torch.mps, "driver_allocated_memory")
            else 0
        )
        peak_alloc = (
            torch.mps.peak_memory_allocated()
            if hasattr(torch.mps, "peak_memory_allocated")
            else curr_alloc
        )
        return float(max(peak_alloc, curr_alloc)) / (1024.0 ** 2), float(driver_mem) / (1024.0 ** 2)
    if dev == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        return (
            float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2),
            float(torch.cuda.max_memory_reserved()) / (1024.0 ** 2),
        )
    return 0.0, 0.0


def _run_arm(
    arm: str,
    fi: ants.ANTsImage,
    mi: ants.ANTsImage,
    aff_path: str,
    dev: str,
    seed: int,
    smoke: bool,
    out_dir: str,
) -> Tuple[Dict[str, Any], float, float, bool]:
    """Execute one benchmark arm on a prepared pair and return ``(res, opt_s, total_s, symmetric)``."""
    t0 = time.perf_counter()
    if arm == "syntx_robust_affine":
        t0_aff = time.perf_counter()
        warped = syntx_apply_transforms(
            fixed=fi, moving=mi, transformlist=[aff_path], interpolator="linear", device=dev
        )
        if dev == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0_aff
        res = {
            "fwdtransforms": [aff_path],
            "invtransforms": [aff_path],
            "whichtoinvert_inv": [True],
            "warpedmovout": warped,
            "provenance": {"method": "syntx_robust_affine"},
        }
        return res, 0.0, dt, False

    if arm == "voxelmorph_pretrained":
        res = run_voxelmorph_pretrained(
            fi, mi, initial_transform=aff_path, polish_steps=0, device=dev, seed=seed, out_dir=out_dir
        )
        total_s = time.perf_counter() - t0
        opt_s = float(res["provenance"].get("vxm_seconds", res["provenance"].get("runtime_seconds", total_s)))
        return res, opt_s, total_s, False

    if arm == "voxelmorph_polished":
        res = run_voxelmorph_pretrained(
            fi,
            mi,
            initial_transform=aff_path,
            polish_steps=(2 if smoke else 30),
            device=dev,
            seed=seed,
            out_dir=out_dir,
        )
        total_s = time.perf_counter() - t0
        opt_s = float(res["provenance"].get("vxm_seconds", res["provenance"].get("runtime_seconds", total_s)))
        return res, opt_s, total_s, False

    if arm == "voxelmorph_hybrid_greedy_turbo":
        res = run_voxelmorph_hybrid_greedy(
            fi,
            mi,
            initial_transform=aff_path,
            scales=(2, 1),
            reg_iterations=((2, 2) if smoke else (25, 10)),
            device=dev,
            seed=seed,
            out_dir=out_dir,
        )
        total_s = time.perf_counter() - t0
        opt_s = float(res["provenance"].get("opt_seconds", total_s))
        return res, opt_s, total_s, False

    if arm == "voxelmorph_hybrid_greedy":
        res = run_voxelmorph_hybrid_greedy(
            fi,
            mi,
            initial_transform=aff_path,
            scales=(2, 1),
            reg_iterations=((2, 2) if smoke else (35, 15)),
            device=dev,
            seed=seed,
            out_dir=out_dir,
        )
        total_s = time.perf_counter() - t0
        opt_s = float(res["provenance"].get("opt_seconds", total_s))
        return res, opt_s, total_s, False

    if arm == "syntx_greedy_default":
        res = syntx.greedy(
            fi,
            mi,
            initial_transform=aff_path,
            scales=([2, 1] if smoke else [4, 2, 1]),
            reg_iterations=([2, 2] if smoke else [100, 100, 20]),
            optimizer="adam",
            learning_rate=0.375,
            flow_sigma=1.8,
            total_sigma=0.28,
            return_inverse=False,
            device=dev,
            seed=seed,
            outprefix=os.path.join(out_dir, ""),
        )
        total_s = time.perf_counter() - t0
        opt_s = float(res["provenance"].get("runtime_deformable_sec", total_s))
        return res, opt_s, total_s, False

    if arm == "syntx_greedy_matched":
        res = syntx.greedy(
            fi,
            mi,
            initial_transform=aff_path,
            scales=([2, 1] if smoke else [4, 2, 1]),
            reg_iterations=([2, 2] if smoke else [100, 100, 50]),
            optimizer="regadam",
            learning_rate=0.5,
            flow_sigma=1.8,
            total_sigma=0.28,
            return_inverse=False,
            device=dev,
            seed=seed,
            outprefix=os.path.join(out_dir, ""),
        )
        total_s = time.perf_counter() - t0
        opt_s = float(res["provenance"].get("runtime_deformable_sec", total_s))
        return res, opt_s, total_s, False

    if arm == "fireants_greedy":
        res = run_fireants_greedy(
            fi,
            mi,
            initial_transform=aff_path,
            scales=([2, 1] if smoke else [4, 2, 1]),
            reg_iterations=([2, 2] if smoke else [100, 100, 50]),
            optimizer="regadam",
            optimizer_lr=0.5,
            return_inverse=False,
            device=dev,
            seed=seed,
            out_dir=out_dir,
        )
        total_s = time.perf_counter() - t0
        opt_s = float(res["provenance"].get("opt_seconds", res["provenance"].get("runtime_seconds", total_s)))
        return res, opt_s, total_s, False

    if arm == "voxelmorph_hybrid_syn":
        res = run_voxelmorph_hybrid_syn(
            fi,
            mi,
            initial_transform=aff_path,
            levels=(2, 1),
            reg_iterations=((2, 2) if smoke else (30, 15)),
            regularizer="beltrami",
            device=dev,
            seed=seed,
            out_dir=out_dir,
        )
        total_s = time.perf_counter() - t0
        opt_s = float(res["provenance"].get("opt_seconds", total_s))
        return res, opt_s, total_s, True

    if arm == "syntx_syn_beltrami":
        cfg = get_model_config("syn_beltrami")
        syn_kw = syn_config_to_syn_kwargs(cfg) if cfg is not None else {"regularizer": "beltrami", "grad_step": 0.7, "dilatation_weight": 0.1}
        syn_kw.pop("reg_iterations", None)
        syn_kw.pop("levels", None)
        res = syntx.syn(
            fi,
            mi,
            initial_transform=aff_path,
            levels=([2, 1] if smoke else [4, 2, 1]),
            reg_iterations=([2, 2] if smoke else [100, 100, 20]),
            backend="pytorch",
            device=dev,
            seed=seed,
            outprefix=os.path.join(out_dir, ""),
            **syn_kw,
        )
        total_s = time.perf_counter() - t0
        opt_s = float(res["provenance"].get("runtime_deformable_sec", res["provenance"].get("fit_time", total_s)))
        return res, opt_s, total_s, True

    raise ValueError(f"Unknown arm: {arm}")


def main() -> None:
    p = argparse.ArgumentParser(description="3-pair Mindboggle-101 Hybrid VoxelMorph Benchmark")
    p.add_argument("--smoke", action="store_true", help="Fast CPU smoke gate over all 10 arms")
    p.add_argument("--device", default="mps", help="Target compute device (default: mps)")
    p.add_argument("--pairs", type=int, nargs="+", default=list(DEFAULT_PAIRS), help="Mindboggle pair indices")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-json",
        default=os.path.join(_REPO_ROOT, "docs", "provenance", "hybrid_voxelmorph_benchmark.json"),
    )
    args = p.parse_args()

    dev = "cpu" if args.smoke else args.device
    out_json = (
        os.path.join(tempfile.gettempdir(), "hybrid_voxelmorph_smoke.json")
        if args.smoke
        else args.out_json
    )

    pair_records: List[Dict[str, Any]] = []
    if args.smoke:
        z, y, x = np.ogrid[:32, :32, :32]
        fi_arr = np.exp(-(((x - 16) / 6.0) ** 2 + ((y - 16) / 6.0) ** 2 + ((z - 16) / 6.0) ** 2)).astype(np.float32)
        mi_arr = np.exp(-(((x - 14) / 6.0) ** 2 + ((y - 17) / 6.0) ** 2 + ((z - 16) / 6.0) ** 2)).astype(np.float32)
        fi = ants.from_numpy(fi_arr)
        mi = ants.from_numpy(mi_arr)
        fl = ants.from_numpy((fi_arr > 0.4).astype(np.uint32))
        ml = ants.from_numpy((mi_arr > 0.4).astype(np.uint32))
        aff_res = syntx.robust_affine(fi, mi, mode="pytorch", device=dev, seed=args.seed)
        dataset = [(999, "smoke_fixed -> smoke_moving", fi, mi, fl, ml, aff_res["fwdtransforms"][0])]
    else:
        dataset = []
        for pid in args.pairs:
            pair = load_mindboggle_pair(pair_idx=pid, use_n4=True)
            fi = normalize_intensity(pair["fixed"])
            mi = normalize_intensity(pair["moving"])
            fl = pair["fixed_label"]
            ml = pair["moving_label"]
            aff_cache = os.path.join(_REPO_ROOT, "results", "canonical_affines", f"pair_{pid:03d}_pt7_affine.mat")
            if os.path.exists(aff_cache):
                aff_path = aff_cache
            else:
                aff_path = syntx.robust_affine(fi, mi, mode="auto", device=dev, seed=args.seed)["fwdtransforms"][0]
            pair_name = f"{pair.get('fixed_subject', f'pair_{pid}')} -> {pair.get('moving_subject', f'pair_{pid}')}"
            dataset.append((pid, pair_name, fi, mi, fl, ml, aff_path))

    print(
        f"Hybrid VoxelMorph Benchmark | device={dev} | smoke={args.smoke} | pairs={[d[0] for d in dataset]} | syntx={syntx.__file__}",
        flush=True,
    )

    for pid, pair_name, fi, mi, fl, ml, aff_path in dataset:
        print(
            f"\n=== Pair {pid}: {pair_name} | fixed={fi.shape} ({fi.orientation}) -> moving={mi.shape} ({mi.orientation}) ===",
            flush=True,
        )
        for arm in ARM_ORDER:
            _reset_device_stats(dev)
            with tempfile.TemporaryDirectory(prefix=f"vxm_bench_p{pid}_{arm}_") as td:
                res, opt_s, total_s, is_sym = _run_arm(
                    arm, fi, mi, aff_path, dev, args.seed + int(pid), args.smoke, td
                )
                peak_mb, driver_mb = _read_device_memory_mb(dev)
                fwd_tx = res["fwdtransforms"]
                inv_tx = res.get("invtransforms") or []
                which_inv = res.get("whichtoinvert_inv") or []

                if is_sym and inv_tx:
                    d_fix_nn, d_mov_nn, d_sym_nn = compute_bidirectional_dice(
                        fl, ml, fi, mi, fwd_tx, inv_tx, which_inv
                    )
                else:
                    d_fix_nn, _, _ = compute_bidirectional_dice(fl, ml, fi, mi, fwd_tx, [], [])
                    d_mov_nn = float("nan")
                    d_sym_nn = float(d_fix_nn)
                d_fix_gl = _label_dice_generic(fl, ml, fwd_tx)

                fwd_warp = next((x for x in fwd_tx if isinstance(x, str) and x.endswith(".nii.gz")), None)
                jac_fd = (
                    compute_jacobian_metrics(fi, fwd_warp)
                    if fwd_warp is not None
                    else {"folding_pct": 0.0, "min": 1.0, "max": 1.0, "mean": 1.0}
                )
                jac_flow = flow_jacobian_metrics(fi, res) or dict(jac_fd, measure="finite_difference")

                ice_phi1 = (res.get("inverse_identity_errors") or {}).get("phi_1", {})
                ice_mean = float(ice_phi1.get("mean_error", ice_phi1.get("mean", float("nan"))))
                ice_max = float(ice_phi1.get("max_error", ice_phi1.get("max", float("nan"))))

                qc_rep = evaluate_registration_qc(fi, mi, res, fixed_labels=fl, moving_labels=ml)
                qc_status = str(qc_rep.status)

                rec = {
                    "pair_idx": int(pid),
                    "pair_name": pair_name,
                    "arm": arm,
                    "device": dev,
                    "symmetric": bool(is_sym),
                    "opt_s": float(opt_s),
                    "total_s": float(total_s),
                    "peak_mps_mb": float(peak_mb),
                    "driver_mps_mb": float(driver_mb),
                    "dice_fixed_nn": float(d_fix_nn),
                    "dice_fixed_gl": float(d_fix_gl),
                    "dice_moving_nn": float(d_mov_nn) if np.isfinite(d_mov_nn) else None,
                    "dice_sym_nn": float(d_sym_nn),
                    "folding_pct": float(jac_flow.get("folding_pct", 0.0)),
                    "min_jac": float(jac_flow.get("min", float("nan"))),
                    "jac_measure": str(jac_flow.get("measure", "finite_difference")),
                    "fd_folding_pct": float(jac_fd.get("folding_pct", 0.0)),
                    "fd_min_jac": float(jac_fd.get("min", float("nan"))),
                    "ice_mean_mm": float(ice_mean) if np.isfinite(ice_mean) else None,
                    "ice_max_mm": float(ice_max) if np.isfinite(ice_max) else None,
                    "antsxqc": qc_status,
                }
                pair_records.append(rec)
                ice_str = f" ICE={ice_mean:.3f}/{ice_max:.3f}mm" if np.isfinite(ice_mean) else ""
                sym_str = f" SymNN={d_sym_nn:.4f}" if is_sym else ""
                print(
                    f"  {arm:32s} opt={opt_s:5.2f}s e2e={total_s:5.2f}s "
                    f"NN={d_fix_nn:.4f} GL={d_fix_gl:.4f}{sym_str} "
                    f"fold={rec['folding_pct']:.4f}% minJ={rec['min_jac']:.4f} "
                    f"FD_fold={rec['fd_folding_pct']:.4f}% "
                    f"mem={peak_mb:6.1f}/{driver_mb:6.1f}MB QC={qc_status}{ice_str}",
                    flush=True,
                )
            _reset_device_stats(dev)

    summary_rows: List[Dict[str, Any]] = []
    for arm in ARM_ORDER:
        rows = [r for r in pair_records if r["arm"] == arm]
        if not rows:
            continue
        ice_means = [r["ice_mean_mm"] for r in rows if r["ice_mean_mm"] is not None]
        ice_maxs = [r["ice_max_mm"] for r in rows if r["ice_max_mm"] is not None]
        summary_rows.append(
            {
                "arm": arm,
                "n_pairs": len(rows),
                "mean_opt_s": float(np.mean([r["opt_s"] for r in rows])),
                "mean_total_s": float(np.mean([r["total_s"] for r in rows])),
                "mean_dice_fixed_nn": float(np.mean([r["dice_fixed_nn"] for r in rows])),
                "mean_dice_fixed_gl": float(np.mean([r["dice_fixed_gl"] for r in rows])),
                "mean_dice_sym_nn": float(np.mean([r["dice_sym_nn"] for r in rows])),
                "mean_folding_pct": float(np.mean([r["folding_pct"] for r in rows])),
                "min_jac_worst": float(np.min([r["min_jac"] for r in rows])),
                "mean_fd_folding_pct": float(np.mean([r["fd_folding_pct"] for r in rows])),
                "max_peak_mps_mb": float(np.max([r["peak_mps_mb"] for r in rows])),
                "max_driver_mps_mb": float(np.max([r["driver_mps_mb"] for r in rows])),
                "mean_ice_mm": float(np.mean(ice_means)) if ice_means else None,
                "max_ice_mm": float(np.max(ice_maxs)) if ice_maxs else None,
                "qc_statuses": [r["antsxqc"] for r in rows],
            }
        )

    payload = {
        "benchmark": "mindboggle_hybrid_voxelmorph_3pair",
        "device": dev,
        "smoke": bool(args.smoke),
        "seed": int(args.seed),
        "pairs": [int(d[0]) for d in dataset],
        "torch_version": torch.__version__,
        "syntx_file": syntx.__file__,
        "summary": summary_rows,
        "records": pair_records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved benchmark JSON -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
