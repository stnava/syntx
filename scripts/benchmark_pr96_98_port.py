#!/usr/bin/env python3
"""
Reproducible Before/After Benchmark for the ANTsTorch PR #96–#99 & Greedy Port into syntx.

Measures Before (`v6.39.0` baseline at HEAD `2c15e8f`) vs. After (current worktree)
across isolated fresh subprocesses on CPU and Apple MPS:

1. `blob.py` (`_separable_gaussian3d` + `detect_sift3d` on `mbhard` `192x256x256`)
2. `smoothing.py` (`separable_gaussian_filter` on CPU `192x256x256x3` / `x1` and MPS `128^3` `sigma=20` / `192x256x256x3` `sigma=3`)
3. `features.py` (`SwinViT` 32-multiple pad on `(1, 1, 18, 256, 256)` MPS vs. CPU)
4. `cross_correlation.py` (`_ChunkedLocalNCC` `metric_chunks=1` vs. `metric_chunks=4` longest-axis parity & memory)
5. Deformable solvers (`3-channel syntx.syn`, `single-channel syntx.syn`, `syntx.tvf`,
   `syntx.syngs`, `syntx.greedy` default/matched/return_inverse) on `mbhard`
   (Mindboggle Pair 44, `192x256x256`, shared affine).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
AFTER_SRC = REPO_ROOT / "src"
DEFAULT_JSON_OUT = REPO_ROOT / "docs" / "provenance" / "pr96_98_port_benchmark.json"


def _wait_for_mps_idle(timeout_s: float = 120.0) -> None:
    """Wait if any external MPS benchmark process is active (GEMINI.md §4)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        proc = subprocess.run(
            ["pgrep", "-fl", "pilot_voxelmorph|fast_syn_ants|vxm"],
            capture_output=True,
            text=True,
        )
        lines = [
            ln for ln in proc.stdout.splitlines()
            if ln.strip() and str(os.getpid()) not in ln and "pgrep" not in ln
        ]
        if not lines:
            return
        time.sleep(2.0)


def _run_worker(src_path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single benchmark worker task in a fresh isolated subprocess."""
    _wait_for_mps_idle()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_path)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-json",
        json.dumps(payload),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker failed (exit {proc.returncode}) for payload={payload}.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("WORKER_RESULT_JSON:"):
            return json.loads(line[len("WORKER_RESULT_JSON:"):])
    raise RuntimeError(f"Worker did not emit WORKER_RESULT_JSON.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def _worker_entry(payload_str: str) -> int:
    payload = json.loads(payload_str)
    task = payload["task"]
    variant = payload.get("variant", "after")

    import ants
    import numpy as np
    import torch
    import torch.nn.functional as F
    import syntx

    if task == "blob_bench":
        from syntx.benchmark.data import load_mindboggle_pair
        from syntx.landmarks import detect_sift3d
        from syntx.landmarks.blob import _separable_gaussian3d, _to_tensor, _normalize_intensity

        pair = load_mindboggle_pair(44)
        fi = pair["fixed"]
        device_name = payload["device"]
        dev = torch.device(device_name)
        vol = _normalize_intensity(_to_tensor(fi, dev))

        for _ in range(2):
            _ = _separable_gaussian3d(vol, (2.0, 2.0, 2.0))
            if device_name == "mps":
                torch.mps.synchronize()

        times_ms = []
        for _ in range(7):
            if device_name == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            out = _separable_gaussian3d(vol, (2.0, 2.0, 2.0))
            if device_name == "mps":
                torch.mps.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        out_cpu = out.detach().cpu()
        chk_path = payload.get("tensor_out")
        if chk_path:
            torch.save(out_cpu, chk_path)

        _ = detect_sift3d(fi, preprocess=False, max_keypoints=200, device=device_name)
        if device_name == "mps":
            torch.mps.synchronize()
        sift_times_s = []
        for _ in range(3):
            if device_name == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            kpts, _ = detect_sift3d(fi, preprocess=False, max_keypoints=200, device=device_name)
            if device_name == "mps":
                torch.mps.synchronize()
            sift_times_s.append(time.perf_counter() - t0)

        res = {
            "blur_median_ms": float(statistics.median(times_ms)),
            "blur_iqr_ms": float(np.percentile(times_ms, 75) - np.percentile(times_ms, 25)),
            "sift3d_median_s": float(statistics.median(sift_times_s)),
            "n_keypoints": int(len(kpts)),
        }
        print("WORKER_RESULT_JSON:" + json.dumps(res))
        return 0

    if task == "smoothing_cpu":
        from syntx.core.smoothing import separable_gaussian_filter

        torch.manual_seed(0)
        shape = tuple(payload["shape"])
        sigma = float(payload["sigma"])
        grid = torch.randn(*shape, dtype=torch.float32)

        for _ in range(2):
            _ = separable_gaussian_filter(grid, sigma=sigma, mode="replicate")

        times_ms = []
        for _ in range(5):
            t0 = time.perf_counter()
            out = separable_gaussian_filter(grid, sigma=sigma, mode="replicate")
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        if payload.get("tensor_out"):
            torch.save(out, payload["tensor_out"])

        res = {
            "median_ms": float(statistics.median(times_ms)),
            "iqr_ms": float(np.percentile(times_ms, 75) - np.percentile(times_ms, 25)),
        }
        print("WORKER_RESULT_JSON:" + json.dumps(res))
        return 0

    if task == "smoothing_mps":
        from syntx.core.smoothing import get_cached_gaussian_kernel_1d, separable_gaussian_filter

        torch.manual_seed(0)
        shape = tuple(payload["shape"])
        sigma = float(payload["sigma"])
        klen = int(get_cached_gaussian_kernel_1d(sigma, "cpu", torch.float32).shape[-1])
        grid_cpu = torch.randn(*shape, dtype=torch.float32)
        torch.mps.empty_cache()
        torch.mps.synchronize()
        grid = grid_cpu.to("mps")
        torch.mps.synchronize()
        base_driver_mb = torch.mps.driver_allocated_memory() / (1024.0 ** 2)

        out = separable_gaussian_filter(grid, sigma=sigma, mode="replicate")
        torch.mps.synchronize()
        after_first_driver_mb = torch.mps.driver_allocated_memory() / (1024.0 ** 2)
        after_first_alloc_mb = torch.mps.current_allocated_memory() / (1024.0 ** 2)

        for _ in range(2):
            out = separable_gaussian_filter(grid, sigma=sigma, mode="replicate")
        torch.mps.synchronize()

        times_ms = []
        peak_driver_mb = after_first_driver_mb
        for _ in range(7):
            torch.mps.synchronize()
            t0 = time.perf_counter()
            out = separable_gaussian_filter(grid, sigma=sigma, mode="replicate")
            torch.mps.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            peak_driver_mb = max(peak_driver_mb, torch.mps.driver_allocated_memory() / (1024.0 ** 2))

        if payload.get("tensor_out"):
            torch.save(out.detach().cpu(), payload["tensor_out"])

        res = {
            "sigma": sigma,
            "klen": klen,
            "median_ms": float(statistics.median(times_ms)),
            "iqr_ms": float(np.percentile(times_ms, 75) - np.percentile(times_ms, 25)),
            "base_driver_mb": float(base_driver_mb),
            "peak_driver_mb": float(peak_driver_mb),
            "driver_delta_mb": float(peak_driver_mb - base_driver_mb),
            "alloc_mb": float(after_first_alloc_mb),
        }
        print("WORKER_RESULT_JSON:" + json.dumps(res))
        return 0

    if task == "features_swinvit_pad":
        from syntx.features import SwinUNETRExtractor

        torch.manual_seed(40)
        x_cpu = torch.randn(1, 1, 18, 256, 256, dtype=torch.float32)
        x_mps = x_cpu.to("mps")

        if variant == "before":
            pad_cpu = F.pad(x_cpu, (0, 0, 0, 0, 0, 14), mode="constant", value=0.0)
            pad_mps = F.pad(x_mps, (0, 0, 0, 0, 0, 14), mode="constant", value=0.0).cpu()
        else:
            pad_cpu = SwinUNETRExtractor._pad_to_multiple_3d(x_cpu, multiple=32)
            pad_mps = SwinUNETRExtractor._pad_to_multiple_3d(x_mps, multiple=32).cpu()

        max_diff = float((pad_cpu - pad_mps).abs().max().item())
        mean_diff = float((pad_cpu - pad_mps).abs().mean().item())
        res = {
            "shape_in": list(x_cpu.shape),
            "shape_out": list(pad_cpu.shape),
            "mps_vs_cpu_max_diff": max_diff,
            "mps_vs_cpu_mean_diff": mean_diff,
        }
        print("WORKER_RESULT_JSON:" + json.dumps(res))
        return 0

    if task == "chunked_lncc_bench":
        import syntx.core.losses.cross_correlation as cc_mod
        from syntx.core.losses import local_ncc_loss_nd

        torch.manual_seed(99)
        chunks = int(payload["metric_chunks"])
        shape = tuple(payload["shape"])
        I_cpu = torch.rand(*shape, dtype=torch.float32)
        J_cpu = torch.rand(*shape, dtype=torch.float32)
        I_m = I_cpu.to("mps").detach().requires_grad_(True)
        J_m = J_cpu.to("mps").detach().requires_grad_(True)

        # Warmup once and clear grad/cache before timing & sampling peak in-flight memory
        w_loss = local_ncc_loss_nd(I_m, J_m, window_size=5, squared=True, metric_chunks=chunks)
        w_loss.backward()
        del w_loss
        I_m.grad = None
        J_m.grad = None
        torch.mps.synchronize()
        torch.mps.empty_cache()
        torch.mps.synchronize()

        base_alloc_mb = torch.mps.current_allocated_memory() / (1024.0 ** 2)
        base_driver_mb = torch.mps.driver_allocated_memory() / (1024.0 ** 2)
        peak_alloc_mb = base_alloc_mb
        peak_driver_mb = base_driver_mb

        orig_sbm = cc_mod._separable_avg_pool_nd

        def _tracked_sbm(*a, **kw):
            nonlocal peak_alloc_mb, peak_driver_mb
            r = orig_sbm(*a, **kw)
            peak_alloc_mb = max(peak_alloc_mb, torch.mps.current_allocated_memory() / (1024.0 ** 2))
            peak_driver_mb = max(peak_driver_mb, torch.mps.driver_allocated_memory() / (1024.0 ** 2))
            return r

        cc_mod._separable_avg_pool_nd = _tracked_sbm
        try:
            torch.mps.synchronize()
            t0 = time.perf_counter()
            loss = local_ncc_loss_nd(I_m, J_m, window_size=5, squared=True, metric_chunks=chunks)
            peak_alloc_mb = max(peak_alloc_mb, torch.mps.current_allocated_memory() / (1024.0 ** 2))
            peak_driver_mb = max(peak_driver_mb, torch.mps.driver_allocated_memory() / (1024.0 ** 2))
            loss.backward()
            torch.mps.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            cc_mod._separable_avg_pool_nd = orig_sbm

        if payload.get("grad_out"):
            torch.save(
                {"loss": loss.detach().cpu(), "gI": I_m.grad.detach().cpu(), "gJ": J_m.grad.detach().cpu()},
                payload["grad_out"],
            )

        res = {
            "metric_chunks": chunks,
            "shape": list(shape),
            "elapsed_ms": float(elapsed_ms),
            "loss": float(loss.detach().cpu().item()),
            "base_alloc_mb": float(base_alloc_mb),
            "peak_alloc_mb": float(peak_alloc_mb),
            "alloc_delta_mb": float(peak_alloc_mb - base_alloc_mb),
            "base_driver_mb": float(base_driver_mb),
            "peak_driver_mb": float(peak_driver_mb),
            "driver_delta_mb": float(peak_driver_mb - base_driver_mb),
        }
        print("WORKER_RESULT_JSON:" + json.dumps(res))
        return 0

    if task == "prepare_mbhard_affine":
        from syntx.benchmark.data import load_mindboggle_pair

        pair = load_mindboggle_pair(44)
        aff = syntx.robust_affine(pair["fixed"], pair["moving"], mode="auto", device="mps", seed=42)
        tx_path = payload["affine_out"]
        import shutil
        shutil.copyfile(aff["fwdtransforms"][0], tx_path)
        print("WORKER_RESULT_JSON:" + json.dumps({"affine_path": tx_path}))
        return 0

    if task == "mbhard_solver":
        from syntx.benchmark.data import load_mindboggle_pair
        from syntx.benchmark.evaluate import normalize_intensity
        from syntx.deformation_metrics import compute_bidirectional_dice
        from syntx.registration_metrics import evaluate_registration_qc
        from syntx.transforms import apply_transforms as syntx_apply_transforms

        pair = load_mindboggle_pair(44)
        fi_raw, mi_raw = pair["fixed"], pair["moving"]
        fl, ml = pair["fixed_label"], pair["moving_label"]
        fi = normalize_intensity(fi_raw)
        mi = normalize_intensity(mi_raw)
        aff_mat = payload["affine_path"]
        solver_name = payload["solver"]
        out_dir = Path(payload["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        outprefix = str(out_dir / f"{solver_name}_{variant}_")

        torch.mps.empty_cache()
        gc.collect()
        torch.mps.synchronize()

        peak_alloc_mb = 0.0
        peak_driver_mb = 0.0

        def _sample_mps_mem():
            nonlocal peak_alloc_mb, peak_driver_mb
            if torch.backends.mps.is_available():
                peak_alloc_mb = max(peak_alloc_mb, torch.mps.current_allocated_memory() / (1024.0 ** 2))
                peak_driver_mb = max(peak_driver_mb, torch.mps.driver_allocated_memory() / (1024.0 ** 2))

        orig_backward = torch.Tensor.backward

        def _tracked_backward(self_t, *args, **kwargs):
            _sample_mps_mem()
            r = orig_backward(self_t, *args, **kwargs)
            _sample_mps_mem()
            return r

        orig_ag_backward = torch.autograd.backward

        def _tracked_ag_backward(*args, **kwargs):
            _sample_mps_mem()
            r = orig_ag_backward(*args, **kwargs)
            _sample_mps_mem()
            return r

        torch.Tensor.backward = _tracked_backward
        torch.autograd.backward = _tracked_ag_backward

        try:
            if solver_name == "syn_3channel":
                fi_s = ants.smooth_image(fi, 1.5)
                mi_s = ants.smooth_image(mi, 1.5)
                fi_g = ants.iMath(fi, "Grad", 1.0)
                mi_g = ants.iMath(mi, "Grad", 1.0)
                torch.mps.synchronize()
                t0 = time.perf_counter()
                reg = syntx.syn(
                    fixed=[fi, fi_s, fi_g],
                    moving=[mi, mi_s, mi_g],
                    syn_metric=["cc2", "mse", "mattes"],
                    syn_metric_weights=[0.5, 0.3, 0.2],
                    reg_iterations=[40, 40, 15],
                    levels=[4, 2, 1],
                    regularizer="beltrami",
                    grad_step=0.7,
                    initial_transform=aff_mat,
                    device="mps",
                    seed=42,
                    outprefix=outprefix,
                )
                torch.mps.synchronize()
                elapsed_s = time.perf_counter() - t0
            elif solver_name == "syn_beltrami":
                torch.mps.synchronize()
                t0 = time.perf_counter()
                reg = syntx.syn(
                    fixed=fi,
                    moving=mi,
                    regularizer="beltrami",
                    grad_step=0.7,
                    reg_iterations=[100, 100, 20],
                    levels=[4, 2, 1],
                    initial_transform=aff_mat,
                    device="mps",
                    seed=42,
                    outprefix=outprefix,
                )
                torch.mps.synchronize()
                elapsed_s = time.perf_counter() - t0
            elif solver_name == "tvf":
                torch.mps.synchronize()
                t0 = time.perf_counter()
                reg = syntx.tvf(
                    fixed=fi,
                    moving=mi,
                    regularizer="beltrami",
                    reg_iterations=[100, 100, 20],
                    levels=[4, 2, 1],
                    initial_transform=aff_mat,
                    device="mps",
                )
                torch.mps.synchronize()
                elapsed_s = time.perf_counter() - t0
            elif solver_name == "syngs":
                torch.mps.synchronize()
                t0 = time.perf_counter()
                reg = syntx.syngs(
                    fixed=fi,
                    moving=mi,
                    reg_iterations=[100, 100, 20],
                    levels=[4, 2, 1],
                    initial_transform=aff_mat,
                    device="mps",
                )
                torch.mps.synchronize()
                elapsed_s = time.perf_counter() - t0
            elif solver_name == "greedy_default":
                torch.mps.synchronize()
                t0 = time.perf_counter()
                reg = syntx.greedy(
                    fixed=fi,
                    moving=mi,
                    reg_iterations=[100, 100, 20],
                    scales=[4, 2, 1],
                    optimizer="adam",
                    initial_transform=aff_mat,
                    return_inverse=False,
                    device="mps",
                )
                torch.mps.synchronize()
                elapsed_s = time.perf_counter() - t0
            elif solver_name == "greedy_matched":
                torch.mps.synchronize()
                t0 = time.perf_counter()
                reg = syntx.greedy(
                    fixed=fi,
                    moving=mi,
                    reg_iterations=[100, 100, 50],
                    scales=[4, 2, 1],
                    optimizer="regadam",
                    initial_transform=aff_mat,
                    return_inverse=False,
                    device="mps",
                )
                torch.mps.synchronize()
                elapsed_s = time.perf_counter() - t0
            elif solver_name == "greedy_with_inv":
                torch.mps.synchronize()
                t0 = time.perf_counter()
                reg = syntx.greedy(
                    fixed=fi,
                    moving=mi,
                    reg_iterations=[100, 100, 20],
                    scales=[4, 2, 1],
                    optimizer="adam",
                    initial_transform=aff_mat,
                    return_inverse=True,
                    device="mps",
                )
                torch.mps.synchronize()
                elapsed_s = time.perf_counter() - t0
            else:
                raise ValueError(f"Unknown solver {solver_name}")
        finally:
            torch.Tensor.backward = orig_backward
            torch.autograd.backward = orig_ag_backward

        _sample_mps_mem()
        post_alloc_mb = torch.mps.current_allocated_memory() / (1024.0 ** 2)
        post_driver_mb = torch.mps.driver_allocated_memory() / (1024.0 ** 2)

        fwd_tx = list(reg["fwdtransforms"])
        inv_tx = list(reg.get("invtransforms") or [])
        fl_c, ml_c = fl.clone(), ml.clone()
        fl_c.set_origin(fi_raw.origin)
        fl_c.set_spacing(fi_raw.spacing)
        fl_c.set_direction(fi_raw.direction)
        ml_c.set_origin(mi_raw.origin)
        ml_c.set_spacing(mi_raw.spacing)
        ml_c.set_direction(mi_raw.direction)

        if inv_tx:
            which_inv = (
                [bool(x) for x in reg["whichtoinvert_inv"]]
                if reg.get("whichtoinvert_inv") is not None
                else [isinstance(t, str) and t.endswith(".mat") for t in inv_tx]
            )
            dice_fix, dice_mov, dice_sym = compute_bidirectional_dice(
                fl_c, ml_c, fi_raw, mi_raw, fwd_tx, inv_tx, whichtoinvert_inv=which_inv
            )
            dice_sym = float(dice_sym)
            dice_fix = float(dice_fix)
            dice_mov = float(dice_mov)
        else:
            ml_w = syntx_apply_transforms(
                fixed=fi_raw, moving=ml_c, transformlist=fwd_tx, interpolator="nearestNeighbor"
            )
            df_ov = ants.label_overlap_measures(fl_c, ml_w)
            df_pos = df_ov[df_ov["Label"] != "All"]
            df_pos = df_pos[df_pos["Label"].astype(float) > 0]
            dice_fix = float(df_pos["MeanOverlap"].mean())
            dice_sym = dice_fix
            dice_mov = float("nan")

        qc = evaluate_registration_qc(fi, mi, reg)
        prov = reg.get("provenance") or {}
        deformable_s = float(prov.get("runtime_deformable_sec", elapsed_s))

        res = {
            "solver": solver_name,
            "variant": variant,
            "runtime_s": float(elapsed_s),
            "runtime_deformable_s": deformable_s,
            "runtime_nonloop_overhead_s": float(max(0.0, elapsed_s - deformable_s)),
            "peak_alloc_mb": float(peak_alloc_mb),
            "peak_driver_mb": float(peak_driver_mb),
            "post_alloc_mb": float(post_alloc_mb),
            "post_driver_mb": float(post_driver_mb),
            "dice_symmetric": dice_sym,
            "dice_fixed_space": dice_fix,
            "dice_moving_space": None if (isinstance(dice_mov, float) and np.isnan(dice_mov)) else dice_mov,
            "folding_pct": float(qc.folding_pct),
            "min_jacobian": float(qc.min_jacobian),
            "inverse_consistency_mean_mm": (
                float(qc.inverse_consistency_error_mean_mm)
                if qc.inverse_consistency_error_mean_mm is not None
                else 0.0
            ),
        }
        print("WORKER_RESULT_JSON:" + json.dumps(res))
        return 0

    raise ValueError(f"Unknown worker task: {task}")


def run_smoke() -> int:
    """Fast <30s smoke check verifying imports and all ported improvements."""
    sys.path.insert(0, str(AFTER_SRC))
    import torch
    import syntx
    from syntx.core.losses import local_ncc_loss_nd, parse_similarity_metric
    from syntx.core.smoothing import separable_gaussian_filter
    from syntx.features import SwinUNETRExtractor
    from syntx.landmarks.blob import _separable_gaussian3d
    from syntx.syn import _backward_similarity_terms_per_term

    torch.manual_seed(0)
    x = torch.randn(1, 1, 8, 10, 12)
    b = _separable_gaussian3d(x, (1.2, 0.0, 1.5), mode="replicate")
    assert b.shape == x.shape and torch.isfinite(b).all()

    g = torch.randn(1, 8, 10, 12, 3)
    s = separable_gaussian_filter(g, [1.5, 0.0, 2.0], mode="replicate")
    assert s.shape == g.shape and torch.isfinite(s).all()

    p = SwinUNETRExtractor._pad_to_multiple_3d(torch.randn(1, 1, 18, 32, 32), multiple=32)
    assert p.shape == (1, 1, 32, 32, 32)

    I_m = torch.rand(1, 2, 8, 8, 8, requires_grad=True)
    J_m = torch.rand(1, 2, 8, 8, 8, requires_grad=True)
    loss_t, _, _ = _backward_similarity_terms_per_term(
        I_m * 1.0,
        J_m * 1.0,
        ["mse1", "mse2"],
        [lambda a, b, mask=None: (a - b).square().mean(), lambda a, b, mask=None: (a - b).abs().mean()],
        [0.5, 0.5],
        propagate_to_mid=True,
    )
    assert torch.isfinite(loss_t) and I_m.grad is not None and J_m.grad is not None

    # Verify PR #99 longest-axis chunked LNCC
    I_thin = torch.rand(1, 1, 8, 24, 20, requires_grad=True)
    J_thin = torch.rand(1, 1, 8, 24, 20, requires_grad=True)
    l_chunk = local_ncc_loss_nd(I_thin, J_thin, window_size=5, squared=True, metric_chunks=3)
    l_chunk.backward()
    assert torch.isfinite(l_chunk) and I_thin.grad is not None
    assert parse_similarity_metric("cc2", metric_chunks=2).metric_chunks == 2

    print(f"[SMOKE PASS] syntx {syntx.__version__} PR #96-#99 + greedy port primitives verified.")
    return 0


def run_full_benchmark(out_json: Path) -> int:
    import torch

    with tempfile.TemporaryDirectory(prefix="syntx_pr96_98_bench_") as tmp_str:
        tmp_dir = Path(tmp_str)
        before_root = tmp_dir / "before_head"
        before_root.mkdir(parents=True, exist_ok=True)
        archive_proc = subprocess.Popen(
            ["git", "-C", str(REPO_ROOT), "archive", "HEAD"],
            stdout=subprocess.PIPE,
        )
        subprocess.run(["tar", "-x", "-C", str(before_root)], stdin=archive_proc.stdout, check=True)
        archive_proc.wait()
        before_src = before_root / "src"

        results: Dict[str, Any] = {
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "baseline_commit": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "suites": {},
        }

        # 1. Suite 1: blob.py (_separable_gaussian3d + detect_sift3d on mbhard 192x256x256)
        print("[1/5] Benchmarking blob.py (_separable_gaussian3d & detect_sift3d on mbhard 192x256x256)...", flush=True)
        blob_suite: Dict[str, Any] = {}
        for dev in ("cpu", "mps"):
            t_bef = str(tmp_dir / f"blob_{dev}_before.pt")
            t_aft = str(tmp_dir / f"blob_{dev}_after.pt")
            r_bef = _run_worker(before_src, {"task": "blob_bench", "device": dev, "variant": "before", "tensor_out": t_bef})
            r_aft = _run_worker(AFTER_SRC, {"task": "blob_bench", "device": dev, "variant": "after", "tensor_out": t_aft})
            tb = torch.load(t_bef, weights_only=True)
            ta = torch.load(t_aft, weights_only=True)
            interior_max_diff = float((tb[:, :, 6:-6, 6:-6, 6:-6] - ta[:, :, 6:-6, 6:-6, 6:-6]).abs().max().item())
            full_max_diff = float((tb - ta).abs().max().item())
            blob_suite[dev] = {
                "before": r_bef,
                "after": r_aft,
                "blur_speedup_x": round(r_bef["blur_median_ms"] / r_aft["blur_median_ms"], 3),
                "sift3d_speedup_x": round(r_bef["sift3d_median_s"] / r_aft["sift3d_median_s"], 3),
                "interior_max_diff": interior_max_diff,
                "full_volume_max_diff": full_max_diff,
            }
        results["suites"]["blob_separable_gaussian3d"] = blob_suite

        # 2. Suite 2: smoothing.py (separable_gaussian_filter CPU & MPS)
        print("[2/5] Benchmarking smoothing.py (separable_gaussian_filter CPU & MPS)...", flush=True)
        smooth_suite: Dict[str, Any] = {"cpu": {}, "mps": {}}
        for label, shape, sig in (
            ("vector_192x256x256x3_sigma3", (1, 192, 256, 256, 3), 3.0),
            ("scalar_192x256x256x1_sigma2", (1, 192, 256, 256, 1), 2.0),
        ):
            t_bef = str(tmp_dir / f"sm_cpu_{label}_bef.pt")
            t_aft = str(tmp_dir / f"sm_cpu_{label}_aft.pt")
            r_bef = _run_worker(before_src, {"task": "smoothing_cpu", "shape": shape, "sigma": sig, "tensor_out": t_bef})
            r_aft = _run_worker(AFTER_SRC, {"task": "smoothing_cpu", "shape": shape, "sigma": sig, "tensor_out": t_aft})
            tb = torch.load(t_bef, weights_only=True)
            ta = torch.load(t_aft, weights_only=True)
            max_diff = float((tb - ta).abs().max().item())
            smooth_suite["cpu"][label] = {
                "before_median_ms": r_bef["median_ms"],
                "after_median_ms": r_aft["median_ms"],
                "speedup_x": round(r_bef["median_ms"] / r_aft["median_ms"], 3),
                "max_diff": max_diff,
            }

        for label, shape, sig in (
            ("large_kernel_128cubed_sigma20", (1, 128, 128, 128, 1), 20.0),
            ("standard_kernel_192x256x256x3_sigma3", (1, 192, 256, 256, 3), 3.0),
        ):
            t_bef = str(tmp_dir / f"sm_mps_{label}_bef.pt")
            t_aft = str(tmp_dir / f"sm_mps_{label}_aft.pt")
            r_bef = _run_worker(before_src, {"task": "smoothing_mps", "shape": shape, "sigma": sig, "tensor_out": t_bef})
            r_aft = _run_worker(AFTER_SRC, {"task": "smoothing_mps", "shape": shape, "sigma": sig, "tensor_out": t_aft})
            tb = torch.load(t_bef, weights_only=True)
            ta = torch.load(t_aft, weights_only=True)
            max_diff = float((tb - ta).abs().max().item())
            smooth_suite["mps"][label] = {
                "before": r_bef,
                "after": r_aft,
                "speedup_x": round(r_bef["median_ms"] / r_aft["median_ms"], 3),
                "driver_mem_ratio_x": round(r_bef["peak_driver_mb"] / max(r_aft["peak_driver_mb"], 1e-6), 3),
                "max_diff": max_diff,
            }
        results["suites"]["smoothing_separable_gaussian_filter"] = smooth_suite

        # 3. Suite 3: features.py (SwinViT 32-multiple pad on (1, 1, 18, 256, 256))
        print("[3/5] Benchmarking features.py (SwinViT 32-multiple pad on (1, 1, 18, 256, 256))...", flush=True)
        f_bef = _run_worker(before_src, {"task": "features_swinvit_pad", "variant": "before"})
        f_aft = _run_worker(AFTER_SRC, {"task": "features_swinvit_pad", "variant": "after"})
        results["suites"]["features_swinvit_pad"] = {
            "before_5d_f_pad": f_bef,
            "after_per_axis_cat": f_aft,
        }

        # 4. Suite 4: PR #99 _ChunkedLocalNCC (metric_chunks=1 vs metric_chunks=4 on (1, 1, 64, 256, 256))
        print("[4/5] Benchmarking PR #99 _ChunkedLocalNCC (metric_chunks=1 vs 4 on (1, 1, 64, 256, 256))...", flush=True)
        g_mc1 = str(tmp_dir / "mc1.pt")
        g_mc4 = str(tmp_dir / "mc4.pt")
        r_mc1 = _run_worker(AFTER_SRC, {"task": "chunked_lncc_bench", "metric_chunks": 1, "shape": (1, 1, 64, 256, 256), "grad_out": g_mc1})
        r_mc4 = _run_worker(AFTER_SRC, {"task": "chunked_lncc_bench", "metric_chunks": 4, "shape": (1, 1, 64, 256, 256), "grad_out": g_mc4})
        d1 = torch.load(g_mc1, weights_only=True)
        d4 = torch.load(g_mc4, weights_only=True)
        results["suites"]["pr99_chunked_lncc"] = {
            "metric_chunks_1": r_mc1,
            "metric_chunks_4": r_mc4,
            "loss_abs_diff": float((d1["loss"] - d4["loss"]).abs().item()),
            "grad_I_max_diff": float((d1["gI"] - d4["gI"]).abs().max().item()),
            "grad_J_max_diff": float((d1["gJ"] - d4["gJ"]).abs().max().item()),
            "alloc_delta_saving_mb": round(r_mc1["alloc_delta_mb"] - r_mc4["alloc_delta_mb"], 2),
            "driver_delta_saving_mb": round(r_mc1["driver_delta_mb"] - r_mc4["driver_delta_mb"], 2),
        }

        # 5. Suite 5: Deformable solvers on mbhard (Pair 44, 192x256x256, shared affine)
        print("[5/5] Benchmarking deformable solvers on mbhard (Pair 44, 192x256x256, shared affine)...", flush=True)
        aff_path = str(tmp_dir / "mbhard_shared_affine.mat")
        _run_worker(AFTER_SRC, {"task": "prepare_mbhard_affine", "affine_out": aff_path})

        solvers_suite: Dict[str, Any] = {}
        for solver in (
            "syn_3channel",
            "syn_beltrami",
            "tvf",
            "syngs",
            "greedy_default",
            "greedy_matched",
            "greedy_with_inv",
        ):
            print(f"  -> Running {solver} (before vs. after) in isolated MPS subprocesses...", flush=True)
            s_bef = _run_worker(
                before_src,
                {
                    "task": "mbhard_solver",
                    "solver": solver,
                    "variant": "before",
                    "affine_path": aff_path,
                    "out_dir": str(tmp_dir / f"out_{solver}_bef"),
                },
            )
            s_aft = _run_worker(
                AFTER_SRC,
                {
                    "task": "mbhard_solver",
                    "solver": solver,
                    "variant": "after",
                    "affine_path": aff_path,
                    "out_dir": str(tmp_dir / f"out_{solver}_aft"),
                },
            )
            solvers_suite[solver] = {
                "before": s_bef,
                "after": s_aft,
                "delta_peak_alloc_mb": round(s_aft["peak_alloc_mb"] - s_bef["peak_alloc_mb"], 2),
                "delta_peak_driver_mb": round(s_aft["peak_driver_mb"] - s_bef["peak_driver_mb"], 2),
                "delta_runtime_s": round(s_aft["runtime_s"] - s_bef["runtime_s"], 3),
                "speedup_x": round(s_bef["runtime_s"] / max(s_aft["runtime_s"], 1e-6), 3),
                "delta_dice_symmetric": round(s_aft["dice_symmetric"] - s_bef["dice_symmetric"], 6),
                "delta_dice_fixed": round(s_aft["dice_fixed_space"] - s_bef["dice_fixed_space"], 6),
            }
        results["suites"]["mbhard_deformable_solvers"] = solvers_suite

        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"[BENCHMARK DONE] Wrote {out_json}", flush=True)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ANTsTorch PR #96-#99 + greedy port in syntx.")
    parser.add_argument("--smoke", action="store_true", help="Run fast <30s smoke check.")
    parser.add_argument("--worker-json", type=str, default=None, help="Internal subprocess worker JSON payload.")
    parser.add_argument(
        "--out-json", "--output", dest="out_json", type=Path, default=DEFAULT_JSON_OUT, help="Output JSON path."
    )
    args = parser.parse_args()

    if args.worker_json is not None:
        return _worker_entry(args.worker_json)
    if args.smoke:
        return run_smoke()
    return run_full_benchmark(args.out_json)


if __name__ == "__main__":
    raise SystemExit(main())
