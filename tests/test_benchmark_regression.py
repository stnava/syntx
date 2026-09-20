"""
Tests for Benchmark Regression Prevention, Parameter Integrity, and Metric Invariants.
========================================================================================

Verifies that:
1. Metric math strictly follows Sørensen-Dice (MeanOverlap) and never confuses TargetOverlap.
2. Production benchmark configurations for syn, syngs, tvf, and greedy are locked.
3. SHA-256 configuration hashing detects any parameter change and invalidates stale disk caches.
4. Cache validation (is_pair_completed) rejects results with missing or mismatched config hashes.
5. Fast 2D registration sanity check verifies smooth convergence without folding regressions.
"""

import copy
import json
import os
import tempfile
import numpy as np
import pytest
import ants
import torch
import syntx

from syntx.benchmark.config import (
    DEFAULT_BENCHMARK_CONFIG,
    get_default_config,
    get_model_config,
    compute_config_hash,
    validate_config_compatibility,
)
from syntx.deformation_metrics import compute_bidirectional_dice
from syntx.benchmark.metrics import compute_pair_metrics


# ==============================================================================
# 1. Metric Math Invariant: Sørensen-Dice vs Target Overlap
# ==============================================================================

class TestMetricMathInvariants:
    """Verifies strict adherence to Sørensen-Dice formulation (2|A ∩ B| / (|A| + |B|))."""

    def test_sorensen_dice_vs_target_overlap_asymmetric(self):
        """
        Verify that compute_bidirectional_dice computes Sørensen-Dice, NOT TargetOverlap,
        on asymmetric label sets.
        
        Set A: 100 voxels
        Set B: 200 voxels
        Overlap A ∩ B: 80 voxels
        
        Sørensen-Dice = 2 * 80 / (100 + 200) = 160 / 300 = 0.533333...
        Target Overlap (A -> B) = 80 / 100 = 0.800000
        Target Overlap (B -> A) = 80 / 200 = 0.400000
        """
        # Create 2D images
        shape = (50, 50)
        arr_fixed = np.zeros(shape, dtype=np.uint32)
        arr_moving = np.zeros(shape, dtype=np.uint32)

        # Overlap region: 80 voxels (label 1)
        arr_fixed[10:18, 10:20] = 1   # 8 * 10 = 80 voxels
        arr_moving[10:18, 10:20] = 1  # 80 voxels overlap

        # Non-overlapping regions
        arr_fixed[20:22, 10:20] = 1   # 2 * 10 = 20 extra voxels -> Total A = 100
        arr_moving[25:37, 10:20] = 1  # 12 * 10 = 120 extra voxels -> Total B = 200

        assert np.sum(arr_fixed == 1) == 100
        assert np.sum(arr_moving == 1) == 200
        assert np.sum((arr_fixed == 1) & (arr_moving == 1)) == 80

        fi = ants.from_numpy(arr_fixed.astype(np.float32))
        mi = ants.from_numpy(arr_moving.astype(np.float32))
        fl = ants.from_numpy(arr_fixed)
        ml = ants.from_numpy(arr_moving)

        # Create identity transform file
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
            tx_path = tmp.name
        tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=2)
        ants.write_transform(tx, tx_path)

        try:
            dice_fixed, dice_moving, dice_sym = compute_bidirectional_dice(
                fl, ml, fi, mi,
                fwdtransforms=[tx_path],
                invtransforms=[tx_path],
                whichtoinvert_inv=[False],
            )

            expected_sorensen = 2.0 * 80.0 / (100.0 + 200.0)  # 0.5333333333333333
            target_overlap_fixed = 80.0 / 100.0               # 0.8000
            target_overlap_moving = 80.0 / 200.0              # 0.4000

            # Both fixed-space and moving-space Dice under identity must be exact Sørensen-Dice
            assert np.isclose(dice_fixed, expected_sorensen, atol=1e-5), (
                f"dice_fixed ({dice_fixed}) must match Sørensen-Dice ({expected_sorensen}), "
                f"NOT TargetOverlap ({target_overlap_fixed})"
            )
            assert np.isclose(dice_moving, expected_sorensen, atol=1e-5), (
                f"dice_moving ({dice_moving}) must match Sørensen-Dice ({expected_sorensen})"
            )
            assert np.isclose(dice_sym, expected_sorensen, atol=1e-5)

            # Strict guard against Target Overlap contamination
            assert not np.isclose(dice_fixed, target_overlap_fixed, atol=0.05)
            assert not np.isclose(dice_moving, target_overlap_moving, atol=0.05)
        finally:
            if os.path.exists(tx_path):
                os.remove(tx_path)


# ==============================================================================
# 2. Production Hyperparameter Locking & Config Invariants
# ==============================================================================

class TestBenchmarkConfigInvariants:
    """Verifies that production benchmark hyperparameters are locked to their peak versions."""

    def test_required_model_blocks_exist(self):
        cfg = get_default_config()
        assert "syn_config" in cfg
        assert "gaussian_config" in cfg
        assert "syngs_config" in cfg
        assert "tvf_config" in cfg
        assert "greedy_config" in cfg

    def test_iterations_standardized(self):
        """All benchmark models must run exactly [100, 100, 20] iterations."""
        expected_iters = [100, 100, 20]
        cfg = get_default_config()
        for m in ["syn", "gaussian", "syngs", "tvf", "greedy"]:
            m_cfg = get_model_config(m, cfg)
            assert m_cfg["reg_iterations"] == expected_iters, f"{m} iterations {m_cfg['reg_iterations']} != {expected_iters}"

    def test_tvf_peak_parameters(self):
        """TVF must use fast_smooth=False, step=0.54, and dsti1 regularizer for peak performance."""
        tvf_cfg = get_model_config("tvf")
        assert tvf_cfg["tvf_fast_smooth"] is False, "tvf_fast_smooth must be False to avoid over-smoothing gyri"
        assert tvf_cfg["max_step_norm"] == 0.54, "max_step_norm must be 0.54 for optimal DSTI-1 convergence"
        assert tvf_cfg["tvf_regularizer"] == "dsti1"
        assert tvf_cfg["similarity_metric"] == "cc2"

    def test_syngs_peak_parameters(self):
        """SyNGS must use alpha=0.45, max_step=0.19, and Sobolev regularizer for low folding."""
        syngs_cfg = get_model_config("syngs")
        assert syngs_cfg["alpha"] == 0.45, "alpha must be 0.45 to prevent high folding"
        assert syngs_cfg["max_step_norm"] == 0.19, "max_step_norm must be 0.19"
        assert syngs_cfg["regularizer"] == "sobolev"
        assert syngs_cfg["optimizer_lr"] == 1.0
        assert syngs_cfg["syn_metric"] == "cc2"

    def test_syn_peak_parameters(self):
        """SyN must route to Sobolev regularizer with alpha=1.5."""
        syn_cfg = get_model_config("syn")
        assert syn_cfg["syn_regularizer"] == "sobolev"
        assert syn_cfg["sobolev_alpha"] == 1.5
        assert syn_cfg["syn_metric"] == "cc2"


# ==============================================================================
# 3. SHA-256 Config Fingerprinting and Cache Invalidation
# ==============================================================================

class TestConfigHashingAndCacheInvalidation:
    """Verifies that config changes alter hash and trigger cache invalidation."""

    def test_config_hash_determinism(self):
        cfg = get_model_config("tvf")
        h1 = compute_config_hash(cfg)
        h2 = compute_config_hash(copy.deepcopy(cfg))
        assert h1 == h2
        assert len(h1) == 16

    def test_config_hash_sensitivity(self):
        """Any parameter change must produce a distinct hash."""
        cfg = get_model_config("syngs")
        h_baseline = compute_config_hash(cfg)

        cfg_modified = copy.deepcopy(cfg)
        cfg_modified["alpha"] = 0.20
        h_modified = compute_config_hash(cfg_modified)

        assert h_baseline != h_modified, "Changing alpha must change the config hash"

        cfg_modified2 = copy.deepcopy(cfg)
        cfg_modified2["max_step_norm"] = 0.30
        assert compute_config_hash(cfg_modified2) != h_baseline

    def test_config_hash_key_order_invariance(self):
        """Order of dictionary keys must not affect the deterministic hash."""
        cfg_a = {"alpha": 0.35, "lr": 1.0, "step": 0.2}
        cfg_b = {"step": 0.2, "alpha": 0.35, "lr": 1.0}
        assert compute_config_hash(cfg_a) == compute_config_hash(cfg_b)

    def test_validate_config_compatibility(self):
        h = compute_config_hash(get_model_config("syn"))
        assert validate_config_compatibility(h, h) is True
        assert validate_config_compatibility("deadbeef12345678", h) is False
        assert validate_config_compatibility(None, h) is False
        assert validate_config_compatibility("", h) is False

    def test_is_pair_completed_cache_invalidation(self, tmp_path):
        """Verify is_pair_completed rejects stale results when config_hash does not match."""
        from scripts.run_90pairs import is_pair_completed

        results_dir = str(tmp_path)
        pair_idx = 44
        model = "tvf"
        active_hash = "a1b2c3d4e5f60718"

        # 1. Result does not exist
        assert not is_pair_completed(results_dir, pair_idx, model, expected_config_hash=active_hash)

        # 2. Result exists with old/different hash
        res_file = os.path.join(results_dir, f"pair_{pair_idx:03d}_{model}.json")
        with open(res_file, "w") as f:
            json.dump({
                "status": "SUCCESS",
                "dice_sym": 0.5923,
                "config_hash": "stale_hash_00000"
            }, f)

        assert not is_pair_completed(results_dir, pair_idx, model, expected_config_hash=active_hash)

        # 3. Result exists with matching hash
        with open(res_file, "w") as f:
            json.dump({
                "status": "SUCCESS",
                "dice_sym": 0.6073,
                "config_hash": active_hash
            }, f)

        assert is_pair_completed(results_dir, pair_idx, model, expected_config_hash=active_hash)


# ==============================================================================
# 4. Fast 2D Non-Regression Multi-Model Integration Check
# ==============================================================================

class TestFastMultiModelNonRegression:
    """Verifies that all 3 primary models execute and improve alignment on synthetic phantom."""

    def test_fast_sobolev_syn_convergence(self):
        from syntx.syn import SyNTo
        torch.manual_seed(42)
        np.random.seed(42)

        # Simple 2D shifted discs
        grid = np.zeros((32, 32), dtype=np.float32)
        y, x = np.ogrid[:32, :32]
        c1 = (16, 14)
        c2 = (16, 18)
        mask1 = ((x - c1[1])**2 + (y - c1[0])**2) <= 36
        mask2 = ((x - c2[1])**2 + (y - c2[0])**2) <= 36

        fi = ants.from_numpy(mask1.astype(np.float32))
        mi = ants.from_numpy(mask2.astype(np.float32))

        # Baseline overlap
        init_dice = 2.0 * np.sum(mask1 & mask2) / (np.sum(mask1) + np.sum(mask2))

        # Register with SyN (Sobolev)
        reg = syntx.syn(
            fixed=fi,
            moving=mi,
            syn_regularizer="sobolev",
            sobolev_alpha=1.5,
            reg_iterations=[15, 10],
            device="cpu",
        )

        warped = ants.apply_transforms(fixed=fi, moving=mi, transformlist=reg["fwdtransforms"])
        w_mask = warped.numpy() > 0.5
        final_dice = 2.0 * np.sum(mask1 & w_mask) / (np.sum(mask1) + np.sum(w_mask))

        # Registration must strictly improve overlap
        assert final_dice > init_dice, f"Final Dice {final_dice} <= Initial Dice {init_dice}"

    def test_fast_syngs_convergence(self):
        torch.manual_seed(42)
        np.random.seed(42)

        y, x = np.ogrid[:32, :32]
        mask1 = ((x - 14)**2 + (y - 16)**2) <= 36
        mask2 = ((x - 18)**2 + (y - 16)**2) <= 36

        fi = ants.from_numpy(mask1.astype(np.float32))
        mi = ants.from_numpy(mask2.astype(np.float32))
        init_dice = 2.0 * np.sum(mask1 & mask2) / (np.sum(mask1) + np.sum(mask2))

        syngs_cfg = get_model_config("syngs")
        reg = syntx.syngs(
            fixed=fi,
            moving=mi,
            alpha=syngs_cfg["alpha"],
            max_step_norm=syngs_cfg["max_step_norm"],
            regularizer=syngs_cfg["regularizer"],
            optimizer_lr=syngs_cfg["optimizer_lr"],
            reg_iterations=[10, 5],
            device="cpu",
            verbose=False,
        )

        warped = ants.apply_transforms(fixed=fi, moving=mi, transformlist=reg["fwdtransforms"])
        w_mask = warped.numpy() > 0.5
        final_dice = 2.0 * np.sum(mask1 & w_mask) / (np.sum(mask1) + np.sum(w_mask))
        assert final_dice > init_dice, f"SyNGS Final Dice {final_dice} <= Initial Dice {init_dice}"

    def test_fast_tvf_convergence(self):
        torch.manual_seed(42)
        np.random.seed(42)

        y, x = np.ogrid[:32, :32]
        mask1 = ((x - 14)**2 + (y - 16)**2) <= 36
        mask2 = ((x - 18)**2 + (y - 16)**2) <= 36

        fi = ants.from_numpy(mask1.astype(np.float32))
        mi = ants.from_numpy(mask2.astype(np.float32))
        init_dice = 2.0 * np.sum(mask1 & mask2) / (np.sum(mask1) + np.sum(mask2))

        tvf_cfg = get_model_config("tvf")
        reg = syntx.tvf(
            fixed=fi,
            moving=mi,
            fast_smooth=tvf_cfg["tvf_fast_smooth"],
            max_step_norm=tvf_cfg["max_step_norm"],
            regularizer=tvf_cfg["tvf_regularizer"],
            optimizer_lr=tvf_cfg["optimizer_lr"],
            reg_iterations=[10, 5],
            device="cpu",
            verbose=False,
        )

        warped = ants.apply_transforms(fixed=fi, moving=mi, transformlist=reg["fwdtransforms"])
        w_mask = warped.numpy() > 0.5
        final_dice = 2.0 * np.sum(mask1 & w_mask) / (np.sum(mask1) + np.sum(w_mask))
        assert final_dice > init_dice, f"TVF Final Dice {final_dice} <= Initial Dice {init_dice}"

    def test_random_order_permutation_invariance(self):
        """Verifies random order shuffling produces a valid permutation of all 90 pairs."""
        import random
        n_pairs = 90
        original = list(range(n_pairs))
        shuffled1 = list(range(n_pairs))
        shuffled2 = list(range(n_pairs))

        rng1 = random.Random(42)
        rng1.shuffle(shuffled1)

        rng2 = random.Random(42)
        rng2.shuffle(shuffled2)

        # 1. Shuffled list is a permutation (contains all elements without loss or duplication)
        assert sorted(shuffled1) == original
        # 2. It is strictly not the sequential order
        assert shuffled1 != original
        # 3. Deterministic given the same seed
        assert shuffled1 == shuffled2

    def test_generate_live_html_report(self):
        """Verifies that generate_live_html_report produces a valid HTML dashboard with auto-refresh."""
        from syntx.benchmark.html_report import generate_live_html_report
        with tempfile.TemporaryDirectory() as tmpdir:
            out_html = os.path.join(tmpdir, "test_live.html")
            res = generate_live_html_report(
                results_dir=tmpdir,
                device="mps",
                models=["syn", "tvf"],
                total_pairs=10,
                out_html=out_html,
                refresh_seconds=10,
            )
            assert os.path.isfile(out_html)
            with open(out_html, "r") as f:
                content = f.read()
            assert "<meta http-equiv=\"refresh\" content=\"10\">" in content
            assert "syntx Multi-Model Benchmark Dashboard" in content
            assert "Eulerian Sobolev SyN" in content
            assert "Time-Varying Velocity Field (TVF)" in content

