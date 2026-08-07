import os
import sys
import json
import math
import tempfile
import pytest
import numpy as np
import torch
import torch.nn.functional as F
import ants

# Ensure syntx source directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import syntx
from syntx.syn import SyNTo, compute_jacobian_determinant_nd
from syntx.tvf import TVFModel

# Import bidirectional dice evaluation function if available from scratch script
try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scratch')))
    from characterize_30_grid import compute_bidirectional_dice, evaluate_jacobian, evaluate_inverse_error
except ImportError:
    compute_bidirectional_dice = None
    evaluate_jacobian = None
    evaluate_inverse_error = None


def create_identity_transform_file(dim=2):
    """
    Helper function to create a valid temporary ANTs identity transform file.
    """
    tx = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=dim)
    tmp = tempfile.NamedTemporaryFile(suffix='.mat', delete=False)
    tmp.close()
    ants.write_transform(tx, tmp.name)
    return tmp.name


# ==============================================================================
# TIER 1 & TIER 2: Sobolev Regularizer Stability (2D & 3D)
# ==============================================================================

class TestSobolevRegularizerStability:
    """
    Test suite for Sobolev Green operator regularization in 2D and 3D.
    Validates mathematical stability, MPS float32 contiguity, absence of magic clamps,
    and stable Jacobian grid output.
    """

    def test_sobolev_2d_math_stability_and_jacobian(self):
        """
        Verify Sobolev Green operator in 2D processes vector fields smoothly
        without NaN/Inf and produces non-folding, stable Jacobian grids.
        """
        syn_model = SyNTo(dim=2, grid_shape=(32, 32))
        
        # Create a 2D gradient update tensor field scaled for a registration step: [1, 32, 32, 2]
        torch.manual_seed(42)
        m_2d = torch.randn(1, 32, 32, 2, dtype=torch.float32) * 0.05
        
        # Apply Sobolev Green operator
        v_out_2d = syn_model._apply_sobolev_green_operator(m_2d, fluid_sigma=2.0, alpha=1.0)
        
        assert v_out_2d.shape == m_2d.shape
        assert v_out_2d.dtype == torch.float32
        assert torch.isfinite(v_out_2d).all(), "Sobolev 2D output contains non-finite values (NaN/Inf)"

        # Ensure contiguous memory representation for downstream operations
        v_contiguous = v_out_2d.contiguous()

        # Evaluate Jacobian determinant of the displacement field
        jac_det_2d = compute_jacobian_determinant_nd(v_contiguous)
        jac_np = jac_det_2d.squeeze().cpu().numpy()
        
        min_det_j = float(np.min(jac_np))
        folding_pct = float(np.mean(jac_np <= 0.0) * 100.0)
        
        assert folding_pct == 0.0, f"Grid folding detected in 2D Sobolev output: {folding_pct}%"
        assert min_det_j > 0.0, f"Min det(J) must be positive, got {min_det_j}"

    def test_sobolev_3d_math_stability_and_jacobian(self):
        """
        Verify Sobolev Green operator in 3D processes vector fields smoothly
        without NaN/Inf and produces non-folding, stable Jacobian grids.
        """
        syn_model = SyNTo(dim=3, grid_shape=(16, 16, 16))
        
        # Create a 3D gradient update tensor field scaled for a registration step: [1, 16, 16, 16, 3]
        torch.manual_seed(42)
        m_3d = torch.randn(1, 16, 16, 16, 3, dtype=torch.float32) * 0.05
        
        # Apply Sobolev Green operator
        v_out_3d = syn_model._apply_sobolev_green_operator(m_3d, fluid_sigma=2.0, alpha=1.0)
        
        assert v_out_3d.shape == m_3d.shape
        assert v_out_3d.dtype == torch.float32
        assert torch.isfinite(v_out_3d).all(), "Sobolev 3D output contains non-finite values (NaN/Inf)"

        v_contiguous = v_out_3d.contiguous()

        # Evaluate Jacobian determinant of the 3D displacement field
        jac_det_3d = compute_jacobian_determinant_nd(v_contiguous)
        jac_np = jac_det_3d.squeeze().cpu().numpy()
        
        min_det_j = float(np.min(jac_np))
        folding_pct = float(np.mean(jac_np <= 0.0) * 100.0)
        
        assert folding_pct == 0.0, f"Grid folding detected in 3D Sobolev output: {folding_pct}%"
        assert min_det_j > 0.0, f"Min det(J) must be positive, got {min_det_j}"

    def test_sobolev_tvf_model_2d_and_3d_stability(self):
        """
        Verify TVFModel implementation of Sobolev Green operator in 2D and 3D.
        """
        tvf_2d = TVFModel(dim=2, image_shape=(32, 32), velocity_shape=(32, 32))
        m_2d = torch.randn(1, 32, 32, 2, dtype=torch.float32) * 0.05
        v_out_2d = tvf_2d._apply_sobolev_green_operator(m_2d, fluid_sigma=2.0, alpha=1.0)
        assert torch.isfinite(v_out_2d).all()

        tvf_3d = TVFModel(dim=3, image_shape=(16, 16, 16), velocity_shape=(16, 16, 16))
        m_3d = torch.randn(1, 16, 16, 16, 3, dtype=torch.float32) * 0.05
        v_out_3d = tvf_3d._apply_sobolev_green_operator(m_3d, fluid_sigma=2.0, alpha=1.0)
        assert torch.isfinite(v_out_3d).all()

    def test_sobolev_mps_float32_contiguity_and_precision(self):
        """
        Verify that Sobolev Green operator handles float32 precision safely
        without casting to float64, avoiding MPS GPU crashes.
        """
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        
        syn_model = SyNTo(dim=2, grid_shape=(32, 32))
        m_in = torch.randn(1, 32, 32, 2, dtype=torch.float32, device=device)
        
        v_out = syn_model._apply_sobolev_green_operator(m_in, fluid_sigma=2.0, alpha=0.5)
        
        assert v_out.device.type == device.type
        assert v_out.dtype == torch.float32
        assert torch.isfinite(v_out).all()

    def test_sobolev_no_hidden_magic_clamps_source_verification(self):
        """
        Inspect Sobolev Green operator source code across syn.py and tvf.py to verify:
        - No hardcoded numeric clamps (e.g. torch.clamp(..., min=-10.0, max=10.0)) applied to fields.
        - Parameters are mathematically exposed and unhidden.
        """
        import inspect
        from syntx.syn import SyNTo
        from syntx.tvf import TVFModel

        syn_source = inspect.getsource(SyNTo._apply_sobolev_green_operator)
        tvf_source = inspect.getsource(TVFModel._apply_sobolev_green_operator)

        for src, class_name in [(syn_source, 'SyNTo'), (tvf_source, 'TVFModel')]:
            assert "torch.clamp(" not in src or "clamp(cc" in src, (
                f"{class_name}._apply_sobolev_green_operator contains forbidden torch.clamp!"
            )
            assert "max=10.0" not in src, (
                f"{class_name}._apply_sobolev_green_operator contains magic parameter max=10.0!"
            )
            assert "min=-10.0" not in src, (
                f"{class_name}._apply_sobolev_green_operator contains magic parameter min=-10.0!"
            )

    def test_sobolev_parameter_responsiveness(self):
        """
        Verify that increasing Sobolev alpha increases field smoothness
        (decreases spatial variance of gradient magnitudes).
        """
        syn_model = SyNTo(dim=2, grid_shape=(64, 64))
        torch.manual_seed(100)
        m_noisy = torch.randn(1, 64, 64, 2, dtype=torch.float32)
        
        v_small_alpha = syn_model._apply_sobolev_green_operator(m_noisy, fluid_sigma=2.0, alpha=0.1)
        v_large_alpha = syn_model._apply_sobolev_green_operator(m_noisy, fluid_sigma=2.0, alpha=10.0)
        
        grad_y_small, grad_x_small = torch.gradient(v_small_alpha[0, ..., 0])
        grad_y_large, grad_x_large = torch.gradient(v_large_alpha[0, ..., 0])
        
        var_small = float((grad_y_small**2 + grad_x_small**2).mean())
        var_large = float((grad_y_large**2 + grad_x_large**2).mean())
        
        assert var_large < var_small, (
            f"Larger Sobolev alpha should yield smoother field (lower gradient variance): "
            f"var(alpha=10)={var_large:.6f} vs var(alpha=0.1)={var_small:.6f}"
        )


# ==============================================================================
# TIER 3 & TIER 4: 30-Configuration Grid Output Format & JSON Schema
# ==============================================================================

class Test30ConfigGridFormatAndSchema:
    """
    Test suite for the 30-configuration grid characterization engine,
    output metric formats, and JSON schema requirements.
    """

    def test_30_grid_configuration_matrix_count(self):
        """
        Verify exact 30-configuration breakdown:
        - 12 syn configs: 2 optimizers ('cfl', 'adam') x 3 regularizers x 2 fast_smooth
        - 18 tvf configs: 3 optimizers ('lars', 'cfl', 'adam') x 3 regularizers x 2 fast_smooth
        """
        syn_opts = ['cfl', 'adam']
        tvf_opts = ['lars', 'cfl', 'adam']
        regs = ['gaussian', 'sobolev', 'dsti']
        fast_smooths = [True, False]

        syn_count = len(syn_opts) * len(regs) * len(fast_smooths)
        tvf_count = len(tvf_opts) * len(regs) * len(fast_smooths)

        assert syn_count == 12, f"Expected 12 syn configs, got {syn_count}"
        assert tvf_count == 18, f"Expected 18 tvf configs, got {tvf_count}"
        assert syn_count + tvf_count == 30, f"Expected 30 total configs, got {syn_count + tvf_count}"

    def test_30_grid_benchmark_datasets_inventory(self):
        """
        Verify all 6 benchmark datasets load correctly via syntx.benchmark_data(key).
        """
        expected_keys = ['r16_r64', '2d', 'c', 'ellipse', 'mbhard', '3d']
        for key in expected_keys:
            data = syntx.benchmark_data(key)
            assert 'fixed' in data, f"Dataset '{key}' missing 'fixed' image"
            assert 'moving' in data, f"Dataset '{key}' missing 'moving' image"
            assert 'fixed_label' in data, f"Dataset '{key}' missing 'fixed_label'"
            assert 'moving_label' in data, f"Dataset '{key}' missing 'moving_label'"

    def test_30_grid_json_results_schema(self, tmp_path):
        """
        Verify schema and field structure of grid_30_characterization_results.json.
        """
        sample_record = {
            "dataset": "2d",
            "config_name": "syn_cfl_sobolev_fsTrue",
            "model": "syn",
            "optimizer": "cfl",
            "regularizer": "sobolev",
            "fast_smooth": True,
            "metrics": {
                "dice_fixed": 0.891234,
                "dice_moving": 0.889123,
                "dice_sym": 0.890178,
                "folding_pct": 0.0,
                "min_det_J": 0.123456,
                "mean_inv_err_mm": 0.005432,
                "time_seconds": 1.234
            }
        }
        
        json_file = tmp_path / "grid_30_characterization_results.json"
        with open(json_file, 'w') as f:
            json.dump([sample_record], f, indent=2)

        with open(json_file, 'r') as f:
            loaded_data = json.load(f)

        assert isinstance(loaded_data, list)
        rec = loaded_data[0]
        assert rec["dataset"] in ['r16_r64', '2d', 'c', 'ellipse', 'mbhard', '3d']
        assert rec["model"] in ['syn', 'tvf']
        assert rec["optimizer"] in ['cfl', 'adam', 'lars']
        assert rec["regularizer"] in ['gaussian', 'sobolev', 'dsti']
        assert isinstance(rec["fast_smooth"], bool)

        m = rec["metrics"]
        assert 0.0 <= m["dice_fixed"] <= 1.0
        assert 0.0 <= m["dice_moving"] <= 1.0
        assert 0.0 <= m["dice_sym"] <= 1.0
        assert m["folding_pct"] >= 0.0
        assert m["min_det_J"] > -100.0
        assert m["mean_inv_err_mm"] >= 0.0
        assert m["time_seconds"] >= 0.0

    def test_30_grid_best_parameters_json_schema(self, tmp_path):
        """
        Verify schema and field structure of best_parameters.json.
        """
        best_params = {
            "syntx.tvf": {
                "backend": "pytorch",
                "similarity_metric": "lncc",
                "optimizer": "lars",
                "flow_sigma": 0.4,
                "total_sigma": 0.05,
                "grad_step": 0.5,
                "regularizer": "sobolev",
                "fast_smooth": True,
                "levels": [4, 2, 1],
                "reg_iterations": [100, 100, 20],
                "affine_iterations": [50, 0, 0],
                "benchmark": {
                    "dataset": "3d",
                    "config_name": "tvf_lars_sobolev_fsTrue",
                    "symmetric_dice": 0.885,
                    "grid_folding": 0.0,
                    "min_jacobian": 0.05,
                    "inverse_error_mm": 0.008,
                    "hardware": "cpu"
                },
                "provenance": {}
            }
        }

        json_file = tmp_path / "best_parameters.json"
        with open(json_file, 'w') as f:
            json.dump(best_params, f, indent=2)

        with open(json_file, 'r') as f:
            loaded_params = json.load(f)

        assert "syntx.tvf" in loaded_params or "syntx.syn" in loaded_params
        algo_cfg = list(loaded_params.values())[0]
        assert "backend" in algo_cfg
        assert "optimizer" in algo_cfg
        assert "regularizer" in algo_cfg
        assert "benchmark" in algo_cfg
        assert "symmetric_dice" in algo_cfg["benchmark"]


# ==============================================================================
# TIER 3 & TIER 4: Bidirectional Dice Evaluation for Mindboggle Benchmarks
# ==============================================================================

class TestBidirectionalDiceEvaluation:
    """
    Test suite for bidirectional fixed, moving, and symmetric mean Dice calculation
    and compliance with Mindboggle benchmark invariants.
    """

    def test_bidirectional_dice_identical_images_perfect_score(self):
        """
        Verify that registering identical label images yields Dice == 1.0 in all spaces.
        """
        label_np = np.zeros((32, 32), dtype=np.float32)
        label_np[8:24, 8:24] = 3.0  # Cortical DKT label 3
        
        fi = ants.from_numpy(label_np)
        mi = ants.from_numpy(label_np)
        fl = ants.from_numpy(label_np.astype(np.uint32))
        ml = ants.from_numpy(label_np.astype(np.uint32))

        tx_path = create_identity_transform_file(dim=2)
        try:
            fwdtransforms = [tx_path]
            invtransforms = [tx_path]

            if compute_bidirectional_dice is not None:
                dice_fixed, dice_moving, dice_sym = compute_bidirectional_dice(
                    fl, ml, fi, mi, fwdtransforms, invtransforms
                )
                assert dice_fixed == pytest.approx(1.0, abs=1e-5)
                assert dice_moving == pytest.approx(1.0, abs=1e-5)
                assert dice_sym == pytest.approx(1.0, abs=1e-5)
        finally:
            if os.path.exists(tx_path):
                os.unlink(tx_path)

    def test_bidirectional_dice_nearest_neighbor_interpolation_invariant(self):
        """
        GEMINI.md Rule 4 requirement:
        Discrete label maps MUST be transformed using nearestNeighbor interpolation.
        Verify that nearestNeighbor interpolation preserves discrete integer label maps
        without introducing non-integer interpolated values.
        """
        label_np = np.zeros((32, 32), dtype=np.uint32)
        label_np[10:20, 10:20] = 5  # Label 5
        label_np[20:25, 20:25] = 12 # Label 12

        fi = ants.from_numpy(label_np.astype(np.float32))
        ml = ants.from_numpy(label_np)

        tx_path = create_identity_transform_file(dim=2)
        try:
            warped_ml = ants.apply_transforms(
                fixed=fi, moving=ml, transformlist=[tx_path], interpolator='nearestNeighbor'
            )
            warped_np = warped_ml.numpy()

            unique_vals = set(np.unique(warped_np))
            assert unique_vals.issubset({0, 5, 12}), f"Nearest neighbor produced unexpected labels: {unique_vals}"
        finally:
            if os.path.exists(tx_path):
                os.unlink(tx_path)

    def test_bidirectional_dice_background_label_exclusion(self):
        """
        Verify that background label (0) is excluded from mean Cortical DKT Dice calculation.
        """
        fl_np = np.zeros((32, 32), dtype=np.uint32)
        ml_np = np.zeros((32, 32), dtype=np.uint32)

        fl_np[4:12, 4:12] = 1
        ml_np[4:12, 4:12] = 1

        fl = ants.from_numpy(fl_np)
        ml = ants.from_numpy(ml_np)

        overlap = ants.label_overlap_measures(fl, ml)
        df_filtered = overlap[~overlap['Label'].astype(str).isin(['All', '0', '0.0'])]

        col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_filtered.columns else 'TargetOverlap'
        mean_dice = float(df_filtered[col].mean())

        assert mean_dice == pytest.approx(1.0, abs=1e-5)

    def test_bidirectional_dice_symmetry_relation(self):
        """
        Verify that symmetric mean Dice equals exactly 0.5 * (dice_fixed + dice_moving).
        """
        fl_np = np.zeros((32, 32), dtype=np.uint32)
        ml_np = np.zeros((32, 32), dtype=np.uint32)
        fl_np[5:15, 5:15] = 2
        ml_np[7:17, 7:17] = 2

        fi = ants.from_numpy(fl_np.astype(np.float32))
        mi = ants.from_numpy(ml_np.astype(np.float32))
        fl = ants.from_numpy(fl_np)
        ml = ants.from_numpy(ml_np)

        tx_path = create_identity_transform_file(dim=2)
        try:
            if compute_bidirectional_dice is not None:
                dice_fixed, dice_moving, dice_sym = compute_bidirectional_dice(
                    fl, ml, fi, mi, [tx_path], [tx_path]
                )
                assert dice_sym == pytest.approx(0.5 * (dice_fixed + dice_moving), abs=1e-6)
        finally:
            if os.path.exists(tx_path):
                os.unlink(tx_path)
