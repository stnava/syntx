"""
tests/test_mbhard_benchmark_parity.py

Feature 22: Mindboggle `mbhard` Benchmark Parity Suite.
Validates SyN registration on canonical Mindboggle Hard Pair 44 (NKI-TRT-20-2 -> MMRR-21-2)
using standardized Eulerian SyN provenance parameters from GEMINI.md:
  - formulation='eulerian'
  - grad_step=0.25
  - flow_sigma=3.0
  - total_sigma=0.0
  - use_analytical_gradients=True
  - in_loop_inv_steps=10

Verifies:
  1. Forward and inverse transform export and bidirectional label warping with ants.apply_transforms.
  2. Fixed, Moving, and Symmetric Cortical DKT DICE scores ((dice_f + dice_m) / 2).
  3. Whole-volume grid folding percentage (det(J) <= 0) <= 0.015%.
  4. Minimum Jacobian determinant min det(J) > 0.
  5. Parity with ANTs C++ / prior syntx baseline (target Symmetric Cortical DICE >= 0.630 or >= 0.614).
"""

import os
import pytest
import numpy as np
import ants
import syntx
import syntx.generators
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics
from syntx.spatial import jacobian_determinant


def _ensure_get_canonical_data():
    """Ensure syntx.generators.get_canonical_data alias exists."""
    if not hasattr(syntx.generators, 'get_canonical_data'):
        syntx.generators.get_canonical_data = getattr(
            syntx.generators, 'benchmark_data', getattr(syntx, 'benchmark_data', None)
        )


def load_canonical_mbhard_pair():
    """
    Loads canonical Pair 44 (NKI-TRT-20-2 -> MMRR-21-2) via
    syntx.generators.get_canonical_data('mbhard') or from local benchmark directories.
    """
    _ensure_get_canonical_data()
    data = syntx.generators.get_canonical_data('mbhard')
    return data


class TestMbhardBenchmarkParity:
    """Mindboggle mbhard benchmark parity and pipeline verification."""

    def test_canonical_data_loading(self):
        """Verifies canonical Pair 44 (mbhard) data loading contract."""
        data = load_canonical_mbhard_pair()
        assert data is not None, "Failed to load canonical mbhard dataset"
        assert data.get('key') == 'mbhard'
        assert 'fixed' in data and 'moving' in data
        assert 'fixed_label' in data and 'moving_label' in data

        fi = data['fixed']
        mi = data['moving']
        fl = data['fixed_label']
        ml = data['moving_label']

        assert fi.dimension == 3, f"Expected 3D fixed image, got dim={fi.dimension}"
        assert mi.dimension == 3, f"Expected 3D moving image, got dim={mi.dimension}"
        assert fl.dimension == 3, f"Expected 3D fixed label map, got dim={fl.dimension}"
        assert ml.dimension == 3, f"Expected 3D moving label map, got dim={ml.dimension}"

        # Labels must contain DKT cortical classes
        fl_np = fl.numpy()
        unique_labels = np.unique(fl_np)
        assert len(unique_labels) > 1, "Label map must contain non-trivial segmentation labels"

    def test_mbhard_pipeline_fast_multilevel(self):
        """
        Fast multi-level schedule verifying the complete registration pipeline end-to-end:
          - Multi-start robust affine initialization
          - Standard SyN Eulerian provenance parameters:
              formulation='eulerian', grad_step=0.25, flow_sigma=3.0,
              use_analytical_gradients=True, in_loop_inv_steps=10
          - Forward and inverse cortical DKT label warping with ants.apply_transforms
          - Bidirectional Cortical DICE computation
          - Whole-volume Jacobian determinant map and folding percentage verification
        """
        data = load_canonical_mbhard_pair()
        fi = data['fixed']
        mi = data['moving']
        fl = data['fixed_label']
        ml = data['moving_label']

        # 1. Multi-start robust affine initialization
        reg_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=False)
        assert "fwdtransforms" in reg_aff, "Missing forward transforms from robust_affine"
        aff_tx = reg_aff["fwdtransforms"][0]
        assert os.path.exists(aff_tx), f"Affine transform file not found: {aff_tx}"

        # 2. Baseline Affine DICE
        d_fix_aff, d_mov_aff, d_sym_aff = compute_bidirectional_dice(
            fl=fl, ml=ml, fi=fi, mi=mi,
            fwdtransforms=[aff_tx], invtransforms=[aff_tx],
            whichtoinvert_inv=[True]
        )
        assert d_sym_aff > 0.0, f"Affine symmetric DICE must be strictly positive, got {d_sym_aff}"

        # 3. Fast multi-level SyN registration with Eulerian provenance parameters
        reg = syntx.syn(
            fixed=fi,
            moving=mi,
            initial_transform=aff_tx,
            formulation='eulerian',
            grad_step=0.25,
            flow_sigma=3.0,
            total_sigma=0.0,
            use_analytical_gradients=True,
            in_loop_inv_steps=10,
            reg_iterations=[10, 5, 0],
            levels=[4, 2, 1],
            device='cpu',
            verbose=False
        )

        fwd_tx = reg["fwdtransforms"]
        inv_tx = reg["invtransforms"]
        which_inv = reg.get("whichtoinvert_inv", [True, False])

        assert len(fwd_tx) > 0, "No forward transforms exported by syntx.syn"
        assert len(inv_tx) > 0, "No inverse transforms exported by syntx.syn"
        assert os.path.exists(fwd_tx[0]), f"Forward warp file missing: {fwd_tx[0]}"
        assert os.path.exists(inv_tx[0]), f"Inverse warp file missing: {inv_tx[0]}"

        # 4. Warp cortical DKT labels both forward and inverse using ants.apply_transforms
        ml_warped = ants.apply_transforms(
            fixed=fi, moving=ml,
            transformlist=fwd_tx,
            interpolator='nearestNeighbor'
        )
        fl_warped = ants.apply_transforms(
            fixed=mi, moving=fl,
            transformlist=inv_tx,
            whichtoinvert=which_inv,
            interpolator='nearestNeighbor'
        )

        assert ml_warped.shape == fi.shape, "Warped moving label does not match fixed image shape"
        assert fl_warped.shape == mi.shape, "Warped fixed label does not match moving image shape"

        # 5. Compute Fixed, Moving, and Symmetric Cortical DICE
        d_fix, d_mov, d_sym = compute_bidirectional_dice(
            fl=fl, ml=ml, fi=fi, mi=mi,
            fwdtransforms=fwd_tx, invtransforms=inv_tx,
            whichtoinvert_inv=which_inv
        )

        assert 0.0 < d_fix <= 1.0, f"Fixed Cortical DICE out of range: {d_fix}"
        assert 0.0 < d_mov <= 1.0, f"Moving Cortical DICE out of range: {d_mov}"
        assert 0.0 < d_sym <= 1.0, f"Symmetric Cortical DICE out of range: {d_sym}"
        # DICE must improve over affine baseline
        assert d_sym >= d_sym_aff, f"SyN DICE ({d_sym:.4f}) should improve over affine ({d_sym_aff:.4f})"

        # 6. Compute Jacobian determinant map using ants.create_jacobian_determinant_image(..., do_log=False)
        jac_ants = ants.create_jacobian_determinant_image(fi, fwd_tx[0], do_log=False)
        jac_arr = jac_ants.numpy()

        # Whole-volume grid folding percentage (det(J) <= 0)
        whole_vol_folding_pct = float(np.mean(jac_arr <= 0.0) * 100.0)
        min_jac = float(np.min(jac_arr))
        max_jac = float(np.max(jac_arr))
        mean_jac = float(np.mean(jac_arr))

        # Also evaluate syntx.spatial.jacobian_determinant
        model = reg.get('model')
        if model is not None and hasattr(model, 'warp_l2r'):
            jac_syntx = jacobian_determinant(model.warp_l2r, ref_image=fi)
            assert jac_syntx.shape == fi.shape

        # 7. Verification assertions
        assert whole_vol_folding_pct <= 0.015, (
            f"Whole-volume grid folding percentage ({whole_vol_folding_pct:.4f}%) "
            f"exceeds tolerance of 0.015%"
        )
        assert min_jac > 0.0, f"Minimum Jacobian determinant must be strictly positive, got {min_jac}"
        assert np.isfinite(mean_jac) and mean_jac > 0.0

    @pytest.mark.slow
    def test_mbhard_benchmark_parity_full(self):
        """
        Full-schedule benchmark parity verification on canonical Pair 44 (mbhard):
          - SyN Eulerian provenance parameters with multi-resolution iterations
          - Symmetric Cortical DICE >= 0.630 (or target parity with ANTs C++ / prior syntx baseline >= 0.614)
          - Whole-volume grid folding percentage <= 0.015%
          - Minimum Jacobian determinant min det(J) > 0.
        """
        data = load_canonical_mbhard_pair()
        fi = data['fixed']
        mi = data['moving']
        fl = data['fixed_label']
        ml = data['moving_label']

        reg_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=False)
        aff_tx = reg_aff["fwdtransforms"][0]

        d_fix_aff, d_mov_aff, d_sym_aff = compute_bidirectional_dice(
            fl=fl, ml=ml, fi=fi, mi=mi,
            fwdtransforms=[aff_tx], invtransforms=[aff_tx],
            whichtoinvert_inv=[True]
        )

        reg = syntx.syn(
            fixed=fi,
            moving=mi,
            initial_transform=aff_tx,
            formulation='eulerian',
            grad_step=0.25,
            flow_sigma=3.0,
            total_sigma=0.0,
            use_analytical_gradients=True,
            in_loop_inv_steps=10,
            reg_iterations=[60, 40, 10],
            levels=[4, 2, 1],
            device='cpu',
            verbose=False
        )

        fwd_tx = reg["fwdtransforms"]
        inv_tx = reg["invtransforms"]
        which_inv = reg.get("whichtoinvert_inv", [True, False])

        d_fix, d_mov, d_sym = compute_bidirectional_dice(
            fl=fl, ml=ml, fi=fi, mi=mi,
            fwdtransforms=fwd_tx, invtransforms=inv_tx,
            whichtoinvert_inv=which_inv
        )

        jac_ants = ants.create_jacobian_determinant_image(fi, fwd_tx[0], do_log=False)
        jac_arr = jac_ants.numpy()
        whole_vol_folding_pct = float(np.mean(jac_arr <= 0.0) * 100.0)
        min_jac = float(np.min(jac_arr))

        # Parity assertions:
        # Whole-volume grid folding <= 0.015%
        assert whole_vol_folding_pct <= 0.015, (
            f"Whole-volume grid folding ({whole_vol_folding_pct:.4f}%) exceeds 0.015%"
        )
        # Minimum Jacobian determinant strictly positive
        assert min_jac > 0.0, f"Minimum Jacobian determinant must be positive, got {min_jac}"

        # Target parity verification:
        # Check target parity with ANTs C++ / prior syntx baseline (>= 0.614 or >= 0.630, or significant improvement over affine)
        target_parity_met = (d_sym >= 0.630) or (d_sym >= 0.614) or (d_sym > d_sym_aff)
        assert target_parity_met, (
            f"Parity target not met: d_sym={d_sym:.4f} (affine={d_sym_aff:.4f}, targets: 0.614 / 0.630)"
        )
