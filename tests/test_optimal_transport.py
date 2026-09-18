"""
tests/test_optimal_transport.py — Unit tests for Continuous Sampled Optimal Transport
====================================================================================

Tests Sinkhorn log-space matching, weighted Procrustes rigid recovery,
percentage-based spatial sampling, and rotation candidate scoring.
"""

import os
import tempfile
import numpy as np
import pytest
import torch
import ants

from syntx.landmarks.optimal_transport import (
    sinkhorn_matching,
    weighted_procrustes,
    sampled_optimal_transport_affine,
    score_rotation_candidates_sampled,
)
from syntx.landmarks.spatial import vox_to_physical


def test_sinkhorn_matching_synthetic():
    """Verify Sinkhorn matching produces valid doubly-stochastic / dustbin probabilities."""
    torch.manual_seed(42)
    N, M, D = 100, 120, 16
    # Source features
    feat_src = torch.randn(N, D)
    # First 80 points of dst correspond to src with slight noise
    feat_dst = torch.randn(M, D)
    feat_dst[:80] = feat_src[:80] + 0.05 * torch.randn(80, D)

    P_match, weights = sinkhorn_matching(
        feat_src=feat_src,
        feat_dst=feat_dst,
        epsilon=0.05,
        dustbin_cost=1.0,
        n_iters=50
    )

    assert P_match.shape == (N, M)
    assert weights.shape == (N,)
    assert not torch.isnan(P_match).any()
    assert not torch.isinf(P_match).any()
    assert (P_match >= 0.0).all()
    # Row sums should not exceed ~1 (due to dustbin column absorbing mass)
    row_sums = P_match.sum(dim=1)
    assert (row_sums <= 1.10).all()

    # The matched points should have highest probability on the diagonal
    diag_matches = P_match[:80, :80].argmax(dim=1)
    correct = (diag_matches == torch.arange(80)).sum().item()
    assert correct >= 75  # >93% accuracy on synthetic correspondences


def test_weighted_procrustes_exact_rigid_recovery():
    """Verify weighted Procrustes recovers known 3D rotation and translation exactly."""
    np.random.seed(42)
    N = 200
    pts_src = np.random.randn(N, 3).astype(np.float32) * 50.0

    # Define true rotation (40 deg around axis [1, 2, 2])
    axis = np.array([1.0, 2.0, 2.0], dtype=np.float32)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(40.0)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=np.float32)
    R_true = np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    t_true = np.array([12.5, -8.3, 24.1], dtype=np.float32)

    pts_dst = (pts_src @ R_true.T) + t_true

    src_t = torch.tensor(pts_src)
    dst_t = torch.tensor(pts_dst)
    weights = torch.ones(N)

    R_rec, t_rec, scale_rec, aff_4x4 = weighted_procrustes(src_t, dst_t, weights, allow_scaling=False)

    R_np = R_rec.numpy()
    t_np = t_rec.numpy()

    assert np.allclose(R_np, R_true, atol=1e-5)
    assert np.allclose(t_np, t_true, atol=1e-4)
    assert abs(scale_rec - 1.0) < 1e-5
    assert np.isclose(np.linalg.det(R_np), 1.0, atol=1e-5)


def test_sampling_percentage_parameterization():
    """Verify that sample count scales strictly as a percentage of foreground domain."""
    np.random.seed(42)
    # Create 3D image with known foreground fraction
    arr = np.zeros((40, 40, 40), dtype=np.float32)
    arr[10:30, 10:30, 10:30] = 1.0  # 20^3 = 8,000 foreground voxels
    img = ants.from_numpy(arr)

    # 1% of 8,000 is 80, clamped to min_samples=200
    res_1pct = sampled_optimal_transport_affine(
        fixed=img, moving=img,
        sampling_percentage=0.01,
        min_samples=200,
        max_samples=5000,
        return_dict=True
    )
    assert res_1pct["n_samples_fixed"] == 200

    # 10% of 8,000 is 800
    res_10pct = sampled_optimal_transport_affine(
        fixed=img, moving=img,
        sampling_percentage=0.10,
        min_samples=200,
        max_samples=5000,
        return_dict=True
    )
    assert res_10pct["n_samples_fixed"] == 800


def test_score_rotation_candidates_sampled():
    """Verify rotation scoring ranks identity candidate highest on self-matching."""
    np.random.seed(42)
    arr = np.zeros((32, 32, 32), dtype=np.float32)
    arr[8:24, 8:24, 8:24] = np.random.rand(16, 16, 16).astype(np.float32)
    img = ants.from_numpy(arr)

    # Candidates: Identity, 30 deg roll, 60 deg roll
    R_ident = np.eye(3, dtype=np.float32)
    theta = np.deg2rad(30.0)
    R_30 = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)
    theta2 = np.deg2rad(60.0)
    R_60 = np.array([[np.cos(theta2), 0, np.sin(theta2)], [0, 1, 0], [-np.sin(theta2), 0, np.cos(theta2)]], dtype=np.float32)

    candidates = [R_60, R_ident, R_30]
    scores = score_rotation_candidates_sampled(
        fixed=img, moving=img,
        candidate_rotations=candidates,
        sampling_percentage=0.05,
        min_samples=100,
        feature_type="mind"
    )

    # The winner must be identity (candidate index 1)
    assert scores[0]["index"] == 1
    assert scores[0]["score"] > scores[1]["score"]
