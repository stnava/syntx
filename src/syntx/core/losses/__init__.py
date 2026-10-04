"""
Canonical high-performance image similarity loss package for syntx.

Centralizes all similarity metrics across the library:
- Cross-Correlation: local_ncc_loss_nd, BoxLNCCLoss, box_lncc_loss_nd, box_cc2_loss_nd,
  AnalyticalLNCC, ANTsPseudoLNCC, box_mean_nd.
- Mutual Information: CanonicalMattesMIFunction, mattes_mi_loss_nd, mattes_mi_loss_core,
  parzen_weights, mattes_mi_from_weights, mattes_sample_indices, get_4tap_splines.
- Pointwise & Pairwise: mse_loss_nd, l2_loss_nd, mae_loss_nd.
- Structural & Labels: soft_dice_loss_nd, compute_soft_distance_transform,
  _soft_signed_distance, compute_image_distance_transform, distance_transform_loss.
- Deep Features: FeatureSpaceLoss.
- Unified Dispatcher: get_similarity_loss, parse_similarity_metric, SimilarityLossConfig.
"""

from __future__ import annotations

from .cross_correlation import (
    AnalyticalLNCC,
    ANTsPseudoLNCC,
    BoxLNCCLoss,
    _box_pool,
    _DeterministicBoxMean,
    _window_counts,
    box_cc2_loss_nd,
    box_lncc_loss_nd,
    box_mean_nd,
    local_ncc_loss_nd,
)
from .deep_features import (
    FeatureSpaceLoss,
)
from .mutual_information import (
    CanonicalMattesMIFunction,
    _default_parzen_weights,
    _parzen_joint_histogram,
    b_spline_3,
    b_spline_3_deriv,
    get_4tap_splines,
    mattes_mi_from_weights,
    mattes_mi_loss_core,
    mattes_mi_loss_nd,
    mattes_sample_indices,
    parzen_weights,
)
from .pointwise import (
    l2_loss_nd,
    mae_loss_nd,
    mse_loss_nd,
)
from .registry import (
    SimilarityLossConfig,
    SimilarityLossWrapper,
    get_similarity_loss,
    parse_similarity_metric,
)
from .structural import (
    _soft_signed_distance,
    compute_image_distance_transform,
    compute_soft_distance_transform,
    distance_transform_loss,
    soft_dice_loss_nd,
)

__all__ = [
    "ANTsPseudoLNCC",
    "AnalyticalLNCC",
    "BoxLNCCLoss",
    "CanonicalMattesMIFunction",
    # Deep Features
    "FeatureSpaceLoss",
    # Registry & Dispatcher
    "SimilarityLossConfig",
    "SimilarityLossWrapper",
    "_DeterministicBoxMean",
    # Cross Correlation
    "_box_pool",
    "_default_parzen_weights",
    "_parzen_joint_histogram",
    "_soft_signed_distance",
    "_window_counts",
    # Mutual Information
    "b_spline_3",
    "b_spline_3_deriv",
    "box_cc2_loss_nd",
    "box_lncc_loss_nd",
    "box_mean_nd",
    "compute_image_distance_transform",
    "compute_soft_distance_transform",
    "distance_transform_loss",
    "get_4tap_splines",
    "get_similarity_loss",
    "l2_loss_nd",
    "local_ncc_loss_nd",
    "mae_loss_nd",
    "mattes_mi_from_weights",
    "mattes_mi_loss_core",
    "mattes_mi_loss_nd",
    "mattes_sample_indices",
    # Pointwise
    "mse_loss_nd",
    "parse_similarity_metric",
    "parzen_weights",
    # Structural
    "soft_dice_loss_nd",
]
