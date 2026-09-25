"""
syntx — Symmetric Normalization & Diffeomorphic Registration Toolkit
=====================================================================

`syntx` provides fast, differentiable, high-accuracy 2D/3D medical image registration
in PyTorch and JAX, featuring Symmetric Normalization (SyN), Time-Varying Velocity Fields
(TVF), and Geodesic Shooting (SyNGS).

Core Entry Points
-----------------
syntx.syn / syntx.registration
    Symmetric Normalization (SyNTo) with optional deep feature loss.
syntx.tvf / syntx.tvf_registration
    Time-Varying Velocity Fields (TVF) with multi-resolution ODE trajectory integration.
syntx.syngs / syntx.syngs_registration
    Geodesic Shooting (SyNGS) using EPDiff Euler integration.

Quick Start
-----------
>>> import syntx
>>> reg = syntx.syn(fixed=fi, moving=mi)
>>> warped = reg['warpedmovout']
>>> transforms = reg['fwdtransforms']
"""

import os

# Force MPS allocator to be unconstrained for large 3D operations
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from .syn import (
    registration,
    SyNTo,
    calculate_inverse_identity_error,
    auto_reg,
    plot_deformation_grid,
    plot_edge_overlay,
    render_standard_4panel,
)
from .core import normalize_tensor, normalize_image
from .syn_jax import SyNTo as SyNToJax
from .transform import SyNToTransform
from .features import (
    FeatureSpaceLoss,
    VGG19Extractor,
    DINOv2Extractor,
    ResNet10Extractor,
    SwinUNETRExtractor,
)
from .image_compare import image_compare
from .generators import CrossProductGenerator, benchmark_data
from .tvf import TVFModel, tvf_registration
from .tvf_jax import TVFModelJAX
from .syngs import (
    GeodesicShootingModel,
    syngs_registration,
    integrate_momentum,
    shoot_geodesic,
    momentum_to_deformation,
)
from .syngs_jax import (
    GeodesicShootingModelJAX,
    integrate_momentum_jax,
    shoot_geodesic_jax,
    momentum_to_deformation_jax,
)
from . import spatial
from .spatial import deformation_gradient
from . import contract
from . import tabulate
from . import viz
from .viz import (
    render_input_pair_figure,
    render_standard_4panel,
    plot_deformation_grid,
    plot_edge_overlay,
    create_registration_report,
    create_population_benchmark_report,
    extract_2d_slice,
)
from .reporting import build_engine_provenance
from .robust_affine import robust_affine, robust_center_of_mass, compute_center_of_mass
from . import benchmark
from .benchmark import (
    run_benchmark_suite,
    high_level_benchmark_run,
    evaluate_affine_benchmark,
)
from .pyramid import build_image_pyramid
from .deformation_metrics import (
    compute_harmonic_energy,
    compute_bending_energy,
    compute_jacobian_metrics,
    compute_bidirectional_dice,
)
from .benchmark.metrics import compute_pair_metrics
from . import scattered
from . import landmarks
from .landmarks import (
    detect_blobs_log,
    detect_blobs_dog,
    detect_sift2d,
    detect_sift3d,
    compute_mind,
    extract_mind_at_points,
    match_landmarks,
    ransac_filter,
    compute_tre,
    sinkhorn_matching,
    weighted_procrustes,
    sampled_optimal_transport_affine,
    score_rotation_candidates_sampled,
)
from .scattered import (
    project_scattered_to_grid,
    ScatteredProjector,
    ProjectionConfig,
    differentiable_grid_projection,
    evaluate_field_at_scattered,
    warp_scattered_coordinates,
    ScatteredWarper,
    pullback_grid_to_scattered,
    pushforward_scattered_to_grid,
    transport_scattered_to_scattered,
    ScatteredRegistrationConfig,
    ScatteredRegistrationResult,
    SyNScattered,
    syn_scattered,
)
from .greedy import (
    greedy,
    greedy_registration,
    GreedyRegistrationModel,
)

from .surface import (
    compute_surface_classes,
    generate_surface_channels,
    extract_sulcal_probability_map,
)
from .diagnose import (
    ImageDiagnosis,
    PairDiagnosis,
    diagnose_image,
    diagnose_pair,
)
from .policy import (
    RegistrationPolicy,
    synthesize_policy,
    auto_policy_for_images,
)
from .classifier import (
    DiagnosticClassifier3D,
    predict_diagnosis_deep,
)
from . import data
from .template import build_template
from .image_utils import reflect_image
from .motion import (
    motion_correction,
    MotionParameters,
    TransformCollection,
    MotionCorrectionResult,
    calculate_framewise_displacement,
    calculate_dvars,
)
from .motion_batched import (
    batched_rigid_register_pass,
    batched_group_bias_register_pass,
)


# Expose syn, registration, auto_reg, and tvf
syn = registration
tvf = tvf_registration
syngs = syngs_registration
affine_benchmark = evaluate_affine_benchmark

__version__ = "5.4.29"


__all__ = [
    "spatial",
    "contract",
    "tabulate",
    "viz",
    "syn",
    "registration",
    "auto_reg",
    "normalize_tensor",
    "normalize_image",
    "plot_deformation_grid",
    "plot_edge_overlay",
    "render_standard_4panel",
    "render_input_pair_figure",
    "create_registration_report",
    "create_population_benchmark_report",
    "build_engine_provenance",
    "SyNTo",
    "SyNToJax",
    "SyNToTransform",
    "calculate_inverse_identity_error",
    "FeatureSpaceLoss",
    "VGG19Extractor",
    "DINOv2Extractor",
    "ResNet10Extractor",
    "SwinUNETRExtractor",
    "image_compare",
    "CrossProductGenerator",
    "benchmark_data",
    "TVFModel",
    "TVFModelJAX",
    "tvf_registration",
    "tvf",
    "GeodesicShootingModel",
    "GeodesicShootingModelJAX",
    "syngs_registration",
    "syngs",
    "integrate_momentum",
    "shoot_geodesic",
    "momentum_to_deformation",
    "integrate_momentum_jax",
    "shoot_geodesic_jax",
    "momentum_to_deformation_jax",
    "robust_affine",
    "robust_center_of_mass",
    "compute_center_of_mass",
    "plot_comparison",
    "plot_structural_comparison",
    "extract_2d_slice",
    "run_benchmark_suite",
    "high_level_benchmark_run",
    "compute_pair_metrics",
    "scattered",
    "landmarks",
    "detect_blobs_log",
    "detect_blobs_dog",
    "detect_sift2d",
    "detect_sift3d",
    "compute_mind",
    "extract_mind_at_points",
    "match_landmarks",
    "ransac_filter",
    "compute_tre",
    "project_scattered_to_grid",
    "ScatteredProjector",
    "ProjectionConfig",
    "differentiable_grid_projection",
    "evaluate_field_at_scattered",
    "warp_scattered_coordinates",
    "ScatteredWarper",
    "pullback_grid_to_scattered",
    "pushforward_scattered_to_grid",
    "transport_scattered_to_scattered",
    "ScatteredRegistrationConfig",
    "ScatteredRegistrationResult",
    "SyNScattered",
    "syn_scattered",
    "greedy",
    "greedy_registration",
    "GreedyRegistrationModel",
    "ImageDiagnosis",
    "PairDiagnosis",
    "diagnose_image",
    "diagnose_pair",
    "RegistrationPolicy",
    "synthesize_policy",
    "auto_policy_for_images",
    "data",
    "build_template",
    "reflect_image",
    "motion_correction",
    "MotionParameters",
    "TransformCollection",
    "MotionCorrectionResult",
    "calculate_framewise_displacement",
    "calculate_dvars",
    "batched_rigid_register_pass",
    "batched_group_bias_register_pass",
    "deformation_gradient",
    "__version__",
]
