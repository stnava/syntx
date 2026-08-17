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
    render_standard_4panel
)
from .core import normalize_tensor, normalize_image
from .syn_jax import SyNTo as SyNToJax
from .transform import SyNToTransform
from .features import FeatureSpaceLoss, VGG19Extractor, DINOv2Extractor, ResNet10Extractor, SwinUNETRExtractor
from .image_compare import image_compare
from .generators import CrossProductGenerator, benchmark_data
from .tvf import TVFModel, tvf_registration
from .tvf_jax import TVFModelJAX
from .syngs import GeodesicShootingModel, syngs_registration
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
from .robust_affine import robust_affine
from . import benchmark
from .benchmark import run_benchmark_suite, high_level_benchmark_run
from .pyramid import build_image_pyramid
from .deformation_metrics import (
    compute_harmonic_energy,
    compute_bending_energy,
    compute_jacobian_metrics,
    compute_bidirectional_dice
)
from .benchmark.metrics import compute_pair_metrics

# Expose syn, registration, auto_reg, and tvf
syn = registration
tvf = tvf_registration
syngs = syngs_registration

__version__ = "3.0.22"



__all__ = [
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
    "syngs_registration",
    "syngs",
    "robust_affine",
    "plot_comparison",
    "plot_structural_comparison",
    "extract_2d_slice",
    "run_benchmark_suite",
    "high_level_benchmark_run",
    "compute_pair_metrics",
    "__version__",
]
