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

from .syn import (
    registration,
    SyNTo,
    calculate_inverse_identity_error,
    auto_reg,
    normalize_tensor,
    plot_deformation_grid,
    plot_edge_overlay,
    render_standard_4panel
)
from .syn_jax import SyNTo as SyNToJax
from .transform import SyNToTransform
from .features import FeatureSpaceLoss, VGG19Extractor, DINOv2Extractor, ResNet10Extractor, SwinUNETRExtractor
from .image_compare import image_compare
from .generators import CrossProductGenerator
from .tvf import TVFModel, tvf_registration
from .tvf_jax import TVFModelJAX
from .syngs import GeodesicShootingModel, syngs_registration
from .reporting import create_registration_report, render_input_pair_figure

# Expose syn, registration, auto_reg, and tvf
syn = registration
tvf = tvf_registration
syngs = syngs_registration

__version__ = "1.1.7"


__all__ = [
    "syn",
    "registration",
    "auto_reg",
    "normalize_tensor",
    "plot_deformation_grid",
    "plot_edge_overlay",
    "render_standard_4panel",
    "render_input_pair_figure",
    "create_registration_report",
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
    "TVFModel",
    "TVFModelJAX",
    "tvf_registration",
    "tvf",
    "GeodesicShootingModel",
    "syngs_registration",
    "syngs",
    "__version__",
]

__version__ = "1.1.7"



