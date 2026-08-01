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
from .reporting import create_registration_report

# Expose syn, registration, auto_reg, and tvf
syn = registration
tvf = tvf_registration
syngs = syngs_registration

__version__ = "1.1.2"


__all__ = [
    "syn",
    "registration",
    "auto_reg",
    "normalize_tensor",
    "plot_deformation_grid",
    "plot_edge_overlay",
    "render_standard_4panel",
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
]


