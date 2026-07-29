from .visualization import plot_comparison, extract_2d_slice
from .syn import (
    registration,
    SyNTo,
    calculate_inverse_identity_error,
    auto_reg,
    normalize_tensor,
    plot_deformation_grid,
    plot_edge_overlay,
    render_standard_4panel,
    plot_structural_comparison
)
from .syn_jax import SyNTo as SyNToJax
from .transform import SyNToTransform
from .features import FeatureSpaceLoss, VGG19Extractor, DINOv2Extractor, ResNet10Extractor, SwinUNETRExtractor
from .image_compare import image_compare
from .generators import CrossProductGenerator
from .tvf import TVFModel
from .tvf_jax import TVFModelJAX
from .shooting import GeodesicShootingModel
from .shooting_jax import GeodesicShootingModelJAX
from .reporting import create_registration_report
from .io import read_registration, write_registration
from . import spatial
from .spatial import (
    disp_tensor_to_itk,
    disp_itk_to_tensor,
    image_to_tensor,
    tensor_to_image,
    jacobian_determinant,
    jacobian_determinant_image,
    deformation_stats,
    reverse_components,
    reverse_metadata,
    get_image_metadata,
)

# Expose syn, registration, and auto_reg
syn = registration

__version__ = "1.0.13"


__all__ = [
    "syn",
    "registration",
    "auto_reg",
    "normalize_tensor",
    "plot_comparison",
    "extract_2d_slice",
    "plot_deformation_grid",
    "plot_edge_overlay",
    "render_standard_4panel",
    "plot_structural_comparison",
    "create_registration_report",
    "read_registration",
    "write_registration",
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
    "GeodesicShootingModel",
    "GeodesicShootingModelJAX",
    "spatial",
    "disp_tensor_to_itk",
    "disp_itk_to_tensor",
    "image_to_tensor",
    "tensor_to_image",
    "jacobian_determinant",
    "jacobian_determinant_image",
    "deformation_stats",
    "reverse_components",
    "reverse_metadata",
    "get_image_metadata",
]


