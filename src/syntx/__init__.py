from .syn import registration, SyNTo
from .syn_jax import SyNTo as SyNToJax
from .transform import SyNToTransform
from .features import FeatureSpaceLoss, VGG19Extractor, DINOv2Extractor, ResNet10Extractor, SwinUNETRExtractor
from .image_compare import image_compare
from .generators import CrossProductGenerator

# Expose both syn and registration to satisfy user requirements
syn = registration

__version__ = "0.1.10"


__all__ = [
    "syn",
    "registration",
    "SyNTo",
    "SyNToJax",
    "SyNToTransform",
    "FeatureSpaceLoss",
    "VGG19Extractor",
    "DINOv2Extractor",
    "ResNet10Extractor",
    "SwinUNETRExtractor",
    "image_compare",
    "CrossProductGenerator",
]
