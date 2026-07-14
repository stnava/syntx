from .syn import registration, SyNTo
from .syn_jax import SyNTo as SyNToJax
from .transform import SyNToTransform
from .features import FeatureSpaceLoss, VGG19Extractor, DINOv2Extractor, ResNet10Extractor, SwinUNETRExtractor

# Expose both syn and registration to satisfy user requirements
syn = registration

__version__ = "0.1.5"


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
]
