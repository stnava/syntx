from .syn import registration, SyNTo
from .syn_jax import SyNTo as SyNToJax
from .transform import SyNToTransform

# Expose both syn and registration to satisfy user requirements
syn = registration

__version__ = "0.1.2"


__all__ = [
    "syn",
    "registration",
    "SyNTo",
    "SyNToJax",
    "SyNToTransform",
]
