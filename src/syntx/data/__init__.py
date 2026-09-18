"""
syntx.data — Medical Imaging Datasets & Ingestion Utilities
==========================================================

Provides automated dataset manifests, download helpers, and PyTorch dataset loaders:
- MSD (Medical Segmentation Decathlon)
"""

from .msd import MSDTask, MSDDataset, list_msd_tasks, get_msd_task_info
from .surrogates import (
    extract_ct_body_trunk,
    extract_ct_lung_parenchyma,
    extract_ct_abdominal_viscera,
    extract_surrogate_target,
)

__all__ = [
    "MSDTask",
    "MSDDataset",
    "list_msd_tasks",
    "get_msd_task_info",
    "extract_ct_body_trunk",
    "extract_ct_lung_parenchyma",
    "extract_ct_abdominal_viscera",
    "extract_surrogate_target",
]
