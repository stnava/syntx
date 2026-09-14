"""
syntx.data — Medical Imaging Datasets & Ingestion Utilities
==========================================================

Provides automated dataset manifests, download helpers, and PyTorch dataset loaders:
- MSD (Medical Segmentation Decathlon)
"""

from .msd import MSDTask, MSDDataset, list_msd_tasks, get_msd_task_info
