"""
syntx.data -- dataset helpers used for registration experiments.

- ``msd``: the Medical Segmentation Decathlon (MSD) task table (``MSDTask``,
  ``list_msd_tasks``, ``get_msd_task_info``), a download/extract helper
  (``msd.download_msd_task``, not re-exported here) and a small PyTorch ``MSDDataset``.
- ``surrogates``: intensity / connected-component masks (CT lung, CT abdominal soft tissue,
  brain) used as overlap targets when an MSD task's own labels are not shared between subjects.
"""

from .msd import MSDTask, MSDDataset, list_msd_tasks, get_msd_task_info
from .surrogates import (
    extract_ct_body_trunk,
    extract_ct_lung_parenchyma,
    extract_ct_abdominal_viscera,
    extract_brain_parenchyma,
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
    "extract_brain_parenchyma",
    "extract_surrogate_target",
]
