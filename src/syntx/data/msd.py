"""
syntx.data.msd — Medical Segmentation Decathlon (MSD) Ingestion & Data Management
================================================================================

Handles task registry, dataset metadata extraction, download utilities, and PyTorch
Dataset interfaces for the 10 Medical Segmentation Decathlon tasks:
- Task01_BrainTumour (MRI 4-ch, Brain)
- Task02_Heart (MRI, Heart)
- Task03_Liver (CT, Abdomen)
- Task04_Hippocampus (MRI T1, Brain)
- Task05_Prostate (MRI T2/ADC, Pelvis)
- Task06_Lung (CT, Thorax)
- Task07_Pancreas (CT, Abdomen)
- Task08_HepaticVessel (CT, Abdomen)
- Task09_Spleen (CT, Abdomen)
- Task10_Colon (CT, Abdomen)
"""

import os
import json
import glob
import urllib.request
import tarfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import ants


@dataclass
class MSDTask:
    task_id: str
    name: str
    target_anatomy: str
    primary_modality: str
    num_channels: int
    channel_names: List[str]
    intensity_domain: str
    archive_name: str
    download_url: str


MSD_TASKS: Dict[str, MSDTask] = {
    "Task01": MSDTask(
        task_id="Task01",
        name="Task01_BrainTumour",
        target_anatomy="BRAIN",
        primary_modality="MRI",
        num_channels=4,
        channel_names=["FLAIR", "T1w", "T1gd", "T2w"],
        intensity_domain="POSITIVE_FLOAT",
        archive_name="Task01_BrainTumour.tar",
        download_url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
    ),
    "Task02": MSDTask(
        task_id="Task02",
        name="Task02_Heart",
        target_anatomy="HEART",
        primary_modality="MRI",
        num_channels=1,
        channel_names=["MRI"],
        intensity_domain="POSITIVE_FLOAT",
        archive_name="Task02_Heart.tar",
        download_url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar"
    ),
    "Task03": MSDTask(
        task_id="Task03",
        name="Task03_Liver",
        target_anatomy="ABDOMEN",
        primary_modality="CT",
        num_channels=1,
        channel_names=["CT"],
        intensity_domain="HOUNSFIELD",
        archive_name="Task03_Liver.tar",
        download_url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar"
    ),
    "Task04": MSDTask(
        task_id="Task04",
        name="Task04_Hippocampus",
        target_anatomy="BRAIN",
        primary_modality="MRI",
        num_channels=1,
        channel_names=["T1w"],
        intensity_domain="POSITIVE_FLOAT",
        archive_name="Task04_Hippocampus.tar",
        download_url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"
    ),
    "Task05": MSDTask(
        task_id="Task05",
        name="Task05_Prostate",
        target_anatomy="PELVIS",
        primary_modality="MRI",
        num_channels=2,
        channel_names=["T2w", "ADC"],
        intensity_domain="POSITIVE_FLOAT",
        archive_name="Task05_Prostate.tar",
        download_url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar"
    ),
    "Task06": MSDTask(
        task_id="Task06",
        name="Task06_Lung",
        target_anatomy="THORAX",
        primary_modality="CT",
        num_channels=1,
        channel_names=["CT"],
        intensity_domain="HOUNSFIELD",
        archive_name="Task06_Lung.tar",
        download_url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar"
    ),
    "Task07": MSDTask(
        task_id="Task07",
        name="Task07_Pancreas",
        target_anatomy="ABDOMEN",
        primary_modality="CT",
        num_channels=1,
        channel_names=["CT"],
        intensity_domain="HOUNSFIELD",
        archive_name="Task07_Pancreas.tar",
        download_url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar"
    ),
    "Task08": MSDTask(
        task_id="Task08",
        name="Task08_HepaticVessel",
        target_anatomy="ABDOMEN",
        primary_modality="CT",
        num_channels=1,
        channel_names=["CT"],
        intensity_domain="HOUNSFIELD",
        archive_name="Task08_HepaticVessel.tar",
        download_url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar"
    ),
    "Task09": MSDTask(
        task_id="Task09",
        name="Task09_Spleen",
        target_anatomy="ABDOMEN",
        primary_modality="CT",
        num_channels=1,
        channel_names=["CT"],
        intensity_domain="HOUNSFIELD",
        archive_name="Task09_Spleen.tar",
        download_url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    ),
    "Task10": MSDTask(
        task_id="Task10",
        name="Task10_Colon",
        target_anatomy="ABDOMEN",
        primary_modality="CT",
        num_channels=1,
        channel_names=["CT"],
        intensity_domain="HOUNSFIELD",
        archive_name="Task10_Colon.tar",
        download_url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar"
    )
}


def list_msd_tasks() -> List[str]:
    """Returns list of available MSD task identifiers ('Task01' through 'Task10')."""
    return list(MSD_TASKS.keys())


def get_msd_task_info(task_key: str) -> MSDTask:
    """Retrieves metadata info for a specific MSD task."""
    normalized_key = task_key.upper().replace("_", "")
    for k, v in MSD_TASKS.items():
        if k.upper() == normalized_key or v.name.upper() == task_key.upper():
            return v
    raise KeyError(f"Unknown MSD task '{task_key}'. Available: {list(MSD_TASKS.keys())}")


def download_msd_task(task_key: str, target_dir: str, progress_bar: bool = True) -> str:
    """
    Downloads and extracts an MSD dataset task archive.

    Parameters:
    -----------
    task_key : str
        Task identifier (e.g. 'Task01' or 'Task01_BrainTumour').
    target_dir : str
        Destination directory to store and unpack the dataset.
    progress_bar : bool, default=True
        Whether to print download progress.

    Returns:
    --------
    str
        Path to unpacked task directory.
    """
    task = get_msd_task_info(task_key)
    os.makedirs(target_dir, exist_ok=True)
    task_dir = os.path.join(target_dir, task.name)
    if os.path.exists(task_dir) and os.path.exists(os.path.join(task_dir, "dataset.json")):
        return task_dir

    tar_path = os.path.join(target_dir, task.archive_name)
    if not os.path.exists(tar_path):
        print(f"[syntx.data] Downloading {task.name} from {task.download_url}...")
        urllib.request.urlretrieve(task.download_url, tar_path)

    print(f"[syntx.data] Extracting {tar_path} into {target_dir}...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=target_dir)

    return task_dir


class MSDDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset interface for training 3D diagnosis classification and policy networks on MSD data.
    """
    def __init__(
        self,
        task_dirs: List[str],
        target_shape: Tuple[int, int, int] = (64, 64, 64),
        split: str = "training",
        transform: Optional[Any] = None
    ):
        self.samples = []
        self.target_shape = target_shape
        self.transform = transform

        for td in task_dirs:
            meta_path = os.path.join(td, "dataset.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r") as f:
                meta = json.load(f)

            task_name = os.path.basename(td.rstrip("/"))
            task_info = None
            for t in MSD_TASKS.values():
                if t.name == task_name or t.task_id == task_name:
                    task_info = t
                    break

            anatomy = task_info.target_anatomy if task_info else "UNKNOWN"
            modality = task_info.primary_modality if task_info else "UNKNOWN"

            entries = meta.get(split, [])
            for entry in entries:
                img_rel = entry["image"] if isinstance(entry, dict) else entry
                lbl_rel = entry.get("label", None) if isinstance(entry, dict) else None

                img_path = os.path.normpath(os.path.join(td, img_rel))
                lbl_path = os.path.normpath(os.path.join(td, lbl_rel)) if lbl_rel else None

                if os.path.exists(img_path):
                    self.samples.append({
                        "image_path": img_path,
                        "label_path": lbl_path,
                        "task_name": task_name,
                        "anatomy": anatomy,
                        "modality": modality
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        img = ants.image_read(item["image_path"])

        # Resample to canonical target_shape for fast batch training
        resampled = ants.resample_image(img, self.target_shape, use_voxels=True, interp_type=1)
        arr = resampled.numpy().astype(np.float32)

        # Handle 4D volumes (e.g. multi-channel BrainTumour or Prostate)
        if arr.ndim == 4:
            arr = arr[..., 0]  # Take primary channel

        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, D, H, W]

        return {
            "image": tensor,
            "anatomy": item["anatomy"],
            "modality": item["modality"],
            "task_name": item["task_name"],
            "path": item["image_path"]
        }
