"""
syntx.data.msd -- Medical Segmentation Decathlon (MSD) task table, download and loader.

``MSD_TASKS`` maps 'Task01' ... 'Task10' to an ``MSDTask`` record (name, anatomy, modality,
channels, archive URL on the ``msd-for-monai`` S3 bucket). ``download_msd_task`` fetches and
unpacks one archive; ``MSDDataset`` reads the ``dataset.json`` of unpacked task folders and
returns resampled image tensors. The 10 tasks:
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
    """Static description of one MSD task.

    Attributes
    ----------
    task_id : str
        Short key, e.g. 'Task06'.
    name : str
        Folder / archive stem, e.g. 'Task06_Lung'.
    target_anatomy : str
        Coarse body region label ('BRAIN', 'HEART', 'ABDOMEN', 'PELVIS', 'THORAX').
    primary_modality : str
        'MRI' or 'CT'.
    num_channels : int
        Number of image channels (4-D images when > 1).
    channel_names : list of str
        Channel names in file order, e.g. ['FLAIR', 'T1w', 'T1gd', 'T2w'] for Task01.
    intensity_domain : str
        Informational tag: 'HOUNSFIELD' for CT, 'POSITIVE_FLOAT' for MRI. Not used by code here.
    archive_name : str
        File name of the .tar archive.
    download_url : str
        URL of the archive.
    """
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
    """Return the task keys of ``MSD_TASKS``: ['Task01', ..., 'Task10']."""
    return list(MSD_TASKS.keys())


def get_msd_task_info(task_key: str) -> MSDTask:
    """Look up an ``MSDTask`` by short key or full name (case-insensitive).

    Parameters
    ----------
    task_key : str
        'Task01' (underscores are dropped before comparing with the short key, so 'task_01'
        also works) or the full name, e.g. 'Task01_BrainTumour'.

    Returns
    -------
    MSDTask

    Raises
    ------
    KeyError
        If no task matches.
    """
    normalized_key = task_key.upper().replace("_", "")
    for k, v in MSD_TASKS.items():
        if k.upper() == normalized_key or v.name.upper() == task_key.upper():
            return v
    raise KeyError(f"Unknown MSD task '{task_key}'. Available: {list(MSD_TASKS.keys())}")


def _safe_extract(tar: tarfile.TarFile, target_dir: str) -> None:
    """Extract ``tar`` into ``target_dir`` refusing members that would land outside it
    (absolute paths, ``..``, links pointing out): the tarfile 'data' filter where available,
    an explicit check otherwise."""
    if hasattr(tarfile, "data_filter"):
        tar.extractall(path=target_dir, filter="data")
        return
    root = os.path.realpath(target_dir)
    for m in tar.getmembers():
        dest = os.path.realpath(os.path.join(root, m.name))
        if os.path.commonpath([root, dest]) != root or m.issym() or m.islnk():
            raise ValueError(f"refusing to extract archive member {m.name!r} (outside {target_dir} or a link)")
    tar.extractall(path=target_dir)


def download_msd_task(task_key: str, target_dir: str, progress_bar: bool = True) -> str:
    """Download (if needed) and unpack one MSD task archive.

    Does nothing and returns at once if ``<target_dir>/<task name>/dataset.json`` already
    exists. Otherwise downloads the .tar into ``target_dir`` unless it is already there, then
    extracts it into ``target_dir`` with path-traversal protection (``_safe_extract``). The
    .tar file is kept.

    Parameters
    ----------
    task_key : str
        Task identifier accepted by ``get_msd_task_info`` ('Task01' or 'Task01_BrainTumour').
    target_dir : str
        Directory to hold the archive and the unpacked task folder (created if missing).
    progress_bar : bool, default True
        Print the download progress (percent) while fetching.

    Returns
    -------
    str
        ``<target_dir>/<task name>``.

    Raises
    ------
    FileNotFoundError
        The archive did not contain ``<task name>/dataset.json``.
    ValueError
        An archive member would be extracted outside ``target_dir``.
    """
    task = get_msd_task_info(task_key)
    os.makedirs(target_dir, exist_ok=True)
    task_dir = os.path.join(target_dir, task.name)
    if os.path.exists(task_dir) and os.path.exists(os.path.join(task_dir, "dataset.json")):
        return task_dir

    tar_path = os.path.join(target_dir, task.archive_name)
    if not os.path.exists(tar_path):
        print(f"[syntx.data] Downloading {task.name} from {task.download_url}...")
        hook = None
        if progress_bar:
            last = [-1]

            def hook(blocks, block_size, total):
                if total > 0:
                    pct = min(100, int(100 * blocks * block_size / total))
                    if pct >= last[0] + 5:
                        last[0] = pct
                        print(f"\r[syntx.data] {pct:3d}%", end="", flush=True)
        urllib.request.urlretrieve(task.download_url, tar_path, reporthook=hook)
        if progress_bar:
            print()

    print(f"[syntx.data] Extracting {tar_path} into {target_dir}...")
    with tarfile.open(tar_path, "r") as tar:
        _safe_extract(tar, target_dir)
    if not os.path.exists(os.path.join(task_dir, "dataset.json")):
        raise FileNotFoundError(f"{tar_path} did not unpack to {task_dir}/dataset.json")
    return task_dir


class MSDDataset(torch.utils.data.Dataset):
    """PyTorch dataset of MSD images (and labels, when listed) resampled to a fixed voxel grid.

    For each folder in ``task_dirs`` that contains ``dataset.json``, every entry of the
    ``split`` list whose image file exists becomes one sample. Folders without
    ``dataset.json`` are skipped silently. Anatomy / modality come from ``MSD_TASKS`` when the
    folder's base name equals a task name or key, otherwise 'UNKNOWN'.

    Parameters
    ----------
    task_dirs : list of str
        Unpacked task folders, e.g. ['/data/Task06_Lung'].
    target_shape : tuple of 3 int, default (64, 64, 64)
        Output grid in voxels, ANTs (x, y, z) order.
    split : str, default 'training'
        Key of ``dataset.json`` to read ('training' or 'test'). Entries may be dicts with
        'image' (and optional 'label') or plain path strings.
    transform : callable, optional
        Applied to each sample dict returned by ``__getitem__`` (its return value is returned).

    Attributes
    ----------
    samples : list of dict
        Keys 'image_path', 'label_path' (None if absent), 'task_name', 'anatomy', 'modality'.
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
        """Number of samples found at construction."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Read sample ``idx`` from disk and resample it to ``target_shape``.

        4-D images (multi-channel tasks 01 / 05) keep their first channel, sliced before
        resampling. The image is resampled linearly, the label (when listed and present)
        nearest-neighbour, both with ``ants.resample_image(..., use_voxels=True)``.

        Returns
        -------
        dict (passed through ``transform`` when given)
            'image': float32 tensor (1, *target_shape) in ANTs (x, y, z) array order;
            'label': float32 tensor (1, *target_shape) or None; 'anatomy', 'modality',
            'task_name': str; 'path': image file path.
        """
        item = self.samples[idx]

        def _read3d(path):
            img = ants.image_read(path)
            if img.dimension == 4:
                img = ants.slice_image(img, axis=3, idx=0)       # primary channel
            return img

        img = _read3d(item["image_path"])
        arr = ants.resample_image(img, self.target_shape, use_voxels=True, interp_type=0).numpy()
        label = None
        if item["label_path"] is not None and os.path.exists(item["label_path"]):
            lab = ants.resample_image(_read3d(item["label_path"]), self.target_shape,
                                      use_voxels=True, interp_type=1).numpy()
            label = torch.from_numpy(lab.astype(np.float32)).unsqueeze(0)

        sample = {
            "image": torch.from_numpy(arr.astype(np.float32)).unsqueeze(0),
            "label": label,
            "anatomy": item["anatomy"],
            "modality": item["modality"],
            "task_name": item["task_name"],
            "path": item["image_path"]
        }
        return self.transform(sample) if self.transform is not None else sample
