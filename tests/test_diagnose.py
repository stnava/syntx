"""
Unit tests for syntx.diagnose, syntx.policy, and syntx.data.msd.
"""

import pytest
import numpy as np
import ants
import torch

import syntx
from syntx.diagnose import diagnose_image, diagnose_pair, ImageDiagnosis, PairDiagnosis
from syntx.policy import synthesize_policy, auto_policy_for_images, RegistrationPolicy
from syntx.data.msd import list_msd_tasks, get_msd_task_info, MSD_TASKS


def test_diagnose_mri_brain_2d():
    fi = ants.image_read(ants.get_data("r16"))
    diag = diagnose_image(fi)

    assert diag.is_mri()
    assert diag.is_brain()
    assert diag.intensity_domain == "POSITIVE_FLOAT"
    assert diag.confidence >= 0.80
    assert diag.dimension == 2


def test_diagnose_pair_mri_intra_brain():
    fi = ants.image_read(ants.get_data("r16"))
    mi = ants.image_read(ants.get_data("r64"))
    pair = diagnose_pair(fi, mi)

    assert pair.relationship == "MONO_MODAL_INTRA"
    assert pair.is_same_anatomy is True
    assert pair.is_same_modality is True

    policy = synthesize_policy(pair)
    assert policy.transform_type == "SyN"
    assert policy.regularizer == "sobolev"
    assert policy.similarity_metric == "cc2"
    assert policy.guided == "sulcal"


def test_diagnose_ct_thorax():
    # Synthetic Thorax CT volume with lung air cavity (-950 HU) and chest wall soft tissue (+40 HU)
    arr = np.full((32, 32, 32), -950.0, dtype=np.float32)
    arr[8:24, 8:24, :] = 40.0  # Soft tissue chest wall
    arr[12:20, 12:20, :] = -900.0  # Lung cavity
    arr[0:4, :, :] = -1000.0  # Ambient air
    arr[-2:, :, :] = 450.0  # Rib bones

    diag = diagnose_image(arr)
    assert diag.is_ct()
    assert diag.is_thorax()
    assert diag.intensity_domain == "HOUNSFIELD"

    pair = PairDiagnosis(fixed=diag, moving=diag, relationship="MONO_MODAL_INTRA", is_same_anatomy=True, is_same_modality=True)
    policy = synthesize_policy(pair)

    assert policy.transform_type == "TVF"
    assert policy.regularizer == "dsti1"
    assert policy.ct_window == (-1000.0, 400.0)


def test_diagnose_ct_abdomen():
    # Synthetic Abdomen CT volume with soft tissue (+50 HU) and mesenteric fat (-80 HU)
    arr = np.full((32, 32, 32), -1000.0, dtype=np.float32)
    arr[6:26, 6:26, 6:26] = 45.0  # Liver/abdominal organs
    arr[4:6, :, :] = -80.0  # Subcutaneous fat
    arr[-2:, :, :] = 500.0  # Spine bone

    diag = diagnose_image(arr)
    assert diag.is_ct()
    assert diag.is_abdomen()

    pair = PairDiagnosis(fixed=diag, moving=diag, relationship="MONO_MODAL_INTRA", is_same_anatomy=True, is_same_modality=True)
    policy = synthesize_policy(pair)

    assert policy.transform_type == "SyN"
    assert policy.regularizer == "sobolev"
    assert policy.ct_window == (-150.0, 250.0)


def test_diagnose_cross_modality_mri_ct():
    mri = ants.image_read(ants.get_data("r16"))
    ct_arr = np.random.uniform(-1000, 500, (64, 64)).astype(np.float32)

    pair = diagnose_pair(mri, ct_arr)
    assert pair.relationship == "CROSS_MODAL"

    policy = synthesize_policy(pair)
    assert policy.similarity_metric == "mattes_mi"
    assert "Cross-Modality" in policy.explanation


def test_auto_policy_convenience_helper():
    fi = ants.image_read(ants.get_data("r16"))
    mi = ants.image_read(ants.get_data("r64"))
    policy = auto_policy_for_images(fi, mi)

    assert isinstance(policy, RegistrationPolicy)
    d = policy.to_dict()
    assert "type_of_transform" in d
    assert "similarity_metric" in d
    assert "regularizer" in d


def test_msd_task_registry():
    tasks = list_msd_tasks()
    assert len(tasks) == 10
    assert "Task01" in tasks
    assert "Task06" in tasks
    assert "Task10" in tasks

    brain_task = get_msd_task_info("Task01")
    assert brain_task.target_anatomy == "BRAIN"
    assert brain_task.primary_modality == "MRI"
    assert brain_task.num_channels == 4

    lung_task = get_msd_task_info("Task06_Lung")
    assert lung_task.target_anatomy == "THORAX"
    assert lung_task.primary_modality == "CT"
    assert lung_task.intensity_domain == "HOUNSFIELD"
