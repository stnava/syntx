"""
syntx.benchmark.data — Mindboggle-101 Dataset Management and Pair Loading
========================================================================

Handles dataset path resolution, integrity checks, and pair loading for the
standardized 90-pair Mindboggle registration benchmark.
"""

import os
import sys
import pandas as pd
import ants
from typing import Dict, Any, Optional, Tuple

DEFAULT_PAIRS_CSV = "examples/pairs.csv"
DEFAULT_DATA_DIR_ENV = "SYNTX_DATA_DIR"
DEFAULT_DATA_DIR = "/Users/stnava/data/mindboggle/volumes"

MINDBOGGLE_SETUP_INSTRUCTIONS = """
================================================================================
                    MINDBOGGLE-101 DATASET SETUP GUIDE
================================================================================

The Mindboggle benchmark requires the 101 manually labeled T1-weighted brain MRI
volumes and DKT31 cortical label maps from the Mindboggle-101 project.

Expected Directory Hierarchy:
-----------------------------
$SYNTX_DATA_DIR/ (default: /Users/stnava/data/mindboggle/volumes)
  ├── OASIS-TRT-20_volumes/
  │   ├── OASIS-TRT-20-1/
  │   │   ├── t1weighted_brain.nii.gz
  │   │   └── labels.DKT31.manual.nii.gz
  │   └── ... (20 subjects)
  ├── NKI-RS-22_volumes/
  │   ├── NKI-RS-22-1/
  │   │   ├── t1weighted_brain.nii.gz
  │   │   └── labels.DKT31.manual.nii.gz
  │   └── ... (22 subjects)
  ├── NKI-TRT-20_volumes/
  │   ├── NKI-TRT-20-1/
  │   │   ├── t1weighted_brain.nii.gz
  │   │   └── labels.DKT31.manual.nii.gz
  │   └── ... (20 subjects)
  └── MMRR-21_volumes/
      ├── MMRR-21-1/
      │   ├── t1weighted_brain.nii.gz
      │   └── labels.DKT31.manual.nii.gz
      └── ... (21 subjects)

Pairs Configuration File:
-------------------------
The benchmark expects `examples/pairs.csv` defining the 90 evaluation pairs
(40 intra-subject and 50 inter-subject pairs).

How to Set the Data Directory:
------------------------------
Option A: Export environment variable in your shell profile:
    export SYNTX_DATA_DIR="/path/to/your/mindboggle/volumes"

Option B: Pass `data_dir` directly to benchmark functions:
    syntx.benchmark.evaluate_mindboggle_pair(pair_idx=0, data_dir="/path/to/volumes")

Download & Reference:
---------------------
Mindboggle-101 Dataset: https://mindboggle.info/data.html
Citation: Klein A, Tourville J. 101 labeled brain images and a consistent human
cortical labeling protocol. Front Neurosci. 2012;6:171.
================================================================================
"""


def resolve_data_dir(data_dir: Optional[str] = None) -> str:
    """
    Resolves the Mindboggle data directory from argument, environment, or default.
    Raises a descriptive FileNotFoundError with instructions if the directory is missing.
    """
    if data_dir is not None and str(data_dir).strip():
        resolved = os.path.abspath(os.path.expanduser(str(data_dir)))
    else:
        resolved = os.environ.get(DEFAULT_DATA_DIR_ENV, DEFAULT_DATA_DIR)
        resolved = os.path.abspath(os.path.expanduser(resolved))

    if not os.path.isdir(resolved):
        print(MINDBOGGLE_SETUP_INSTRUCTIONS, file=sys.stderr)
        raise FileNotFoundError(
            f"Mindboggle data directory not found at: '{resolved}'\n"
            f"Please set the {DEFAULT_DATA_DIR_ENV} environment variable or pass `data_dir`."
        )
    return resolved


def check_mindboggle_data(
    pairs_csv: str = DEFAULT_PAIRS_CSV,
    data_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verifies that the Mindboggle pairs CSV and required volume files exist.

    Parameters
    ----------
    pairs_csv : str
        Path to the pairs CSV file defining the 90 pairs.
    data_dir : str, optional
        Root directory containing the cohort subdirectories.
    verbose : bool
        If True, prints diagnostic setup instructions on missing data.

    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        (is_valid, report_dictionary)
    """
    report = {
        "pairs_csv_path": os.path.abspath(pairs_csv),
        "pairs_csv_exists": os.path.exists(pairs_csv),
        "data_dir": None,
        "data_dir_exists": False,
        "total_pairs_in_csv": 0,
        "available_pairs": 0,
        "missing_pairs": [],
        "missing_files": []
    }

    if not os.path.exists(pairs_csv):
        if verbose:
            print(f"[syntx.benchmark] ERROR: Pairs CSV not found at '{pairs_csv}'", file=sys.stderr)
            print(MINDBOGGLE_SETUP_INSTRUCTIONS, file=sys.stderr)
        return False, report

    try:
        data_dir_resolved = resolve_data_dir(data_dir)
        report["data_dir"] = data_dir_resolved
        report["data_dir_exists"] = True
    except FileNotFoundError:
        return False, report

    df = pd.read_csv(pairs_csv)
    report["total_pairs_in_csv"] = len(df)

    missing_pairs = []
    missing_files = []

    for idx, row in df.iterrows():
        c1, s1 = str(row["cohort1"]), str(row["subject1"])
        c2, s2 = str(row["cohort2"]), str(row["subject2"])

        p_fix = os.path.join(data_dir_resolved, f"{c1}_volumes", s1, "t1weighted_brain.nii.gz")
        p_flab = os.path.join(data_dir_resolved, f"{c1}_volumes", s1, "labels.DKT31.manual.nii.gz")
        p_mov = os.path.join(data_dir_resolved, f"{c2}_volumes", s2, "t1weighted_brain.nii.gz")
        p_mlab = os.path.join(data_dir_resolved, f"{c2}_volumes", s2, "labels.DKT31.manual.nii.gz")

        pair_missing = []
        for pth in [p_fix, p_flab, p_mov, p_mlab]:
            if not os.path.exists(pth):
                pair_missing.append(pth)
                missing_files.append(pth)

        if pair_missing:
            missing_pairs.append({"pair_idx": int(idx), "missing": pair_missing})
        else:
            report["available_pairs"] += 1

    report["missing_pairs"] = missing_pairs
    report["missing_files"] = list(set(missing_files))

    is_valid = (len(missing_pairs) == 0 and report["total_pairs_in_csv"] > 0)
    if not is_valid and verbose:
        print(f"[syntx.benchmark] Incomplete Mindboggle dataset! Found {report['available_pairs']}/{report['total_pairs_in_csv']} pairs.", file=sys.stderr)
        print(f"[syntx.benchmark] {len(report['missing_files'])} missing image/label files detected.", file=sys.stderr)
        print(MINDBOGGLE_SETUP_INSTRUCTIONS, file=sys.stderr)

    return is_valid, report


def load_mindboggle_pair(
    pair_idx: int,
    pairs_csv: str = DEFAULT_PAIRS_CSV,
    data_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Loads a single image pair and ground-truth segmentation pair from the CSV.

    Parameters
    ----------
    pair_idx : int
        Index of the pair in the CSV file (0 to 89).
    pairs_csv : str
        Path to the pairs CSV file.
    data_dir : str, optional
        Root directory containing the cohort subdirectories.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing ANTsImage objects for 'fixed', 'moving',
        'fixed_label', 'moving_label', and metadata.
    """
    if not os.path.exists(pairs_csv):
        print(MINDBOGGLE_SETUP_INSTRUCTIONS, file=sys.stderr)
        raise FileNotFoundError(f"Pairs CSV not found: '{pairs_csv}'")

    data_dir_resolved = resolve_data_dir(data_dir)
    df = pd.read_csv(pairs_csv)

    if pair_idx < 0 or pair_idx >= len(df):
        raise IndexError(
            f"pair_idx={pair_idx} out of range [0, {len(df) - 1}]. "
            f"CSV has {len(df)} pairs."
        )

    row = df.iloc[pair_idx]
    c1, s1 = str(row["cohort1"]), str(row["subject1"])
    c2, s2 = str(row["cohort2"]), str(row["subject2"])

    paths = {
        "fixed": os.path.join(data_dir_resolved, f"{c1}_volumes", s1, "t1weighted_brain.nii.gz"),
        "fixed_label": os.path.join(data_dir_resolved, f"{c1}_volumes", s1, "labels.DKT31.manual.nii.gz"),
        "moving": os.path.join(data_dir_resolved, f"{c2}_volumes", s2, "t1weighted_brain.nii.gz"),
        "moving_label": os.path.join(data_dir_resolved, f"{c2}_volumes", s2, "labels.DKT31.manual.nii.gz"),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            print(MINDBOGGLE_SETUP_INSTRUCTIONS, file=sys.stderr)
            raise FileNotFoundError(f"Missing Mindboggle {name} volume: '{path}'")

    return {
        "pair_idx": int(pair_idx),
        "fixed": ants.image_read(paths["fixed"]),
        "moving": ants.image_read(paths["moving"]),
        "fixed_label": ants.image_read(paths["fixed_label"]),
        "moving_label": ants.image_read(paths["moving_label"]),
        "fixed_id": s1,
        "moving_id": s2,
        "cohort1": c1,
        "cohort2": c2,
        "pair_type": str(row.get("type", "intra" if c1 == c2 else "inter")),
    }
