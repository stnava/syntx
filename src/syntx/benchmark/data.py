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
    verbose: bool = False
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
    if is_valid and verbose:
        print(f"[syntx.benchmark] Dataset Location: '{data_dir_resolved}'", flush=True)
        print(f"[syntx.benchmark] Pairs Configuration: '{os.path.abspath(pairs_csv)}'", flush=True)
    elif not is_valid and verbose:
        print(f"[syntx.benchmark] Incomplete Mindboggle dataset! Found {report['available_pairs']}/{report['total_pairs_in_csv']} pairs at '{data_dir_resolved}'.", file=sys.stderr)
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


def organize_mindboggle_data(
    source_path: str,
    target_dir: str,
    mode: str = "auto",
    pairs_csv: str = DEFAULT_PAIRS_CSV,
    verbose: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Discovers Mindboggle-101 T1 brain MRI and DKT31 cortical label volumes in `source_path`
    (which can be a directory of extracted folders, nested subdirectories, or archive files),
    and standardizes them into the exact directory hierarchy expected by `syntx.benchmark`.

    Parameters
    ----------
    source_path : str
        Path to raw downloads, unzipped directories, or directory containing Mindboggle archives.
    target_dir : str
        Target root directory to organize into (e.g. `~/data/mindboggle/volumes`).
    mode : str, default='auto'
        File transfer mode ('auto', 'link', 'copy', 'symlink'). 'auto' attempts hard links first
        for instant zero-copy organization, falling back to copy if cross-filesystem.
    pairs_csv : str
        Path to pairs.csv to validate against.
    verbose : bool
        If True, prints progress details.

    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        (is_valid, report_dict)
    """
    import tarfile
    import zipfile
    import shutil

    source_path = os.path.abspath(os.path.expanduser(str(source_path)))
    target_dir = os.path.abspath(os.path.expanduser(str(target_dir)))
    os.makedirs(target_dir, exist_ok=True)

    if verbose:
        print(f"[syntx.benchmark] Organizing Mindboggle dataset from '{source_path}' -> '{target_dir}'...", flush=True)

    # 1. Handle single archive or directory of archives
    extract_dirs = [source_path]
    if os.path.isfile(source_path):
        if source_path.endswith((".tar.gz", ".tgz", ".tar")):
            tmp_ext = os.path.join(target_dir, "_tmp_extracted")
            os.makedirs(tmp_ext, exist_ok=True)
            if verbose:
                print(f"[syntx.benchmark] Extracting tar archive '{source_path}'...", flush=True)
            with tarfile.open(source_path, "r:*") as tar:
                tar.extractall(tmp_ext)
            extract_dirs.append(tmp_ext)
        elif source_path.endswith(".zip"):
            tmp_ext = os.path.join(target_dir, "_tmp_extracted")
            os.makedirs(tmp_ext, exist_ok=True)
            if verbose:
                print(f"[syntx.benchmark] Extracting zip archive '{source_path}'...", flush=True)
            with zipfile.ZipFile(source_path, "r") as zf:
                zf.extractall(tmp_ext)
            extract_dirs.append(tmp_ext)

    # Also check if source_path is a directory containing archives
    if os.path.isdir(source_path):
        for fname in os.listdir(source_path):
            fpath = os.path.join(source_path, fname)
            if fname.endswith((".tar.gz", ".tgz", ".tar")):
                tmp_ext = os.path.join(target_dir, "_tmp_extracted", os.path.splitext(fname)[0])
                os.makedirs(tmp_ext, exist_ok=True)
                if verbose:
                    print(f"[syntx.benchmark] Extracting archive '{fname}'...", flush=True)
                with tarfile.open(fpath, "r:*") as tar:
                    tar.extractall(tmp_ext)
                extract_dirs.append(tmp_ext)
            elif fname.endswith(".zip"):
                tmp_ext = os.path.join(target_dir, "_tmp_extracted", os.path.splitext(fname)[0])
                os.makedirs(tmp_ext, exist_ok=True)
                if verbose:
                    print(f"[syntx.benchmark] Extracting archive '{fname}'...", flush=True)
                with zipfile.ZipFile(fpath, "r") as zf:
                    zf.extractall(tmp_ext)
                extract_dirs.append(tmp_ext)

    # 2. Known cohorts and subject matching
    cohort_names = ["OASIS-TRT-20", "NKI-RS-22", "NKI-TRT-20", "MMRR-21", "Extra-18"]
    
    # 3. Recursive search for T1 brain volumes and manual DKT31 label files
    found_brains = {}
    found_labels = {}

    for s_dir in extract_dirs:
        if not os.path.exists(s_dir):
            continue
        for root, _, files in os.walk(s_dir):
            for file in files:
                f_lower = file.lower()
                full_path = os.path.join(root, file)

                # Skip MNI normalized or atlas files
                if "mni" in f_lower or "+aseg" in f_lower or file.startswith("."):
                    continue

                # Identify subject and cohort from path
                rel_parts = full_path.replace("\\", "/").split("/")
                matched_subj = None
                matched_cohort = None

                for part in rel_parts:
                    for c_name in cohort_names:
                        if part.startswith(c_name):
                            matched_cohort = c_name
                            matched_subj = part
                            break
                    if matched_subj:
                        break

                if not matched_subj or not matched_cohort:
                    continue

                # Clean subject ID (e.g. OASIS-TRT-20-1)
                # Sometimes subfolder is "OASIS-TRT-20_volumes" -> look for sub-part
                for part in rel_parts:
                    for c_name in cohort_names:
                        if part.startswith(f"{c_name}-"):
                            matched_subj = part
                            matched_cohort = c_name
                            break

                # Brain volume detection
                if f_lower == "t1weighted_brain.nii.gz" or (
                    "t1" in f_lower and "brain" in f_lower and f_lower.endswith((".nii.gz", ".nii"))
                ):
                    found_brains[matched_subj] = (matched_cohort, full_path)

                # Label volume detection
                elif f_lower == "labels.dkt31.manual.nii.gz" or (
                    "dkt31" in f_lower and "manual" in f_lower and f_lower.endswith((".nii.gz", ".nii"))
                ):
                    found_labels[matched_subj] = (matched_cohort, full_path)

    # 4. Transfer files into target structure
    organized_subjects = 0
    for subj_id in set(found_brains.keys()).union(found_labels.keys()):
        if subj_id not in found_brains or subj_id not in found_labels:
            continue

        cohort, brain_src = found_brains[subj_id]
        _, label_src = found_labels[subj_id]

        subj_target_dir = os.path.join(target_dir, f"{cohort}_volumes", subj_id)
        os.makedirs(subj_target_dir, exist_ok=True)

        brain_dst = os.path.join(subj_target_dir, "t1weighted_brain.nii.gz")
        label_dst = os.path.join(subj_target_dir, "labels.DKT31.manual.nii.gz")

        for src, dst in [(brain_src, brain_dst), (label_src, label_dst)]:
            if os.path.exists(dst):
                continue
            
            transferred = False
            if mode in ("auto", "link"):
                try:
                    os.link(src, dst)
                    transferred = True
                except (OSError, NotImplementedError):
                    pass

            if not transferred and mode == "symlink":
                try:
                    os.symlink(src, dst)
                    transferred = True
                except (OSError, NotImplementedError):
                    pass

            if not transferred:
                shutil.copyfile(src, dst)

        organized_subjects += 1

    # Cleanup temporary extraction directory if created
    tmp_ext_dir = os.path.join(target_dir, "_tmp_extracted")
    if os.path.exists(tmp_ext_dir):
        shutil.rmtree(tmp_ext_dir, ignore_errors=True)

    if verbose:
        print(f"[syntx.benchmark] Organized {organized_subjects} subjects into '{target_dir}'.", flush=True)

    # 5. Validate resulting directory
    is_valid, report = check_mindboggle_data(pairs_csv=pairs_csv, data_dir=target_dir, verbose=verbose)
    report["organized_subjects"] = organized_subjects
    report["target_dir"] = target_dir

    if is_valid and verbose:
        print("\n================================================================================")
        print("                 MINDBOGGLE DATASET SUCCESSFULLY ORGANIZED!")
        print("================================================================================")
        print(f"Target Directory: {target_dir}")
        print(f"Set environment variable to use this dataset across all benchmarks:")
        print(f"    export SYNTX_DATA_DIR=\"{target_dir}\"")
        print("================================================================================\n")

    return is_valid, report
