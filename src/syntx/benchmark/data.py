"""
syntx.benchmark.data — Mindboggle-101 data directory, pair loading, N4 cache
============================================================================

Layout expected under the data directory: ``<cohort>_volumes/<subject>/t1weighted_brain.nii.gz``
and ``.../labels.DKT31.manual.nii.gz`` (see ``MINDBOGGLE_SETUP_INSTRUCTIONS``). A pair is a row
of the pairs CSV (columns ``cohort1, subject1, cohort2, subject2`` and optionally ``type``);
subject 1 is the fixed image, subject 2 the moving one.

The data directory is resolved as: explicit argument > environment variable
``SYNTX_DATA_DIR`` > ``DEFAULT_DATA_DIR``. N4-corrected brains are cached under
``<data_dir>/.n4_cache/<cohort>_volumes/<subject>/t1weighted_brain_n4.nii.gz``.
"""

import os
import sys
import pandas as pd
import ants
from typing import Dict, Any, Optional, Tuple

DEFAULT_PAIRS_CSV = "examples/pairs.csv"
DEFAULT_DATA_DIR_ENV = "SYNTX_DATA_DIR"
DEFAULT_DATA_DIR = os.path.expanduser("~/data/mindboggle/volumes")

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
    """Absolute Mindboggle data directory: ``data_dir`` if given and non-blank, else
    ``$SYNTX_DATA_DIR``, else ``DEFAULT_DATA_DIR`` (``~`` expanded).

    Raises
    ------
    FileNotFoundError
        If the directory does not exist (``MINDBOGGLE_SETUP_INSTRUCTIONS`` is printed to
        stderr first).
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
    """Check that the pairs CSV exists and that every pair's four files (two brains, two
    DKT31 label maps) exist. Files are only tested for existence, not read.

    Parameters
    ----------
    pairs_csv : str, default ``DEFAULT_PAIRS_CSV`` ('examples/pairs.csv', relative to the
        working directory)
        Pairs CSV.
    data_dir : str, optional
        Data directory; resolved with ``resolve_data_dir``.
    verbose : bool, default False
        Print the data location on success, or a summary plus the setup instructions on
        failure (to stderr). A missing data directory always prints the instructions (from
        ``resolve_data_dir``).

    Returns
    -------
    (is_valid, report) : (bool, dict)
        ``is_valid`` is True when the CSV has at least one row and no pair is missing a file.
        ``report`` keys: 'pairs_csv_path', 'pairs_csv_exists', 'data_dir' (None if not
        found), 'data_dir_exists', 'total_pairs_in_csv', 'available_pairs', 'missing_pairs'
        (list of ``{'pair_idx', 'missing': [paths]}``), 'missing_files' (unique paths). If the
        CSV or the directory is missing, the function returns early with the counts at 0.
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


def get_n4_cached_subject_volume(
    cohort: str,
    subject: str,
    raw_brain_path: str,
    data_dir: str,
    use_n4: bool = True,
    device: Optional[str] = None,
    verbose: bool = False
) -> ants.ANTsImage:
    """Return a subject's brain volume, N4-corrected via a disk cache.

    With ``use_n4=False``, the raw file is read. Otherwise the cached
    ``<data_dir>/.n4_cache/<cohort>_volumes/<subject>/t1weighted_brain_n4.nii.gz`` is read if
    it exists; if not, ``antstorch.n4_bias_field_correction`` is run on the image (mask =
    intensity > 0.01; shrink_factor 4; 4 levels x 50 iterations; tol 1e-7), the result is
    written to the cache (side effect) and returned with the raw image's header.

    Parameters
    ----------
    cohort, subject : str
        Cohort (e.g. 'MMRR-21') and subject id (e.g. 'MMRR-21-2'); used for the cache path.
    raw_brain_path : str
        Path of the raw ``t1weighted_brain.nii.gz``.
    data_dir : str
        Resolved data directory (cache root).
    use_n4 : bool, default True
        Apply / use the N4 cache.
    device : str, optional
        Torch device for the N4 computation; None means 'cpu'.
    verbose : bool, default False
        Print progress and the fallback warning.

    Returns
    -------
    ANTsImage
        The corrected (or cached, or raw) volume.

    Raises
    ------
    RuntimeError
        If N4 fails for any reason (including antstorch not being installed); nothing is
        cached (the raw volume was returned silently, so a "use_n4" record held raw data).
    """
    if not use_n4:
        return ants.image_read(raw_brain_path)

    cache_dir = os.path.join(data_dir, ".n4_cache", f"{cohort}_volumes", subject)
    cache_file = os.path.join(cache_dir, "t1weighted_brain_n4.nii.gz")

    if os.path.exists(cache_file):
        return ants.image_read(cache_file)

    # Not in cache: compute N4 with antstorch
    raw_img = ants.image_read(raw_brain_path)
    try:
        import antstorch
        mask = ants.threshold_image(raw_img, 0.01, float("inf"))
        if verbose:
            print(f"[syntx.benchmark] Computing antstorch N4 correction for {subject}...", flush=True)
        # ANTsImage interface (antstorch >= 2026-09-24; the tensor call it replaced failed, and
        # the failure was swallowed, so uncached subjects were silently left uncorrected)
        corrected_img = antstorch.n4_bias_field_correction(
            raw_img,
            mask=mask,
            shrink_factor=4,
            convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
            device=device or "cpu",
        )
        corrected_img = ants.copy_image_info(raw_img, corrected_img)
        os.makedirs(cache_dir, exist_ok=True)
        ants.image_write(corrected_img, cache_file)
        return corrected_img
    except Exception as e:
        raise RuntimeError(f"N4 correction failed for {subject} ({e}); pass use_n4=False to use "
                           "the raw volume") from e


def precompute_mindboggle_n4(
    pairs_csv: str = DEFAULT_PAIRS_CSV,
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Fill the N4 cache for every distinct subject named in ``pairs_csv`` (both columns).

    Subjects already cached are skipped; the others go through
    ``get_n4_cached_subject_volume``. A subject whose N4 fails is counted under 'failed'
    (with its error) and nothing is cached for it.

    Parameters
    ----------
    pairs_csv : str, default ``DEFAULT_PAIRS_CSV``
        Pairs CSV whose subjects are processed.
    data_dir : str, optional
        Data directory (``resolve_data_dir``; raises FileNotFoundError if missing).
    device : str, optional
        Torch device for N4.
    verbose : bool, default True
        Print one line per subject and a summary.

    Returns
    -------
    dict
        'total_subjects', 'computed', 'already_cached', 'failed' ({subject: error}),
        'total_time_seconds', 'cache_dir'.
    """
    import time
    data_dir_resolved = resolve_data_dir(data_dir)
    df = pd.read_csv(pairs_csv)

    subjects = set()
    for _, row in df.iterrows():
        subjects.add((str(row["cohort1"]), str(row["subject1"])))
        subjects.add((str(row["cohort2"]), str(row["subject2"])))

    subjects_sorted = sorted(list(subjects))
    total = len(subjects_sorted)

    if verbose:
        print("=" * 80)
        print(f"      PRECOMPUTING ANTSTORCH N4 BIAS CORRECTION ({total} SUBJECTS)")
        print("=" * 80)
        print(f"Data Directory: {data_dir_resolved}")
        print(f"Cache Location: {os.path.join(data_dir_resolved, '.n4_cache')}\n", flush=True)

    t0_all = time.time()
    computed_count = 0
    cached_count = 0
    failed = {}

    for idx, (cohort, subj) in enumerate(subjects_sorted, start=1):
        raw_path = os.path.join(data_dir_resolved, f"{cohort}_volumes", subj, "t1weighted_brain.nii.gz")
        cache_path = os.path.join(data_dir_resolved, ".n4_cache", f"{cohort}_volumes", subj, "t1weighted_brain_n4.nii.gz")

        if os.path.exists(cache_path):
            cached_count += 1
            if verbose:
                print(f"[{idx:3d}/{total}] {cohort}/{subj:<18} [CACHED]", flush=True)
        else:
            t0 = time.time()
            try:
                get_n4_cached_subject_volume(
                    cohort=cohort,
                    subject=subj,
                    raw_brain_path=raw_path,
                    data_dir=data_dir_resolved,
                    use_n4=True,
                    device=device,
                    verbose=False
                )
            except RuntimeError as e:
                failed[subj] = str(e)
                print(f"[{idx:3d}/{total}] {cohort}/{subj:<18} [FAILED] {e}", file=sys.stderr, flush=True)
                continue
            elapsed = time.time() - t0
            computed_count += 1
            if verbose:
                print(f"[{idx:3d}/{total}] {cohort}/{subj:<18} [COMPUTED in {elapsed:5.1f}s]", flush=True)

    total_time = time.time() - t0_all
    if verbose:
        print("\n" + "=" * 80)
        print(f"N4 PRECOMPUTATION COMPLETE: {computed_count} computed, {cached_count} already cached, "
              f"{len(failed)} failed in {total_time:.1f}s")
        print("=" * 80 + "\n", flush=True)

    return {
        "total_subjects": total,
        "computed": computed_count,
        "already_cached": cached_count,
        "failed": failed,
        "total_time_seconds": total_time,
        "cache_dir": os.path.join(data_dir_resolved, ".n4_cache")
    }


def load_mindboggle_pair(
    pair_idx: int,
    pairs_csv: str = DEFAULT_PAIRS_CSV,
    data_dir: Optional[str] = None,
    use_n4: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """Load row ``pair_idx`` of the pairs CSV: fixed (subject 1) and moving (subject 2)
    brains and their DKT31 manual label maps.

    Parameters
    ----------
    pair_idx : int
        Row index in the CSV (0-based; 0..89 for the 90-row ``examples/pairs.csv``).
    pairs_csv : str, default ``DEFAULT_PAIRS_CSV``
        Pairs CSV.
    data_dir : str, optional
        Data directory (``resolve_data_dir``).
    use_n4 : bool, default True
        Brains via ``get_n4_cached_subject_volume`` (computed and cached on first use, on
        CPU; raw volume if N4 fails). False reads the raw brains.
    verbose : bool, default False
        Passed to the N4 loader.

    Returns
    -------
    dict
        'pair_idx', 'fixed', 'moving', 'fixed_label', 'moving_label' (ANTsImage),
        'fixed_id', 'moving_id' (subject ids), 'cohort1', 'cohort2', 'pair_type' (the CSV
        'type' column, else 'intra' if both cohorts are equal, else 'inter'), 'use_n4'.

    Raises
    ------
    FileNotFoundError
        Missing CSV, data directory or any of the four files (setup instructions printed).
    IndexError
        ``pair_idx`` outside the CSV's rows.
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

    fixed_img = get_n4_cached_subject_volume(c1, s1, paths["fixed"], data_dir_resolved, use_n4=use_n4, verbose=verbose)
    moving_img = get_n4_cached_subject_volume(c2, s2, paths["moving"], data_dir_resolved, use_n4=use_n4, verbose=verbose)

    return {
        "pair_idx": int(pair_idx),
        "fixed": fixed_img,
        "moving": moving_img,
        "fixed_label": ants.image_read(paths["fixed_label"]),
        "moving_label": ants.image_read(paths["moving_label"]),
        "fixed_id": s1,
        "moving_id": s2,
        "cohort1": c1,
        "cohort2": c2,
        "pair_type": str(row.get("type", "intra" if c1 == c2 else "inter")),
        "use_n4": use_n4,
    }



def organize_mindboggle_data(
    source_path: str,
    target_dir: str,
    mode: str = "auto",
    pairs_csv: str = DEFAULT_PAIRS_CSV,
    verbose: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """Find Mindboggle brains and DKT31 manual labels under ``source_path`` and place them in
    the layout this package expects under ``target_dir``, then validate with
    ``check_mindboggle_data``.

    ``source_path`` may be a directory (searched recursively), a ``.tar`` / ``.tar.gz`` /
    ``.tgz`` / ``.zip`` archive, or a directory containing such archives (top level only);
    archives are extracted into ``<target_dir>/_tmp_extracted``, which is deleted at the end.
    A file is assigned to the first path component starting with a known cohort name
    ('OASIS-TRT-20', 'NKI-RS-22', 'NKI-TRT-20', 'MMRR-21', 'Extra-18'), preferring a
    component of the form '<cohort>-...'. Brains: 't1weighted_brain.nii.gz' or any .nii /
    .nii.gz name containing 't1' and 'brain'; labels: 'labels.DKT31.manual.nii.gz' or a name
    containing 'dkt31' and 'manual' (case-insensitive). Names containing 'mni' or '+aseg' and
    hidden files are skipped. Only subjects with both files are placed; existing target
    files are not overwritten.

    Parameters
    ----------
    source_path : str
        Directory or archive with the raw data.
    target_dir : str
        Destination data directory (created).
    mode : {'auto', 'link', 'symlink', 'copy'}, default 'auto'
        'auto' and 'link': hard link, falling back to a copy when linking fails; 'symlink':
        symbolic link, falling back to a copy; 'copy': copy. Any other value raises
        ValueError. Files extracted from an archive are always moved into place (the
        extraction directory is deleted, so links to it would dangle). Archives are
        extracted with path-traversal protection.
    pairs_csv : str, default ``DEFAULT_PAIRS_CSV``
        CSV used for the final validation.
    verbose : bool, default False
        Print progress and, on success, the ``export SYNTX_DATA_DIR=...`` line.

    Returns
    -------
    (is_valid, report) : (bool, dict)
        As ``check_mindboggle_data`` for ``target_dir``, plus 'organized_subjects' (subjects
        with both files found, including ones already present) and 'target_dir'.
    """
    import tarfile
    import zipfile
    import shutil
    from syntx.data.msd import _safe_extract

    if mode not in ("auto", "link", "symlink", "copy"):
        raise ValueError(f"mode must be 'auto', 'link', 'symlink' or 'copy'; got {mode!r}")

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
                _safe_extract(tar, tmp_ext)
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
                    _safe_extract(tar, tmp_ext)
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

        tmp_root = os.path.join(target_dir, "_tmp_extracted")
        for src, dst in [(brain_src, brain_dst), (label_src, label_dst)]:
            if os.path.exists(dst):
                continue

            transferred = False
            if os.path.commonpath([os.path.abspath(src), tmp_root]) == tmp_root:
                # extracted files are deleted below: move them (a symlink would dangle)
                shutil.move(src, dst)
                continue
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
