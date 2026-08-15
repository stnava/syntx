import os
import pandas as pd
import ants

DEFAULT_PAIRS_CSV = "examples/pairs.csv"
DEFAULT_DATA_DIR_ENV = "SYNTX_DATA_DIR"
DEFAULT_DATA_DIR = "/Users/stnava/data/mindboggle/volumes"

def resolve_data_dir() -> str:
    """Resolves the Mindboggle data directory from environment or default."""
    data_dir = os.environ.get(DEFAULT_DATA_DIR_ENV, DEFAULT_DATA_DIR)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Set {DEFAULT_DATA_DIR_ENV} environment variable or ensure "
            f"the default path exists."
        )
    return data_dir

def load_mindboggle_pair(pair_idx: int, pairs_csv: str = DEFAULT_PAIRS_CSV, data_dir: str = None) -> dict:
    """Loads a single image pair from the pairs CSV file."""
    if data_dir is None:
        data_dir = resolve_data_dir()

    df = pd.read_csv(pairs_csv)
    if pair_idx < 0 or pair_idx >= len(df):
        raise IndexError(
            f"pair_idx={pair_idx} out of range [0, {len(df) - 1}]. "
            f"CSV has {len(df)} pairs."
        )

    row = df.iloc[pair_idx]
    c1, s1 = row["cohort1"], row["subject1"]
    c2, s2 = row["cohort2"], row["subject2"]

    paths = {
        "fixed": os.path.join(data_dir, f"{c1}_volumes", s1, "t1weighted_brain.nii.gz"),
        "fixed_label": os.path.join(data_dir, f"{c1}_volumes", s1, "labels.DKT31.manual.nii.gz"),
        "moving": os.path.join(data_dir, f"{c2}_volumes", s2, "t1weighted_brain.nii.gz"),
        "moving_label": os.path.join(data_dir, f"{c2}_volumes", s2, "labels.DKT31.manual.nii.gz"),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name}: {path}")

    return {
        "fixed": ants.image_read(paths["fixed"]),
        "moving": ants.image_read(paths["moving"]),
        "fixed_label": ants.image_read(paths["fixed_label"]),
        "moving_label": ants.image_read(paths["moving_label"]),
        "fixed_id": s1,
        "moving_id": s2,
        "pair_type": str(row.get("type", "unknown")),
    }
