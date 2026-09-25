"""syntx.tabulate — generic, atlas-agnostic label-based tabulation.

Shared by every ANTsX-family modality package (antsxdwi, antsxslowflow, antsxfunctional)
and antsxmm itself, for turning per-label (long-format) measurements into the single wide
row each modality contributes to a session's study CSV.

Makes no assumption about which atlas, species, or organ system produced the labels --
ported and generalised from ``blindantspymm.mm.widen_summary_dataframe`` /
``rsfmri_to_correlation_matrix_wide`` (a genuinely species/atlas-agnostic precedent),
reimplemented independently here rather than imported (that package depends on the legacy
TensorFlow/dipy stack this ecosystem is moving away from).
"""

from __future__ import annotations

import itertools
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def widen_summary_dataframe(
    long_df: pd.DataFrame,
    description: str = "Label",
    value: str = "VolumeInMillimeters",
    skip_first: bool = False,
    prefix: str = "",
) -> pd.DataFrame:
    """Convert a long-format per-label dataframe into a single-row wide-format dataframe.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format dataframe with at least a categorical identifier column (e.g. ``Label``)
        and a numeric value column (e.g. ``VolumeInMillimeters``).
    description : str
        Column name holding the per-row category (label id).
    value : str
        Column name holding the numeric value to spread into wide columns.
    skip_first : bool
        Drop the first row before pivoting (e.g. to skip a background/label-0 row).
    prefix : str
        Prefix prepended to every output column name (e.g. a modality tag like ``"dwi."``).

    Returns
    -------
    pd.DataFrame
        One row; one column per label, named ``f"{prefix}{description}{label}.{value}"``.
    """
    if description not in long_df.columns or value not in long_df.columns:
        raise ValueError(f"Dataframe must contain '{description}' and '{value}' columns.")

    df = long_df.copy()
    if df[description].dtype == float:
        df[description] = df[description].astype(int)
    if skip_first:
        df = df.drop(df.index[0])

    wide = df.pivot_table(index=None, columns=description, values=value, aggfunc="first")
    wide.columns = [f"{description}{col}.{value}" for col in wide.columns]
    wide = wide.reset_index(drop=True)
    wide = wide.add_prefix(prefix)
    return wide


def correlation_matrix_wide_from_timeseries(
    timeseries: "np.ndarray | torch.Tensor",
    roi_ids: Sequence[int],
    prefix: str = "Corr_",
    device: str = "cpu",
) -> pd.DataFrame:
    """Compute a one-row wide dataframe of pairwise ROI correlations from an already-extracted
    per-ROI mean timeseries matrix.

    Parameters
    ----------
    timeseries : array-like, shape (n_timepoints, n_rois)
        Per-ROI mean signal timeseries, columns in the same order as ``roi_ids``.
    roi_ids : sequence of int
        Label ids corresponding to each column of ``timeseries``, used only for naming.
    prefix : str
        Prefix prepended to every output column name.
    device : str
        Torch device used for the correlation computation when torch is available; falls
        back to numpy if torch is not installed.

    Returns
    -------
    pd.DataFrame
        One row; one column per unordered ROI pair, named ``f"{prefix}Label{a}_Label{b}"``.
    """
    n_rois = len(roi_ids)
    if torch is not None:
        t = torch.as_tensor(np.asarray(timeseries), dtype=torch.float64, device=device)
        t = t - t.mean(dim=0, keepdim=True)
        std = t.std(dim=0, keepdim=True, unbiased=False)
        std = torch.where(std == 0, torch.ones_like(std), std)
        t = t / std
        corr = (t.T @ t) / t.shape[0]
        corr = corr.cpu().numpy()
    else:  # pragma: no cover
        corr = np.corrcoef(np.asarray(timeseries), rowvar=False)

    roi_pairs = list(itertools.combinations(range(n_rois), 2))
    cor_values = [corr[a, b] for a, b in roi_pairs]
    col_names = [f"Label{int(roi_ids[a])}_Label{int(roi_ids[b])}" for a, b in roi_pairs]
    df_wide = pd.DataFrame([cor_values], columns=col_names)
    return df_wide.add_prefix(prefix)


def correlation_matrix_wide_from_image(
    timeseries_image: Any,
    roi_label_image: Any,
    prefix: str = "Corr_",
    device: str = "cpu",
) -> pd.DataFrame:
    """Compute a one-row wide dataframe of pairwise ROI correlations directly from a 4-D
    timeseries ANTsImage and a 3-D ROI label ANTsImage.

    Extracts each ROI's mean timeseries via boolean masking (no ANTs-specific
    ``timeseries_to_matrix`` dependency), then delegates to
    :func:`correlation_matrix_wide_from_timeseries`.

    Parameters
    ----------
    timeseries_image : ants.ANTsImage
        4-D timeseries image (e.g. resting-state fMRI, ASL control/label series).
    roi_label_image : ants.ANTsImage
        3-D image with integer ROI labels (0 = background, excluded).
    prefix : str
        Prefix prepended to every output column name.
    device : str
        Torch device used for the correlation computation.

    Returns
    -------
    pd.DataFrame
        One row; one column per unordered ROI pair.
    """
    ts_arr = timeseries_image.numpy()
    label_arr = roi_label_image.numpy()
    roi_ids = sorted(int(v) for v in np.unique(label_arr) if v > 0)

    n_t = ts_arr.shape[-1]
    mean_roi = np.zeros((n_t, len(roi_ids)), dtype=np.float64)
    for i, label in enumerate(roi_ids):
        mask = label_arr == label
        mean_roi[:, i] = ts_arr[mask, :].mean(axis=0)

    return correlation_matrix_wide_from_timeseries(mean_roi, roi_ids, prefix=prefix, device=device)
