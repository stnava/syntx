"""syntx.tabulate — label-based tabulation into one-row "wide" dataframes.

Turns per-label (long-format) measurements and ROI timeseries into the single wide row a
modality contributes to a session's study CSV. Label ids are used only for column names, so
nothing depends on which atlas produced them. (At the time of writing antsxfunctional's
rsfMRI connectivity code uses it.)

- ``widen_summary_dataframe``: long per-label table -> one row.
- ``roi_mean_timeseries``: 4-D image + 3-D label image -> per-ROI mean timeseries.
- ``correlation_matrix``: Pearson correlation of timeseries columns.
- ``correlation_matrix_wide_from_timeseries`` / ``_from_image``: upper-triangle
  correlations as one row.
- ``connectivity_summary_stats``: scalar summaries of a correlation matrix.

Re-implemented (not imported) from ``blindantspymm.mm.widen_summary_dataframe`` /
``rsfmri_to_correlation_matrix_wide``.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import re

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def normalize_label_name(name: str, *, lowercase: bool = False) -> str:
    """Canonicalize an atlas region name for wide-format column headers.

    Strips outer whitespace, collapses internal whitespace to underscores, and corrects
    known upstream label-table typos (e.g. ``cerebellem`` -> ``cerebellum`` in ``dkt.csv``).
    """
    cleaned = re.sub(r"\s+", "_", str(name).strip()).replace("cerebellem", "cerebellum")
    return cleaned.lower() if lowercase else cleaned


def label_df_to_lookup(
    label_df: pd.DataFrame | Mapping[int, str] | None,
    *,
    lowercase: bool = False,
) -> dict[int, str]:
    """Convert a ``(label, label_name)`` DataFrame or mapping into a sanitized ``int -> str``
    lookup, excluding non-positive IDs (``<= 0``), ``background``, and bare numeric strings.
    """
    if label_df is None:
        return {}
    if isinstance(label_df, Mapping):
        pairs = list(label_df.items())
    else:
        lab_col = "label" if "label" in label_df.columns else "Label"
        name_col = "label_name" if "label_name" in label_df.columns else "Description"
        pairs = list(zip(label_df[lab_col], label_df[name_col], strict=False))

    lookup: dict[int, str] = {}
    for lab, raw_name in pairs:
        lab_int = int(lab)
        name_str = normalize_label_name(str(raw_name), lowercase=lowercase)
        if lab_int <= 0 or not name_str or name_str.lower() == "background" or name_str == str(lab_int):
            continue
        lookup[lab_int] = name_str
    return lookup


def map_intensity_to_dataframe(
    scalar_img: Any,
    label_img: Any,
    label_df: pd.DataFrame,
    stat: str = "mean",
) -> pd.DataFrame:
    """Extract regional statistics of a scalar image by integer atlas labels.

    Generic, atlas-agnostic -- works for any labeled parcellation (DKT, CIT168,
    Harvard-Oxford, JHU white matter, or anything else) given its label-name table.
    The one shared primitive every per-repo "region tabulation" helper in this
    ecosystem should compose on top of, instead of reimplementing per-region masking
    independently (confirmed duplicated, with slightly different signatures, in
    antsxstructural.tabulate and antsxdwi.atlas before this consolidation).

    Parameters
    ----------
    scalar_img : ants.ANTsImage
        Scalar image (FA, MD, PET SUV, fixel FD, ...) in the SAME physical space as
        label_img.
    label_img : ants.ANTsImage
        Integer label/parcellation image, same space as scalar_img.
    label_df : pandas.DataFrame
        Must have at least ``label`` (int) and ``label_name`` (str) columns.
    stat : {"mean", "median"}, default "mean"

    Returns
    -------
    pandas.DataFrame
        Long-format, columns ``label``, ``label_name``, ``value`` -- one row per
        label_df row, in the same order. A label with zero voxels in label_img gets
        value=NaN, not an error or a skipped row.
    """
    scalar_arr = scalar_img.numpy().astype(float)
    label_arr = label_img.numpy().astype(int)

    if stat != "median" and not np.isnan(scalar_arr).any():
        lbl_flat = label_arr.ravel()
        val_flat = scalar_arr.ravel()
        if (lbl_flat < 0).any():
            non_neg = lbl_flat >= 0
            lbl_non_neg = lbl_flat[non_neg]
            val_non_neg = val_flat[non_neg]
        else:
            lbl_non_neg = lbl_flat
            val_non_neg = val_flat
        counts = np.bincount(lbl_non_neg)
        sums = np.bincount(lbl_non_neg, weights=val_non_neg)
        n_bins = len(counts)
        rows = []
        for _, row in label_df.iterrows():
            lval = int(row["label"])
            lname = str(row.get("label_name", row.get("name", str(lval))))
            if 0 <= lval < n_bins and counts[lval] > 0:
                val = float(sums[lval] / counts[lval])
            elif lval < 0:
                voxels = val_flat[lbl_flat == lval]
                val = float(voxels.mean()) if voxels.size > 0 else float("nan")
            else:
                val = float("nan")
            rows.append({"label": lval, "label_name": lname, "value": val})
        return pd.DataFrame(rows)

    rows = []
    for _, row in label_df.iterrows():
        lval = int(row["label"])
        lname = str(row.get("label_name", row.get("name", str(lval))))
        voxels = scalar_arr[label_arr == lval]
        if voxels.size == 0:
            val = float("nan")
        elif stat == "median":
            val = float(np.median(voxels))
        else:
            val = float(voxels.mean())
        rows.append({"label": lval, "label_name": lname, "value": val})
    return pd.DataFrame(rows)


def region_wide_row(
    scalar_img: Any,
    label_img: Any,
    label_df: pd.DataFrame,
    column_prefix: str,
    stat: str = "mean",
) -> dict[str, float]:
    """Flatten any scalar image's per-region statistics into wide-CSV columns, excluding
    non-positive / background rows and normalizing whitespace in ``label_name``.

    Centralized here so ``antsxdwi.atlas.region_wide_row`` and other sibling packages share
    one implementation rather than maintaining per-repo loops.
    """
    df = map_intensity_to_dataframe(scalar_img, label_img, label_df, stat=stat)
    row: dict[str, float] = {}
    for _, r in df.iterrows():
        lab = int(r["label"])
        lname = normalize_label_name(str(r["label_name"]))
        if lab <= 0 or not lname or lname.lower() == "background":
            continue
        col = f"{column_prefix}_{lname}" if column_prefix else lname
        row[col] = float(r["value"])
    return row


def roi_summary_wide(
    scalar_img: Any,
    label_img: Any,
    *,
    prefix: str = "",
    value_name: str = "Mean",
    stat: str = "mean",
    include_label_names: bool = True,
    label_df: pd.DataFrame | Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Single-row wide DataFrame of per-ROI scalar statistics present in ``label_img``.

    Centralizes the wide ROI tabulation used across ``antsxfunctional`` (PET, ASL, rsfMRI,
    neuromelanin):
    - Filters to positive labels present in ``label_img`` (never emits ``Label0`` or
      ``background``).
    - When ``include_label_names=True`` (default), emits ``{prefix}{name}.{value_name}`` for
      every label in ``label_df`` without emitting redundant numeric ``Label{N}`` duplicates;
      falls back to ``{prefix}Label{N}.{value_name}`` only for unknown label IDs or when
      ``include_label_names=False``.
    """
    lab_arr = label_img.numpy()
    present_labels = sorted(int(v) for v in np.unique(lab_arr) if v > 0)
    if not present_labels:
        return pd.DataFrame([{}])

    names_lookup = label_df_to_lookup(label_df)
    region_df = pd.DataFrame({
        "label": present_labels,
        "label_name": [names_lookup.get(lab, str(lab)) for lab in present_labels],
    })
    long_df = map_intensity_to_dataframe(scalar_img, label_img, region_df, stat=stat)
    if not include_label_names:
        long_df = long_df.rename(columns={"label": "Label", "value": value_name})
        return widen_summary_dataframe(
            long_df[["Label", value_name]], description="Label", value=value_name, prefix=prefix
        )

    cols: dict[str, float] = {}
    suffix = f".{value_name}" if value_name else ""
    for row in long_df.itertuples():
        lab = int(row.label)
        val = float(row.value)
        if np.isnan(val):
            continue
        col_token = names_lookup.get(lab, f"Label{lab}")
        cols[f"{prefix}{col_token}{suffix}"] = val
    return pd.DataFrame([cols])


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
        Long-format dataframe with at least a label-id column and a value column. Not
        modified.
    description : str, default "Label"
        Column holding the label id. A float64 column is cast to int (so 2.0 -> "2"; NaN ids
        raise).
    value : str, default "VolumeInMillimeters"
        Column holding the value to spread into wide columns.
    skip_first : bool, default False
        Drop the first row (by position) before pivoting, e.g. a background row.
    prefix : str, default ""
        Prepended to every output column name (e.g. ``"dwi."``).

    Returns
    -------
    pd.DataFrame
        One row; one column per distinct label (sorted), named
        ``f"{prefix}{description}{label}.{value}"``. For a repeated label the first value is
        kept; labels whose value is NaN get no column (``pivot_table`` drops them).

    Raises
    ------
    ValueError
        If either column is missing.
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


def correlation_matrix(timeseries: "np.ndarray | torch.Tensor", device: str = "cpu") -> np.ndarray:
    """N x N Pearson correlation matrix of the columns of a (n_timepoints, n_rois) matrix.

    With torch (the normal case): columns are centred and divided by their population standard
    deviation, then ``Z^T Z / n_timepoints``, in float64 (float32 on MPS, which has no float64).
    A constant column has no defined correlation: its row and column, diagonal included, are
    NaN -- the same as ``np.corrcoef``, used without torch.

    Parameters
    ----------
    timeseries : array-like or torch.Tensor, shape (n_timepoints, n_rois)
    device : str, default "cpu"
        Torch device.

    Returns
    -------
    np.ndarray, shape (n_rois, n_rois), float64
    """
    if torch is not None:
        dtype = torch.float32 if str(device).startswith("mps") else torch.float64
        t = torch.as_tensor(np.asarray(timeseries), dtype=dtype, device=device)
        t = t - t.mean(dim=0, keepdim=True)
        std = t.std(dim=0, keepdim=True, unbiased=False)
        t = t / torch.where(std == 0, torch.ones_like(std), std)
        corr = (t.T @ t) / t.shape[0]
        constant = (std == 0).squeeze(0)
        corr[constant, :] = float("nan")
        corr[:, constant] = float("nan")
        return corr.cpu().numpy().astype(np.float64)
    return np.corrcoef(np.asarray(timeseries), rowvar=False)  # pragma: no cover


def connectivity_summary_stats(matrix: np.ndarray) -> dict[str, float]:
    """Scalar summaries of the off-diagonal entries of an N x N correlation matrix.

    Useful as a QC flag: e.g. near-uniform strong positive correlation suggests a global
    artefact (motion, drift) survived nuisance regression.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_rois, n_rois)
        Non-finite off-diagonal entries are ignored.

    Returns
    -------
    dict of float
        ``mean_abs_connectivity`` (mean |r|), ``median_connectivity`` (median r),
        ``positive_fraction`` / ``negative_fraction`` (fraction of entries > 0 / < 0, so exact
        zeros count in neither). All NaN if there are fewer than 2 ROIs or no finite
        off-diagonal entry.
    """
    n = matrix.shape[0]
    if n < 2:
        return {
            "mean_abs_connectivity": float("nan"), "median_connectivity": float("nan"),
            "positive_fraction": float("nan"), "negative_fraction": float("nan"),
        }
    off_diag = matrix[~np.eye(n, dtype=bool)]
    off_diag = off_diag[np.isfinite(off_diag)]
    if off_diag.size == 0:
        return {
            "mean_abs_connectivity": float("nan"), "median_connectivity": float("nan"),
            "positive_fraction": float("nan"), "negative_fraction": float("nan"),
        }
    return {
        "mean_abs_connectivity": float(np.mean(np.abs(off_diag))),
        "median_connectivity": float(np.median(off_diag)),
        "positive_fraction": float(np.mean(off_diag > 0)),
        "negative_fraction": float(np.mean(off_diag < 0)),
    }


def correlation_matrix_wide_from_timeseries(
    timeseries: "np.ndarray | torch.Tensor",
    roi_ids: Sequence[int],
    prefix: str = "Corr_",
    device: str = "cpu",
    label_names: Mapping[int, str] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One-row dataframe of the pairwise ROI correlations (upper triangle, ``correlation_matrix``)
    of a per-ROI timeseries matrix.

    Parameters
    ----------
    timeseries : array-like, shape (n_timepoints, n_rois)
        Per-ROI mean signal timeseries, columns in the same order as ``roi_ids``.
    roi_ids : sequence of int
        Label id of each column, used only for naming (must have n_rois entries).
    prefix : str, default "Corr_"
        Prefix prepended to every output column name.
    device : str, default "cpu"
        Passed to ``correlation_matrix``.
    label_names : mapping of int -> str or DataFrame, optional
        Human-readable region name per integer ROI id (or a ``(label, label_name)`` DataFrame).
        When given, each ROI ``rid`` present in ``label_names`` is formatted as its sanitized
        name instead of ``Label{rid}``; any ROI id absent from ``label_names`` (or all ROIs
        when ``label_names`` is ``None``) falls back to ``Label{rid}``.

    Returns
    -------
    pd.DataFrame
        One row; one column per pair i < j (in ``roi_ids`` order), named
        ``f"{prefix}{name_i}_{name_j}"`` (or ``f"{prefix}Label{roi_ids[i]}_Label{roi_ids[j]}"``
        when ``label_names`` is not supplied).
    """
    n_rois = len(roi_ids)
    corr = correlation_matrix(timeseries, device=device)

    iu_a, iu_b = np.triu_indices(n_rois, k=1)
    vals_2d = np.ascontiguousarray(corr[iu_a, iu_b][None, :], dtype=np.float64)
    lookup = label_df_to_lookup(label_names)
    roi_tokens = [f"{prefix}{lookup.get(int(rid), f'Label{int(rid)}')}" for rid in roi_ids]
    names_raw = [lookup.get(int(rid), f"Label{int(rid)}") for rid in roi_ids]
    col_names = [f"{roi_tokens[a]}_{names_raw[b]}" for a, b in zip(iu_a.tolist(), iu_b.tolist())]
    return pd.DataFrame(vals_2d, columns=col_names, copy=False)


def _project_voxel_timeseries_to_rois(
    fg_ts: np.ndarray,
    fg_lbl: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Project ``(n_fg_voxels, n_timepoints)`` array onto sorted positive integer ROI IDs."""
    from scipy.sparse import csr_matrix

    n_t = int(fg_ts.shape[1])
    if fg_lbl.size == 0:
        return np.zeros((n_t, 0), dtype=np.float64), []
    u_pos, inv = np.unique(fg_lbl, return_inverse=True)
    roi_ids = [int(v) for v in u_pos]
    n_rois = len(roi_ids)
    n_fg = int(fg_lbl.shape[0])
    counts = np.bincount(inv, minlength=n_rois).astype(np.float64)
    work_dtype = np.float32 if fg_ts.dtype == np.float32 else np.float64
    ts_work = np.ascontiguousarray(fg_ts, dtype=work_dtype)
    proj = csr_matrix(
        (np.ones(n_fg, dtype=work_dtype), (inv, np.arange(n_fg, dtype=np.int64))),
        shape=(n_rois, n_fg),
    )
    sums = proj.dot(ts_work).astype(np.float64)
    mean_roi = np.ascontiguousarray((sums / counts[:, None]).T, dtype=np.float64)
    return mean_roi, roi_ids


def _as_readonly_array(img_or_arr: Any) -> np.ndarray:
    """Return a zero-copy NumPy view of an ``ANTsImage`` (via ``.view()``) or array."""
    if hasattr(img_or_arr, "view") and callable(img_or_arr.view) and hasattr(img_or_arr, "spacing"):
        return img_or_arr.view()
    if hasattr(img_or_arr, "numpy") and callable(img_or_arr.numpy):
        return img_or_arr.numpy()
    return np.asarray(img_or_arr)


def roi_mean_from_masked_timeseries(
    flat_timeseries: np.ndarray,
    mask: Any,
    roi_label_image: Any,
) -> tuple[np.ndarray, list[int]]:
    """Extract ``(mean_roi, roi_ids)`` directly from a ``(n_timepoints, n_mask_voxels)`` masked array.

    Avoids materializing a multi-gigabyte 4-D ``ANTsImage`` when the caller already holds the
    brain-masked ``(T, V_mask)`` time-series matrix (e.g. ``antsxfunctional.rsfmri``). Voxels with
    ``roi_label > 0`` outside ``mask`` (if any) have zero signal in a brain-masked residual volume;
    pass ``roi_label_image`` already masked to ``mask`` (standard in ``rsfmri``/``TaskfMRI``) for
    exact equivalence with :func:`roi_mean_timeseries` on the zero-filled 4-D volume.
    """
    mask_arr = _as_readonly_array(mask) > 0
    label_arr = _as_readonly_array(roi_label_image)
    if mask_arr.shape != label_arr.shape:
        raise ValueError(
            f"roi_mean_from_masked_timeseries: label grid {label_arr.shape} != mask grid {mask_arr.shape}"
        )
    if not np.array_equal(label_arr, np.floor(label_arr)):
        values = np.unique(label_arr)
        raise ValueError(
            "roi_mean_from_masked_timeseries: labels must be integer-valued "
            f"(got e.g. {values[values != np.round(values)][:3].tolist()})"
        )
    ts_tv = np.asarray(flat_timeseries)
    n_t, n_m = int(ts_tv.shape[0]), int(ts_tv.shape[1])
    lbl_in_mask = label_arr[mask_arr]
    if lbl_in_mask.shape[0] != n_m:
        raise ValueError(
            f"roi_mean_from_masked_timeseries: flat_timeseries has {n_m} columns, mask has {lbl_in_mask.shape[0]}"
        )
    pos_in_mask = lbl_in_mask > 0
    if not np.any(pos_in_mask):
        return np.zeros((n_t, 0), dtype=np.float64), []

    # Check if any positive ROI voxel lies outside `mask_arr` (if so, its signal is 0 in a masked 4-D image,
    # so denominator is total voxel count of that ROI in `label_arr`, while numerator sums over `mask_arr`).
    fg_lbl_in_mask = lbl_in_mask[pos_in_mask].astype(np.int64, copy=False)
    fg_ts_vt = ts_tv[:, pos_in_mask].T
    mean_in_mask, roi_ids = _project_voxel_timeseries_to_rois(fg_ts_vt, fg_lbl_in_mask)
    outside_pos = (label_arr > 0) & (~mask_arr)
    if np.any(outside_pos):
        all_pos_lbl = label_arr[label_arr > 0].astype(np.int64, copy=False)
        u_all, cnt_all = np.unique(all_pos_lbl, return_counts=True)
        u_in, cnt_in = np.unique(fg_lbl_in_mask, return_counts=True)
        in_map = dict(zip(u_in.tolist(), range(len(u_in))))
        out_full = np.zeros((n_t, len(u_all)), dtype=np.float64)
        for j, (lid, c_tot) in enumerate(zip(u_all.tolist(), cnt_all.tolist())):
            if lid in in_map:
                idx_in = in_map[lid]
                out_full[:, j] = mean_in_mask[:, idx_in] * (float(cnt_in[idx_in]) / float(c_tot))
        return out_full, [int(v) for v in u_all]
    return mean_in_mask, roi_ids


def roi_mean_timeseries(timeseries_image: Any, roi_label_image: Any) -> tuple[np.ndarray, list[int]]:
    """Mean signal over each ROI at every time point (vectorized O(V_fg * T) sparse projection).

    Parameters
    ----------
    timeseries_image : ants.ANTsImage
        4-D timeseries (time on the last axis), e.g. rsfMRI or an ASL series.
    roi_label_image : ants.ANTsImage
        3-D integer-valued label image on the same voxel grid: its shape, spacing and origin
        must match the first three axes of the timeseries (ValueError otherwise). Labels <= 0
        are excluded.

    Returns
    -------
    (mean_roi, roi_ids) : (np.ndarray of shape (n_timepoints, n_rois), list of int)
        ``roi_ids`` are the sorted positive label values; ``mean_roi`` is float64.
    """
    ts_arr = _as_readonly_array(timeseries_image)
    label_arr = _as_readonly_array(roi_label_image)
    if ts_arr.shape[:3] != label_arr.shape:
        raise ValueError(f"roi_mean_timeseries: label grid {label_arr.shape} does not match the "
                         f"timeseries grid {ts_arr.shape[:3]}")
    for name, a, b in (("spacing", timeseries_image.spacing[:3], roi_label_image.spacing),
                       ("origin", timeseries_image.origin[:3], roi_label_image.origin)):
        if not np.allclose(a, b, atol=1e-4):
            raise ValueError(f"roi_mean_timeseries: label {name} {tuple(b)} does not match the "
                             f"timeseries grid {tuple(a)}")
    if not np.array_equal(label_arr, np.floor(label_arr)):
        values = np.unique(label_arr)
        raise ValueError("roi_mean_timeseries: labels must be integer-valued "
                         f"(got e.g. {values[values != np.round(values)][:3].tolist()})")

    n_t = int(ts_arr.shape[-1])
    fg_mask = label_arr > 0
    if not np.any(fg_mask):
        return np.zeros((n_t, 0), dtype=np.float64), []

    fg_lbl = label_arr[fg_mask].astype(np.int64, copy=False)
    fg_ts = ts_arr[fg_mask, :]
    return _project_voxel_timeseries_to_rois(fg_ts, fg_lbl)


def correlation_matrix_wide_from_image(
    timeseries_image: Any,
    roi_label_image: Any,
    prefix: str = "Corr_",
    device: str = "cpu",
    label_names: Mapping[int, str] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One-row dataframe of pairwise ROI correlations from a 4-D timeseries ANTsImage and a
    3-D ROI label ANTsImage.

    Extracts each ROI's mean timeseries via :func:`roi_mean_timeseries`, then delegates to
    :func:`correlation_matrix_wide_from_timeseries`.
    """
    mean_roi, roi_ids = roi_mean_timeseries(timeseries_image, roi_label_image)
    return correlation_matrix_wide_from_timeseries(
        mean_roi, roi_ids, prefix=prefix, device=device, label_names=label_names
    )
