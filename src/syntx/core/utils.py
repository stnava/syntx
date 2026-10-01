"""
Intensity normalisation: ``normalize_tensor`` (torch, several statistics) and
``normalize_image`` (ANTsImage / NumPy, for registration input; exported as
``syntx.normalize_image``), with ``auto_select_intensity_percentiles`` choosing its clipping
percentiles.
"""

import torch

def normalize_tensor(
    tensor: torch.Tensor,
    method: str = 'minmax',
    eps: float = 1e-8,
    p_min: float = 1.0,
    p_max: float = 99.0,
    dim=None,
    keepdim: bool = True
) -> torch.Tensor:
    """
    Normalise a tensor's intensities.

    Parameters
    ----------
    tensor : Tensor or array-like
        Converted with ``torch.as_tensor`` if needed.
    method : str, default 'minmax'
        - 'minmax' / '01': (t - min) / (max - min), in [0, 1].
        - 'zscore' / 'standard': (t - mean) / std (population std).
        - 'robust' / 'percentile': (t - q_low) / (q_high - q_low) with the ``p_min`` /
          ``p_max`` percentiles, clamped to [0, 1].
        - 'l2' / 'unit_norm', 'l1' / 'unit_sum': t / ||t||.
        - 'sigmoid' / 'logistic': sigmoid(t) (no statistics; ``dim`` ignored).
        Case-insensitive. Any other value raises ValueError.
    eps : float, default 1e-8
        Added to each denominator.
    p_min, p_max : float, default 1, 99
        Percentiles (0-100) for 'robust'.
    dim : int or tuple, optional
        Axes the statistics are taken over (per-slice / per-channel normalisation). None: all
        elements. 'robust' accepts a single int only (``torch.quantile``).
    keepdim : bool, default True
        Keep the reduced axes so the statistics broadcast. Leave True when ``dim`` is set.

    Returns
    -------
    Tensor of the input shape. Integer input gives a float result.
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)

    method = method.lower().strip()

    if method in ('minmax', '01'):
        if dim is None:
            t_min = tensor.min()
            t_max = tensor.max()
        else:
            t_min = tensor.amin(dim=dim, keepdim=keepdim)
            t_max = tensor.amax(dim=dim, keepdim=keepdim)
        return (tensor - t_min) / (t_max - t_min + eps)

    elif method in ('zscore', 'standard'):
        if dim is None:
            t_mean = tensor.mean()
            t_std = tensor.std(unbiased=False)
        else:
            t_mean = tensor.mean(dim=dim, keepdim=keepdim)
            t_std = tensor.std(dim=dim, keepdim=keepdim, unbiased=False)
        return (tensor - t_mean) / (t_std + eps)

    elif method in ('robust', 'percentile'):
        if dim is None:
            q_min = torch.quantile(tensor.float(), p_min / 100.0).to(tensor.dtype)
            q_max = torch.quantile(tensor.float(), p_max / 100.0).to(tensor.dtype)
        else:
            q_min = torch.quantile(tensor.float(), p_min / 100.0, dim=dim, keepdim=keepdim).to(tensor.dtype)
            q_max = torch.quantile(tensor.float(), p_max / 100.0, dim=dim, keepdim=keepdim).to(tensor.dtype)
        res = (tensor - q_min) / (q_max - q_min + eps)
        return torch.clamp(res, 0.0, 1.0)

    elif method in ('l2', 'unit_norm'):
        if dim is None:
            norm = torch.linalg.vector_norm(tensor, ord=2)
        else:
            norm = torch.linalg.vector_norm(tensor, ord=2, dim=dim, keepdim=keepdim)
        return tensor / (norm + eps)

    elif method in ('l1', 'unit_sum'):
        if dim is None:
            norm = torch.linalg.vector_norm(tensor, ord=1)
        else:
            norm = torch.linalg.vector_norm(tensor, ord=1, dim=dim, keepdim=keepdim)
        return tensor / (norm + eps)

    elif method in ('sigmoid', 'logistic'):
        return torch.sigmoid(tensor)

    else:
        raise ValueError(f"Unknown normalization method '{method}'. Options: 'minmax', 'zscore', 'robust', 'l2', 'l1', 'sigmoid'.")


def auto_select_intensity_percentiles(
    image,
    num_bins: int = 32,
    p_low_candidates: tuple = (0.5, 1.0, 2.0, 3.0, 5.0),
    p_high_candidates: tuple = (95.0, 97.0, 98.0, 99.0, 99.5),
    saturation_weight: float = 0.5
):
    """
    Choose the clipping percentiles (p_low, p_high) used by ``normalize_image(method='auto')``.

    For every candidate pair, the positive voxels are clipped to that percentile range and
    rescaled to [0, 1]. The pair with the highest score wins:
    score = (Shannon entropy, in bits, of a ``num_bins`` histogram)
    - ``saturation_weight`` * (fraction of voxels in the first + last bin).
    This favours a range that spreads intensities over the histogram without piling them up
    at the clip points.

    Parameters
    ----------
    image : ANTsImage or ndarray
        Only voxels > 0 are used, subsampled to about 100 000 by striding.
    num_bins : int, default 32
        Histogram bins (32, as in Mattes MI).
    p_low_candidates, p_high_candidates : tuple of float
        Candidate percentiles (0-100).
    saturation_weight : float, default 0.5
        Weight of the saturation penalty.

    Returns
    -------
    (p_low, p_high) : tuple of float
        (2.0, 98.0) if there are fewer than 100 positive voxels.
    """
    import numpy as np
    arr = image.numpy() if hasattr(image, "numpy") else np.asarray(image)
    pos = arr[arr > 0]
    if len(pos) < 100:
        return (2.0, 98.0)

    # Subsample if large for sub-millisecond evaluation
    if len(pos) > 100000:
        stride = len(pos) // 100000
        sample_vox = pos[::stride]
    else:
        sample_vox = pos

    low_vals = np.percentile(sample_vox, p_low_candidates)
    high_vals = np.percentile(sample_vox, p_high_candidates)

    best_score = -float('inf')
    best_pair = (2.0, 98.0)

    for i, p_l in enumerate(p_low_candidates):
        v_l = float(low_vals[i])
        for j, p_h in enumerate(p_high_candidates):
            v_h = float(high_vals[j])
            if v_h <= v_l + 1e-4:
                continue

            scaled = np.clip((sample_vox - v_l) / (v_h - v_l), 0.0, 1.0)
            hist, _ = np.histogram(scaled, bins=num_bins, range=(0.0, 1.0))
            p_dist = hist.astype(np.float64) / hist.sum()

            p_active = p_dist[p_dist > 0]
            entropy = -np.sum(p_active * np.log2(p_active))
            saturation = p_dist[0] + p_dist[-1]

            score = entropy - saturation_weight * saturation
            if score > best_score:
                best_score = score
                best_pair = (float(p_l), float(p_h))

    return best_pair


def normalize_image(
    image,
    method: str = 'auto',
    p_min: float = 2.0,
    p_max: float = 98.0,
    foreground_only: bool = True,
    eps: float = 1e-6,
    force: bool = False
):
    """
    Normalise an image's intensities for registration.

    Parameters
    ----------
    image : ANTsImage or ndarray
    method : str, default 'auto'
        - 'auto' / 'entropy': clip to percentiles chosen by
          ``auto_select_intensity_percentiles`` (``p_min`` / ``p_max`` ignored), rescale to
          [0, 1].
        - 'robust' / 'percentile': clip to the ``p_min`` / ``p_max`` percentiles, rescale to
          [0, 1].
        - 'minmax' / '01': (x - min) / (max - min).
        - 'zscore' / 'standard': (x - mean) / std. Not clipped, so not in [0, 1].
        Case-insensitive. Any other value raises ValueError.
    p_min, p_max : float, default 2, 98
        Percentiles (0-100) for 'robust'.
    foreground_only : bool, default True
        Percentiles / mean / std from voxels > 0 only ('auto', 'robust', 'zscore'). If the
        two percentiles nearly coincide, the range [0, max] is used instead.
    eps : float, default 1e-6
        Added to each denominator.
    force : bool, default False
        False: for the [0, 1] methods ('auto', 'robust', 'minmax'), an input already in [0, 1]
        (with max >= 0.5) is returned clipped to [0, 1] and otherwise unchanged, so repeated
        calls are idempotent ('zscore' always z-scores). True: always normalise.

    Returns
    -------
    ANTsImage (same geometry) or ndarray, float32.
    """
    import numpy as np

    is_ants = hasattr(image, "numpy") and hasattr(image, "new_image_like")
    arr = image.numpy() if is_ants else np.asarray(image)

    method = method.lower().strip()

    # Idempotency for the [0, 1] methods: an input already in [0, 1] with an active range is
    # returned clipped (not re-normalised). Not applied to 'zscore', whose output is not [0, 1].
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if (not force and method in ('auto', 'entropy', 'robust', 'percentile', 'minmax', '01')
            and arr_min >= -1e-4 and arr_max <= 1.0 + 1e-4 and arr_max >= 0.5):
        norm_arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
        return image.new_image_like(norm_arr) if is_ants else norm_arr

    if method in ('auto', 'entropy'):
        p_min, p_max = auto_select_intensity_percentiles(arr)
        pos = arr[arr > 0] if foreground_only else arr
        if len(pos) > 0:
            q_min = float(np.percentile(pos, p_min))
            q_max = float(np.percentile(pos, p_max))
            if q_max <= q_min + 1e-4:
                q_min = 0.0
                q_max = float(pos.max())
        else:
            q_min = float(arr.min())
            q_max = float(arr.max())
        norm_arr = np.clip((arr - q_min) / (q_max - q_min + eps), 0.0, 1.0).astype(np.float32)

    elif method in ('robust', 'percentile'):
        pos = arr[arr > 0] if foreground_only else arr
        if len(pos) > 0:
            q_min = float(np.percentile(pos, p_min))
            q_max = float(np.percentile(pos, p_max))
            if q_max <= q_min + 1e-4:
                q_min = 0.0
                q_max = float(pos.max())
        else:
            q_min = float(arr.min())
            q_max = float(arr.max())
        norm_arr = np.clip((arr - q_min) / (q_max - q_min + eps), 0.0, 1.0).astype(np.float32)

    elif method in ('minmax', '01'):
        q_min = float(arr.min())
        q_max = float(arr.max())
        norm_arr = np.clip((arr - q_min) / (q_max - q_min + eps), 0.0, 1.0).astype(np.float32)

    elif method in ('zscore', 'standard'):
        pos = arr[arr > 0] if foreground_only else arr
        mean = float(pos.mean()) if len(pos) > 0 else float(arr.mean())
        std = float(pos.std()) if len(pos) > 0 else float(arr.std())
        norm_arr = ((arr - mean) / (std + eps)).astype(np.float32)

    else:
        raise ValueError(f"Unknown normalization method '{method}'. Options: 'auto', 'robust', 'minmax', 'zscore'.")

    return image.new_image_like(norm_arr) if is_ants else norm_arr
