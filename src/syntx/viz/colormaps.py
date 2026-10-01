"""
syntx.viz.colormaps — categorical colours for label maps (e.g. DKT parcellations)
==================================================================================

Deterministic label colours. Hues are stepped by the golden-ratio fraction (0.618...) starting
from 0.125, in HSV space with fixed saturation 0.88 and value 0.68 and alpha 0.90, so that
consecutive labels get well-separated hues. Label 0 is transparent black.

Two schemes exist and they do not give the same colour to the same label:

- ``get_dkt_colormap`` / ``dkt_colormap``: colour is a function of the label ID (entry ``i`` is
  the ``i``-th hue step).
- ``build_dkt_label_palette`` / ``get_dkt_label_color_dict``: colour is a function of the
  label's rank among the labels passed in (the ``k``-th sorted label gets the ``k``-th step).
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def build_dkt_label_palette(unique_labels: List[Union[int, str]]) -> Tuple[Dict[Union[int, str], Tuple[float, ...]], np.ndarray]:
    """Build a colour for each label in ``unique_labels``, by rank among those labels.

    Each entry is converted with ``int()`` when possible (so floats are truncated) and kept
    only if > 0; entries that cannot be converted are kept as stripped, non-empty strings
    (so ``"3.0"`` stays the string ``"3.0"``). The cleaned set is sorted (integers first,
    ascending, then strings) and the ``k``-th label gets the ``k``-th golden-ratio hue step
    (HSV saturation 0.88, value 0.68, alpha 0.90).

    Parameters
    ----------
    unique_labels : list of int or str
        Label IDs or region names. Duplicates and labels <= 0 are ignored.

    Returns
    -------
    color_map : dict
        Label -> RGBA tuple of 4 floats. Every label is stored under both its cleaned value
        and ``str()`` of it (e.g. ``3`` and ``"3"``).
    lut : np.ndarray, float32, shape (max_int_label + 1, 4)
        RGBA lookup table indexed by integer label (row 0 and unused rows are transparent
        zeros). ``max_int_label`` is the largest integer label, or 256 if there are none.
    """
    clean_labels = []
    for l in unique_labels:
        try:
            val = int(l)
            if val > 0:
                clean_labels.append(val)
        except (ValueError, TypeError):
            if str(l).strip():
                clean_labels.append(str(l).strip())

    # Sort labels deterministically
    sorted_labels = sorted(list(set(clean_labels)), key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)))

    golden_ratio = 0.618033988749895
    color_map = {}
    
    int_labels = [l for l in sorted_labels if isinstance(l, int)]
    max_id = max(int_labels) if int_labels else 256
    lut = np.zeros((max_id + 1, 4), dtype=np.float32)
    lut[0] = [0.0, 0.0, 0.0, 0.0]  # Background = transparent

    h = 0.125
    for idx, lid in enumerate(sorted_labels):
        h = (h + golden_ratio) % 1.0
        rgb = mcolors.hsv_to_rgb((h, 0.88, 0.68))
        rgba = (*rgb, 0.90)

        color_map[lid] = rgba
        color_map[str(lid)] = rgba
        if isinstance(lid, int) and 0 <= lid <= max_id:
            lut[lid] = rgba

    return color_map, lut


def get_dkt_colormap(max_label: int = 2050, lightness: float = 0.68, saturation: float = 0.88) -> mcolors.ListedColormap:
    """Build a ``ListedColormap`` with ``max_label + 1`` colours indexed by label ID.

    Entry 0 is transparent black; entry ``i`` (1..max_label) is the ``i``-th golden-ratio hue
    step starting from hue 0.125, with alpha 0.90. Use with integer data and a norm / vmin-vmax
    that maps label ``i`` to entry ``i`` (e.g. ``vmin=0, vmax=max_label``,
    ``interpolation='nearest'``).

    Parameters
    ----------
    max_label : int, default 2050
        Largest label ID that gets its own colour.
    lightness : float, default 0.68
        HSV *value* (brightness) passed to ``hsv_to_rgb`` (not HSL lightness).
    saturation : float, default 0.88
        HSV saturation.

    Returns
    -------
    matplotlib.colors.ListedColormap
        Named ``"dkt_colormap"``.
    """
    golden_ratio = 0.618033988749895
    colors = [(0.0, 0.0, 0.0, 0.0)]  # Label 0 = transparent background

    h = 0.125
    for i in range(1, max_label + 1):
        h = (h + golden_ratio) % 1.0
        rgb = mcolors.hsv_to_rgb((h, saturation, lightness))
        colors.append((*rgb, 0.90))

    cmap = mcolors.ListedColormap(colors, name="dkt_colormap")
    return cmap


# Standardized singleton instance exported across syntx.viz
dkt_colormap = get_dkt_colormap(2050)


def get_dkt_label_color_dict(unique_labels: List[Union[int, str]]) -> Dict[Union[int, str], Tuple[float, ...]]:
    """Return only the ``color_map`` dict of ``build_dkt_label_palette(unique_labels)``
    (label -> RGBA tuple, colours assigned by rank among the given labels)."""
    color_map, _ = build_dkt_label_palette(unique_labels)
    return color_map
