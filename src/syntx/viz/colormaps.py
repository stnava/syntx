"""
syntx.viz.colormaps — categorical colours for label maps (e.g. DKT parcellations)
==================================================================================

Deterministic label colours, one scheme everywhere: integer label ``i`` gets hue
``(0.125 + i * 0.618...) mod 1`` (golden-ratio steps, so consecutive labels get
well-separated hues) in HSV space (saturation 0.88, value 0.68, alpha 0.90). Label 0 is
transparent black. ``get_dkt_colormap`` / ``dkt_colormap`` and ``build_dkt_label_palette`` /
``get_dkt_label_color_dict`` give the same colour to the same integer label.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


_GOLDEN = 0.618033988749895
_HUE0 = 0.125


def _label_rgba(i: int, saturation: float = 0.88, value: float = 0.68) -> Tuple[float, float, float, float]:
    """RGBA of integer label ``i`` (``i >= 1``)."""
    h = (_HUE0 + i * _GOLDEN) % 1.0
    return (*mcolors.hsv_to_rgb((h, saturation, value)), 0.90)


def _clean_label(l):
    """int for integer-valued labels (ints, integral floats, numeric strings such as "3.0"),
    stripped str for names, None for empty strings; non-integral numbers raise ValueError."""
    if isinstance(l, (bool, np.bool_)):
        raise ValueError(f"label {l!r} is not a label ID or name")
    if isinstance(l, (int, np.integer)):
        return int(l)
    if isinstance(l, (float, np.floating)):
        if not float(l).is_integer():
            raise ValueError(f"label {l!r} is not an integer label ID")
        return int(l)
    t = str(l).strip()
    if not t:
        return None
    try:
        f = float(t)
    except ValueError:
        return t
    if not f.is_integer():
        raise ValueError(f"label {l!r} is not an integer label ID")
    return int(f)


def build_dkt_label_palette(unique_labels: List[Union[int, str]]) -> Tuple[Dict[Union[int, str], Tuple[float, ...]], np.ndarray]:
    """Build a colour for each label in ``unique_labels``.

    Integer-valued labels (ints, integral floats, numeric strings such as ``"3.0"``) are
    converted to ``int`` and kept if > 0; a non-integral number raises ValueError. Integer
    label ``i`` gets the same colour as entry ``i`` of ``get_dkt_colormap``. Other strings
    are region names (stripped, empty ones dropped); the ``k``-th name in sorted order gets
    the colour of integer ``max_int_label + 1 + k``, so names never share a colour with the
    integer labels passed alongside them.

    Parameters
    ----------
    unique_labels : list of int, float or str
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
    clean = {_clean_label(l) for l in unique_labels}
    ints = sorted(l for l in clean if isinstance(l, int) and l > 0)
    names = sorted(l for l in clean if isinstance(l, str))

    max_id = max(ints) if ints else 256
    lut = np.zeros((max_id + 1, 4), dtype=np.float32)
    color_map = {}
    for lid in ints:
        rgba = _label_rgba(lid)
        color_map[lid] = color_map[str(lid)] = rgba
        lut[lid] = rgba
    base = max(ints) if ints else 0
    for k, name in enumerate(names):
        color_map[name] = _label_rgba(base + 1 + k)
    return color_map, lut


def get_dkt_colormap(max_label: int = 2050, value: float = 0.68, saturation: float = 0.88) -> mcolors.ListedColormap:
    """Build a ``ListedColormap`` with ``max_label + 1`` colours indexed by label ID.

    Entry 0 is transparent black; entry ``i`` (1..max_label) is label ``i``'s colour (the same
    as ``build_dkt_label_palette``'s with the default ``value`` / ``saturation``), alpha
    0.90. Use with integer data and a norm / vmin-vmax that maps label ``i`` to entry ``i``
    (e.g. ``vmin=0, vmax=max_label``, ``interpolation='nearest'``).

    Parameters
    ----------
    max_label : int, default 2050
        Largest label ID that gets its own colour.
    value : float, default 0.68
        HSV value (brightness).
    saturation : float, default 0.88
        HSV saturation.

    Returns
    -------
    matplotlib.colors.ListedColormap
        Named ``"dkt_colormap"``.
    """
    colors = [(0.0, 0.0, 0.0, 0.0)] + [_label_rgba(i, saturation, value) for i in range(1, max_label + 1)]
    return mcolors.ListedColormap(colors, name="dkt_colormap")


# Standardized singleton instance exported across syntx.viz
dkt_colormap = get_dkt_colormap(2050)


def get_dkt_label_color_dict(unique_labels: List[Union[int, str]]) -> Dict[Union[int, str], Tuple[float, ...]]:
    """Return only the ``color_map`` dict of ``build_dkt_label_palette(unique_labels)``
    (label -> RGBA tuple)."""
    color_map, _ = build_dkt_label_palette(unique_labels)
    return color_map
