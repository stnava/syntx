"""
syntx.viz.colormaps — Standardized Perceptually Uniform Categorical Colormaps
=============================================================================

Provides deterministic 1-to-1 color mapping for DKT anatomical labels.
Colors are generated using golden ratio hue distribution in HSL space
(constant lightness L=0.68, saturation S=0.88) to maximize visual contrast 
between all adjacent and nearby integer or string region IDs.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def build_dkt_label_palette(unique_labels: List[Union[int, str]]) -> Tuple[Dict[Union[int, str], Tuple[float, ...]], np.ndarray]:
    """
    Deterministically builds a 1-to-1 high-contrast discrete color mapping for a set of DKT labels.
    
    Returns:
        (color_map_dict, lut_array)
        - color_map_dict: Maps label ID (int or str) -> RGBA tuple
        - lut_array: 2D RGBA array of shape (max_label_id + 1, 4) for direct index lookup.
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
    """
    Constructs a standardized ListedColormap with deterministic golden ratio hue spacing.
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
    """
    Constructs a deterministic 1-to-1 mapping from unique label IDs or region names to high-contrast discrete RGBA colors.
    """
    color_map, _ = build_dkt_label_palette(unique_labels)
    return color_map
