"""
syntx.viz.colormaps — Standardized Perceptually Uniform Categorical Colormaps
=============================================================================

Provides the standardized `dkt_colormap` for anatomical segmentations (Mindboggle DKT labels).
Colors are constructed using golden ratio hue sampling in perceptually uniform HSL space
(constant lightness L=0.68, saturation S=0.85) to maximize visual contrast between all 
adjacent and nearby integer region IDs.
"""

from typing import Dict, List, Tuple, Union
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def get_dkt_colormap(max_label: int = 256, lightness: float = 0.68, saturation: float = 0.85) -> mcolors.ListedColormap:
    """
    Constructs a standardized categorical colormap for DKT labels with colors
    equally spaced in perceptually uniform color space (golden ratio hue spacing)
    to maximize visual distinctiveness across anatomical brain regions.
    """
    golden_ratio = 0.618033988749895
    colors = [(0.0, 0.0, 0.0, 0.0)]  # Label 0 = transparent background

    h = 0.125
    for i in range(1, max_label + 1):
        h = (h + golden_ratio) % 1.0
        rgb = mcolors.hsv_to_rgb((h, saturation, lightness))
        colors.append((*rgb, 0.90))

    cmap = mcolors.ListedColormap(colors, name="dkt_colormap")
    try:
        if hasattr(plt, 'colormaps'):
            plt.colormaps.register(cmap, force=True)
        else:
            plt.cm.register_cmap(name="dkt_colormap", cmap=cmap)
    except Exception:
        pass

    return cmap


# Standardized singleton instance exported across syntx.viz
dkt_colormap = get_dkt_colormap()


def get_dkt_label_color_dict(unique_labels: List[Union[int, str]]) -> Dict[Union[int, str], Tuple[float, ...]]:
    """
    Constructs a deterministic mapping from unique label IDs or region names to high-contrast discrete RGBA colors.
    """
    cmap = get_dkt_colormap()
    color_map = {}
    for idx, l in enumerate(unique_labels):
        try:
            val = int(l)
            if 0 < val < len(cmap.colors):
                color_map[val] = cmap.colors[val]
            else:
                color_map[str(l)] = cmap.colors[(idx + 1) % len(cmap.colors)]
        except Exception:
            color_map[str(l)] = cmap.colors[(idx + 1) % len(cmap.colors)]

    return color_map
