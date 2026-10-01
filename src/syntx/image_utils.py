"""
syntx.image_utils — General-purpose ANTsImage manipulation utilities
====================================================================

Provides:
  reflect_image — reflect an image along a physical axis using ANTsPy's reflection
                  matrix, preserving exact physical space header metadata.
"""

from __future__ import annotations

from typing import Union, Optional, Any, Dict
import ants

__all__ = ["reflect_image"]


def reflect_image(
    image: ants.ANTsImage,
    axis: Union[int, str] = 0,
    tx: Optional[str] = None,
    metric: str = "mattes",
) -> Union[ants.ANTsImage, Dict[str, Any]]:
    """
    Reflect an image along an axis, preserving the exact image header metadata.

    A thin wrapper of ``ants.reflect_image`` that also accepts axis names.

    Parameters
    ----------
    image : ants.ANTsImage
        Image to reflect.
    axis : int or str, default 0
        Physical (LPS world) axis to reflect across, about the image's centre of mass:
        0 / 'x' / 'LR', 1 / 'y' / 'AP', 2 / 'z' / 'SI' (names in any case). The axis is a
        physical one (``ants.reflect_image`` builds the reflection in world coordinates), so
        'LR' is left-right whatever the array storage order or direction matrix.
    tx : str, optional
        Transformation type to estimate after reflection (e.g. 'Rigid', 'Affine').
        If None (default), returns reflected ANTsImage.
    metric : str
        Similarity metric if tx is specified. Default: 'mattes'.

    Returns
    -------
    ants.ANTsImage or dict
        Reflected ANTsImage with the same header (origin, spacing, direction)
        as input, or registration dict if tx is not None.
    """
    _alias = {
        'x': 0, 'X': 0, 'LR': 0, 'lr': 0,
        'y': 1, 'Y': 1, 'AP': 1, 'ap': 1,
        'z': 2, 'Z': 2, 'SI': 2, 'si': 2,
    }
    if isinstance(axis, str):
        if axis not in _alias:
            raise ValueError(f"Unknown axis string '{axis}'. Use 0/1/2, 'x'/'y'/'z', or 'LR'/'AP'/'SI'.")
        axis_int = _alias[axis]
    else:
        axis_int = int(axis)

    return ants.reflect_image(image, axis=axis_int, tx=tx, metric=metric)
