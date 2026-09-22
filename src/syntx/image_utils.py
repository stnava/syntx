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

    Directly uses ANTsPy's ``ants.reflect_image`` implementation with support
    for named physical axes ('x', 'y', 'z', 'LR', 'AP', 'SI').

    Parameters
    ----------
    image : ants.ANTsImage
        Image to reflect.
    axis : int or str
        Axis to reflect across. Can be integer (0 to dim-1), or
        'x'/'X'/'LR' (0), 'y'/'Y'/'AP' (1), 'z'/'Z'/'SI' (2).
        Default: 0.
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
