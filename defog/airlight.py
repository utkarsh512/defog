"""
airlight.py - Module for airlight estimation
"""

import numpy as np
from .constants import MAX_INTENSITY_LEVEL

def histogram(edge_image: np.ndarray,
              depth_map: np.ndarray,
              depth_label: float) -> float:
    """Computes the histogram of smoothness wrt given depth label

    (Refer to pg. 7, eqn. 23)

    :param edge_image: edge image of input image (two-dimensional)
    :param depth_map: estimated depth map (two-dimensional)
    :param depth_label: depth label
    """
    hist = (edge_image * (depth_map == depth_label)).sum()
    return hist

def fog_opaque_region(edge_image: np.ndarray,
                      depth_map: np.ndarray) -> np.ndarray:
    """Generating fog-opaque regions

    (Refer to pg. 7, eqn. 22)

    :param edge_image: edge image of input image (two-dimensional)
    :param depth_map: estimated depth map (two-dimensional)
    """
    hist = np.fromfunction(lambda i: histogram(edge_image, depth_map, i), (MAX_INTENSITY_LEVEL,))
    return depth_map == np.argmin(hist)

def atmospheric_luminance(in_image: np.ndarray,
                          fog_opaque_region_: np.ndarray) -> tuple[float, float, float]:
    """Computing color vectors of the atmospheric luminance

    (Refer to pg. 7, eqn. 21)

    :param in_image: input image (three-dimensional)
    :param fog_opaque_region_: pixels in fog-opaque region
    """
    return tuple(np.sum(in_image, axis=(1, 2), where=fog_opaque_region_) / fog_opaque_region_.sum())

