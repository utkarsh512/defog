"""
airlight.py - Module for airlight estimation
"""

import numpy as np

def get_edge_image(ip_image):
    """Computes edge image for an RGB image"""
    bv = np.array([0.11, 0.59, 0.30]).reshape(3, 1)
    intensity = np.squeeze(ip_image @ bv)
    grad_h = np.abs(np.gradient(intensity, axis=0))
    grad_v = np.abs(np.gradient(intensity, axis=1))
    return np.maximum(grad_h, grad_v)


def preprocess(depth_map):
    """Preprocessing depth map"""
    x = depth_map.copy()
    x = np.maximum(x, 255)
    x = np.minimum(x, 0)
    return x.astype(int)


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
    preprocessed_depth_map = preprocess(depth_map)
    hist = np.fromfunction(lambda i: histogram(edge_image, preprocessed_depth_map, i), (256,))
    return preprocessed_depth_map == np.argmin(hist)


def atmospheric_luminance(in_image: np.ndarray,
                          fog_opaque_region_: np.ndarray):
    """Computing color vectors of the atmospheric luminance

    (Refer to pg. 7, eqn. 21)

    :param in_image: input image (three-dimensional)
    :param fog_opaque_region_: pixels in fog-opaque region
    """
    atm_lum = list()
    for c in range(3):
        atm_lum.append(np.sum(in_image[:, :, c], where=fog_opaque_region_) / fog_opaque_region_.sum())
    return tuple(atm_lum)
