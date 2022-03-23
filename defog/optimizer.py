""" optimizer.py - Module for computing regularized depth map via energy minimization

Author: Utkarsh Patel (18EC35034)
This module is part of IP lab final project
"""

import numpy as np

def depth_map_delta(depth_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes absolute gradient of depth map
    along horizontal and vertical direction

    (Refer to pg. 5, paragraph just before eqn. 15)

    :param depth_map: depth map (two-dimensional)
    """
    delta_h = np.abs(np.gradient(depth_map, axis=0))
    delta_v = np.abs(np.gradient(depth_map, axis=1))
    return delta_h, delta_v

def line_field(delta_h: np.ndarray,
               delta_v: np.ndarray,
               threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Computes horizontal and vertical component of line field

    (Refer to pg. 5, eqn. 15)

    :param delta_h: absolute gradient of depth map along horizontal direction (two-dimensional)
    :param delta_v: absolute gradient of depth map along vertical direction (two-dimensional)
    :param threshold: threshold in the eqn.
    """
    b_h = 1 - (delta_h - threshold >= 0).astype(float)
    b_v = 1 - (delta_v - threshold >= 0).astype(float)
    return b_h, b_v

def base_potential(delta_h: np.ndarray,
                   delta_v: np.ndarray,
                   v_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Computes horizontal and vertical component of base potential

    (Refer to pg. 5, eqn. 15)

    :param delta_h: absolute gradient of depth map along horizontal direction (two-dimensional)
    :param delta_v: absolute gradient of depth map along vertical direction (two-dimensional)
    :param v_max: upper bound for edge-preserving potential
    """
    phi_h = np.min(delta_h, v_max)
    phi_v = np.min(delta_v, v_max)
    return phi_h, phi_v

def noise_variance(prior_maps: np.ndarray,
                   depth_map: np.ndarray) -> np.ndarray:
    """Estimates noise variance using maximum likelihood criterion

    (Refer to pg.6, eqn. 17)

    :param prior_maps: collection of all prior maps (three-dimensional)
    :param depth_map: depth map (two-dimensional)
    """
    variance = (prior_maps - depth_map) / (depth_map.shape[0] * depth_map.shape[1])
    variance = np.linalg.norm(variance, axis=(1, 2))
    return variance

def energy(depth_map: np.ndarray,
           prior_maps: np.ndarray,
           lambda_: float,
           threshold: float,
           v_max: float) -> float:
    """Computes energy for given prior maps and depth map

    (Refer to pg. 6, eqn. 16)

    :param depth_map: depth map (two-dimensional)
    :param prior_maps: collection of all prior maps (three-dimensional)
    :param lambda_: regularization term
    :param threshold: `T` in pg. 5, eqn. 15
    :param v_max: upper bound for edge-preserving potential
    """
    diff = prior_maps - depth_map                    # (pi - D)
    diff_t = np.array([x.transpose() for x in diff]) # (pi - D).T
    variance = noise_variance(prior_maps, depth_map)
    delta_h, delta_v = depth_map_delta(depth_map)
    b_h, b_v = line_field(delta_h, delta_v, threshold)
    phi_h, phi_v = base_potential(delta_h, delta_v, v_max)
    prior_term = ((diff_t @ diff) / variance).sum()
    smoothening_term = (b_h * phi_h + b_v * phi_v).sum() * lambda_
    return prior_term + smoothening_term
