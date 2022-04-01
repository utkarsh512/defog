"""
optimizer.py - Module for computing regularized depth map via energy minimization
"""

import numpy as np
from .graph import expansion_move

def depth_map_delta(depth_map: np.ndarray):
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
               threshold: float):
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
                   v_max: float):
    """Computes horizontal and vertical component of base potential

    (Refer to pg. 5, eqn. 15)

    :param delta_h: absolute gradient of depth map along horizontal direction (two-dimensional)
    :param delta_v: absolute gradient of depth map along vertical direction (two-dimensional)
    :param v_max: upper bound for edge-preserving potential
    """
    phi_h = np.minimum(delta_h, v_max)
    phi_v = np.minimum(delta_v, v_max)
    return phi_h, phi_v


def noise_variance(prior_maps: np.ndarray,
                   depth_map: np.ndarray) -> np.ndarray:
    """Estimates noise variance using maximum likelihood criterion

    (Refer to pg.6, eqn. 17)

    :param prior_maps: collection of all prior maps (three-dimensional)
    :param depth_map: depth map (two-dimensional)
    """
    vars = list()
    print(f"P:{prior_maps.shape}, D:{depth_map.shape}")
    for i in range(prior_maps.shape[0]):
        vars.append(np.linalg.norm((prior_maps[i, :, :] - depth_map) / (depth_map.shape[0] * depth_map.shape[1])))
    return np.array(vars)


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


def converged(cur_val: np.ndarray,
              prv_val: np.ndarray,
              tolerance: float) -> bool:
    diff = cur_val - prv_val
    norm = np.linalg.norm(diff)
    return norm < tolerance


def optimize(depth_map: np.ndarray,
             prior_maps: np.ndarray,
             max_iter: int,
             tolerance: float,
             lambda_: float,
             threshold: float,
             v_max: float) -> np.ndarray:
    """Computing a labeling extremely close to global solution
    via expansion-move algorithm

    :param depth_map: depth map (two-dimensional)
    :param prior_maps: collection of all prior maps (three-dimensional)
    :param max_iter: maximum number of iterations to run
    :param tolerance: error until convergence
    :param lambda_: regularization term (Refer to pg. 6, eqn. 16)
    :param threshold: `T` in pg. 5, eqn, 15
    :param v_max: upper bound for edge-preserving potential
    """
    cur_depth_map = depth_map.copy()
    prv_depth_map = depth_map.copy()
    it = 0
    while max_iter > 0 and not converged(cur_depth_map, prv_depth_map, tolerance):
        prv_depth_map = cur_depth_map.copy()
        variance = noise_variance(prior_maps, prv_depth_map)
        cur_depth_map = expansion_move(prv_depth_map,
                                       cur_depth_map,
                                       prior_maps,
                                       it + 1,
                                       lambda_,
                                       threshold,
                                       v_max,
                                       variance)

        energy_val = energy(cur_depth_map, prior_maps, lambda_,
                            threshold, v_max)
        energy_val = np.around(energy_val, decimals=4)
        print(f'[*] iter = {it + 1:7}, energy = {energy_val}')
        max_iter -= 1
        it += 1
    return cur_depth_map
