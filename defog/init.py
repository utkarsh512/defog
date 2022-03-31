"""
init.py - initializing depth map close to the solution for faster convergence
"""

import numpy as np
from copy import deepcopy

prior_map_cache = None


def isvalid(x: int, y: int, w: int, h: int):
    """Check whether (x, y) lies inside (w, h)"""
    return x >= 0 and x < w and y >= 0 and y < h


def minimum_neighbor(image, nbr_size):
    """Finds the minimum element in k-th neighborhood of all pixel"""
    dx = (1, 0, -1, 0)
    dy = (0, 1, 0, -1)
    cur = deepcopy(image)
    w = len(cur)
    h = len(cur[0])
    for _ in range(nbr_size):
        nxt = deepcopy(cur)
        for x in range(w):
            for y in range(h):
                for k in range(4):
                    i, j = x + dx[k], y + dy[k]
                    if isvalid(i, j, w, h):
                        nxt[x][y] = min(nxt[x][y], cur[i][j])
        cur = nxt
    return cur


def gen_prior_map(ip_image: np.ndarray, nbr_size: int) -> np.ndarray:
    """Generates prior map using non-linear filtering on input image for a given patch size"""
    channel_B = np.array(minimum_neighbor(ip_image[:, :, 0].tolist(), nbr_size))
    channel_G = np.array(minimum_neighbor(ip_image[:, :, 1].tolist(), nbr_size))
    channel_R = np.array(minimum_neighbor(ip_image[:, :, 2].tolist(), nbr_size))
    return np.min(np.min(channel_B, channel_G), channel_R)


def gen_prior_maps(ip_image: np.ndarray, nbr_sizes: list) -> np.ndarray:
    """Generates all prior maps for given collection of patch sizes"""
    prior_maps = list()
    for nbr_size in nbr_sizes:
        prior_maps.append(gen_prior_map(ip_image, nbr_size))
    return np.array(prior_maps)


def get_neighborhood(x, y, nbr_size, h, w):
    """Generates neighborhood for given manhattan distance constraint"""
    nbrs = list()
    start_x, start_y = x, y - nbr_size
    length = nbr_size * 2 + 1
    while length > 0:
        for k in range(length):
            i, j = start_x, start_y + k
            if isvalid(i, j, h, w):
                nbrs.append((i, j))
        length -= 2
        start_x += 1
        start_y += 1
    start_x, start_y = x, y - nbr_size
    start_x -= 1
    start_y += 1
    length = nbr_size * 2 - 1
    while length > 0:
        for k in range(length):
            i, j = start_x, start_y + k
            if isvalid(i, j, h, w):
                nbrs.append((i, j))
        length -= 2
        start_x -= 1
        start_y += 1
    return nbrs


def estimate_variance(ip_image: np.ndarray, x: int, y: int, nbr_size: int) -> float:
    """Estimates local variances as described in pg. 6, eqn. 20"""
    nbrs = get_neighborhood(x, y, nbr_size, ip_image.shape[0], ip_image.shape[1])
    vars = list()
    for channel in range(3):
        pixel_avg = 0
        for i, j in nbrs:
            pixel_avg += ip_image[i, j, channel]
        pixel_avg /= len(nbrs)
        pixel_var = 0
        for i, j in nbrs:
            pixel_var += (ip_image[i, j, channel] - pixel_avg) * (ip_image[i, j, channel] - pixel_avg)
        pixel_var /= len(nbrs) - 1
        vars.append(pixel_var)
    return np.average(vars)


def get_weights(ip_image: np.ndarray, x: int, y: int, nbr_sizes: list):
    """Computes W(x) - pg. 6, eqn. 20"""
    A = list()
    for nbr_size in nbr_sizes:
        A.append(1.0 / estimate_variance(ip_image, x, y, nbr_size))
    A = np.array(A)
    A = A.reshape(1, len(nbr_sizes))
    return np.linalg.inv(A.T @ A) @ A.T


def get_prior_maps(ip_image: np.ndarray, x: int, y: int, nbr_sizes: list):
    """Computes P(x) - pg. 6, eqn. 20"""
    global prior_map_cache
    if prior_map_cache is None:
        prior_map_cache = gen_prior_maps(ip_image, nbr_sizes)
    P = prior_map_cache[:, x, y]
    P = P.reshape(len(nbr_sizes), 1)
    return P


def get_depth_map(ip_image: np.ndarray, x: int, y: int, nbr_sizes: list):
    """Computes D(x) - pg. 6, eqn. 20"""
    W = get_weights(ip_image, x, y, nbr_sizes)
    P = get_prior_maps(ip_image, x, y, nbr_sizes)
    return W @ P


def init_depth_map(ip_image: np.ndarray, nbr_sizes: list):
    """Computes initial value for depth map"""
    depth_map = [[0 for j in range(ip_image.shape[1])] for i in range(ip_image.shape[0])]
    for i in range(ip_image.shape[0]):
        for j in range(ip_image.shape[1]):
            depth_image[i][j] = get_depth_map(ip_image, i, j, nbr_sizes)
    depth_map = np.array(depth_map)
    return depth_map
    #return np.fromfunction(lambda i, j: get_depth_map(ip_image, i, j, nbr_sizes), (ip_image.shape[0], ip_image.shape[1]))
