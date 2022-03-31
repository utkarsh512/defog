"""
graph.py - using min-cut max-flow algorithm to minimize energy
"""

import maxflow as mf
import numpy as np
import time
import sys
import os.path
from random import shuffle
from PIL import Image


def image_to_array(img):
    '''input: path to image
       output: array of grayscale
       reference: https://stackoverflow.com/questions/40727793/how-to-convert-a-grayscale-image-into-a-list-of-pixel-values
       '''
    # from PIL import Image
    img = Image.open(img).convert('L')
    w, h = img.size

    data = list(img.getdata())
    data = [data[off:off + w] for off in range(0, w * h, w)]

    return data


def arr_to_image(img_work, fname):
    '''Saves image arr as image'''
    temp = np.asarray(img_work)
    im = Image.fromarray(temp);  # https://stackoverflow.com/questions/33658709/convert-an-array-into-image
    # img_work was originally a list
    im.save(fname)
    return 0


def calculate_energy(img_work, prior_maps: np.ndarray, lambda_: float, threshold: float, v_max: float,
                     sigma: np.ndarray):
    '''Calculates Energy of image.
       img: is input array'''

    E_data = 0
    for i in range(len(img_work)):
        for j in range(len(img_work[0])):
            E_data += D_p(img_work[i][j], prior_maps, j, i, sigma)

    E_smooth = 0
    for i in range(len(img_work)):
        for j in range(len(img_work[0])):
            ns = give_neighbours(img_work, j, i)
            E_smooth += sum([V_p_q(v, img_work[i][j], lambda_, threshold, v_max) for v in ns])

    return E_data + E_smooth


def V_p_q(label1, label2, lambda_: float, threshold: float, v_max: float) -> float:
    # Returns V_p_q when we calculate between just D(levels), and not the neighbourhood
    # V_p_q is calculated differently for at each point and between edges
    # value = lambda_*min((label1 - label2),v_max)*(1 - (abs(label1 - label2) - threshold >= 0).astype(float))
    value = lambda_ * min((label1 - label2), v_max) * (1 - (abs(label1 - label2) - threshold >= 0))
    return value


def D_p(label, prior_maps: np.ndarray, x, y, sigma: np.ndarray) -> float:
    '''Returns the quadratic difference between label and real intensity of pixel as defined in Eq 16'''
    # prior_maps is a 3D block
    # sigma is equal to depth of prior_maps
    # return (abs(label**2-graph[y][x]**2))*sigma**2
    value = (abs(label ** 2 - prior_maps[:, y, x] ** 2)) * sigma ** 2
    return value.sum()


def give_neighbours(image, x, y):
    '''Returns a list of all neighbour intensities'''
    if x >= len(image[0]) or x < 0 or y >= len(image) or y < 0:
        raise ValueError('Pixel is not in image. x and/or y are to large')
    ns = []
    for a, b in zip([1, 0, -1, 0], [0, 1, 0, -1]):
        if (x + a < len(image[0]) and x + a >= 0) and (y + b < len(image) and y + b >= 0):
            ns.append(image[y + b][x + a])
    return ns


def return_mapping_of_image(image, alpha, beta):
    # map does the position in graph map to (y,x) position in image
    map = {}
    # other way
    revmap = {}
    # loop over all pixels and add them to maps
    map_parameter = 0
    for y in range(len(image)):
        for x in range(len(image[0])):
            # extract pixel which have the wanted label
            if image[y][x] == alpha or image[y][x] == beta:
                map[map_parameter] = (y, x)
                revmap[(y, x)] = map_parameter
                map_parameter += 1

    return map, revmap


def alpha_beta_swap_new(alpha, beta, img_work, prior_maps: np.ndarray, lambda_: float, threshold: float, v_max: float,
                        sigma: np.ndarray):
    ''' Performs alpha-beta-swap
        img_orig: input image
        img_work: denoised image in each step
        time_measure: flag if you want measure time
        prior_maps: Prior maps block'''

    # extract position of alpha or beta pixels to mapping
    map, revmap = return_mapping_of_image(img_work, alpha, beta)

    # graph of maxflow
    graph_mf = mf.Graph[float](len(map))
    # add nodes
    nodes = graph_mf.add_nodes(len(map))

    # add n-link edges
    weight = V_p_q(alpha, beta, lambda_, threshold, v_max)
    # Have to modify V_p_q to get depth map

    for i in range(0, len(map)):
        y, x = map[i]
        # top, left, bottom, right
        for a, b in zip([1, 0, -1, 0], [0, 1, 0, -1]):
            if (y + b, x + a) in revmap:
                graph_mf.add_edge(i, revmap[(y + b, x + a)], weight, 0)

    # add all the terminal edges
    for i in range(0, len(map)):
        y, x = map[i]
        # find neighbours
        neighbours = give_neighbours(img_work, x, y)
        # consider only neighbours which are not having alpha or beta label
        fil_neigh = list(filter(lambda i: i != alpha and i != beta, neighbours))
        # calculation of weight
        t_weight_alpha = sum([V_p_q(alpha, v, lambda_, threshold, v_max) for v in fil_neigh]) + D_p(alpha, prior_maps,
                                                                                                    x, y, sigma)
        t_weight_beta = sum([V_p_q(beta, v, lambda_, threshold, v_max) for v in fil_neigh]) + D_p(beta, prior_maps, x,
                                                                                                  y, sigma)
        graph_mf.add_tedge(nodes[i], t_weight_alpha, t_weight_beta)

    # calculating flow
    flow = graph_mf.maxflow()
    res = [graph_mf.get_segment(nodes[i]) for i in range(0, len(nodes))]

    # depending on cut assign new label
    for i in range(0, len(res)):
        y, x = map[i]
        if res[i] == 1:
            img_work[y][x] = alpha
        else:
            img_work[y][x] = beta

    return img_work


def expansion_move(img_orig, img_work, prior_maps: np.ndarray, cycles, lambda_: float, threshold: float, v_max: float,
                   sigma: np.ndarray):
    '''This methods implements the energy minimization via alpha-beta-swaps
       img_orig: is original input image (Depth map)
       img_work: optimized image (Depth map)
       prior_maps: prior_maps block (3D)
       lambda: weight of smoothing term (eqn 15)
       threshold: T (eqn 15)
       v_max: (eqn 15), maximum value of V in the graph.
       sigma: (eqn 16), prior_maps' sigma values, 1D array equal to depth of prior_maps (eqn 16)
       cycles: how often to iterate over all labels'''

    import time
    # find all labels of image
    start = time.time()
    labels = []
    for i in range(0, len(img_orig)):
        for j in range(0, len(img_orig[0])):
            if img_orig[i][j] not in labels:
                labels.append(img_orig[i][j])  # The labels are being created according to the intensity value
    labels = np.array(labels)
    stop = time.time()
    print(stop - start)
    T = 0
    # do iteration of all pairs a few times
    for u in range(0, cycles):
        # shuffle(labels)
        # iterate over all pairs of labels
        for i in range(0, len(labels) - 1):
            for j in range(i + 1, len(labels)):
                # computing intensive swapping and graph cutting part
                # img_work  = alpha_beta_swap_new(labels[i],labels[j], img_orig, img_work)
                img_work = alpha_beta_swap_new(labels[i], labels[j], img_work, prior_maps, lambda_, threshold, v_max,
                                               sigma)
                # user output and interims result image
        print(str(u + 1) + "\t\t\t", calculate_energy(img_work, prior_maps, lambda_, threshold, v_max, sigma))
        # print("Energy after " + str(u+1) + "/" + str(cycles) + " cylces:", calculate_energy(img_orig, img_work))
        # arr_to_image(img_work, "bad_denoised_"+output_name+"_"+str(u+1)+"_cycle"+".png")

    return img_work
