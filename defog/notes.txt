NOTES ON VARIABLES AND THEIR DIMENSIONS
=======================================

Assumptions
-----------

For the notes we assume the following are true:
    * The dimension of the input image is (N, M, 3). Here, 3 denotes that there are three channels (BGR).
    * There are a total of m prior maps known


Caution
-------

* In the implementation, it is assumed that `dtype` of `depth_map` is `int`, not `float`.


Variables
---------

ip_image
    Variable `ip_image` is used to represent the input image as a 3-D numpy array with dimensions (N, M, 3)

edge_image
    Variable `edge_image` is used to represent the edge image of the input image.
    It is a 2-D numpy array with dimensions (N, M)

op_image
    Variable `op_image` is used to represent the image obtained by using the algorithm.
    It is a 3-D numpy array with dimensions (N, M, 3)

prior_maps
    Variable `prior_maps` is used to represent collection of all the prior maps p_i(x), i = 1, 2, ..., m.
    Since each prior map is a 2-D array with dimensions (N, M), this results in variable `prior_maps` being a
    3-D numpy array with dimensions (m, N, M).
    Each p_i(x) can be accessed as `prior_maps[i, :, :]`

depth_map
    Variable `depth_map` is used to represent the depth map we are estimating via the algorithm.
    It is a 2-D numpy array with dimensions (N, M).

b_h, b_v
    Variables `b_h` and `b_v` are used to represent the horizontal and vertical component of line field.
    Both of them are 2-D numpy arrays with dimensions (N, M).

phi_h, phi_v
    Variable `phi_h` and 'phi_v' are used to represent the horizontal and vertical component of base potential.
    Both of them are 2-D numpy arrays with dimensions (N, M).

delta_h, delta_v
    Variable `delta_h` and `delta_v` are used to represent the absolute gradient of depth map in horizontal and vertical
    direction respectively.
    Both of them are 2-D numpy arrays with dimesions (N, M).

