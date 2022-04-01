"""
defog - single image defogging by multiscale depth fusion
"""

import sys
import os
import argparse
import cv2
import numpy as np

from . import init
from . import optimizer
from . import airlight


def error(_error, message):
    """
    Print errors to stdout
    """
    print("[-] {}: {}".format(_error, message))
    sys.exit(0)


def normalize(v):
    """
    Normalize for saving as image file
    """
    v_max = np.max(v)
    v = np.divide(v, v_max)
    v *= 255
    v = v.astype(int)
    return v


def options():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(prog="defog",
                                     usage="python3 %(prog)s [options]",
                                     description="defog - single image defogging by multiscale depth fusion")
    parser.add_argument("-i", "--input", type=str, required=True, help="path to input image")
    parser.add_argument("-o", "--output", type=str, required=True, help="path for output image")
    parser.add_argument("-p", "--patches", nargs="+", type=int, default=[4, 25], help="size of patches")
    parser.add_argument("-r", "--max_iter", type=int, required=True, help="maximum iteration to run")
    parser.add_argument("-t", "--threshold", type=int, default=128, help="`T` in pg. 5, eqn. 15")
    parser.add_argument("-l", "--lambda_", type=float, required=True, help="regularization factor")
    parser.add_argument("-v", "--v_max", type=float, required=True, help="upper bound on potential")
    args = parser.parse_args()
    return args


def main():
    """
    main
    """
    args = options()
    print(f"[*] Reading image from {args.input}")
    ip_image = cv2.imread(args.input)
    if ip_image is None:
        error('CV::IMREAD', 'Unable to read image file!')
    print(f"[DONE] Input image shape = {ip_image.shape}.")
    print(f"[*] Initializing depth map...")
    depth_map = init.init_depth_map(ip_image, args.patches)
    print(f"[DONE] Depth map initialized.")
    print(f"[*] Optimizing depth map...")
    depth_map = optimizer.optimize(depth_map=depth_map,
                                   prior_maps=init.prior_map_cache,
                                   max_iter=args.max_iter,
                                   lambda_=args.lambda_,
                                   threshold=args.threshold,
                                   v_max=args.v_max)
    print(f"[DONE] Depth map optimized.")
    depth_map = normalize(depth_map)
    print(f"[*] Estimating airlight...")
    edge_image = airlight.get_edge_image(ip_image)
    fog_region = airlight.fog_opaque_region(edge_image, depth_map)
    atm_lum = airlight.atmospheric_luminance(ip_image, fog_region)
    atm_image = ip_image.copy()
    h, w, c = ip_image.shape
    for i in range(h):
        for j in range(w):
            for k in range(c):
                atm_image[i, j, k] = (255 - depth_map[i, j]) * atm_lum[k]
    print(f"[DONE] Airlight estimation done.")
    print(f"[*] Writing output image...")
    op_image = ip_image - atm_image
    for k in range(c):
        op_image[:, :, k] = np.divide(op_image[:, :, k], depth_map)
    op_image = normalize(op_image)
    op_image = airlight.preprocess(op_image)
    cv2.imwrite(args.output, op_image)
    print(f"[DONE] Output image written at {args.output}.")


def run_as_command():
    version = ".".join(str(v) for v in sys.version_info[:2])
    if float(version) < 3.6:
        print("[-] defog requires Python version 3.6+.")
        sys.exit(0)
    main()

if __name__ == '__main__':
    main()
