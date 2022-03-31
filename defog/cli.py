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


def options():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(prog="defog",
                                     usage="python3 %(prog)s [options]",
                                     description="defog - single image defogging by multiscale depth fusion")
    parser.add_argument("-i", "--image", type=str, required=True, help="input image")
    parser.add_argument("-p", "--patches", nargs="+", type=int, default=[4, 25], help="size of patches")
    parser.add_argument("-r", "--max_iter", type=int, required=True, help="maximum iteration to run")
    parser.add_argument("-t", "--threshold", type=int, default=128, help="`T` in pg. 5, eqn. 15")
    parser.add_argument("-l", "--lambda_", type=float, required=True, help="regularization factor")
    parser.add_argument("-v", "--v_max", type=float, required=True, help="upper bound on potential")
    parser.add_argument("-e", "--tolerance", type=float, default=0.01, help="error upto which iterations continue")
    args = parser.parse_args()
    return args

def main():
    """
    main
    """
    args = options()
    ip_image = cv2.imread(args.image)
    if ip_image is None:
        error('CV::IMREAD', 'Unable to read image file!')
    depth_map = init.init_depth_map(ip_image, args.patches)
    depth_map = optimizer.optimize(depth_map=depth_map,
                                   prior_maps=init.prior_map_cache,
                                   max_iter=args.max_iter,
                                   tolerance=args.tolerance,
                                   lambda_=args.lambda_,
                                   threshold=args.threshold,
                                   v_max=args.v_max)
    edge_image = airlight.get_edge_image(ip_image)
    fog_region = airlight.fog_opaque_region(edge_image, depth_map)
    atm_lum = airlight.atmospheric_luminance(ip_image, fog_region)
    atm_image = np.fromfunction(lambda i, j, c: (255 - depth_map[i, j]) * atm_lum[c], ip_image.shape)
    op_image = ip_image - atm_image
    op_image /= depth_map
    op_image *= 255
    op_image = airlight.preprocess(op_image)
    cv2.imshow('Input image', ip_image)
    cv2.imshow('Output image', op_image)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image

def run_as_command():
    version = ".".join(str(v) for v in sys.version_info[:2])
    if float(version) < 3.6:
        print("[-] TWINT requires Python version 3.6+.")
        sys.exit(0)

    main()

if __name__ == '__main__':
    main()
