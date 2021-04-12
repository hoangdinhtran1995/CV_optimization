import lab3
import numpy as np
def get_interest_points(img, block_size=10, kernel_size=3, suppression_window_size = 17, threshold = 0.01):
    # harris
    harris = lab3.harris(img, block_size, kernel_size)

    # non-max suppression
    harris = lab3.non_max_suppression(harris,suppression_window_size)

    # normalize maxes and threshold
    harris /= np.max(harris)
    points = np.zeros_like(harris)
    points[harris > threshold] = 1

    # get coords for interest points
    pt_coords = np.nonzero(points)

    return pt_coords
