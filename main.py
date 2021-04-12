# imports
import scipy
import cv2
import lab3
import matplotlib.pyplot as plt

# load and show
img = lab3.load_stereo_pair()
# plt.figure(0)
# plt.imshow(img[0])
# plt.figure(1)
# plt.imshow(img[1])
# plt.show()

"""
Compute the fundamental matrix between two images
"""

"""
1. Interest points
"""
# harris
block_size = 10
kernel_size = 3
harris_0 = lab3.harris(img[0], block_size, kernel_size)
harris_1 = lab3.harris(img[1], block_size, kernel_size)

# non-max suppression
suppression_window_size = 5
harris_0 = lab3.non_max_suppression(harris_0,suppression_window_size)
harris_1 = lab3.non_max_suppression(harris_1,suppression_window_size)

# plt.figure(0)
# plt.imshow(harris_0)
# plt.figure(1)
# plt.imshow(harris_1)
# plt.show()


