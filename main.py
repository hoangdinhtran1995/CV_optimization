# imports
import scipy
import numpy as np
import cv2
import lab3
import matplotlib.pyplot as plt
from get_interest_points import get_interest_points

# load and show
img = lab3.load_stereo_pair()
img_0 = img[0]
img_1 = img[1]
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
pt_coord_0 = get_interest_points(img_0)
pt_coord_1 = get_interest_points(img_1)

"""
2. Putative correspondences
"""
roi_0 = lab3.cut_out_rois(img[0], pt_coord_0[1], pt_coord_0[0],15)
roi_1 = lab3.cut_out_rois(img[1], pt_coord_1[1], pt_coord_1[0],15)

#

#### testing area ####

diff_sq = np.square(roi_0[20] - roi_1[20])
print(np.sum(diff_sq))
plt.figure('diff')
plt.imshow(diff_sq)


# plt.figure(0)
# plt.imshow(harris_0)
# plt.figure(1)
# plt.imshow(harris_1)
# plt.figure(2)
# plt.imshow(points_0)
# plt.figure(3)
# plt.imshow(points_1)
plt.figure('roi0')
plt.imshow(roi_0[20])
plt.figure('roi1')
plt.imshow(roi_1[20])
plt.show()


