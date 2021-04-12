# imports
import scipy
import numpy as np
import cv2
import lab3
import matplotlib.pyplot as plt
from get_interest_points import get_interest_points
from get_corr import get_corr

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
p0, p1 = get_corr(img_0,img_1,pt_coord_0,pt_coord_1) #[x,y]


#

#### testing area ####



# plt.figure(0)
# plt.imshow(harris_0)
# plt.figure(1)
# plt.imshow(harris_1)
# plt.figure(2)
# plt.imshow(points_0)
# plt.figure(3)
# plt.imshow(points_1)
lab3.show_corresp(img_0,img_1,p0,p1)
plt.show()


