# imports
import scipy
import numpy as np
import cv2
import lab3
import matplotlib.pyplot as plt
import PIL
from pathlib import Path

from get_interest_points import get_interest_points
from get_corr import get_corr
from ransac import ransac


# load data
img = lab3.load_stereo_pair()
img_0 = img[0]
img_1 = img[1]

"""
Compute the fundamental matrix between two images
"""

"""
1. Interest points
"""
pt_coord_0 = get_interest_points(img_0, block_size=10, kernel_size=5, suppression_window_size = 15, threshold = 0.005)
pt_coord_1 = get_interest_points(img_1, block_size=10, kernel_size=5, suppression_window_size = 15, threshold = 0.005)
"""
2. Putative correspondences
"""
p0, p1 = get_corr(img_0,img_1,pt_coord_0,pt_coord_1) #[x,y]

"""
3. RANSAC
"""
F, pl, pr = ransac(p0, p1, N = 2000 , t = 5)

"""
Visualize epipolar lines
"""
lab3.show_corresp(img_0,img_1,p0,p1)
lab3.show_corresp(img_0,img_1,pl,pr)

plt.figure('epipolar lines 0')
plt.imshow(img_0)
lab3.plot_eplines(F, p1, img_0.shape)
plt.plot(pl[0], pl[1], 'o')
plt.figure('epipolar lines 1')
plt.imshow(img_1)
lab3.plot_eplines(F.T, p0, img_1.shape)
plt.plot(pr[0], pr[1], 'o')


#### testing area ####
plt.show()


