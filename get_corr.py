import lab3
import numpy as np
import matplotlib.pyplot as plt

def get_corr(img_0, img_1, points_0, points_1, ROI_size = 15):
    roi_0 = lab3.cut_out_rois(img_0, points_0[1], points_0[0], ROI_size)
    roi_1 = lab3.cut_out_rois(img_1, points_1[1], points_1[0], ROI_size)

    # construct match matrix
    match_mat = np.zeros((len(roi_0),len(roi_1)))
    for i in range(len(roi_0)):
        for j in range(len(roi_1)):
            match_mat[i,j] = np.sum(np.square(roi_0[i] - roi_1[j]))

    # find minimum
    _, pt_index_0, pt_index_1 = lab3.joint_min(match_mat)

    # construct output
    p0 = points_0[:,pt_index_0]
    p1 = points_1[:,pt_index_1]

    return p0, p1
