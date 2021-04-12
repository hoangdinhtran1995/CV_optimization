import lab3
import random
import numpy as np

def ransac(p0, p1, N, t = 1):
    points = len(p0[0])
    cur_max_inliers = 0
    cur_F = None
    for _ in range(N):
        rand_points = random.sample(range(points),8)
        pl = p0[:,rand_points]
        pr = p1[:,rand_points]

        F = lab3.fmatrix_stls(pl, pr)

        residuals = lab3.fmatrix_residuals(F, p0, p1)
        inliers = np.zeros_like(residuals)
        inliers[residuals < t] = 1

        if np.sum(inliers) > cur_max_inliers:
            cur_max_inliers = np.sum(inliers)
            cur_F = F

    return F