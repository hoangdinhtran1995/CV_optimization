import lab3
import random
import numpy as np

def ransac(p0, p1, N, t = 10):
    points = len(p0[0])
    cur_max_inliers = 0
    cur_F = None
    cur_pl = None
    cur_pr = None
    for _ in range(N):
        rand_points = random.sample(range(points),8)
        pl = p0[:,rand_points]
        pr = p1[:,rand_points]

        F = lab3.fmatrix_stls(pl, pr)

        residuals = lab3.fmatrix_residuals(F, p0, p1)
        # print(residuals)
        inliers = np.zeros_like(residuals)
        inliers[abs(residuals) < t] = 1
        # print(inliers)

        if np.sum(inliers) > cur_max_inliers:
            cur_max_inliers = np.sum(inliers)
            cur_F = F
            cur_pl = pl
            cur_pr = pr

    return cur_F, cur_pl, cur_pr
