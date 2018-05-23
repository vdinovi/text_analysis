import numpy as np

def single_link(c1, c2, dist_mat):
    for row in range(0, len(dist_mat)):
        dist_mat[row, c1] = min(dist_mat[row, c1], dist_mat[row, c2])
        dist_mat[c1, row] = dist_mat[row, c1]
    dist_mat = np.delete(dist_mat, c2, axis=0)
    dist_mat = np.delete(dist_mat, c2, axis=1)
    return dist_mat

def complete_link(c1, c2, dist_mat):
    for row in range(0, len(dist_mat)):
        dist_mat[row, c1] = max(dist_mat[row, c1], dist_mat[row, c2])
        dist_mat[c1, row] = dist_mat[row, c1]
    dist_mat = np.delete(dist_mat, c2, axis=0)
    dist_mat = np.delete(dist_mat, c2, axis=1)
    return dist_mat


def average_link(c1, c2, dist_mat):
    for row in range(0, len(dist_mat)):
        dist_mat[row, c1] = (dist_mat[row, c1] + dist_mat[row, c2]) / 2
        dist_mat[c1, row] = dist_mat[row, c1]
    dist_mat = np.delete(dist_mat, c2, axis=0)
    dist_mat = np.delete(dist_mat, c2, axis=1)
    return dist_mat

