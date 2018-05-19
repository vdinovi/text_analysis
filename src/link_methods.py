import numpy as np

def single_link(c1, c2, dist_mat):
    # clusters, data unused -- passed for consistency with other methods
    for row in range(0, len(dist_mat)):
        dist_mat[row, c1] = min(dist_mat[row, c1], dist_mat[row, c2])
        dist_mat[c1, row] = dist_mat[row, c1]
    dist_mat = np.delete(dist_mat, c2, axis=0)
    dist_mat = np.delete(dist_mat, c2, axis=1)
    return dist_mat

def complete_link(c1, c2, dist_mat, clusters, data):
    # clusters, data unused -- passed for consistency with other methods
    for row in range(0, len(dist_mat)):
        dist_mat[row, c1] = max(dist_mat[row, c1], dist_mat[row, c2])
        dist_mat[c1, row] = dist_mat[row, c1]
    dist_mat = np.delete(dist_mat, c2, axis=0)
    dist_mat = np.delete(dist_mat, c2, axis=1)
    return dist_mat


def average_link(c1, c2, dist_mat, clusters, data):
    # clusters, data unused -- passed for consistency with other methods
    for row in range(0, len(dist_mat)):
        dist_mat[row, c1] = (dist_mat[row, c1] + dist_mat[row, c2]) / 2
        dist_mat[c1, row] = dist_mat[row, c1]
    dist_mat = np.delete(dist_mat, c2, axis=0)
    dist_mat = np.delete(dist_mat, c2, axis=1)
    return dist_mat

def centroid(c1, c2, dist_mat, clusters, data):
    # This methods require reading the data to compute centroids
    cluster = clusters[c1] + clusters[c2]
    centroid = sum([data.values[p] for p in cluster]) / len(cluster)
    for row in range(0, len(dist_mat)):
        orig = sum([data.values[p] for p in clusters[row]]) / len(clusters[row])
        dist_mat[row, c1] = sum(abs(orig - centroid))
        dist_mat[c1, row] = dist_mat[row, c1]
    dist_mat = np.delete(dist_mat, c2, axis=0)
    dist_mat = np.delete(dist_mat, c2, axis=1)
    return dist_mat

def wards(c1, c2, dist_mat, clusters, data):
    # This methods require reading the data to compute centroids
    cluster = clusters[c1] + clusters[c2]
    centroid = sum([data.values[p] for p in cluster]) / len(cluster)
    for row in range(0, len(dist_mat)):
        orig = sum([data.values[p] for p in clusters[row]]) / len(clusters[row])
        coef = len(clusters[row]) * len(cluster) / float(len(clusters[row]) + len(cluster))
        dist_mat[row, c1] = coef * math.sqrt(sum(orig - centroid) ** 2)
        dist_mat[c1, row] = dist_mat[row, c1]
    dist_mat = np.delete(dist_mat, c2, axis=0)
    dist_mat = np.delete(dist_mat, c2, axis=1)
    return dist_mat


