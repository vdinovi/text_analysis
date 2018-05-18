from argparse import ArgumentParser
from heapq import heappush, nlargest
from textVectorizer import Vector, read_model, read_truths
import numpy as np
import pdb

class KNNClassifer:
    def __init__(self, data_filename, dist_method, k):
        assert(data_filename)
        assert(dist_method in ['COSINE', 'OKAPI'])
        assert(k > 0)

        self.k = k
        self.dist_method = dist_method
        dfs, vectors = read_model(data_filename)
        if dist_method == 'COSINE':
            self.dist_matrix, self.indices = self._cosine_sim_mat(dfs, vectors)
        else:
            self.dist_matrix, self.indices = self._okapi_mat(dfs, vectors)

    def classify(self, vector):
        closest = self.dist_matrix[self.indices.index(vector.label)].argsort()[-self.k:][::-1]
        return [self.indices[c][0] for c in closest]

    @classmethod
    def _cosine_sim_mat(cls, dfs, vectors):
        indices = [v for v in vectors]
        num_docs = len(vectors)
        dist_mat = np.zeros(shape=(len(indices), len(indices)))
        for row in range(0, len(indices)):
            for col in range(row + 1, len(indices)):
                # cosine similairty is commutative, copy across main diagonal
                dist_mat[row, col] = vectors[indices[row]].cosine_sim(dfs, num_docs, vectors[indices[col]])
                dist_mat[col, row] = dist_mat[row, col]
        return dist_mat, indices

    @classmethod
    def _okapi_mat(cls, dfs, vectors):
        return None, [v.label for v in vectors]

def classify_vectors(knn, vectors):
    return [knn.classify(v) for v in vectors]

if __name__ == "__main__":
    #parser = ArgumentParser(description="Generates a KNN classifer")
    #parser.add_argument("vectorfile", help="Vectors to be classified")
    #parser.add_argument("distance_method", help="The type of distance measure to use. Options: COSINE | OKAPI")
    #args = parser.parse_args()

    knn = KNNClassifer("small.csv", 'COSINE', 3)
    dfs, model = read_model("small.csv")
    vectors = model.values()
    truths = read_truths("small_truths.csv")
    results = classify_vectors(knn, vectors)
    for i in range(0, len(truths)):
        print("{} -> {}".format(truths[i], results[i]))




