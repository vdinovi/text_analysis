from argparse import ArgumentParser
from heapq import heappush, nlargest
from textVectorizer import Vector, read_model, read_truths
import numpy as np
import threading
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
            self.dist_matrix, self.indices = self.cosine_sim_mat(dfs, vectors)
        else:
            self.dist_matrix, self.indices = self.okapi_mat(dfs, vectors)

    def classify(self, vector):
        closest = self.dist_matrix[self.indices.index(vector.label)].argsort()[-self.k:][::-1]
        return [self.indices[c][0] for c in closest]

    @classmethod
    def cosine_sim_mat(cls, dfs, vectors):
        indices = [v for v in vectors]
        num_docs = len(vectors)
        dist_mat = np.zeros(shape=(len(indices), len(indices)))
        thread_keys = {index[0] for index in indices}
        threads = []
        # Parellelize the work by author
        for key in thread_keys:
            thd = threading.Thread(target=process_cosine_sim, args=(key, dfs, vectors, dist_mat, indices, num_docs))
            thd.daemon = True
            threads.append(thd)
            print("   + spawning processing thread for ", key)
            thd.start()
        for thd in threads:
            thd.join()
        return dist_mat, indices

    @classmethod
    def okapi_mat(cls, dfs, vectors):
        return None, [v.label for v in vectors]

def process_cosine_sim(key, dfs, vectors, dist_mat, indices, num_docs):
    work = [i for i in range(0, len(indices)) if indices[i][0] == key]
    for row in work:
        for col in range(row + 1, len(indices)):
            # cosine similairty is commutative, copy across main diagonal
            dist_mat[row, col] = vectors[indices[row]].cosine_sim(dfs, num_docs, vectors[indices[col]])
            dist_mat[col, row] = dist_mat[row, col]
    print("   - ending processing thread for ", key)



def classify_vectors(knn, vectors):
    print("-> classifying input")
    return [knn.classify(v) for v in vectors]

if __name__ == "__main__":
    parser = ArgumentParser(description="Generates a KNN classifer")
    parser.add_argument("data_file", help="Vectors to be classified")
    parser.add_argument("outfile", help="Write results of classification")
    #parser.add_argument("distance_method", help="The type of distance measure to use. Options: COSINE | OKAPI")
    args = parser.parse_args()

    knn = KNNClassifer(args.data_file, 'COSINE', 5)
    dfs, model = read_model(args.data_file)
    vectors = model.values()
    truths = read_truths(args.data_file.split('.')[0] + "_truths.csv")
    results = classify_vectors(knn, vectors)

    with open(args.outfile, 'w') as file:
        for i in range(0, len(truths)):
            file.write("{} -> {}\n".format(truths[i], results[i]))




