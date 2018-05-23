from argparse import ArgumentParser
from textVectorizer import Vector, read_vectors, read_truths
import numpy as np
import pickle
import string
import pdb

class KNNClassifier:

    def __init__(self, dist_method, dist_mat, indices, rev_indices):
        self.dist_method = dist_method
        self.dist_mat = dist_mat
        self.indices = indices
        self.rev_indices = rev_indices

    def classify(self, vector, k):
        assert(k > 0)
        closest = self.dist_mat[self.rev_indices[vector.label]].argsort()[-k:][::-1]
        candidates = [self.indices[c][0] for c in closest]
        return max(set(candidates), key=candidates.count)

    @classmethod
    def generate(cls, data_filename, dist_method):
        assert(data_filename)
        assert(dist_method in ['COSINE', 'OKAPI'])

        print("-> generating model from ", data_filename)
        dfs, vectors = read_vectors(data_filename)
        if dist_method == 'COSINE':
            indices, rev_indices, dist_mat = cls.generate_cosine_sim_mat(dfs, vectors)
        else:
            #indices, rev_indices, dist_mat = cls.generate_okapi_mat(dfs, vectors)
            indices, rev_indices, dist_mat = cls.generate_cosine_sim_mat(dfs, vectors)
        return KNNClassifier(dist_method, dist_mat, indices, rev_indices)

    @staticmethod
    def generate_cosine_sim_mat(dfs, vectors):
        indices = [v for v in vectors]
        rev_indices = {indices[i]: i for i in range(0, len(indices))}
        dist_mat = np.zeros(shape=(len(indices), len(indices)))
        num_docs = len(vectors)

        print("-> generating model")
        for i in range(0, len(dist_mat)):
            for j in range(i + 1, len(dist_mat)):
                dist_mat[i, j] = vectors[indices[i]].cosine_sim(dfs, num_docs, vectors[indices[j]])
                dist_mat[j, i] = dist_mat[i, j]
        return indices, rev_indices, dist_mat

def process_cosine_sim(outq, key, work, num_docs, dfs, vectors):
    results = []
    for item1 in work:
        v = vectors[item1]
        for item2, w in vectors.items():
            if item1 != item2:
                results.append(((item1, item2), v.cosine_sim(dfs, num_docs, w)))
    outq.put(results)
    print("   - ending process for authors in ", key)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def classify_vectors(knn, vectors, k):
    print("-> classifying input")
    return [knn.classify(v, k) for v in vectors]

def write_results(filename, results):
    with open(filename, 'w') as file:
        print("-> writing results to ", filename)
        for r in results:
            file.write(r + "\n")

def read_results(filename):
    with open(filename, 'r') as file:
        print("-> reading results from ", filename)
        return [line.strip('\n') for line in file.readlines()]



if __name__ == "__main__":
    parser = ArgumentParser(description="Generates a KNN classifer")
    parser.add_argument("datafile", help="Vectors to be classified")
    parser.add_argument("--k", help="The number of nearest neighbors to check")
    parser.add_argument("--dist-method", help="The type of distance measure to use. Options: COSINE | OKAPI")
    parser.add_argument("--outfile", help="Write results of classification to a file")
    parser.add_argument("--save", help="Pickle the model to a file")
    parser.add_argument("--load", help="Load a pickled model from a file")
    args = parser.parse_args()

    if args.load:
        print("-> loading knn from ", args.load)
        with open(args.load, 'rb') as file:
            knn = pickle.load(file)
    else:
        knn = KNNClassifier.generate(args.datafile, args.dist_method or 'COSINE')
    if args.save:
        print("-> saving knn to ", args.save)
        with open(args.save, 'wb') as file:
            pickle.dump(knn, file)

    dfs, vectors = read_vectors(args.datafile)
    vectors = vectors.values()
    results = classify_vectors(knn, vectors, args.k or 5)

    if args.outfile:
        print("-> writing classication results to ", args.outfile)
        with open(args.outfile, 'w') as file:
            for r in results:
                file.write(r + "\n")
    else:
        for r in results:
            print(r)




