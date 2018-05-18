from argparse import ArgumentParser
from textVectorizer import Vector, read_model, read_truths
import numpy as np
#import threading
from multiprocessing import Process, Queue
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
            self.generate_cosine_sim_mat(dfs, vectors)
        else:
            self.generate_okapi_mat(dfs, vectors)

    def classify(self, vector):
        closest = self.dist_mat[self.rev_indices[vector.label]].argsort()[-self.k:][::-1]
        return [self.indices[c][0] for c in closest]

    def generate_cosine_sim_mat(self, dfs, vectors):
        self.indices = [v for v in vectors]
        self.rev_indices = {self.indices[i]: i for i in range(0, len(self.indices))}
        self.dist_mat = np.zeros(shape=(len(self.indices), len(self.indices)))

        num_docs = len(vectors)
        # Parellelize the work by author
        procs = []
        proc_keys = { index[0] for index in self.indices }
        result_q = Queue()
        for key in proc_keys:
            work = [(index[0], index[1]) for index in self.indices if index[0] == key]
            proc = Process(target=process_cosine_sim, args=(result_q, key, work, num_docs, dfs, vectors))
            procs.append(proc)
            print("   + spawning process for ", key)
            proc.start()
        # Consume from workers
        for _ in range(len(procs)):
            for res in result_q.get():
                self.dist_mat[self.rev_indices[res[0][0]], self.rev_indices[res[0][1]]] = res[1]
                self.dist_mat[self.rev_indices[res[0][1]], self.rev_indices[res[0][0]]] = res[1]
        for proc in procs:
            proc.join()

    @classmethod
    def generate_okapi_mat(cls, dfs, vectors):
        return None, [v.label for v in vectors]

def process_cosine_sim(outq, key, work, num_docs, dfs, vectors):
    results = []
    for item1 in work:
        v = vectors[item1]
        for item2, w in vectors.items():
            if item1 != item2:
                results.append(((item1, item2), v.cosine_sim(dfs, num_docs, w)))
    print("   - ending processing for ", key)
    outq.put(results)



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
        print("-> writing classication results to ", args.outfile)
        for i in range(0, len(truths)):
            file.write("{} -> {}\n".format(truths[i], results[i]))




