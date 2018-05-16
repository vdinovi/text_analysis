from argparse import ArgumentParser
from heapq import heappush, nlargest
from textVectorizer import Vector, read_vectors
import pdb

class KNNClassifer:
    def __init__(self, model_filename, dist_method, k):
        assert(model_filename)
        assert(dist_method in ['COSINE', 'OKAPI'])
        assert(k > 0)

        self.vectors = read_vectors(model_filename)
        self.dist_method = dist_method
        self.k = k

    def classify(self, vector):
        queue = []
        for v in self.vectors:
            if self.dist_method == 'COSINE':
                dist = v.cosine_similarity(vector)
            else:
                dist = v.okapi(vector)
            heappush(queue, (dist, v.label[0]))
        nearest = [x[1]  for x in nlargest(self.k, queue, lambda x: x[0])]
        return max(set(nearest), key= nearest.count)


def classify_vectors(knn, vectors):
    return [knn.classify(v) for v in vectors]


if __name__ == "__main__":
    parser = ArgumentParser(description="Generates a KNN classifer")
    parser.add_argument("vectorfile", help="Vectors to be classified")
    #parser.add_argument("distance_method", help="The type of distance measure to use. Options: COSINE | OKAPI")
    args = parser.parse_args()

    knn = KNNClassifer("model.csv", 'COSINE', 3)
    vectors = read_vectors(args.vectorfile)
    results = classify_vectors(knn, vectors)
    with open("results.txt", "r") as file:
        for result in results:
            file.write("{}\n".format(result))
        print("-> wrote results to results.txt")



