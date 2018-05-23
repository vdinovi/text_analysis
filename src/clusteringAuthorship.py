from textVectorizer import Vector, read_vectors, read_truths
from link_methods import single_link, complete_link, average_link
from argparse import ArgumentParser
import numpy as np
import pickle
import sys
import json
import pdb

class Node:
    def __init__(self, node_type, height, data=None):
        if node_type not in ['LEAF', 'NODE', 'ROOT']:
            raise Exception("invalid node type: {}".format(node_type))
        self.node_type = node_type
        self.data = data
        self.height = float(height)

    # Using pickle instead, this is just an inneficient
    # the json tree is so large that its barely decipherable anyway
    # -- pickle requires less storage and is faster
    #def to_dict(self):
    #    node = {
    #        "type": self.node_type,
    #        "height": self.height,
    #    }
    #    if self.node_type == "LEAF":
    #        node["data"] = self.data
    #    else:
    #        node["nodes"] = [child.to_dict() for child in self.data]
    #    return node

def generate_cosine_sim_mat(dfs, vectors):
    indices = [v for v in vectors]
    clusters = [(index,) for index in range(0, len(indices))]
    #rev_indices = {indices[i]: i for i in range(0, len(indices))}
    dist_mat = np.zeros(shape=(len(indices), len(indices)))
    num_docs = len(vectors)

    print("-> generating model")
    for i in range(0, len(dist_mat)):
        for j in range(i + 1, len(dist_mat)):
            dist_mat[i, j] = vectors[indices[i]].cosine_sim(dfs, num_docs, vectors[indices[j]])
            dist_mat[j, i] = dist_mat[i, j]
    return clusters, indices, dist_mat


def find_closest(dist_mat):
    x, y, dist = 0, 0, sys.float_info.max
    for ai in range(0, len(dist_mat)):
        for bi in range(ai, len(dist_mat)):
            if ai != bi and dist_mat[ai, bi] <= dist:
                dist = dist_mat[ai, bi]
                x, y = ai, bi
    return x, y, dist

def generate(dfs, vectors, dist_func):
    print("-> generating dendrogram")
    clusters, indices, dist_mat = generate_cosine_sim_mat(dfs, vectors)
    tree = [Node("LEAF", 0, indices[clusters[ci][0]]) for ci in range(0, len(clusters))]
    while len(clusters) > 1:
        c1, c2, dist = find_closest(dist_mat)
        dist_mat = dist_func(c1, c2, dist_mat)
        tree[c1] = Node("NODE", dist, (tree[c1], tree[c2]))
        tree.remove(tree[c2])
        clusters[c1] += clusters[c2]
        clusters.pop(c2)
    tree[0].node_type = "ROOT"
    return tree[0]

if __name__ == "__main__":
    parser = ArgumentParser(description="Generates a hierarchical cluster dendrogram")
    parser.add_argument("datafile", help="Vectors to be clustered")
    parser.add_argument("outfile", help="Write the resulting dendrogram to this file")
    parser.add_argument("--link-method", default="SINGLE", help="Specify link-method for agglomeration. Allowed values: SINGLE | COMPLETE | AVERAGE. By default SINGLE.")
    args = parser.parse_args()
    sys.setrecursionlimit(25000)
    link_methods = {
        "SINGLE":   single_link,
        "COMPLETE": complete_link,
        "AVERAGE":  average_link
    }

    dfs, vectors = read_vectors(args.datafile)
    root = generate(dfs, vectors, link_methods[args.link_method])
    with open(args.outfile, 'w') as file:
        print("-> writing dendrogram to ", args.outfile)
        with open(args.outfile, 'wb') as file:
            pickle.dump(root, file)


