from multiprocessing import Process, Queue
from textVectorizer import Vector, read_model, read_truths
from link_methods import single_link, complete_link, average_link, centroid, wards
from argparse import ArgumentParser
from os import path
from datetime import datetime
import numpy as np
import math
from pprint import pprint
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

    def to_dict(self):
        node = {
            "type": self.node_type,
            "height": self.height,
        }
        if self.node_type == "LEAF":
            node["data"] = self.data
        else:
            node["nodes"] = [child.to_dict() for child in self.data]
        return node

def cosine_sim_mat(dfs, vectors):
    indices = [v for v in vectors]
    clusters = [(index,) for index in range(0, len(indices))]
    rev_indices = {indices[i]: i for i in range(0, len(indices))}
    dist_mat = np.zeros(shape=(len(indices), len(indices)))

    num_docs = len(vectors)
    # Parellelize the work by author
    procs = []
    proc_keys = { index[0] for index in indices }
    result_q = Queue()
    for key in proc_keys:
        work = [(index[0], index[1]) for index in indices if index[0] == key]
        proc = Process(target=process_cosine_sim, args=(result_q, key, work, num_docs, dfs, vectors))
        procs.append(proc)
        print("   + spawning process for ", key)
        proc.start()
    # Consume from workers
    for _ in range(len(procs)):
        for res in result_q.get():
            dist_mat[rev_indices[res[0][0]], rev_indices[res[0][1]]] = res[1]
            dist_mat[rev_indices[res[0][1]], rev_indices[res[0][0]]] = res[1]
    for proc in procs:
        proc.join()
    return clusters, indices, rev_indices, dist_mat

def process_cosine_sim(outq, key, work, num_docs, dfs, vectors):
    results = []
    for item1 in work:
        v = vectors[item1]
        for item2, w in vectors.items():
            if item1 != item2:
                results.append(((item1, item2), v.cosine_sim(dfs, num_docs, w)))
    outq.put(results) print("   - ending processing for ", key) def find_closest(dist_mat):
    x, y, dist = 0, 0, sys.float_info.max
    for ai in range(0, len(dist_mat)):
        for bi in range(ai, len(dist_mat)):
            if ai != bi and dist_mat[ai, bi] <= dist:
                dist = dist_mat[ai, bi]
                x, y = ai, bi
    return x, y, dist

def generate(dfs, vectors, dist_func):
    print("-> generating dendrogram")
    clusters, indices, rev_indices, dist_mat = cosine_sim_mat(dfs, vectors)
    tree = [Node("LEAF", 0, indices[clusters[ci][0]]) for ci in range(0, len(clusters))]
    while len(clusters) > 1:
        c1, c2, dist = find_closest(dist_mat)
        #dist_mat = dist_func(c1, c2, dist_mat, clusters, data)
        dist_mat = single_link(c1, c2, dist_mat)
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
    parser.add_argument("--link-method", default="SINGLE", help="Specify link-method for agglomeration. Allowed values: SINGLE | COMPLETE | AVERAGE | CENTROID | WARDS]. By default SINGLE.")
    args = parser.parse_args()

    link_methods = {
        "SINGLE":   single_link,
        "COMPLETE": complete_link,
        "AVERAGE":  average_link,
        "CENTROID": centroid,
        "WARDS":    wards
    }

    dfs, vectors = read_model(args.datafile)
    root = generate(dfs, vectors, link_methods[args.link_method])
    with open(args.outfile, 'w') as file:
        print("-> writing dendrogram to ", args.outfile)
        file.write(json.dumps(root.to_dict(), indent=4, separators=(',', ': ')))


