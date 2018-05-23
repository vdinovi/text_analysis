from argparse import ArgumentParser
import pandas as pd
import numpy as np
import pdb
import json
import queue
from random import shuffle
from pprint import pprint

def gather(node):
    # Recursively gathers up all the leafs from a given nodes to get the cluster
    if node['type'] == "LEAF":
        return [node['data']]
    else:
        cluster = []
        for child in node['nodes']:
            cluster += gather(child)
        return cluster

def size(node):
    # calculates the size of the cluster generated from the subtree of this node
    #
    # this could be made much more efficient if it was tail-recursive and stopped
    # when either it visited the entire tree OR the min_size was hit
    # for now just visit the whole tree to get size
    if node['type'] == 'LEAF':
        return 1
    else:
        return size(node['nodes'][0]) + size(node['nodes'][1])

#  This function is wrong -- I think the right solution is 
#    1. Use BFS starting at the root to get list of size=num_clusters nodes
#    2. Use the size function to remove clusters that are too small
#    3. Repeat step 1 with the remaining clusters in place of the root
#    4. End when you the remaining cluster list is >= num_clusters
def extract_clusters(root, num_clusters, min_size):
    clusters = [root]
    while len(clusters) < num_clusters:
        expandable = [n for n in clusters if n['type'] != 'LEAF']
        if not expandable:
            break
        node = expandable[0]
        clusters.remove(node)
        clusters.append(node['nodes'][0])
        clusters.append(node['nodes'][1])
        for node in clusters:
            if size(node) < min_size:
                clusters.remove(node)
        if not len(clusters):
            # reaches end without finding enough good sized clusters
            break
    return [gather(node) for node in clusters]


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluates a Hierarchical clustering")
    parser.add_argument("dendrogram", help="File containing the dendogram in JSON format")
    #parser.add_argument("outfile", help="Filename for the confusion matrix output")
    args = parser.parse_args()

    with open(args.dendrogram) as file:
        tree = json.load(file)
    clusters = extract_clusters(tree, 2, 1)
    print(len(clusters))

    #with open(args.confusion_outfile, 'w') as file:
    #    print("-> writing results to ", args.outfile)
