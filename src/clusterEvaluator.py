from argparse import ArgumentParser
from clusteringAuthorship import Node
#from queue import Queue
import pdb
import json
import sys
import pickle

def gather(node):
    # Recursively gathers up all the leafs from a given nodes to get the cluster
    if node.node_type == "LEAF":
        return [node.data]
    else:
        cluster = []
        for child in node.data:
            cluster += gather(child)
        return cluster

def size(node):
    # calculates the size of the cluster generated from the subtree of this node
    #
    # this could be made much more efficient if it was tail-recursive and stopped
    # when either it visited the entire tree OR the min_size was hit
    # for now just visit the whole tree to get size
    if node.node_type == 'LEAF':
        return 1
    else:
        return size(node.data[0]) + size(node.data[1])

def addPQ(clusters,node):
    idx = 0
    s = size(node)
    currSize = size(clusters[idx]) if len(clusters) > idx else 0
    while idx < len(clusters) and s < currSize:
        idx+=1
        currSize = size(clusters[idx]) if len(clusters) > idx else 0
    clusters.insert(idx,node)

def extract_clusters(root, num_clusters, min_size):
    clusters = [root]
    while len(clusters) < num_clusters:
        expandable = [n for n in clusters if n.node_type != 'LEAF']
        if not expandable:
            break
        node = expandable[0]
        s = size(node)
        sl = size(node.data[0])
        sr = size(node.data[1])
        if(s < min_size):
            break
        #print("{} -> {},{}".format(s,sl,sr))
        clusters.remove(node)
        if sl >= min_size:
            addPQ(clusters,node.data[0])
        if sr >= min_size:
            addPQ(clusters,node.data[1])
        if len(clusters) >= num_clusters:
            removal_count = 0
            for node in clusters:
                s = size(node)
                if s < min_size:
                    removal_count +=1
                    clusters.remove(node)
    return [gather(node) for node in clusters]
#  This function is wrong -- I think the right solution is 
#    1. Use BFS starting at the root to get list of size=num_clusters nodes
#    2. Use the size function to remove clusters that are too small
#    3. Repeat step 1 with the remaining clusters in place of the root
#    4. End when you the remaining cluster list is >= num_clusters

def extract_clusters_BFS(root, num_clusters, min_size):
    clusters = [root]
    while len(clusters) < num_clusters:
        expandable = [n for n in clusters if n.node_type != 'LEAF']
        if not expandable:
            break
        node = expandable[0]
        clusters.remove(node)
        clusters.append(node.data[0])
        clusters.append(node.data[1])
        if len(clusters) >= num_clusters:
            clusters = [c for c in clusters if size(c) < min_size]
    return [gather(node) for node in clusters]

'''def extract_clusters(root, num_clusters, min_size):
    clusters = [root]
    while len(clusters) < num_clusters:
        expandable = [n for n in clusters if n.node_type != 'LEAF']
        if not expandable:
            break
        node = expandable[0]
        clusters.remove(node)
        clusters.append(node.data[0])
        clusters.append(node.data[1])
        for node in clusters:
            if size(node) < min_size:
                clusters.remove(node)
        if not len(clusters):
            # reaches end without finding enough good sized clusters
            break
    return [gather(node) for node in clusters]
'''

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluates a Hierarchical clustering")
    parser.add_argument("dendrogram", help="File containing the dendogram in JSON format")
    #parser.add_argument("outfile", help="Filename for the confusion matrix output")
    args = parser.parse_args()
    sys.setrecursionlimit(2000)

    print("-> loading dendrogram from ", args.dendrogram)
    with open(args.dendrogram, 'rb') as file:
       root  = pickle.load(file)
    #clusters = extract_clusters(root, 40, 2)
    clusters = extract_clusters_BFS(root, 50, 2)
    print(len(clusters))

    #with open(args.confusion_outfile, 'w') as file:
    #    print("-> writing results to ", args.outfile)
