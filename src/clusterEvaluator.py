from argparse import ArgumentParser
import pandas as pd
import numpy as np
import pdb
import json
from random import shuffle
from pprint import pprint

def gather(node):
    if node['type'] == "LEAF":
        return [node['data']]
    else:
        cluster = []
        for child in node['nodes']:
            cluster += gather(child)
        return cluster

def check_size(node):
    if node['type'] == 'LEAF':
        return 1
    else:
        return sum([check_size(child) for child in node['nodes']])

def extract_clusters(root, num_clusters, min_size):
    clusters = [root]
    while len(clusters) < num_clusters:
        print(len(clusters))
        expandable = [n for n in clusters if 'nodes' in n]
        if not expandable:
            break
        new_nodes = []
        for node in expandable:
            temp = []
            for child in node['nodes']:
                if child['type'] != 'LEAF' and check_size(child) >= min_size:
                    new_nodes.append(child)
            if temp:
                clusters.remove(node)
            new_nodes += temp
        if new_nodes:
            clusters += new_nodes
        else:
            break
        if len(clusters) >= num_clusters:
            break
    return [gather(node) for node in clusters]


"""
def extract_clusters(root, num_clusters):
    assert(num_clusters > 0)
    clusters = [root]
    while len(clusters) < num_clusters:
        accum = []
        while clusters:
            # search for expandable node in list
            nodes = [n for n in clusters if 'nodes' in n]
            if not nodes:
                break
            # remove from orig list and accumulate children
            node = nodes[0]
            clusters.remove(node)
            if node['nodes'][0]['type'] != 'LEAF':
                accum.append(node['nodes'][0])
            if node['nodes'][1]['type'] != 'LEAF':
                accum.append(node['nodes'][1])
            # Break if target hit
            if len(accum) + len(clusters) >= num_clusters:
                break
        clusters += accum
        # break if target hit or no remaining expandable nodes
        if len(clusters) >= num_clusters or not [n for n in clusters if 'nodes' in n]:
            break
    return [gather(node) for node in clusters]
"""

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluates a Hierarchical clustering")
    parser.add_argument("dendrogram", help="File containing the dendogram in JSON format")
    #parser.add_argument("outfile", help="Filename for the confusion matrix output")
    args = parser.parse_args()

    with open(args.dendrogram) as file:
        tree = json.load(file)
    clusters = extract_clusters(tree, 4, 2)
    pdb.set_trace()



    #with open(args.confusion_outfile, 'w') as file:
    #    print("-> writing results to ", args.outfile)
