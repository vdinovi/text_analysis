from argparse import ArgumentParser
from clusteringAuthorship import Node
from textVectorizer import read_truths
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

def ClusterNames(clusters,truths):
    data={key:{'hits': 0, 'misses': 0, 'strikes': 0,'clusters':[],'expected': len([a for a in truths if a == key])} for key in set(truths)}
    names = []
    totalCorrect=0
    totalIncorrect=0
    for i in range(len(clusters)):
        cluster = clusters[i]
        d = {}
        for item in cluster:
            if item[0] in d:
                d[item[0]]+=1
            else:
                d[item[0]]=1
        bestName = max(d, key=lambda key: d[key])
        names.append(bestName)
        data[bestName]['clusters'].append(i)
    for i in range(len(clusters)):
        cluster = clusters[i]
        clusterName = names[i]
        for item in cluster:
            name = item[0]
            if name == clusterName:
                data[clusterName]['hits']+=1
                totalCorrect+=1
            else:
                data[clusterName]['strikes']+=1
                data[name]['misses']+=1
                totalIncorrect+=1
    return names,data,totalCorrect,totalIncorrect
        # hits - Same as true positives
        # misses - same as false negatives
        # strikes - same as false positives
        # TP - items that match their cluster
        # FP - items that don't match their cluster
        # TN - items that aren't in the cluster that do shouldn't be there
        # FN - items that aren't in the cluster that should be

def test_all(root):
    vals=[2,5,10,20,35,50]
    for i in vals:
        print("--testing for min_size={}".format(i))
        down=True
        j = 50
        while(1):
            if j==1:
                print("Unable to find any clusters...")
                break
            clusters = extract_clusters(root,j,i)
            print("    for {} clusters requested: {} found".format(j,len(clusters)))
            if len(clusters) > 0:
                if j == 50:
                    print("found {} clusters".format(len(clusters)))
                    break
                down=False
            if down:
                j -= j//2
            else:
                if len(clusters) <= 0:
                    print("      Final answer: {} clusters".format(j-1))
                    break
                j += 1

def precision(clusterDatum):
    # TP/(TP+FP)
    TP = float(clusterDatum['hits'])
    FP = float(clusterDatum['misses'])
    if (TP+FP) == 0:
        return 0
    return TP/(TP+FP)

def recall(clusterDatum):
    # TP/(TP+FN)
    TP = float(clusterDatum['hits'])
    FN = float(clusterDatum['strikes'])
    if (TP+FN) == 0:
        return 0
    return TP/(TP+FN)

def fmeasure(clusterDatum,beta=None):
    if beta==None:
        beta = 1
    p=precision(clusterDatum)
    r=recall(clusterDatum)
    if (beta**2)*p+r == 0:
        return 0
    return (1+beta**2)*((p*r)/((beta**2)*p+r))

def printDatum(name,cd):
    h,s,m = (cd['hits'],cd['strikes'],cd['misses'])
    p = precision(cd)
    r = recall(cd)
    f = fmeasure(cd)
    print("  - {}".format(name))
    print("      hits = {}, strikes = {}, misses = {}".format(h,s,m))
    print("      precision = {:8.4f}, recall = {:8.4f}, f-measure = {:8.4f}".format(p,r,f))

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluates a Hierarchical clustering")
    parser.add_argument("dendrogram", help="File containing the dendogram in JSON format")
    parser.add_argument("truthfile", help="Actual truth classes")
    #parser.add_argument("outfile", help="Resulting classes from the classifier")
    #parser.add_argument("confusion_outfile", help="Filename for the confusion matrix output")
    args = parser.parse_args()
    sys.setrecursionlimit(3000)

    print("-> loading dendrogram from ", args.dendrogram)
    with open(args.dendrogram, 'rb') as file:
       root  = pickle.load(file)
    truths = read_truths(args.truthfile)
    #test_all(root)
    #clusters = extract_clusters(root,50,20)
    clusters = extract_clusters_BFS(root,50,20)
    clusterNames,clusterData,tc,ti = ClusterNames(clusters,truths)
    print("-- Evaluation Results --")
    print("Total correct:    {}".format(tc))
    print("Total incorrect:  {}".format(ti))
    print("Overall accuracy: {:8.4f}".format(float(tc)/(tc+ti)))
    print("Authors:")
    for key in clusterData:
        printDatum(key,clusterData[key])
    #with open(args.confusion_outfile, 'w') as file:
    #    print("-> writing results to ", args.outfile)
