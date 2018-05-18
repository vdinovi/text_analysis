from argparse import ArgumentParser
import numpy as np
import pdb
from textVectorizer import read_truths
from knnAuthorship import read_results
#
#    AT  |  AF
# ET 0,0 | 0,1
# -------+-------
# EF 1,0 | 1,1
#        |
#
#     AT  |  AF
# ET  TP  |  FN
# --------+-------
# EF  FP  |  TN
#         |
def confusion_matrix(target, actual, expected):
    arr = [[0,0], [0,0]]
    for i in range(len(actual)):
        y = 0 if actual[i] == target else 1
        x = 0 if expected[i] == target else 1
        arr[x][y] += 1
        return arr

# TP/(TP+FP)
def precision(cmatrix):
    TP = cmatrix[0][0]
    FP = cmatrix[1][0]
    return float(TP)/(TP+FP)

#TP/(TP+FN)
def recall(cmatrix):
    TP = cmatrix[0][0]
    FN = cmatrix[0][1]
    return float(TP)/(TP+FN)

def fmeasure(cmatrix, beta=None):
    if beta == None:
        beta = 1
        p = precision(cmatrix)
        r = recall(cmatrix)
        return (1+beta**2)*((p*r)/((beta**2)*p+r))


def parse_results(truths, results):
    authors = {author: {'hits': 0, 'strikes': 0, 'misses': 0} for author in set(truths)}
    for i in range(0, len(truths)):
        if truths[i] == results[i]:
            authors[truths[i]]['hits'] += 1
        else:
            authors[truths[i]]['misses'] += 1
            authors[results[i]]['strikes'] += 1
    return authors


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluates a KNN classifier")
    parser.add_argument("truthfile", help="Actual truth classes")
    parser.add_argument("resultfile", help="Resulting classes from classifier")
    args = parser.parse_args()

    truths = read_truths(args.truthfile)
    results = read_results(args.resultfile)
    authors = parse_results(truths, results)
    for name, result in authors.items():
        print("{}: {} hits, {} strikes, {} misses".format(name, result['hits'], result['strikes'], result['misses']))




