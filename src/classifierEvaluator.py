from argparse import ArgumentParser
import numpy as np
import pdb
from textVectorizer import read_truths
from knnAuthorship import read_results
#
#    ET  |  EF
# AT 0,0 | 0,1
# -------+-------
# AF 1,0 | 1,1
#        |
#
#     ET  |  EF
# AT  TP  |  FN
# --------+-------
# AF  FP  |  TN
#         |
def confusion_matrix(authors, actual, expected):
    arr = [[0,0], [0,0]]
    for author in authors:
        for i in range(len(actual)):
            r = 0 if actual[i] == author else 1
            c = 0 if expected[i] == author else 1
            arr[r][c] += 1
    return arr

def evaluate(cmatrix, beta = None):
    TP = cmatrix[0][0]
    FP = cmatrix[1][0]
    precision =  float(TP) / (TP+FP)
    TP = cmatrix[0][0]
    FN = cmatrix[0][1]
    recall = float(TP) / (TP+FN)
    beta = beta or 1
    f_measure = (1 + beta ** 2) * ((precision * recall)/((beta ** 2) * precision + recall))
    return precision, recall, f_measure

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
    conf_mat = confusion_matrix(set(authors.keys()), truths, results)
    precision, recall, f_measure = evaluate(conf_mat)
    print("Confusion Matrix:")
    print("  |TP={}|FN={}|".format(conf_mat[0][0], conf_mat[0][1]))
    print("  |FP={}|TN={}|".format(conf_mat[1][0], conf_mat[1][1]))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F-measure: {}".format(f_measure))

    for name, result in authors.items():
        print("{}: {} hits, {} strikes, {} misses".format(name, result['hits'], result['strikes'], result['misses']))




