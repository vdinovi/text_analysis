from argparse import ArgumentParser
import pandas as pd
import numpy as np
import pdb
from textVectorizer import read_truths
from knnAuthorship import read_results

def confusion_matrix(authors, truths, results):
    matr = pd.DataFrame(np.zeros((len(authors), len(authors))), index=authors.keys(), columns=authors.keys())
    for index in range(0, len(truths)):
        matr[results[index]][truths[index]] += 1
    return matr

def eval_precision_recall(authors, conf_matr, beta = 1):
    #  TP  |  FN
    #  FP  |  TN  
    names = conf_matr.columns
    matr = conf_matr.values
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    trace = np.trace(matr)
    for row in range(0, len(matr)):
        for col in range(0, len(matr)):
            if row == col:
                true_pos += matr[row][col]
                true_neg += trace - matr[row][col]
            else:
                false_neg += matr[row][col]
                false_pos += matr[row][col]
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        authors[names[row]]['f_measure'] = (1 + beta ** 2) * (precision * recall / (beta ** 2 * precision + recall))
        authors[names[row]]['precision'] = precision
        authors[names[row]]['recall'] = precision

def eval_accuracy(conf_matr):
    matr = conf_matr.values
    correct = 0
    incorrect = 0
    for row in range(0, len(matr)):
        for col in range(0, len(matr)):
            if row == col:
                correct += matr[row][col]
            else:
                incorrect += matr[row][col]
    accuracy = float(correct) / (correct + incorrect)
    return correct, incorrect, accuracy

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
    parser.add_argument("confusion_outfile", help="Filename for the confusion matrix output")
    args = parser.parse_args()

    truths = read_truths(args.truthfile)
    results = read_results(args.resultfile)
    # calculate stats
    authors = parse_results(truths, results)
    conf_mat = confusion_matrix(authors, truths, results)
    eval_precision_recall(authors, conf_mat)
    total_correct, total_incorrect, accuracy = eval_accuracy(conf_mat)

    print("-- Evaluation Results --")
    print("Total Correct:    {}".format(total_correct))
    print("Total Incorrect:  {}".format(total_incorrect))
    print("Overall Accuracy: {:.4f}".format(accuracy))
    print("Authors:")
    for name, result in authors.items():
        print("  - {}:".format(name))
        print("      hits = {},  strikes = {},  misses = {}".format(result['hits'], result['strikes'], result['misses']))
        print("      precision = {:4f},  recall = {:.4f},  f-measure = {:.4f}".format(result['precision'], result['recall'], result['f_measure']))

    with open(args.confusion_outfile, 'w') as file:
        print("-> writing confusion matrix to ", args.confusion_outfile)
        file.write(conf_mat.to_csv())
