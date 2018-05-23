import matplotlib.pyplot as plt
from argparse import ArgumentParser
from textVectorizer import read_truths
from classifierEvaluator import parse_results, confusion_matrix, eval_accuracy
from knnAuthorship import classify_vectors
import subprocess
import pickle


def plot_k(knn, vectors, truths, outfile):
    k_values = []
    accuracies = []
    MAX = 50

    k = 0
    while k < MAX:
        results = classify_vectors(knn, vectors, k)
        authors = parse_results(truths, results)
        conf_mat = confusion_matrix(authors, truths, results)
        _, _, accuracy = eval_accuracy(conf_mat)

        k_values.append(k)
        accuracies.append(accuracy)
        k += 2
    plt.style.use('ggplot')
    plt.clf()
    fig = plt.figure()
    fig.suptitle('K vs. Overall Accuracy')
    plt.plot(k_values, accuracies)
    plt.xlabel("k")
    plt.ylabel("Overall Accuracy item sets found")
    plt.savefig(outfile)
    print("-> writing plot to ", outfile)


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot KNN accuries against various k's")
    parser.add_argument("modelfile", help="KNN pickles model file")
    parser.add_argument("vectorsfile", help="KNN pickles model file")
    parser.add_argument("outfile", help="Output file")
    args = parser.parse_args()

    print("-> loading knn from ", args.load)
    with open(args.modelfile, 'rb') as file:
        knn = pickle.load(file)

    truths = read_truths(args.truthfile)
    _, vectors = read_vectors(args.vectorsfile)
    print("-> generating plot ")
    plot_k(knn, vectors, truths, args.outfile)





