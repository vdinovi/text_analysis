from argparse import ArgumentParser
from textVectorizer import read_truths, read_vectors
from classifierEvaluator import parse_results, confusion_matrix, eval_accuracy
from knnAuthorship import KNNClassifier, classify_vectors
import subprocess
import pickle

def test_k(knn, vectors, truths, outfile):
    k_values = []
    accuracies = []
    MAX = 25

    k = 1
    while k <= MAX:
        results = classify_vectors(knn, vectors, k)
        authors = parse_results(truths, results)
        conf_mat = confusion_matrix(authors, truths, results)
        _, _, accuracy = eval_accuracy(conf_mat)

        k_values.append(k)
        accuracies.append(accuracy)
        k += 2
    with open(outfile, 'w') as file:
        for i in range(0, len(k_values)):
            file.write("{}, {}\n".format(k_values[i], accuracies[i]))
    print("-> writing results to ", outfile)


if __name__ == "__main__":
    parser = ArgumentParser(description="Test KNN accuries against various k's")
    parser.add_argument("modelfile", help="KNN pickled model file")
    parser.add_argument("vectorsfile", help="vectors to classify")
    parser.add_argument("truthfile", help="ground truths")
    parser.add_argument("outfile", help="Output file")
    args = parser.parse_args()

    print("-> loading knn from ", args.modelfile)
    with open(args.modelfile, 'rb') as file:
        knn = pickle.load(file)

    truths = read_truths(args.truthfile)
    _, vectors = read_vectors(args.vectorsfile)
    print("-> testing ")
    test_k(knn, vectors.values(), truths, args.outfile)





