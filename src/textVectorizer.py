from argparse import ArgumentParser
from os import listdir
from os.path import basename
from porter_stem import PorterStemmer
import csv
import numpy as np
import pdb
import math
import re


class Vector:
    def __init__(self, tfidf):
        self.tfidf = tfidf

    def cosine_similarity(self, other):
        numerator = 0
        self_denom = 0
        other_denom = 0
        for word, val in self.tfidf.items():
            numerator += val * other.tfidf.get(word, 0)
            self_denom += self.tfidf[word] ** 2
        for word, val in other.tfidf.items():
            other_denom += other.tfidf[word] ** 2
        return float(numerator) / math.sqrt(self_denom * other_denom)


    def okapi(self, other):
        return 0;

def read_stopwords():
    with open("stopwords.txt", "r") as file:
        return set(file.read().split("\n"))

def parse_datadir(dirname, stopword, stemming):
    vectors = []
    data = {}
    vocab = {}
    num_docs = 0
    if stopword:
        stopwords = read_stopwords()
    if stemming:
        stemmer = PorterStemmer()
    for author in listdir(dirname):
        data[author] = {}
        for document in listdir(dirname + '/' + author):
            num_docs += 1
            data[author][document] = {}
            with open(dirname + '/' + author + '/' + document, 'r') as file:
                words = list(filter(lambda w: w != '', re.split(" |\.|\n", file.read())))
                if stopword:
                    words = [word for word in words if word not in stopwords]
                if stemming:
                    words = [stemmer.stem(word, 0, len(word) - 1) for word in words]
                for word in words:
                    if word not in vocab:
                        # word does not exist, add to vocab with one appearence in this document 
                        vocab[word] = {document}
                    elif document not in vocab[word]:
                        # word exists, but first appearance in this document
                        vocab[word].add(document)
                    if word not in data[author][document]:
                        # first word occurrence in document, init to 1
                        data[author][document][word] = 1
                    else:
                        # not first word occurrence in document, increment
                        data[author][document][word] += 1
    # calculate idfs
    for word, docs in vocab.items():
        vocab[word] = math.log(num_docs / len(docs))
    # calculate tf-idfs and create vectors
    for author, works in data.items():
        for document, words in works.items():
            for word, freq in words.items():
                data[author][document][word] = freq * vocab[word]
            vectors.append(((author, document), Vector(data[author][document])))
    return vocab.keys(), vectors


# TODO: This stores tfidf vectors in full which results in a very large (>300MB) file...
#       Consider:
#           (a) serializing it with pickle (this seems totally acceptable but may want to run it by Dehktyar
#           (b) convert to scipy sparse matrix and store as binary
#           (c) figure out another way to store the sparse matrices (can't think of a way right now)
#def write_vectors(filename, vocab, vectors):
#    with open(filename, 'w') as file:
#        writer = csv.writer(file)
#        for v in vectors:
#            row = [v.tfidf[word] if word in word in v.tfidf else 0 for word in vocab]
#            writer.writerow(row)

def write_truths(filename, truths):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        for t in truths:
            writer.writerow(list(t))



if __name__ == "__main__":
    parser = ArgumentParser(description="A text vectorizer -- produces a file of vectorized tf-idf documents and a ground truth file from the Reuter 50-50 dataset.")
    parser.add_argument("datadir", help="The location of the Reuters 50-50 dataset")
    parser.add_argument("outfile", help="The name of the output file containing tf-idf vectors (the ground truth filename will also be derived from this")
    parser.add_argument("--stopword", action="store_true", help="Include stopword removal")
    parser.add_argument("--stemming", action="store_true", help="Include word stemming")
    args = parser.parse_args()

    # generate ground-truths and corresponding vectors as parallel lists
    vocab, result = parse_datadir(args.datadir, args.stopword, args.stemming)
    truths, vectors = zip(*result)
    #write_vectors(args.outfile, vocab, vectors)
    write_truths(args.outfile.split('.')[0] + "_truths.csv", truths)

