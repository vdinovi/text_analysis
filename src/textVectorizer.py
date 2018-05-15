from argparse import ArgumentParser
from os import listdir
from os.path import basename
from functools import reduce
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


def write_vectors(filename, vectors):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        for v in vectors:
            # store comma separated parise -> word,tfidf,word,tfidf,...
            row = reduce(lambda a, b: a + b, [[word, val] for word, val in v.tfidf.items()])
            writer.writerow(row)

def read_vectors(filename):
    vectors = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            tfidf = {}
            for index in range(0, len(row), 2):
                tfidf[row[index]] = float(row[index + 1])
            vectors.append(Vector(tfidf))
    return vectors

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
    #vocab, result = parse_datadir(args.datadir, args.stopword, args.stemming)
    #truths, vectors = zip(*result)
    #write_vectors(args.outfile, vectors)
    #write_truths(args.outfile.split('.')[0] + "_truths.csv", truths)
    #vectors = read_vectors("out.csv")


