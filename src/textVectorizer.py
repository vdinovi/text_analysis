from argparse import ArgumentParser
from os import listdir
from porter_stem import PorterStemmer
import numpy as np
import pdb
import math
import re


class Vector:
    def __init__(self, tfidf):
        self.tfidf = tfidf

    def cosine_similarity(self, other):
        return 0;

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
    return vectors


if __name__ == "__main__":
    parser = ArgumentParser(description="A text vectorizer -- produces a file of vectorized tf-idf documents and a ground truth file from the Reuter 50-50 dataset.")
    parser.add_argument("datadir", help="The location of the Reuters 50-50 dataset")
    #parser.add_argument("outfile", help="The name of the output file containing tf-idf vectors (the ground truth filename will also be derived from this")
    parser.add_argument("--stopword", action="store_true", help="Include stopword removal")
    parser.add_argument("--stemming", action="store_true", help="Include word stemming")
    args = parser.parse_args()

    vectors = parse_datadir(args.datadir, args.stopword, args.stemming)
    pdb.set_trace()

