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
    def __init__(self, label, term_freqs):
        self.label = label
        self.tfs = term_freqs

    def cosine_sim(self, dfs, num_docs, other):
        numerator = 0
        self_denom = 0
        other_denom = 0
        for word, freq in self.tfs.items():
            idf = math.log(num_docs / dfs[word])
            if word in other.tfs:
                numerator += freq * idf * other.tfs[word] * idf
            self_denom += (freq * idf) ** 2
        for word, freq in other.tfs.items():
            other_denom += (other.tfs[word] * idf) ** 2
        return float(numerator) / math.sqrt(self_denom * other_denom)

    def okapi(self, other):
        return 0

def read_stopwords():
    with open("stopwords.txt", "r") as file:
        return set(file.read().split("\n"))

def parse_datadir(dirname, stopword, stemming):
    vectors = []
    labels = []
    dfs = {}

    data = {}
    num_docs = 0
    if stopword:
        stopwords = read_stopwords()
    if stemming:
        stemmer = PorterStemmer()
    # Filter out words containing invalid characters
    wf = re.compile(".*[0-9\`\@\#\$\%\^\&\*\+\_\{\}\[\]]+.*")
    # Generate
    for author in listdir(dirname):
        data[author] = {}
        for document in listdir(dirname + '/' + author):
            num_docs += 1
            data[author][document] = {}
            with open(dirname + '/' + author + '/' + document, 'r') as file:
                words = list(filter(lambda w: w != '' and not re.match(wf, w), re.split(" |\.|\n", file.read())))
                if stopword:
                    words = [word for word in words if word not in stopwords and not re.match(wf, word)]
                if stemming:
                    words = [stemmer.stem(word, 0, len(word) - 1) for word in words]
                for word in words:
                    if word not in dfs:
                        # word does not exist, add to dfs with one appearence in this document 
                        dfs[word] = {document}
                    elif document not in dfs[word]:
                        # word exists, but first appearance in this document
                        dfs[word].add(document)
                    if word not in data[author][document]:
                        # first word occurrence in document, init to 1
                        data[author][document][word] = 1
                    else:
                        # not first word occurrence in document, increment
                        data[author][document][word] += 1
    # Create vectors from term freqs
    for author, works in data.items():
        for document, words in works.items():
            for word, freq in words.items():
                data[author][document][word] = freq
            labels.append(author)
            vectors.append(Vector((author, document), data[author][document]))
    # Convert to doc freqs
    for word in dfs:
        dfs[word] = len(dfs[word])
    return labels, dfs, vectors


# Stores doc-freqs and tf-vectors in the form:
#   w1,df1,w2,df2,...                     <-- doc freqs per word
#   Author,Document,vw1,vfw1,vw2,vfw2,... <-- sparse vector format
def write_model(filename, dfs, vectors):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        # Write doc freqs for all words in vocab
        row = reduce(lambda a, b: a + b, [[word, df] for word, df in dfs.items()])
        writer.writerow(row)
        # Write tf-vectors in sparse form (word, freq), ...
        for v in vectors:
            row = reduce(lambda a, b: a + b, [[word, freq] for word, freq in v.tfs.items()])
            writer.writerow([v.label[0], v.label[1]] + row)

def read_model(filename):
    vectors = {}
    dfs = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        row = next(reader)
        # Read doc freqs
        for index in range(0, len(row), 2):
            dfs[row[index]] = int(row[index + 1])
        # Read tf-vectors
        for row in reader:
            tfs = {}
            label = (row[0], row[1])
            for index in range(2, len(row), 2):
                tfs[row[index]] = int(row[index + 1])
            vectors[label] = Vector(label, tfs)
    return dfs, vectors

# Store truths as one author per line
def write_truths(filename, truths):
    with open(filename, 'w') as file:
        for t in truths:
            file.write(t + "\n")

def read_truths(filename):
    with open(filename, 'r') as file:
        return [line.strip('\n') for line in file.readlines()]

if __name__ == "__main__":
    parser = ArgumentParser(description="A text vectorizer -- produces a file of vectorized tf-idf documents and a ground truth file from the Reuter 50-50 dataset.")
    parser.add_argument("datadir", help="The location of the Reuters 50-50 dataset")
    parser.add_argument("outfile", help="The name of the output file containing tf-idf vectors (the ground truth filename will also be derived from this")
    parser.add_argument("--stopword", action="store_true", help="Include stopword removal")
    parser.add_argument("--stemming", action="store_true", help="Include word stemming")
    args = parser.parse_args()

    # generate ground-truths and corresponding vectors as parallel lists
    labels, dfs, vectors = parse_datadir(args.datadir, args.stopword, args.stemming)
    write_model(args.outfile, dfs, vectors)
    write_truths(args.outfile.split('.')[0] + "_truths.csv", labels)
    #rdfs, rvectors = read_model(args.outfile)
    #rtruths = read_truths(args.outfile.split('.')[0] + "_truths.csv")


