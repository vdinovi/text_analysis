from argparse import ArgumentParser
from os import listdir
import numpy as np
import pdb

def parse_datadir(dirname):
    for filename in listdir(dirname):
        print(filename)




if __name__ == "__main__":
    parser = ArgumentParser(description="A text vectorizer -- produces a file of vectorized tf-idf documents and a ground truth file from the Reuter 50-50 dataset.")
    parser.add_argument("datadir", help="The location of the Reuters 50-50 dataset")
    parser.add_argument("outfile", help="The name of the output file containing tf-idf vectors (the ground truth filename will also be derived from this")
    args = parser.parse_args()

    data = parse_data(args.dirname)

