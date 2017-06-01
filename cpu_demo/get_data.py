import numpy as np
import os
from os import path

def line_count(fname):
    with open(fname) as f:
        return sum(1 for _ in f)

def get_indices(count):
    result = np.empty(count, dtype=np.uint8)
    r = np.random.rand(count)
    result[r <= 0.98] = 0
    result[np.logical_and(r > 0.98, r <= 0.99)] = 1
    result[r > 0.99] = 2
    return result

def preprocess(fname):
    os.system("unzip {}".format(fname))
    # Strip off `.zip`
    dirname = fname[:-4]
    datafile = path.join(dirname, 'ratings.csv')
    print "Counting data..."
    count = line_count(datafile)
    print "Done. Generating data hierarchy..."
    indices = get_indices(count)
    print "Done. Saving results..."
    i_file = path.join(dirname, 'indices.csv')
    np.savetxt(i_file, indices, fmt='%d')
    print "Done."

DATA_URL = 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'

if __name__ == "__main__":
    print "Downloading data..."
    if not (os.system("wget " + DATA_URL) or preprocess("ml-20m.zip")):
        os.unlink("ml-20m.zip")
        print "Data successfully downloaded and preprocessed."
