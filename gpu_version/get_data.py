import csv
import itertools
import numpy as np
import os
from os import path
import sys

def read_array(fname):
    with open(fname) as f:
        r = itertools.islice(csv.reader(f), 1, None, 1)
        l = [(int(x[0]), int(x[1]), float(x[2])) for x in r]
    result = np.array(l, dtype=np.dtype([('user',   np.uint32),
                                         ('movie',  np.uint32),
                                         ('rating', np.float32)]))
    result['user'] -= 1
    result['movie'] -= 1
    return result

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
    print "Parsing data..."
    ratings = read_array(datafile)
    print "Done. Fixing movie IDs..."
    fix_movie_ids(ratings)
    print "Done. Generating data hierarchy..."
    indices = get_indices(len(ratings))
    print "Done. Saving results..."
    ratings[indices == 0].tofile('base.dat')
    ratings[indices == 1].tofile('valid.dat')
    ratings[indices == 2].tofile('test.dat')
    print "Done. Batching training data..."
    if not path.isfile('./batch'):
        res = os.system('make batch')
        if res or not path.isfile('./batch'):
            sys.stderr.write('Unable to compile batching program! Aborting...')
            exit(1)
    os.system('./batch base.dat batches.dat')
    print "Done."

def fix_movie_ids(data):
    y = np.empty(np.max(data['movie']) + 1, dtype=np.uint32)
    u = np.unique(data['movie'])
    y[u] = np.arange(len(u))
    data['movie'] = y[data['movie']]

DATA_URL = 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'

if __name__ == "__main__":
    print "Downloading data..."
    if not (os.system("wget " + DATA_URL) or preprocess("ml-20m.zip")):
        os.unlink("ml-20m.zip")
        print "Data successfully downloaded and preprocessed."
