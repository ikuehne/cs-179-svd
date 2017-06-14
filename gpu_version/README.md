SVD on the GPU
==============

Building
--------

Simple:

```
make
```

Requires nvcc (obviously) and a compiler that supports C++11, such as any recent
clang or gcc.

Getting data
------------

A script (`get_data.py`) is included which fetches the data (the MovieLens 20M
dataset) from the Interwebs, groups it into training, validation, and testing
sets, and batches it (see below).  It can be invoked simply

```
python get_data.py
```

assuming you have Python, Numpy, and wget installed.  Be warned that it will
take a couple of minutes and use up a few hundred MB of disk space.

Running
-------

You can invoke the demo after the above steps with

```
./demo batches.dat valid.dat
```

The first argument is the (batched) training set, the second is a validation set
on which it will calculate its RMSE.

On a 2012 Intel Core i7 processor at 2.6 GHz with a NVIDIA GeForce 650M GPU, it
takes about 4 minutes with the default parameters (256 features, 50 iterations),
and gets a final RMSE of 0.786.  For comparison, the CPU version with the same
parameters gets roughly the same RMSE, but takes well over an hour.

Technical Overview
------------------

As described in the proposal, this program implements an approximate matrix
factorization algorithm for extremely large, sparse matrices, using a variant of
stochastic gradient descent.

The algorithm has been thoroughly modified to run efficiently on the GPU.  I
initially tried simply parallelizing over the input data by assigning one thread
to each input point.  That has obvious caching issues (each point corresponds to
a random access into the input data), as well as concurrency problems (there is
no guarantee that the gradients from two points are independent).  Shuffling the
data and running the algorithm "hog-wild" gave results similar to the CPU
version, but took about 3x as long.

To solve those problems, we use the following algorithm:

    1. Batch the data.  Split it into chunks of 256 points, where no two points
       in a batch have a user or movie in common (i.e., the gradients of all
       points in a batch are independent).  This is done in the preprocessing
       stage, with a randomized greedy algorithm.
    2. Running one batch at a time, assign each point in the batch to one block,
       and each feature at that point to one thread.
    3. Have the threads load the relevant user and movie features into shared
       memory.  The accesses are parallel and coalesced.
    4. Run a reduction to compute the dot product of the user and movie
       features.
    5. With the error computed in shared memory, each thread computes the
       gradient at its feature, and does the update.

Thus, in contrast to the CPU version, all of the features are updated at each
iteration.  That means that if we initialized them to a constant, as in the CPU
version, they would all be identical after training, so instead we use cuRand to
initialize the feature vectors randomly.

Also, note that for the shared memory behavior to be friendly to the hardware,
the number of features must be a multiple of 32, and that the performance
benefit of the GPU gets greater (less overhead, more work in each thread) with
more features.
