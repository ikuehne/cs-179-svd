SVD CPU Demo
============

Building
--------

Simple:

```
make demo
```

The code is written to use multiple cores for a substantial speedup
(approximately equal to the number of cores) when compiled with an OpenMP-aware
compiler; however, I have never been able to write a Makefile which portably
builds with OpenMP.  On GNU/Linux with a reasonably recent version of GCC,
simply adding -fopenmp should usually work.  On MacOS, I bid you good luck.
Probably whoever grades this doesn't care and will run it single-threaded.

Getting data
------------

A script (`get_data.py`) is included which fetches the data (the MovieLens 20M
dataset) from the Interwebs and groups it into training, validation, and testing
sets.  It can be invoked simply

```
python get_data.py
```

assuming you have Python, Numpy, and wget installed.  Be warned that it will
take a couple of minutes and use up ~300MB of disk space.

Running
-------

You can invoke the demo after the above steps with

```
./demo ml-20m/ratings.csv ml-20m/indices.csv
```

It will take 1-10 minutes depending on your computer, printing training RMSEs
(in units of stars) along the way, and on the last line printing the validation
RMSE.  The parameters used are 8 features, learning rate 0.005, regularization
0.02, and 40 iterations.  Better results can be achieved, given more time, with
more features and a lower learning rate; I chose these to quickly achieve a
reasonable result for the demonstration.
