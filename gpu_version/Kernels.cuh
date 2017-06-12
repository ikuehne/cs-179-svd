#pragma once

#include <cstdint>

#include "Data.hh"

void train_batch(Batch b,
                 float *user_features, float *movie_features,
                 float lrate, float reg, int nfeatures);

void fill_array(float *arr, float val, int size);
