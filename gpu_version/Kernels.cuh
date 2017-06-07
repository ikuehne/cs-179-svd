#pragma once

#include <cstdint>

#include "Data.hh"

void train_feature(float *user_features, float *movie_features,
                   Dataset data, int current_feature,
                   float lrate, float reg, int nfeatures,
                   int niters);

void fill_array(float *arr, float val, int size);
