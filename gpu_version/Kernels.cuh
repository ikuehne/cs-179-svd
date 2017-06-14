/**
 * @file Kernels.cuh
 *
 * Interface to the GPU code.
 */

#pragma once

#include <cstdint>

#include "Data.hh"

/**
 * @brief Train the given features on the given batch of data.
 */
void train_batch(Batch b,
                 float *user_features, float *movie_features,
                 float lrate, float reg, int nfeatures);
