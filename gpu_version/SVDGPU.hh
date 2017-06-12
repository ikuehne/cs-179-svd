#pragma once

#include "Data.hh"

class SVDGPU {
public:
    SVDGPU(int nusers, int nmovies, int nfeatures);

    void fit(DataPoint *batches, int nbatches,
             float lrate, float reg, int niters);

    float *predict(uint32_t *users, uint32_t *movies, int npoints);

    ~SVDGPU(void);

private:
    int nusers;
    int nmovies;
    float *user_features;
    float *movie_features;
    int nfeatures;
};
