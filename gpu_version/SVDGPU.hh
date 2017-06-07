#pragma once

#include "Data.hh"

class SVDGPU {
public:
    SVDGPU(int nusers, int nmovies, int nfeatures);

    void fit(uint32_t *users, uint32_t *movies, float *ratings,
             int npoints, float lrate, float reg, int niters);

    float *predict(uint32_t *users, uint32_t *movies, int npoints);
    ~SVDGPU(void);
private:
    Dataset to_gpu(uint32_t *users, uint32_t *movies, float *ratings,
                   int count);

    int nusers;
    int nmovies;
    float *user_features;
    float *movie_features;
    int nfeatures;
};
