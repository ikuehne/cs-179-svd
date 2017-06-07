#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

#include "Kernels.cuh"
#include "SVDGPU.hh"

/*
 * Error checking on the GPU.
 */
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
    cudaError_t code,
    const char *file,
    int line,
    bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}


SVDGPU::SVDGPU(int nusers, int nmovies, int nfeatures)
      : nusers(nusers), nmovies(nmovies), nfeatures(nfeatures) {
    gpuErrChk (
        cudaMallocManaged(&user_features, nusers * nfeatures * sizeof(float))
    );

    fill_array(user_features, 0.1, nusers * nfeatures);

    gpuErrChk (
        cudaMallocManaged(&movie_features, nmovies * nfeatures * sizeof(float))
    );

    fill_array(movie_features, 0.1, nmovies * nfeatures);

    cudaDeviceSynchronize();
}

SVDGPU::~SVDGPU(void) {
    cudaFree(user_features);
    cudaFree(movie_features);
}

void SVDGPU::fit(uint32_t *users, uint32_t *movies, float *ratings,
                 int npoints, float lrate, float reg, int niters) {
    auto data = to_gpu(users, movies, ratings, npoints);
    for (int feature = 0; feature < nfeatures; ++feature) {
        std::cerr << "Calling kernel for feature " << feature << "...\n";
        train_feature(user_features, movie_features, data,
                      feature, lrate, reg, nfeatures, niters);
    }

    cudaFree(data.users);
    cudaFree(data.movies);
    cudaFree(data.ratings);

    for (int user = 0; user < nusers; ++user) {
        for (int feature = 0; feature < nfeatures; ++feature) {
            float u = user_features[user * nfeatures + feature];

            /*
            if (u < -1 || u > 1) {
                std::cerr << "We got a problem (" << u << ") at "
                          << "user " << user << ", "
                          << "feature " << feature << ".\n";
            }
            */
        }
    }

    for (int movie = 0; movie < nmovies; ++movie) {
        for (int feature = 0; feature < nfeatures; ++feature) {
            float m = movie_features[movie * nfeatures + feature];

            /*
            if (m < -1 || m > 1) {
                std::cerr << "We got a problem (" << m << ") at "
                          << "movie " << movie << ", "
                          << "feature " << feature << ".\n";
            }*/
        }
    }
}

float *SVDGPU::predict(uint32_t *users, uint32_t *movies, int npoints) {
    auto *result = new float[npoints];

    for (int i = 0; i < npoints; ++i) {
        result[i] = 0;
        for (int f = 0; f < nfeatures; ++f) {
            result[i] += user_features[users[i] * nfeatures + f]
                       * movie_features[movies[i] * nfeatures + f];
        }
    }

    return result;
}

Dataset SVDGPU::to_gpu(uint32_t *users, uint32_t *movies, float *ratings,
                       int count) {
   uint32_t *dev_users, *dev_movies;
   float *dev_ratings; 

   std::cerr << "Allocating device memory...\n";

   gpuErrChk(
       cudaMalloc(&dev_users, count * sizeof *dev_users)
   );
   gpuErrChk(
       cudaMalloc(&dev_movies, count * sizeof *dev_movies)
   );
   gpuErrChk(
       cudaMalloc(&dev_ratings, count * sizeof *dev_ratings)
   );

   std::cerr << "Done. Copying to device...\n";

   gpuErrChk(
       cudaMemcpy(dev_users, users, count * sizeof *dev_users,
                  cudaMemcpyHostToDevice)
   );
   gpuErrChk(
       cudaMemcpy(dev_movies, movies, count * sizeof *dev_movies,
                  cudaMemcpyHostToDevice)
   );
   gpuErrChk(
       cudaMemcpy(dev_ratings, ratings, count * sizeof *dev_ratings,
                  cudaMemcpyHostToDevice)
   );

   cudaDeviceSynchronize();

   std::cerr << "Done.\n";

   return Dataset(dev_users, dev_movies, dev_ratings, count);
}

