#include <cuda_runtime.h>
#include <curand.h>
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

SVDGPU::SVDGPU(int nusers, int nmovies, int nfeatures): nfeatures(nfeatures) {
    curandGenerator_t gen;

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    gpuErrChk (
        cudaMallocManaged(&user_features, nusers * nfeatures * sizeof(float))
    );

    curandGenerateNormal(gen, user_features, nusers * nfeatures, 0.1, 0.05);

    gpuErrChk (
        cudaMallocManaged(&movie_features, nmovies * nfeatures * sizeof(float))
    );

    curandGenerateNormal(gen, movie_features, nmovies * nfeatures, 0.1, 0.05);

    gpuErrChk(
        cudaDeviceSynchronize()
    );
}

SVDGPU::~SVDGPU(void) {
    cudaFree(user_features);
    cudaFree(movie_features);
}

void SVDGPU::fit(DataPoint *batches, int nbatches,
                 float lrate, float reg, int niters) {
    int data_bytes = nbatches * BATCH_SIZE * sizeof(DataPoint); 
    DataPoint *dev_batches;
    cudaMalloc(&dev_batches, data_bytes);

    std::cerr << "Copying data to device...\n";
    cudaMemcpy(dev_batches, batches, data_bytes, cudaMemcpyHostToDevice);
    std::cerr << "Done copying data...\n";
    for (int i = 0; i < niters; ++i) {
        std::cerr << "Iteration " << i + 1 << ".\n";
        for (int batch = 0; batch < nbatches; ++batch) {

            Batch b;

            b.data = dev_batches + batch * BATCH_SIZE;

            train_batch(b, user_features, movie_features,
                        lrate, reg, nfeatures);
        }
    }

    gpuErrChk(
        cudaDeviceSynchronize()
    );

    std::cerr << "All done!\n";

    cudaFree(dev_batches);
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
