#include "Kernels.cuh"

#include <cstdio>

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

__global__ void compute_gradient_kernel(
        float *user_features, float *movie_features,
        Dataset data,
        int current_feature,
        float lrate, float reg, int nfeatures) {
    int stride = blockDim.x * gridDim.x;
    int start = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = start; i < data.count; i += stride) {
        uint32_t user = data.get_user(i);
        uint32_t movie = data.get_movie(i);

        /* Get `error` as actual - predicted. */
        float error = data.get_rating(i);

        for (int f = 0; f <= current_feature; ++f) {
            error -= user_features[user * nfeatures + f]
                   * movie_features[movie * nfeatures + f];
        }

        float *user_fp = user_features
                       + (user * nfeatures + current_feature);
        float *movie_fp = movie_features
                        + (movie * nfeatures + current_feature);

        atomicAdd(user_fp,
                  lrate * (error * (*movie_fp) - reg * (*user_fp)));
        atomicAdd(movie_fp,
                  lrate * (error * (*user_fp) - reg * (*movie_fp)));

        if (*user_fp < -5) *user_fp = -5;
        if (*user_fp > 5)  *user_fp = 5;
        if (isnan(*user_fp)) *user_fp = 10;

        if (*movie_fp < -5) *movie_fp = -5;
        if (*movie_fp > 5)  *movie_fp = 5;
        if (isnan(*movie_fp)) *movie_fp = 10;

    }
}

__global__ void fill_array_kernel(float *arr, float val, int size) {
    int start = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < size; i += stride) {
        arr[i] = val;
    }
}

void train_feature(float *user_features, float *movie_features,
                   Dataset data, int current_feature,
                   float lrate, float reg, int nfeatures,
                   int niters) {

    for (int i = 0; i < niters; ++i) {
        compute_gradient_kernel<<<1, 1024>>>(user_features, movie_features,
                                             data, current_feature,
                                             lrate, reg, nfeatures);
    }

    gpuErrChk(
        cudaDeviceSynchronize()
    );
}

void fill_array(float *arr, float val, int size) {
    fill_array_kernel<<<1024, 1024>>>(arr, val, size);
}
