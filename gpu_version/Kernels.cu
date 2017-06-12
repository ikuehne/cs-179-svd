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

__global__ void train_kernel(Batch batch,
                             float *user_features, float *movie_features,
                             float lrate, float reg) {
    extern __shared__ float shmem[];

    int nfeatures = blockDim.x;

    float *user_feature = shmem;
    float *movie_feature = shmem + nfeatures;
    float *scratch = shmem + 2 * nfeatures;

    // Each block does one data point.
    DataPoint entry = batch.data[blockIdx.x];

    uint32_t user = entry.user;
    uint32_t movie = entry.movie;

    int f = threadIdx.x;

    // Check if this is an empty entry.
    if (user == DONE_CODE) return;

    // Load in the features.
    user_feature[f] = user_features[user * nfeatures + f];
    movie_feature[f] = movie_features[movie * nfeatures + f];

    __syncthreads();

    // Do the dot product.
    scratch[f] = user_feature[f] * movie_feature[f];

    __syncthreads();

    // ... With a reduction for the sum.
    for (unsigned stride = nfeatures / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }

        __syncthreads();
    }

    float err = entry.rating - scratch[0];

    float user_grad = lrate * (err * movie_feature[f]
                             - reg * user_feature[f]);
    float movie_grad = lrate * (err * user_feature[f]
                              - reg * movie_feature[f]);

    /*
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("user: %d\n", user);
        printf("movie: %d\n", movie);
        printf("rating: %g\n", entry.rating);

        printf("prediction: %g\n", scratch[0]);
        printf("user_grad: %g\n", user_grad);
        printf("movie_grad: %g\n", movie_grad);
    }
    */

    user_features[user * nfeatures + f] += user_grad;
    movie_features[movie * nfeatures + f] += movie_grad;

    /*
    float uf = user_features[user * nfeatures + f];
    if (threadIdx.x == 0 && blockIdx.x == 0 && (fabsf(uf) > 1 || isnan(uf))) {
        printf("Got a weird feature: %g at %d, %d\n", uf, user, f);
    }
    */
}

__global__ void fill_array_kernel(float *arr, float val, int size) {
    int start = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < size; i += stride) {
        arr[i] = val;
    }
}

void train_batch(Batch b,
                 float *user_features, float *movie_features,
                 float lrate, float reg, int nfeatures) {
    int grid_dim = BATCH_SIZE;
    int block_dim = nfeatures;
    int shmem = nfeatures * 3 * sizeof(float);

    train_kernel<<<grid_dim, block_dim, shmem>>>
        (b, user_features, movie_features, lrate, reg);

    gpuErrChk(
        cudaDeviceSynchronize()
    );
}

void fill_array(float *arr, float val, int size) {
    fill_array_kernel<<<1024, 1024>>>(arr, val, size);
}
