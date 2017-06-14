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

/**
 * @brief The actual kernel that does the training.
 *
 * Each block trains with one input point.  Each thread in the block trains
 * one feature in the input.  Thus, gridDim has to be BATCH_SIZE, and blockDim
 * has to be nfeatures.
 *
 * We load all features for the given point into shared memory in parallel,
 * then perform the dot product with a reduction (thus shared memory has to
 * have space for 3 * BATCH_SIZE * nfeatures floats).  Then each thread
 * computes the gradient for its feature and performs the corresponding
 * update.
 */
__global__ void train_kernel(Batch batch,
                             float *user_features, float *movie_features,
                             float lrate, float reg) {
    extern __shared__ float shmem[];

    int nfeatures = blockDim.x;

    // Save some offsets into shared memory.
    float *user_feature = shmem;
    float *movie_feature = shmem + nfeatures;
    float *scratch = shmem + 2 * nfeatures;

    // Each block does one data point.
    DataPoint entry = batch.data[blockIdx.x];
    uint32_t user = entry.user;
    uint32_t movie = entry.movie;

    // The feature this thread is working on.
    int f = threadIdx.x;

    // Check if this is an empty entry.
    if (user == DONE_CODE) return;

    // Load in the features.
    user_feature[f] = user_features[user * nfeatures + f];
    movie_feature[f] = movie_features[movie * nfeatures + f];

    // (sync once the features are loaded).
    __syncthreads();

    // Do the products.
    scratch[f] = user_feature[f] * movie_feature[f];

    // (sync once the products are done).
    __syncthreads();

    // ... With a reduction for the sum.
    for (unsigned stride = nfeatures / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }

        __syncthreads();
    }

    // Compute the error in the present prediction.
    float err = entry.rating - scratch[0];

    // Based on that, compute the gradients.
    float user_grad = lrate * (err * movie_feature[f]
                             - reg * user_feature[f]);
    float movie_grad = lrate * (err * user_feature[f]
                              - reg * movie_feature[f]);

    // Perform the update.
    user_features[user * nfeatures + f] += user_grad;
    movie_features[movie * nfeatures + f] += movie_grad;
}

void train_batch(Batch b,
                 float *user_features, float *movie_features,
                 float lrate, float reg, int nfeatures) {
    // Just get the kernel parameters, which are computed straightforwardly
    // from the model parameters,
    int grid_dim = BATCH_SIZE;
    int block_dim = nfeatures;
    int shmem = nfeatures * 3 * sizeof(float);

    // and pass the data on to the kernel with those parameters.
    train_kernel<<<grid_dim, block_dim, shmem>>>
        (b, user_features, movie_features, lrate, reg);
}
