/**
 * @file SVDGPU.hh
 *
 * Interface to the GPU implementation of SVD.
 */

#pragma once

#include "Data.hh"

/**
 * @brief The "Singular Value Decomposition" collaborative filtering model.
 *
 * Despite the name, this model does not actually compute an SVD in the
 * traditional linear algebra sense.  To understand where the name comes from,
 * consider the data as an nusers x nmovies matrix D where each entry is a
 * rating.  Thus, for any real dataset we only know some of the entries, and
 * we wish to predict some of the others.  This model computes an
 * nusers x f matrix A (for some positive integer f) and an f x nmovies matrix
 * B, such that the quantity
 *
 * sum_{user, movie}(AB_{user, movie} - D_{user, movie})^2
 *
 * is minimized; that is, such that the product AB approximates the known
 * entries of D as well as possible.
 *
 * This version is GPU-accelerated.
 */
class SVDGPU {
public:
    /**
     * @brief Create a new SVDGPU model.
     *
     * @param nusers The total number of distinct users, or equivalently the
     *               maximum user plus one.
     * @param nmovies The total number of distinct movies.
     * @param nfeatures The number of features to learn.  Performance will be
     *                  best for multiples of the size of a warp (i.e. 32).
     */
    SVDGPU(int nusers, int nmovies, int nfeatures);

    /**
     * @brief Fit the model to the provided data.
     *
     * @param batches The input data, as an array of `DataPoint`s.  Should be
     *                arranged such that each `BATCH_SIZE` chunk (i.e., 0-255,
     *                256-511, ...) has `BATCH_SIZE` distinct users and
     *                `BATCH_SIZE` distinct movies.  (This allows each block
     *                to work independently).  Any entries with the user set
     *                to `DONE_CODE` will be ignored, which allows for partial
     *                batches if not all points can be arranged in
     *                that manner..
     * @param nbatches The number of batches in `batches.
     * @param lrate The learning rate.
     * @param reg The regularization parameter.
     * @param niters The number of iterations to run over the whole training
     *               set.
     */
    void fit(DataPoint *batches, int nbatches,
             float lrate, float reg, int niters);

    /**
     * @brief Make predictions for the ratings of the given user/movie pairs.
     *
     * @param users An array of users.
     * @param movies An array of movies.
     * @param npoints The number of user/movie pairs to predict.
     *
     * @return An array of ratings such that `ratings[i]` is `users[i]`'s
     *         predicted rating of `movies[i]`.  Allocated on the heap, and
     *         owned by the caller.
     */
    float *predict(uint32_t *users, uint32_t *movies, int npoints);

    ~SVDGPU(void);

private:
    /**
     * @brief The user features array.
     *
     * Managed memory, which is transferred between the CPU and GPU as needed.
     * Each row is a user, each column is a feature.
     */
    float *user_features;

    /**
     * @brief The movie features array.
     *
     * Managed memory, which is transferred between the CPU and GPU as needed.
     * Each row is a movie, each column is a feature.
     */
    float *movie_features;

    /**
     * @brief The number of features to compute for each user and movie.
     */
    int nfeatures;
};
