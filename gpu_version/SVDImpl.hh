/**
 * @file SVDImpl.hh
 */

#pragma once

#include <cstdint>
#include <random>

#include <eigen3/Eigen/Core>

/**
 * @brief Contains the actual implementations of the methods used in SVD.
 *
 * See SVD.hh for documentation on what the public methods do.
 */
class SVDImpl {
public:
    SVDImpl(int nusers, int nmovies, int nfeatures);

    void fit(uint32_t *users, uint32_t *movies, float *ratings,
               int npoints, float lrate, float reg, int niters, bool verbose);

    float *predict(uint32_t *users, uint32_t *movies, int npoints);

private:

    /**
     * @brief Make a prediction at the given (user, movie) pair.
     */
    inline float predict(uint32_t user, uint32_t movie) const;

    /**
     * @brief As `predict`, but in-place (to avoid allocations) and on Eigen
     *        types.
     *
     * @param result Overwritten with the predictions on `users` and `movies`.
     */
    inline void predict_inplace(
            const Eigen::Array<uint32_t, 1, Eigen::Dynamic> &users,
            const Eigen::Array<uint32_t, 1, Eigen::Dynamic> &movies,
            Eigen::Array<float, 1, Eigen::Dynamic> &result);

    int nfeatures;

    /**
     * @brief The user-feature matrix.
     *
     * Each row is a feature, and each column is a user.
     */
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> user_features;

    /**
     * @brief The movie-feature matrix.
     *
     * Each row is a feature, and each column is a movie.
     */
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> movie_features;

    std::mt19937 gen;
    std::normal_distribution<float> gaussian;
};
