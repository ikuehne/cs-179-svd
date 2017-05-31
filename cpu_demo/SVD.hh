/**
 * @file SVD.hh
 */

#pragma once

#include <memory>
#include <vector>

// Forward-declare implemenatation.
class SVDImpl;

/**
 * @brief The C++ interface to the SVD engine.
 *
 * Allows building a model, training it on (user, movie, rating) tuples, and
 * making rating predictions for (user, movie) tuples.
 */
class SVD {
public:
    /**
     * @brief Create a new SVD model.
     *
     * @param nusers The total number user IDs.  Should be equal to the
     *               maximum user ID plus one.
     * @param nmovies Same, for movies.
     */
    SVD(int nusers, int nmovies, int nfeatures);

    /**
     * @brief Train on the provided data.
     *
     * @param users An array of user IDs.
     * @param movies An array of movie IDs corresponding to `users`.
     * @param ratings An array of user ratings corresponding to `users` and
     *                `movies`, in the sense that `ratings[i]` is the rating
     *                `users[i]` gave `movies[i]`.
     * @param npoints The size of the input arrays (must all be the same).
     * @param lrate The learning rate.  Higher settings learn faster, but tend
     *              to find local minima.
     * @param niters The number of iterations to run on each feature.
     * @param verbose Whether to print progress while training.
     */
    void fit(uint32_t *users, uint16_t *movies, float *ratings,
               int npoints, float lrate, float reg, int niters, bool verbose);

    /**
     * @brief Make a prediction of the ratings of the given users on the given
     *        movies.
     *
     * @param users As in `train`.
     * @param movies As in `train`.
     * @param npoints As in `train`.
     *
     * @return An array (call it `ratings`) such that ratings[i] is the
     *         predicted rating of `users[i]` for `movies[i]`.  The result has
     *         length `npoints`, and is owned by the caller.
     */
    float *predict(uint32_t *users, uint16_t *movies, int npoints);

    // Forward-declare this, since we don't yet know the size of SVDImpl.
    ~SVD();

private:

    /**
     * @brief A pointer to the SVDImpl object that actually does the work.
     */
    SVDImpl *pimpl;
};
