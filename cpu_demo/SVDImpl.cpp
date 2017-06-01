/**
 * @file SVDImpl.cpp
 *
 * The actual implementation of the SVD engine.  Heavily based on the
 * algorithms described in Simon Funk's blog post (at
 * http://sifter.org/~simon/journal/20061211.html).
 */

#include <cmath>
#include <cstring>
#include <ctime>

#include <iostream>

#include "SVDImpl.hh"

SVDImpl::SVDImpl(int nusers, int nmovies, int nfeatures)
      : nfeatures(nfeatures), gen(std::random_device()()),
        user_features(nfeatures, nusers), movie_features(nfeatures, nmovies) {
    // Set the features to 0.
    user_features.block(0, 0, nfeatures, nusers).setConstant(0);
    movie_features.block(0, 0, nfeatures, nmovies).setConstant(0);

}

void SVDImpl::fit(uint32_t *users, uint32_t *movies, float *ratings,
                    int npoints, float lrate, float reg, int niters, bool verbose) {
    /* Convert the different C arrays to Eigen types. */
    Eigen::Map<Eigen::Array<uint32_t, 1, Eigen::Dynamic> >
        user_arr(users, 1, npoints);
    Eigen::Map<Eigen::Array<uint32_t, 1, Eigen::Dynamic> >
        movie_arr(movies, 1, npoints);
    Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic> >
        rating_arr(ratings, 1, npoints);

    /* Allocate this once at the beginning, to speed things up. */
    Eigen::Array<float, 1, Eigen::Dynamic> error(1, npoints);
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> residuals(1, npoints);
    residuals.block(0, 0, 1, npoints).setConstant(0);

    /* Time how long it takes to train each feature. */
    time_t start = time(0);

    for (int i = 0; i < nfeatures; ++i) {
        if (verbose) {
            std::cerr << "==============================" << std::endl;
            std::cerr << "Feature " << i << std::endl;
            std::cerr << "==============================" << std::endl;
        }

        user_features.row(i).setConstant(0.1);
        movie_features.row(i).setConstant(0.1);

        // Initialize the feature.
        // user_features.row(i) = user_features.row(i).unaryExpr([=](float) -> float {
        //     return 0.01 * gaussian(gen);
        // });

        // movie_features.row(i) = movie_features.row(i).unaryExpr([=](float) -> float {
        //     return 0.01 * gaussian(gen);
        // });

        // float prev_rmse;
        // float rmse;

        for (int j = 0; j < niters; ++j) {
            // if ((j > 0) && (rmse >= prev_rmse)) {
            //     break;
            // }

            if (verbose && j % 10 == 0) {
                // Calculate the training set RMSE.
                predict_inplace(user_arr, movie_arr, error);
                // After this, error = predicted - actual
                error -= rating_arr;

                // So this is the RMSE.
                float rmse = sqrt(error.square().mean());
                // if (j > 0) {
                //     prev_rmse = rmse;
                // }

                std::cerr << "RMSE: " << rmse << std::endl;
                std::cerr << "Iteration " << j << std::endl << std::endl;
            }

#pragma omp parallel for
            // k -- index into the input data.  So user_arr(k) is the kth
            // user in the training data.
            for (int k = 0; k < npoints; k++) {
                // Get the error as the actual rating
                float err = rating_arr(k);
                // minus the dot product of the user features and the movie
                // features.
                // err -= predict(user_arr(k), movie_arr(k));

                if (j != niters - 1) {
                    err -= residuals(k);
                    err -= movie_features(i, movie_arr(k)) * user_features(i, user_arr(k));
                } else {
                    residuals(k) += movie_features(i, movie_arr(k)) * user_features(i, user_arr(k));
                    err -= residuals(k);
                }

                // Now we actually modify the features: save the old user
                // feature,
                float old = user_features(i, user_arr(k));
                // Adjust the user feature according to the movie feature,
                user_features(i, user_arr(k)) +=
                    lrate * (err * movie_features(i, movie_arr(k))
                    - reg * user_features(i, user_arr(k)));
                // and adjust the movie feature according to the saved user
                // feature.
                movie_features(i, movie_arr(k)) +=
                    lrate * (err * old - reg * movie_features(i, movie_arr(k)));
            }
        }

        time_t current = time(0);
        double elapsed = difftime(current, start);
        if (verbose) {
            std::cerr << "Elapsed time: " << elapsed << "s" << std::endl << std::endl;
        }
    }
}

inline float SVDImpl::predict(uint32_t user, uint32_t movie) const {
    return (user_features.col(user).array()
          * movie_features.col(movie).array()).sum();
}

float *SVDImpl::predict(uint32_t *users, uint32_t *movies, int npoints) {
    // Convert the C pointers to Eigen arrays.
    Eigen::Map<Eigen::Array<uint32_t, 1, Eigen::Dynamic> >
        user_arr(users, 1, npoints);
    Eigen::Map<Eigen::Array<uint32_t, 1, Eigen::Dynamic> >
        movie_arr(movies, 1, npoints);


    // In particular, allocate a new array for the result.
    Eigen::Array<float, 1, Eigen::Dynamic>result_arr (1, npoints);

    // Perform the prediction in-place.
    predict_inplace(user_arr, movie_arr, result_arr);

    // Copy the result to a C array.
    float *result = new float[npoints];
    memcpy(result, result_arr.data(), npoints * sizeof(float));

    // And return that.
    return result;
}

inline void SVDImpl::predict_inplace(
        const Eigen::Array<uint32_t, 1, Eigen::Dynamic> &users,
        const Eigen::Array<uint32_t, 1, Eigen::Dynamic> &movies,
        Eigen::Array<float, 1, Eigen::Dynamic> &result) {
    // Go through each point to predict.
    for (int i = 0; i < users.cols(); i++) {
        // The result is the dot product of this user's features and this
        // movie's features.
        result(i) = predict(users[i], movies[i]);
    }
}
