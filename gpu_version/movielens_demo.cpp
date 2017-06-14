#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <istream>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ta_utilities.hpp"
#include "SVDGPU.hh"

/**
 * Find the maximum user and movie from the given dataset.
 */
std::pair<uint32_t, uint32_t> maxima(DataPoint *data, int count) {
    uint32_t max_user = 0;
    uint32_t max_movie = 0;
    for (int i = 0; i < count; ++i) {
        auto entry = data[i];
        if (entry.user == DONE_CODE) {
            continue;
        }
        if (entry.user > max_user) {
            max_user = entry.user;
        }
        if (entry.movie > max_movie) {
            max_movie = entry.movie;
        }
    }

    return std::pair<uint32_t, uint32_t>(max_user, max_movie);
}

int main(int argc, char **argv) {
    // Check the command-line arguments.
    const std::string usage = std::string("usage: ")
                            + argv[0]
                            + " batches valid";

    if (argc != 3) {
        std::cerr << usage << '\n';

        return 1;
    }

    // Read the data in.
    auto base = read_file(argv[1]);
    auto valid = read_file(argv[2]);

    // Compute the max user/movie.
    auto base_maxima = maxima((DataPoint *)base.second,
                              base.first / sizeof(DataPoint));
    auto valid_maxima = maxima((DataPoint *)valid.second,
                               valid.first / sizeof(DataPoint));

    auto max_user = std::max(base_maxima.first, valid_maxima.first);
    auto max_movie = std::max(base_maxima.second, valid_maxima.second);
    
    // Create the model.
    SVDGPU model(max_user + 1, max_movie + 1, 256);

    // Count the batches.
    int nbatches = base.first / (sizeof(DataPoint) * BATCH_SIZE);

    //TA_Utilities::select_coldest_GPU();

    // Fit the model.
    model.fit((DataPoint *)base.second, nbatches, 0.001, 0.02, 50);

    // Get the validation set ready for predicting.
    int valid_count = valid.first / sizeof(DataPoint);

    uint32_t *valid_users = new uint32_t[valid_count];
    uint32_t *valid_movies = new uint32_t[valid_count];

    auto valid_data = (DataPoint *)valid.second;

    for (int i = 0; i < valid_count; ++i) {
        valid_users[i] = valid_data[i].user;
        valid_movies[i] = valid_data[i].movie;

        assert(valid_data[i].user <= max_user);
    }

    // Get predictions on the validation set.
    auto predictions = model.predict(valid_users, valid_movies, valid_count);

    delete[] valid_users;
    delete[] valid_movies;

    // Compute the RMSE.
    float total = 0;
    for (int i = 0; i < valid_count; ++i) {
        float diff = predictions[i] - valid_data[i].rating;
        total += diff * diff;
    }

    total /= valid_count;

    // Print it out.
    std::cerr << "RMSE: " << sqrt(total) << "\n";

    delete[] base.second;
    delete[] valid.second;
}
