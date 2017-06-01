#include <algorithm>
#include <cmath>
#include <cstdint>
#include <istream>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "SVD.hh"

struct DataPoint {
    uint32_t user;
    uint32_t movie;
    float rating;
};

struct Dataset {
    uint32_t *users;
    uint32_t *movies;
    float *ratings;
    int count;

    Dataset(void): users(nullptr), movies(nullptr), ratings(nullptr) {}

    Dataset(std::vector<DataPoint> &v): count(v.size()) {
        users = new uint32_t[count];
        movies = new uint32_t[count];
        ratings = new float[count];

        for (int i = 0; i < count; ++i) {
            users[i] = v[i].user;
            movies[i] = v[i].movie;
            ratings[i] = v[i].rating;
        }
    }

    void debug(void) {
        std::cerr << "Size: " << count << '\n';

        std::cerr << "5: (" << users[5]
                    << ", " << movies[5]
                    << ", " << (int)ratings[5] << ")\n";
    }

    uint32_t max_user(void) {
        return *std::max_element(users, users + count - 1);
    }

    uint32_t max_movie(void) {
        return *std::max_element(movies, movies + count - 1);
    }
};

void read_csv(std::istream &data_in, std::istream &indices_in,
              Dataset &base, Dataset &valid, Dataset &test) {
    std::vector<DataPoint> base_v;
    std::vector<DataPoint> valid_v;
    std::vector<DataPoint> test_v;

    std::string throwaway;

    // Toss the header.
    std::getline(data_in,    throwaway);
    std::getline(indices_in, throwaway);

    while (!data_in.eof()) {
        std::string user;
        std::string movie;
        std::string rating;
        std::string index;

        std::getline(data_in, user, ',');
        std::getline(data_in, movie, ',');
        std::getline(data_in, rating, ',');

        // Toss the timestamp.
        std::getline(data_in, throwaway);

        // Get the index from the other file.
        std::getline(indices_in, index);

        if (data_in.eof()) break;

        int i = std::stoi(index);

        auto point = DataPoint{(uint32_t)std::stoi(user) - 1,
                               (uint32_t)std::stoi(movie) - 1,
                               std::stof(rating)};
        switch (i) {
            case 0: base_v.push_back(point);
                    break;
            case 1: valid_v.push_back(point);
                    break;
            case 2: test_v.push_back(point);
                    break;
        }
    }

    base = Dataset(base_v);
    valid = Dataset(valid_v);
    test = Dataset(test_v);
}

int main(int argc, char **argv) {
    const std::string usage = std::string("usage: ")
                            + argv[0]
                            + " movielens_ratings movielens_indices";

    if (argc != 3) {
        std::cerr << usage << '\n';

        return 1;
    }

    std::ifstream data_in(argv[1]);
    std::ifstream indices_in(argv[2]);

    Dataset base;
    Dataset valid;
    Dataset test;

    read_csv(data_in, indices_in, base, valid, test);

    SVD model(base.max_user() + 1, base.max_movie() + 1, 8);

    model.fit(base.users, base.movies, base.ratings,
              base.count, 0.005, 0.02, 40, true);

    auto predictions = model.predict(valid.users, valid.movies, valid.count);

    float total = 0;
    for (int i = 0; i < valid.count; ++i) {
        float diff = predictions[i] - valid.ratings[i];

        total += diff * diff;
    }

    total /= valid.count;

    std::cerr << "RMSE: " << sqrt(total) << "\n";
}
