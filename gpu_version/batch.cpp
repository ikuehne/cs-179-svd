#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <istream>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Data.hh"

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

    std::cerr << "Shuffling..." << std::endl;

    std::random_shuffle(base_v.begin(), base_v.end());

    std::cerr << "Done shuffling." << std::endl;

    base = Dataset(base_v);
    valid = Dataset(valid_v);
    test = Dataset(test_v);
}

struct IndependentDataset {
    std::vector<bool> users;
    std::vector<bool> movies;

    IndependentDataset(uint32_t nusers, uint32_t nmovies)
          : users(nusers), movies(nmovies) {}

    bool try_add(uint32_t user, uint32_t movie) {
        bool result = !(users[user] || movies[movie]);

        if (result) {
            users[user] = true;
            movies[movie] = true;
        }

        return result;
    }
};

int count_independent_sets(Dataset base) {
    std::vector<IndependentDataset> sets;

    uint32_t nusers = base.max_user() + 1;
    uint32_t nmovies = base.max_movie() + 1;

    for (int i = 0; i < base.count; ++i) {
        if (i % 50000 == 0) {
            std::cerr << "At point " << i << 
                         " with " << sets.size() << ".\n";
        }

        uint32_t user = base.users[i];
        uint32_t movie = base.movies[i];

        bool success = false;

        for (int j = 0; j < (int)sets.size(); ++j) {
            if (sets[j].try_add(user, movie)) {
                success = true;
                break;
            }
        }

        if (!success) {
            sets.push_back(IndependentDataset(nusers, nmovies));
            assert(sets.back().try_add(user, movie));
        }
    }

    return sets.size();
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

    std::cout << "Count: " << count_independent_sets(base) << "\n";
}
