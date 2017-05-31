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
    uint8_t rating;
};

struct Dataset {
    uint32_t *users;
    uint32_t *movies;
    uint8_t *ratings;
    int count;

    Dataset(void): users(nullptr), movies(nullptr), ratings(nullptr) {}

    Dataset(std::vector<DataPoint> &v): count(v.size()) {
        users = new uint32_t[count];
        movies = new uint32_t[count];
        ratings = new uint8_t[count];

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
                               (uint8_t)(std::stof(rating) * 2)};
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

    base.debug();
    valid.debug();
    test.debug();
}
