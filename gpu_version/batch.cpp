#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <istream>
#include <fstream>
#include <iostream>
#include <list>
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

std::vector<Batch> get_batches(DataPoint *base, int count) {
    std::list<Batcher> batchers;
    std::vector<Batch> batches;

    int nadded = 0;
    for (int i = 0; i < count; ++i) {
        if (i % 50000 == 0) {
            std::cerr << "At point " << i << 
                         " with " << batches.size() + batchers.size()
                      << ".\n";
        }

        uint32_t user = base[i].user;
        uint32_t movie = base[i].movie;
        float rating = base[i].rating;

        bool success = false;

        for (auto batcheri = batchers.begin();
             batcheri != batchers.end();
             batcheri++) {
            auto &batcher = *batcheri;
            if (batcher.add_point(DataPoint{user, movie, rating})) {
                success = true;

                if (batcher.full()) {
                    batches.push_back(batcher.finish_batch());
                    batchers.erase(batcheri);
                }
                break;
            }
        }

        if (!success) {
            batchers.push_back(Batcher());
            nadded++;
            assert(batchers.back().add_point(DataPoint{user, movie, rating}));
            if (nadded % 10 == 0) {
                batchers.reverse();
            }
        }
    }

    for (auto &batcher: batchers) {
        batches.push_back(batcher.finish_batch());
    }

    return batches;
}

int main(int argc, char **argv) {
    const std::string usage = std::string("usage: ")
                            + argv[0]
                            + " base_ratings";

    if (argc != 2) {
        std::cerr << usage << '\n';

        return 1;
    }

    std::ifstream data_in(argv[1], std::ios::binary | std::ios::ate);
    int size = data_in.tellg();
    int count = size / sizeof(DataPoint);
    data_in.seekg(0);

    DataPoint *base = new DataPoint[count];

    data_in.read((char *)base, size);

    std::cerr << "Shuffling..." << std::endl;

    std::random_shuffle(base, base + count);

    std::cerr << "Done shuffling." << std::endl;

    auto batches = get_batches(base, count);

    uint32_t test_user = batches[6].data[4].user;
    uint32_t test_movie = batches[6].data[4].movie;
    float test_rating = batches[6].data[4].rating;

    {
        std::ofstream batch_out("batches.out");
        for (auto batch: batches) {
            batch.serialize(batch_out);
        }
    }

    std::ifstream batch_in("batches.out", std::ios::binary | std::ios::ate);

    std::vector<Batch> test_batches;

    size = batch_in.tellg();
    batch_in.seekg(0);

    char *buf = new char[size];

    batch_in.read(buf, size);

    int incr = BATCH_SIZE * sizeof(DataPoint);
    for (int i = 0; i + incr <= size; i += incr) {
        Batch b;
        b.data = (DataPoint *)(buf + i);
        test_batches.push_back(b);
    }

    uint32_t new_user = test_batches[6].data[4].user;
    uint32_t new_movie = test_batches[6].data[4].movie;
    float new_rating = test_batches[6].data[4].rating;

    std::cout << "Old count " << batches.size()
              << "; now " << test_batches.size() << ".\n";
    std::cout << "Old user " << test_user << "; now " << new_user << ".\n";
    std::cout << "Old movie " << test_movie << "; now " << new_movie << ".\n";
    std::cout << "Old rating " << test_rating
              << "; now " << new_rating << ".\n";
}
