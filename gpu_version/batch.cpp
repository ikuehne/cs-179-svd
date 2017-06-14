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

std::vector<Batch> get_batches(DataPoint *base, int count) {
    std::list<Batcher> batchers;
    std::vector<Batch> batches;

    int nadded = 0;
    for (int i = 0; i < count; ++i) {
        if (i % 1000000 == 0) {
            std::cerr << "Batched " << i << " points with "
                      << batches.size() + batchers.size() << " batches.\n";
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
                            + " base_ratings"
                            + " output_file";

    if (argc != 3) {
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

    {
        std::ofstream batch_out(argv[2]);
        for (auto batch: batches) {
            batch.serialize(batch_out);
        }
    }
}
