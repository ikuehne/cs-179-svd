#pragma once

#include <istream>
#include <limits>
#include <ostream>
#include <unordered_set>
#include <vector>

const uint32_t DONE_CODE = 0xFFFFFFFF;

const int BATCH_SIZE = 256;

struct DataPoint {
    uint32_t user;
    uint32_t movie;
    float rating;
};

struct Batch {
    DataPoint *data;

    Batch(void) {
        data = new DataPoint[BATCH_SIZE];
    }

    Batch(std::istream &in) {
        data = new DataPoint[BATCH_SIZE];

        in.read((char *)data, BATCH_SIZE * sizeof(DataPoint));
    }

    void serialize(std::ostream &out) {
        out.write((char *)data, BATCH_SIZE * sizeof(DataPoint));
    }
};

class Batcher {
    std::unordered_set<uint32_t> users;
    std::unordered_set<uint32_t> movies;
    Batch batch;
    int count;

public:
    Batch finish_batch(void) {
        for (; count < BATCH_SIZE; ++count) {
            batch.data[count].user = DONE_CODE;
        }

        return batch;
    }

    bool add_point(DataPoint p) {
        if (full()) return false;

        if (users.count(p.user)) return false;
        if (movies.count(p.movie)) return false;

        users.insert(p.user);
        movies.insert(p.movie);

        batch.data[count] = p;

        ++count;

        return true;
    }

    bool full(void) {
        return count >= BATCH_SIZE;
    }
};

struct Dataset {
    uint32_t *users;
    uint32_t *movies;
    float *ratings;
    int count;

    Dataset(void): users(nullptr), movies(nullptr), ratings(nullptr) {}

    Dataset(uint32_t *users, uint32_t *movies, float *ratings, int count)
        : users(users), movies(movies), ratings(ratings), count(count) {}

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

#ifdef __CUDACC__
    __host__ __device__
#endif
    uint32_t get_user(int i) {
        return users[i];
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    uint32_t get_movie(int i) {
        return movies[i];
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    float get_rating(int i) {
        return ratings[i];
    }

    uint32_t max_user(void) {
        return *std::max_element(users, users + count - 1);
    }

    uint32_t max_movie(void) {
        return *std::max_element(movies, movies + count - 1);
    }
};
