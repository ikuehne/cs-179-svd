#pragma once

#include <vector>

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


