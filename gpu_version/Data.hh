#pragma once

#include <istream>
#include <limits>
#include <fstream>
#include <ostream>
#include <unordered_set>
#include <vector>

/**
 * @brief A code to indicate when a DataPoint is intended as padding.
 */
const uint32_t DONE_CODE = 0xFFFFFFFF;

/**
 * @brief The number of points in each batch.
 */
const int BATCH_SIZE = 256;

/**
 * @brief A single rating.
 */
struct DataPoint {
    uint32_t user;
    uint32_t movie;
    float rating;
};

/**
 * @brief A single batch, containing no two equal user or movie IDs.
 */
struct Batch {
    /**
     * @brief The actual data.
     */
    DataPoint *data;

    Batch(void) { data = NULL; }

    /**
     * @brief Write the batch's data to the given stream in binary form.
     *
     * (Just a bytewise copy of the data--endianness will match the CPU doing
     * the serialization).
     */
    void serialize(std::ostream &out) {
        out.write((char *)data, BATCH_SIZE * sizeof(DataPoint));
    }
};

/**
 * @brief Utility for creating batches.
 */
class Batcher {
    /**
     * @brief The user IDs the batch contains so far.
     */
    std::unordered_set<uint32_t> users;

    /**
     * @brief The movie IDs the batch contains so far.
     */
    std::unordered_set<uint32_t> movies;

    /**
     * @brief The batch itself.
     */
    Batch batch;

    /**
     * @brief The number of points in the batch.
     */
    int count;

public:
    Batcher(void) {
        batch.data = new DataPoint[BATCH_SIZE];
        count = 0;
    }

    /**
     * @brief Fill the batch up (potentially with padding) to `BATCH_SIZE` and
     *        return it.
     */
    Batch finish_batch(void) {
        for (; count < BATCH_SIZE; ++count) {
            batch.data[count].user = DONE_CODE;
        }

        return batch;
    }

    /**
     * @brief Add a point and return `true` if possible; otherwise return
     *        `false`.
     */
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

    /**
     * @brief Is the batch in progress full yet?
     */
    bool full(void) {
        return count >= BATCH_SIZE;
    }
};

/**
 * @brief Read the entire contents of a binary file, returning them as a byte
 *        array.
 */
std::pair<int, char *>read_file(std::string fname);
