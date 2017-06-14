#include <utility>
#include <fstream>

#include "Data.hh"

std::pair<int, char *>read_file(std::string fname) {
    std::ifstream in(fname, std::ios::binary | std::ios::ate);
    int size = in.tellg();
    char *data = new char[size];

    in.seekg(0);

    in.read(data, size);

    return std::pair<int, char *>(size, data);
}
