/**
 * @file SVD.cpp
 */

//
// `SVD` does not (presently) do any work; everything is outsourced to an
// `SVDImpl` object (PImpl idiom).  See SVDImpl.cpp for actual
// implementations.
//

#include "SVD.hh"
#include "SVDImpl.hh"

SVD::SVD(int nusers, int nmovies, int nfeatures) 
    : pimpl(new SVDImpl(nusers, nmovies, nfeatures)) {}

void SVD::fit(uint32_t *users, uint32_t *movies, float *ratings,
                int npoints, float lrate, float reg, int niters, bool verbose) {
    pimpl->fit(users, movies, ratings, npoints, lrate, reg, niters, verbose);
}

float *SVD::predict(uint32_t *users, uint32_t *movies, int npoints) {
    return pimpl->predict(users, movies, npoints);
}

SVD::~SVD() { delete pimpl; }
