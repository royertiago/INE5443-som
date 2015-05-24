#include <cstdlib>
#include <ctime>
#include "std_rand_wrapper.h"

std_rand_wrapper::result_type std_rand_wrapper::min() const {
    return 0;
}

std_rand_wrapper::result_type std_rand_wrapper::max() const {
    return RAND_MAX;
}

std_rand_wrapper::result_type std_rand_wrapper::operator()() {
    return std::rand();
}

namespace {
    unsigned current_seed = 1;
} // anonymous namespace

void std_rand_wrapper::seed( unsigned new_seed ) {
    std::srand( new_seed );
    current_seed = new_seed;
}

void std_rand_wrapper::time_seed() {
    seed( std::time(0) );
}

unsigned std_rand_wrapper::seed() {
    return current_seed;
}
