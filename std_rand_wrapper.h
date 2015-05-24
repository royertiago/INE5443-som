#ifndef STD_RAND_WRAPPER_H
#define STD_RAND_WRAPPER_H

/* Class that wraps the std::rand and std::srand function calls
 * into an interface compatible with the C++ standard library's
 * concept UniformRandomNumberGenerator.
 *
 * (Thus, in can be used with std::shuffle.)
 */

struct std_rand_wrapper {
    typedef unsigned result_type;
    result_type min() const;
    result_type max() const;
    result_type operator()();

// Utility functions

    /* Calls std::srand with the given value.
     */
    static void seed( unsigned new_seed );

    /* Calls std::srand with a seed based on current time.
     */
    static void time_seed();

    /* Retrieves the last seed value chosen through this class.
     */
    static unsigned seed();
};

#endif // STD_RAND_WRAPPER_H
