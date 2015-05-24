#include "std_rand_wrapper.h"
#include "kohonen_pallete.h"

KohonenPallete::KohonenPallete(
    std::size_t rows,
    std::size_t columns,
    const std::vector<color>& input_pallete
)
    :
    _activation_function( 2.0, 1 ),
    _distance_function(),
    _network_container(),
    _data( input_pallete.size(), data_type(3) ),

    // Training-related functors
    _topology(),
    _space_function( 100, 1 ), // This 100 seems to have no effect in the training...
    _network_function( 10, 1 ), // This 10 is the influence area.
    _weights( _network_function, _space_function, _topology, _distance_function ),
    _training_function( _weights, 0.3 ),
    _training_algorithm( _training_function )
{
    /* _data will be the same as input_pallete, but normalized to [0, 1].
     */
    for( int i = 0; i < input_pallete.size(); i++ ) {
        _data[i][0] = input_pallete[i].r / 255.0;
        _data[i][1] = input_pallete[i].g / 255.0;
        _data[i][2] = input_pallete[i].b / 255.0;
    }

    /* KNNL uses std::rand to generate random numbers.
     * There are two "randomization policies", the Internal_randomize
     * (that calls std::srand with std::time(0))
     * and External_randomize
     * (that does nothing).
     * Using External_randomize we are able to control the seeding process externally,
     * so we will use this seeding procedure.
     */
    neural_net::External_randomize disable_seeding;

    neural_net::generate_kohonen_network(
        rows,
        columns,
        _activation_function,
        _distance_function,
        _data,
        _network_container,
        disable_seeding
    );
}

void KohonenPallete::train( double influence_radius ) {
    _training_algorithm.
        training_functional.
        generalized_training_weight.
        network_function.
        sigma = influence_radius;

    _training_algorithm( _data.begin(), _data.end(), &_network_container );
    std::shuffle( _data.begin(), _data.end(), std_rand_wrapper() );
}

color KohonenPallete::operator()( std::size_t i, std::size_t j ) const {
    return color{
        // Adding 0.5 at end correctly rounds the weights.
        (unsigned) (_network_container.objects[i][j].weights[0] * 255 + 0.5),
        (unsigned) (_network_container.objects[i][j].weights[1] * 255 + 0.5),
        (unsigned) (_network_container.objects[i][j].weights[2] * 255 + 0.5),
    };
}

void KohonenPallete::activation_function_scaling_factor( double factor ) {
    _activation_function.sigma = factor;
}

double KohonenPallete::activation_function_scaling_factor() const {
    return _activation_function.sigma;
}
