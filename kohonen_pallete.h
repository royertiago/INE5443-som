#ifndef KOHONEN_PALLETE_H
#define KOHONEN_PALLETE_H

#include <knnl/neural_net_headers.hpp>

/* Class that encapsulates a Kohonen network
 * ready to learn a color pallete.
 */

/* Data structure returned by the network
 * to represent which color some neuron learned.
 */
struct color {
    unsigned r;
    unsigned g;
    unsigned b;
};


class KohonenPallete {
public:
    typedef std::vector<double> data_type;

    // Neural net typedefs
    typedef
        neural_net::Cauchy_function
        <
            double,
            double,
            int
        >
        activation_function_type;
    typedef
        distance::Euclidean_distance_function
        <
            data_type
        >
        distance_function_type;
    typedef
        neural_net::Basic_neuron
        <
            activation_function_type,
            distance_function_type
        >
        neuron_type;
    typedef
        neural_net::Rectangular_container
        <
            neuron_type
        >
        network_container_type;

    // Winner-takes-all algorithm typedefs
    typedef
        neural_net::Wta_proportional_training_functional
        <
            data_type,
            double,
            int
        >
        wta_training_function_type;
    typedef
        neural_net::Wta_training_algorithm
        <
            network_container_type,
            data_type,
            std::vector<data_type>::iterator,
            wta_training_function_type
        >
        wta_training_algorithm_type;

    /* Constructs a Kohonen network in a rectangular grid
     * with the given amount or rows and columns,
     * to be trained with the specified color list.
     */
    KohonenPallete(
        std::size_t rows,
        std::size_t columns,
        const std::vector< color >& input_pallete
    );

    /* Run one training round with the Winner-takes-all algogrithm.
     */
    void train();

    /* Returns the weights of the specified node within the network;
     * that is, which color that node had learned.
     */
    color operator()( std::size_t i, std::size_t j ) const;

    /* Gets/sets the activation function scaling factor.
     * Default value: 2.0.
     */
    void activation_function_scaling_factor( double );
    double activation_function_scaling_factor() const;

private:
    activation_function_type _activation_function;
    distance_function_type _distance_function;
    network_container_type _network_container;
    std::vector< data_type > _data;
    wta_training_function_type _training_function;
    wta_training_algorithm_type _training_algorithm;
};

#endif // KOHONEN_PALLETE_H
