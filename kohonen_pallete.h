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
            data_type::value_type,
            data_type::value_type,
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

    // Winner Takes Most algorithm typedefs
    typedef neural_net::Max_topology< std::size_t > topology_type;
    typedef
        neural_net::Gauss_function
        <
            data_type::value_type,
            data_type::value_type,
            int
        >
        space_function_type;
    typedef
        neural_net::Gauss_function
        <
            std::size_t,
            data_type::value_type,
            int
        >
        network_function_type;
    typedef
        neural_net::Classic_training_weight
        <
            data_type,
            int,
            network_function_type,
            space_function_type,
            topology_type,
            distance_function_type,
            std::size_t
        >
        weight_type;
    typedef
        neural_net::Wtm_classical_training_functional
        <
            data_type,
            double,
            int,
            std::size_t,
            weight_type
        >
        wtm_training_function_type;
    typedef
        neural_net::Wtm_training_algorithm
        <
            network_container_type,
            data_type,
            std::vector<data_type>::iterator,
            wtm_training_function_type,
            std::size_t
        >
        wtm_training_algorithm_type;

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
    topology_type _topology;
    space_function_type _space_function;
    network_function_type _network_function;
    weight_type _weights;
    wtm_training_function_type _training_function;
    wtm_training_algorithm_type _training_algorithm;
};

#endif // KOHONEN_PALLETE_H
