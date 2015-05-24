#include <vector>
#include <algorithm>
#include <iostream>
#include <knnl/functors.hpp>
#include <knnl/basic_neuron.hpp>
#include <knnl/rectangular_container.hpp>
#include <knnl/randomize_policy.hpp>
#include <knnl/euclidean_distance_function.hpp>
#include <knnl/kohonen_network.hpp>
#include <knnl/wta_training_algorithm.hpp>

int main() {
    typedef std::vector<double> data_type;
    typedef neural_net::Cauchy_function<double, double, int> activation_function;
    typedef distance::Euclidean_distance_function<data_type> distance_function;
    typedef neural_net::Basic_neuron<activation_function, distance_function> neuron;
    typedef neural_net::Rectangular_container< neuron > network_container;

    activation_function a( 2.0, 1 );
    distance_function d;
    network_container network;
    std::vector< data_type > data{
        {1, 0, 1},
        {0, 1, 0},
        {1, 0, 0},
    };

    neural_net::External_randomize disable_seeding;

    neural_net::generate_kohonen_network(
        2, 2, a, d, data, network, disable_seeding
    );

    typedef neural_net::Wta_proportional_training_functional< data_type, double, int >
        wta_training_function;
    typedef neural_net::Wta_training_algorithm<
        network_container,
        data_type, std::vector<data_type>::iterator,
        wta_training_function
    > training_algorithm;

    wta_training_function training_function(0.2, 0);
    training_algorithm training( training_function );

    for( int i = 0; i < 1000; i++ ) {
        training( data.begin(), data.end(), &network );
        std::random_shuffle( data.begin(), data.end() );
    }

    for( int i = 0; i < network.objects.size(); i++ ) {
        for( int j = 0; j < network.objects[i].size(); j++ ) {
            std::cout << "[" << i << "][" << j << "]";
            for( int k = 0; k < network.objects[i][j].weights.size(); k++ )
                std::cout << " " << network.objects[i][j].weights[k];
            std::cout << '\n';
        }
    }

    return 0;
}

