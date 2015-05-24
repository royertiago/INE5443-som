#include <iostream>
#include "kohonen_pallete.h"

int main() {
    std::vector< color > data{
        {255,   0,   0},
        {  0, 255,   0},
        {  0,   0, 255},
    };

    KohonenPallete knn( 2, 2, data ); // Kohonen Neural Network

    for( int i = 0; i < 1000; i++ )
        knn.train();

    for( int i = 0; i < 2; i++ ) {
        for( int j = 0; j < 2; j++ ) {
            std::cout << '[' << i << "][" << j << ']'
                << ' ' << knn(i, j).r
                << ' ' << knn(i, j).g
                << ' ' << knn(i, j).b
                << '\n';
        }
    }

    return 0;
}

