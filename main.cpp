#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "kohonen_pallete.h"

void draw_network( const KohonenPallete & knn, cv::Mat& img ) {
    for( int i = 0; i < img.rows; i++ )
        for( int j = 0; j < img.cols; j++ ) {
            img.at<cv::Vec3b>(i, j)[2] = knn(i, j).r;
            img.at<cv::Vec3b>(i, j)[1] = knn(i, j).g;
            img.at<cv::Vec3b>(i, j)[0] = knn(i, j).b;
        }
}

int main() {
    std::vector< color > data{
        {255,   0,   0},
        {  0, 255,   0},
        {  0,   0, 255},
    };

    KohonenPallete knn( 30, 40, data ); // Kohonen Neural Network

    cv::namedWindow( "Kohonen", cv::WINDOW_NORMAL);
    cv::Mat img( 30, 40, CV_8UC3 );

    for( int i = 0; i < 1000; i++ ) {
        draw_network( knn, img );
        cv::imshow( "Kohonen", img );
        cv::waitKey( 0 );

        std::cout << "Training " << i << '\n';
        knn.train();
    }
    cv::waitKey(0);

    return 0;
}

