namespace command_line {
    const char help_message[] =
" [options]\n"
"Make a Kohonen neural network learn a color pallete,\n"
"using the Winner Takes Most alogrithm.\n"
"\n"
"Options:\n"
"--nodes <rows> <columns>\n"
"--size <rows> <columns>\n"
"    Choose the size of the rectangular network.\n"
"    Defaults: 30 rows, 40 columns.\n"
"\n"
"--epochs <N>\n"
"    Choose the number of training epochs.\n"
"    Default: 100\n"
"\n"
"--initial-radius <F>\n"
"    Choose the initial influene radius the winner will have in the training.\n"
"    <F> can be any positive floating-point number.\n"
"    Default: 10.0.\n"
"\n"
"--delta <F>\n"
"    Choose how fast the influence radius will shrink throughout the training.\n"
"    The influence radius will be multiplied by this value every epoch.\n"
"    Default: 0.95.\n"
"\n"
"--input <filename>\n"
"    Choose a different color pallete to be learnt by the network.\n"
"    The file must be in white-space separated RGB format.\n"
"    For instance, the following text generates the default colors:\n"
"        255    0    0\n"
"          0  255    0\n"
"          0    0  255\n"
"        255  255    0\n"
"          0  255  255\n"
"        255    0  255\n"
"        255  255  255\n"
"          0    0    0\n"
"\n"
"--seed <N>\n"
"    Choose N as the seed used by the random number generator.\n"
"    Default: Generate a time-based seed.\n"
"\n"
"--verbose\n"
"    Tell the current epoch in the command line.\n"
"    Default: non-verbose.\n"
"\n"
"--help\n"
"    Show this text and quit.\n"
"\n"
;
}

#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "cmdline/args.hpp"
#include "kohonen_pallete.h"
#include "std_rand_wrapper.h"

namespace command_line {
    bool seed_set = false;
    unsigned seed = 0;
    int rows = 30, cols = 40;
    int epochs = 100;
    bool verbose = false;
    double initial_radius = 10.0;
    double delta = 0.95;
    bool input_set = false;

    std::vector< color > data{
        {255,   0,   0},
        {  0, 255,   0},
        {  0,   0, 255},
        {255, 255,   0},
        {  0, 255, 255},
        {255,   0, 255},
        {255, 255, 255},
        {  0,   0,   0},
    };

    void file_parse( std::string filename ) {
        data.clear();
        std::ifstream file( filename );
        if( ! file ) {
            std::cerr << "Could not open " << filename << std::endl;
            std::exit(1);
        }
        color c;
        while( file >> c.r >> c.g >> c.b )
            data.push_back( c );
    }

    void parse( cmdline::args && args ) {
        while( args.size() > 0 ) {
            std::string arg = args.next();
            if( arg == "--size" || arg == "--nodes" ) {
                args >> rows >> cols;
                continue;
            }
            if( arg == "--epochs" ) {
                args >> epochs;
                continue;
            }
            if( arg == "--initial-radius" ) {
                args.range(0) >> initial_radius;
                continue;
            }
            if( arg == "--delta" ) {
                args.range(0, 1) >> delta;
                continue;
            }
            if( arg == "--input" ) {
                if( input_set ) {
                    args.log() << "--input suplied twice!" << std::endl;
                    std::exit(1);
                }
                input_set = true;
                file_parse( args.next() );
                continue;
            }
            if( arg == "--seed" ) {
                args >> seed;
                seed_set = true;
                continue;
            }
            if( arg == "--verbose" ) {
                verbose = true;
                continue;
            }
            if( arg == "--help" ) {
                std::cout << args.program_name() << help_message;
                std::exit(0);
            }
            args.log() << "Unknown command line option " << arg << std::endl;
            std::exit(1);
        }
    }
}

void draw_network( const KohonenPallete & knn, cv::Mat& img ) {
    for( int i = 0; i < img.rows; i++ )
        for( int j = 0; j < img.cols; j++ ) {
            img.at<cv::Vec3b>(i, j)[2] = knn(i, j).r;
            img.at<cv::Vec3b>(i, j)[1] = knn(i, j).g;
            img.at<cv::Vec3b>(i, j)[0] = knn(i, j).b;
        }
}

int main( int argc, char ** argv ) {
    command_line::parse( cmdline::args(argc, argv) );

    if( command_line::seed_set )
        std_rand_wrapper::seed( command_line::seed );
    else
        std_rand_wrapper::time_seed();
    std::cout << "Seed: " << std_rand_wrapper::seed() << '\n';

    // Kohonen Neural Network
    KohonenPallete knn( command_line::rows, command_line::cols, command_line::data );

    cv::namedWindow( "Kohonen", cv::WINDOW_NORMAL);
    cv::Mat img( command_line::rows, command_line::cols, CV_8UC3 );

    double radius = command_line::initial_radius;
    double delta = command_line::delta;

    for( int i = 0; i < command_line::epochs; i++ ) {
        draw_network( knn, img );
        cv::imshow( "Kohonen", img );
        cv::waitKey( 1 );

        if( command_line::verbose )
            std::cout << "Epoch " << i + 1 << '\n';

        knn.train( radius );
        radius *= delta;
    }
    cv::waitKey(0);

    return 0;
}

