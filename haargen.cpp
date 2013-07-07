#include <iostream>
#include <string>

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lib/haarwavelet.h"

#define SAMPLE_SIZE 20
#define MIN_RECT_SIZE 3



int main()
{
    cv::Point position(0,0); //always like that during SRFS production
    cv::Size sampleSize(SAMPLE_SIZE, SAMPLE_SIZE); //size in pixels of the trainning images

    std::vector<HaarWavelet*> wavelets; //list of haar wavelets

    /*
     * Pavani's restrictions:
     * 1) only 2 to 4 rectangles
     * 2) detector size = 20x20
     * 3) no rotated rectangles
     * 4) disjoint rectangles are integer multiples of rectangle size
     * 5) all rectangles in a HW have the same size
     * 6) no rectangles smaller than 3x3
     */
    std::vector<float> weights(2);
    weights[0] = 1;
    weights[1] = -1;

    //First, let's produce simple haar wavelets with 2 features
    for(int x = 0; x < SAMPLE_SIZE - MIN_RECT_SIZE; x++) //x position of the first rectangle
    {
        for(int y = 0; y < SAMPLE_SIZE - MIN_RECT_SIZE; y++) //y position of the first rectangle
        {
            for(int w = 1; w <= SAMPLE_SIZE / MIN_RECT_SIZE; w++) //width of both rectangles
            {
                for(int h = 1; h <= SAMPLE_SIZE / MIN_RECT_SIZE; h++) //height of both rectangles
                {
                    for(int dx = 0; dx <= SAMPLE_SIZE/MIN_RECT_SIZE; dx++) //dx = horizontal displacement multiplier of the second rectangle.
                    {                                                      //If bigger than 1 the rectangles will be disjoint. See Pavani's restriction #4.
                        for(int dy = 0; dy <= SAMPLE_SIZE/MIN_RECT_SIZE; dy++) //dy is similar to dx but in the vertical direction
                        {
                            const int xOther = x + dx * w;
                            const int yOther = y + dy * h;

                            if ( !( (w * h < MIN_RECT_SIZE * MIN_RECT_SIZE) //Pavani's restriction #6
                                 || (dx == 0 && dy == 0)
                                 || x + w >= SAMPLE_SIZE
                                 || y + h >= SAMPLE_SIZE
                                 || xOther + w >= SAMPLE_SIZE
                                 || yOther + y >= SAMPLE_SIZE) )
                            {
                                //create a haar wavelet
                                std::vector<cv::Rect> rects(2);
                                rects[0] = cv::Rect(     x,      y, w, h);
                                rects[1] = cv::Rect(xOther, yOther, w, h);

                                HaarWavelet * wavelet = new HaarWavelet(&sampleSize, &position, rects, weights);
                                wavelets.push_back(wavelet);

                                //std::cout << x      << " " << y      << " " << w << " " << h << std::endl;
                                //std::cout << xOther << " " << yOther << " " << w << " " << h << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }

    //TODO release memory??? meh...
    std::cout << wavelets.size() << std::endl;

    return 0;
}
