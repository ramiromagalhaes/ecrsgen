#include <iostream>
#include <string>

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lib/haarwavelet.h"

#define SAMPLE_SIZE 20
#define MIN_RECT_SIZE 3
#define MAX_DIMENSIONS 4



int main(int argc, char * args[])
{
    if (argc != 2) {
        return 1;
    }

    cv::FileStorage waveletStorage(args[1], cv::FileStorage::WRITE);

    cv::Point position(0,0); //always like that during SRFS production
    cv::Size sampleSize(SAMPLE_SIZE, SAMPLE_SIZE); //size in pixels of the trainning images

    int generetedWaveletCounter = 0;

    /*
     * Pavani's restrictions:
     * 1) only 2 to 4 rectangles
     * 2) detector size = 20x20
     * 3) no rotated rectangles
     * 4) disjoint rectangles are integer multiples of rectangle size
     * 5) all rectangles in a HW have the same size
     * 6) no rectangles smaller than 3x3
     */
    std::vector<float> weights[3]; // will vary depending on the amount of rects
    weights[0].resize(2);
    weights[0][0] = -1;
    weights[0][1] = 1;
    weights[1].resize(3);
    weights[1][0] = -1;
    weights[1][1] = 2;
    weights[1][2] = -1;
    weights[2].resize(4);
    weights[2][0] = -1;
    weights[2][1] = 1;
    weights[2][2] = -1;
    weights[2][3] = 1;

    //First, let's produce simple haar wavelets with 2 features
    for(int w = 1; w <= SAMPLE_SIZE; w++) //width of both rectangles.
    {
        for(int h = 1; h <= SAMPLE_SIZE; h++) //height of both rectangles
        {
            if ( w * h < MIN_RECT_SIZE * MIN_RECT_SIZE ) //Check against Pavani's restriction #6
            {
                continue;
            }
            int x[MAX_DIMENSIONS], //x and y positions of each rectangle.
                y[MAX_DIMENSIONS];

            for(x[0] = 0; x[0] <= SAMPLE_SIZE - w; x[0]++) //for each x...
            {
                for(y[0] = 0; y[0] <= SAMPLE_SIZE - h; y[0]++) //...and y of the first rectangle...
                {
                    for(int k = 1; k < MAX_DIMENSIONS; k++) //...create Haar wavelets of 1 + k dimensions...
                    {
                        int dx[k], //dx[i] is the horizontal displacement multiplier of the rectangle i in relation to rectangle i - 1.
                                   //If bigger than 1 the rectangles are disjoint. If equal to 1, they touch each other.
                                   //Notice that the first rectangle doesn't need this parameter. See Pavani's restriction #4.
                            dy[k]; //Same as dx, but in the vertical direction

                        for(int d = 0; d < k; d++) //...where the rectangle d + 1 is (px, py) pixels away from rectangle's d upper left corner...
                        {
                            for(dx[d] = 0; dx[d] < SAMPLE_SIZE/w; dx[d]++) //dx displacement of rectangle d + 1 in regard to d
                            {
                                for(dy[d] = 0; dy[d] < SAMPLE_SIZE/h; dy[d]++)  //dy displacement of rectangle d + 1 in regard to d
                                {
                                    bool overlap = false;
                                    for (int i = 0; i < k; i++) //...as long as no rectangles overlap...
                                    {
                                        if (dx[i] == 0 && dy[i] == 0) //If both displacements == 0 the rectangles are overlaped.
                                        {
                                            overlap = true;
                                            break;
                                        }
                                    }
                                    if (overlap)
                                    {
                                        continue;
                                    }

                                    for (int i = 1; i <= k; i++) //...where px[d+1] = dx[d] * w and py[d+1] = dy[d] * h (as per restriction #4)...
                                    {
                                        x[i] = x[i-1] + dx[i-1] * w;
                                        y[i] = y[i-1] + dy[i-1] * h;
                                    }

                                    bool overflow = false;
                                    for (int i = 0; i <= k; i++) //...and all rectangles fit into the sampling window...
                                    {
                                        if(x[i] >= SAMPLE_SIZE || y[i] >= SAMPLE_SIZE)
                                        {
                                            overflow = true;
                                            break;
                                        }
                                    }
                                    if(overflow)
                                    {
                                        continue;
                                    }

                                    //...then create the wavelet.
                                    std::vector<cv::Rect> rects(1 + k);
                                    for (int i = 0; i <= k; i++)
                                    {
                                        rects[i] = cv::Rect(x[i], y[i], w, h);
                                    }
                                    HaarWavelet wavelet(&sampleSize, &position, rects, weights[k - 1]);
                                    wavelet.write(waveletStorage);
                                    generetedWaveletCounter++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << "Wavelets generated: " << generetedWaveletCounter << std::endl;

    return 0;
}
