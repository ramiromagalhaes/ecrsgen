#include <string>
#include <iostream>
#include <fstream>

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lib/haarwavelet.h"

#define SAMPLE_SIZE 20
#define MIN_RECT_SIZE 3



/*
 * Pavani's restrictions on Haar wavelets generation:
 * 1) only 2 to 4 rectangles
 * 2) detector size = 20x20
 * 3) no rotated rectangles
 * 4) disjoint rectangles are integer multiples of rectangle size
 * 5) all rectangles in a HW have the same size
 * 6) no rectangles smaller than 3x3
 */



inline void persistWavelet(HaarWavelet &wavelet, std::ostream &output)
{
    wavelet.write(output);
}


inline void persistWavelet(HaarWavelet &wavelet, cv::FileStorage &waveletStorage, int counter)
{
    std::stringstream waveletEntityName;
    waveletEntityName << "wavelet" << counter; //opencv persistence sucks hard
    waveletStorage << waveletEntityName.str();
    wavelet.write(waveletStorage);
}


/**
 * Generates Haar wavelets with 2 rectangles.
 */
void gen2d(cv::Size * const sampleSize, cv::Point * const position,
           std::ostream &output, int &counter)
{
    std::vector<float> weights(2);
    weights[0] = 1;
    weights[1] = -1;

    for(int w = 1; w <= SAMPLE_SIZE; w++) //width of both rectangles
    {
        for(int h = 1; h <= SAMPLE_SIZE; h++) //height of both rectangles
        {
            if ( w * h < MIN_RECT_SIZE * MIN_RECT_SIZE ) //Check against Pavani's restriction #6
            {
                continue;
            }

            for(int x = 0; x <= SAMPLE_SIZE - w; x++) //x position of the first rectangle
            {
                for(int y = 0; y <= SAMPLE_SIZE - h; y++) //y position of the first rectangle
                {
                    if (   x + w > SAMPLE_SIZE
                        || y + h > SAMPLE_SIZE)
                    {
                        continue;
                    }

                    for(int dx = 0; dx < SAMPLE_SIZE; dx++) //dx = horizontal displacement multiplier of the second rectangle.
                    {                                                      //If bigger than 1 the rectangles will be disjoint. See Pavani's restriction #4.
                        for(int dy = 0; dy < SAMPLE_SIZE; dy++) //dy is similar to dx but in the vertical direction
                        {
                            if (dx == 0 && dy == 0) //rectangles will overlap
                            {
                                continue;
                            }

                            const int xOther = x + dx * w;
                            const int yOther = y + dy * h;

                            if (   xOther >= SAMPLE_SIZE
                                || yOther >= SAMPLE_SIZE
                                || xOther + w > SAMPLE_SIZE
                                || yOther + h > SAMPLE_SIZE)
                            {
                                continue;
                            }

                            //create the wavelet
                            std::vector<cv::Rect> rects(2);
                            rects[0] = cv::Rect(     x,      y, w, h);
                            rects[1] = cv::Rect(xOther, yOther, w, h);

                            HaarWavelet wavelet(sampleSize, position, rects, weights);
                            persistWavelet(wavelet, output);
                            counter++;
                        }
                    }
                }
            }
        }
    }
}



/**
 * Generates Haar wavelets with 3 rectangles.
 */
void gen3d(cv::Size * const sampleSize, cv::Point * const position,
           std::ostream &output, int &counter)
{
    const int K = 3; //number of dimensions of the generated wavelets

    std::vector<float> weights(K);
    weights[0] = 1;
    weights[1] = -1;
    weights[2] = 1;

    for(int w = 1; w <= SAMPLE_SIZE; w++) //width of both rectangles
    {
        for(int h = 1; h <= SAMPLE_SIZE; h++) //height of both rectangles
        {
            if ( w * h < MIN_RECT_SIZE * MIN_RECT_SIZE ) //Check against Pavani's restriction #6
            {
                continue;
            }

            int x[K], //x and y positions of each rectangle.
                y[K];

            for(x[0] = 0; x[0] <= SAMPLE_SIZE - w; x[0]++) //for each x...
            {
                for(y[0] = 0; y[0] <= SAMPLE_SIZE - h; y[0]++) //...and y of the first rectangle...
                {
                    if (   x[0] + w > SAMPLE_SIZE
                        || y[0] + h > SAMPLE_SIZE)
                    {
                        continue;
                    }

                    int dx[K - 1], //dx = horizontal displacement multiplier of the second rectangle.
                        dy[K - 1]; //If bigger than 1 the rectangles will be disjoint. See Pavani's restriction #4.
                                   //dy is similar to dx but in the vertical direction

                    for(dx[0] = 0; dx[0] < SAMPLE_SIZE; dx[0]++)
                    {
                        for(dy[0] = 0; dy[0] < SAMPLE_SIZE; dy[0]++)
                        {
                            for(dx[1] = 0; dx[1] < SAMPLE_SIZE; dx[1]++)
                            {
                                for(dy[1] = 0; dy[1] < SAMPLE_SIZE; dy[1]++)
                                {
                                    //avoids rectangle overlapping
                                    if (   (dx[0] == 0 && dy[0] == 0)
                                        || (dx[1] == 0 && dy[1] == 0))
                                    {
                                        continue;
                                    }

                                    //sets the values of the x, y position of the rectangles
                                    for (int i = 1; i < K; i++)
                                    {
                                        x[i] = x[i-1] + dx[i-1] * w;
                                        y[i] = y[i-1] + dy[i-1] * h;
                                    }

                                    {
                                        bool overflow = false;
                                        for (int i = 1; i < K; i++) //...and all rectangles fit into the sampling window...
                                        {
                                            if(    x[i] >= SAMPLE_SIZE //x and y must be at least 1 pixel away from the window's last pixel
                                                || y[i] >= SAMPLE_SIZE
                                                || x[i] + w > SAMPLE_SIZE //and the rectangle must fully fit the window
                                                || y[i] + h > SAMPLE_SIZE)
                                            {
                                                overflow = true;
                                                break;
                                            }
                                        }
                                        if(overflow)
                                        {
                                            continue;
                                        }
                                    }


                                    //create the wavelet
                                    std::vector<cv::Rect> rects(K);
                                    for (int i = 0; i < K; i++)
                                    {
                                        rects[i] = cv::Rect(x[i], y[i], w, h);
                                    }

                                    HaarWavelet wavelet(sampleSize, position, rects, weights);
                                    persistWavelet(wavelet, output);
                                    counter++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}



/**
 * Generates Haar wavelets with 4 rectangles.
 */
void gen4d(cv::Size * const sampleSize, cv::Point * const position,
           std::ostream &output, int &counter)
{
    const int K = 4; //number of dimensions of the generated wavelets

    std::vector<float> weights(K);
    weights[0] = 1;
    weights[1] = -1;
    weights[2] = 1;

    for(int w = 1; w <= SAMPLE_SIZE; w++) //width of both rectangles
    {
        for(int h = 1; h <= SAMPLE_SIZE; h++) //height of both rectangles
        {
            if ( w * h < MIN_RECT_SIZE * MIN_RECT_SIZE ) //Check against Pavani's restriction #6
            {
                continue;
            }

            int x[K], //x and y positions of each rectangle.
                y[K];

            for(x[0] = 0; x[0] <= SAMPLE_SIZE - w; x[0]++) //for each x...
            {
                for(y[0] = 0; y[0] <= SAMPLE_SIZE - h; y[0]++) //...and y of the first rectangle...
                {
                    if (   x[0] + w > SAMPLE_SIZE
                        || y[0] + h > SAMPLE_SIZE)
                    {
                        continue;
                    }

                    int dx[K - 1], //dx = horizontal displacement multiplier of the second rectangle.
                        dy[K - 1]; //If bigger than 1 the rectangles will be disjoint. See Pavani's restriction #4.
                                   //dy is similar to dx but in the vertical direction

                    for(dx[0] = 0; dx[0] < SAMPLE_SIZE; dx[0]++)
                    {
                        for(dy[0] = 0; dy[0] < SAMPLE_SIZE; dy[0]++)
                        {
                            if (dx[0] == 0 && dy[0] == 0)
                            {
                                continue;
                            }

                            x[1] = x[0] + dx[0] * w;
                            y[1] = y[0] + dy[0] * h;

                            if (   x[1] >= SAMPLE_SIZE
                                || y[1] >= SAMPLE_SIZE
                                || x[1] + w > SAMPLE_SIZE
                                || y[1] + h > SAMPLE_SIZE)
                            {
                                continue;
                            }

                            for(dx[1] = 0; dx[1] < SAMPLE_SIZE; dx[1]++)
                            {
                                for(dy[1] = 0; dy[1] < SAMPLE_SIZE; dy[1]++)
                                {
                                    if (dx[1] == 0 && dy[1] == 0)
                                    {
                                        continue;
                                    }

                                    x[2] = x[1] + dx[1] * w;
                                    y[2] = y[1] + dy[1] * h;

                                    if (   x[2] >= SAMPLE_SIZE
                                        || y[2] >= SAMPLE_SIZE
                                        || x[2] + w > SAMPLE_SIZE
                                        || y[2] + h > SAMPLE_SIZE)
                                    {
                                        continue;
                                    }

                                    for(dx[2] = 0; dx[2] < SAMPLE_SIZE; dx[2]++)
                                    {
                                        for(dy[2] = 0; dy[2] < SAMPLE_SIZE; dy[2]++)
                                        {
                                            //avoids rectangle overlapping
                                            if ( dx[2] == 0 && dy[2] == 0 )
                                            {
                                                continue;
                                            }

                                            x[3] = x[2] + dx[2] * w;
                                            y[3] = y[2] + dy[2] * h;

                                            if (   x[3] >= SAMPLE_SIZE
                                                || y[3] >= SAMPLE_SIZE
                                                || x[3] + w > SAMPLE_SIZE
                                                || y[3] + h > SAMPLE_SIZE)
                                            {
                                                continue;
                                            }


                                            {
                                                bool overflow = false;
                                                for (int i = 1; i < K; i++) //...and all rectangles fit into the sampling window...
                                                {
                                                    if(    x[i] >= SAMPLE_SIZE //x and y must be at least 1 pixel away from the window's last pixel
                                                        || y[i] >= SAMPLE_SIZE
                                                        || x[i] + w > SAMPLE_SIZE //and the rectangle must fully fit the window
                                                        || y[i] + h > SAMPLE_SIZE)
                                                    {
                                                        overflow = true;
                                                        break;
                                                    }
                                                }
                                                if(overflow)
                                                {
                                                    continue;
                                                }
                                            }


                                            //create the wavelet
                                            std::vector<cv::Rect> rects(K);
                                            for (int i = 0; i < K; i++)
                                            {
                                                rects[i] = cv::Rect(x[i], y[i], w, h);
                                            }

                                            HaarWavelet wavelet(sampleSize, position, rects, weights);
                                            persistWavelet(wavelet, output);
                                            counter++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}



int main(int argc, char * args[])
{
    if (argc != 2) {
        return 1;
    }

    cv::Size sampleSize(SAMPLE_SIZE, SAMPLE_SIZE); //size in pixels of the trainning images
    cv::Point position(0,0); //always like that during SRFS production

    //cv::FileStorage waveletStorage(args[1], cv::FileStorage::WRITE);
    std::ofstream ofs;
    ofs.open(args[1], std::ofstream::out | std::ofstream::app);

    int generetedWaveletCounter = 0;

    //TODO generate in threads?
    gen2d(&sampleSize, &position, ofs, generetedWaveletCounter);
    gen3d(&sampleSize, &position, ofs, generetedWaveletCounter);
    gen4d(&sampleSize, &position, ofs, generetedWaveletCounter);

    ofs.close();

    std::cout << "Wavelets generated: " << generetedWaveletCounter << std::endl;

    return 0;
}
