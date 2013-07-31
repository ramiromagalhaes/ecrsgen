#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/unordered_map.hpp>

#include "lib/haarwavelet.h"

#define SAMPLE_SIZE 20

#define MIN_RECT_HEIGHT 3 //Minimum = 3 thanks to Pavani's restriction #6.
#define MIN_RECT_WIDTH 3 //Minimum = 3 thanks to Pavani's restriction #6.



/*
 * Pavani's restrictions on Haar wavelets generation:
 * 1) only 2 to 4 rectangles
 * 2) detector size = 20x20
 * 3) no rotated rectangles
 * 4) disjoint rectangles are integer multiples of rectangle size
 * 5) all rectangles in a HW have the same size
 * 6) no rectangles smaller than 3x3
 */



typedef boost::unordered_map<int, HaarWavelet * > WaveletMap;



int hash(HaarWavelet * w)
{
    const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71};
    int x[SAMPLE_SIZE], y[SAMPLE_SIZE], width[SAMPLE_SIZE], height[SAMPLE_SIZE];

    for(int i = 0; i < SAMPLE_SIZE; ++i)
    {
        x[i] = y[i] = width[i] = height[i] = 0;
    }

    std::vector<cv::Rect>::const_iterator it = w->rects_begin();
    const std::vector<cv::Rect>::const_iterator end = w->rects_end();
    for(; it != end; ++it)
    {
        const cv::Rect r = *it;

        x[r.x]++;
        y[r.y]++;
        width[r.width]++;
        height[r.height]++;
    }

    int xVal = 1, yVal = 1, wVal = 1, hVal = 1;
    for(int i = 0; i < SAMPLE_SIZE; ++i)
    {
        xVal *= pow(primes[i], x[i]);
        yVal *= pow(primes[i], y[i]);
        wVal *= pow(primes[i], width[i]);
        hVal *= pow(primes[i], height[i]);
    }
    yVal += (71*71*71*71);
    wVal += (71*71*71*71*2);
    hVal += (71*71*71*71*3);

    return xVal + yVal + wVal + hVal;
}



void writeToFile(char * filename, WaveletMap &wavelets)
{
    std::ofstream ofs;
    ofs.open(filename, std::ofstream::out | std::ofstream::trunc);
    WaveletMap::const_iterator it = wavelets.begin();
    const WaveletMap::const_iterator end = wavelets.end();
    for(; it != end; ++it)
    {
        it->second->write(ofs);
    }
    ofs.close();
}



/**
 * Generates Haar wavelets with 2 rectangles.
 */
void gen2d(cv::Size * const sampleSize, cv::Point * const position, WaveletMap &wavelets)
{
    std::vector<float> weights(2);
    weights[0] = 1;
    weights[1] = -1;

    for(int w = MIN_RECT_WIDTH; w <= SAMPLE_SIZE; w++)
    {
        for(int h = MIN_RECT_HEIGHT; h <= SAMPLE_SIZE; h++)
        {
            for(int x = 0; x <= SAMPLE_SIZE - w; x++) //x position of the first rectangle
            {
                for(int y = 0; y <= SAMPLE_SIZE - h; y++) //y position of the first rectangle
                {
                    if (   x + w > SAMPLE_SIZE
                        || y + h > SAMPLE_SIZE)
                    {
                        continue;
                    }

                    for(int dx = -SAMPLE_SIZE - x / w; dx < SAMPLE_SIZE / w; dx++) //dx = horizontal displacement multiplier of the second rectangle.
                    {                                           //If bigger than 1 the rectangles will be disjoint. See Pavani's restriction #4.
                        for(int dy = -SAMPLE_SIZE / h; dy < SAMPLE_SIZE / h; dy++) //dy is similar to dx but in the vertical direction
                        {
                            if (dx == 0 && dy == 0) //rectangles will overlap
                            {
                                continue;
                            }

                            const int xOther = x + dx * w;
                            const int yOther = y + dy * h;

                            if (   xOther < 0
                                || yOther < 0
                                || xOther >= SAMPLE_SIZE
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

                            HaarWavelet * wavelet = new HaarWavelet(sampleSize, position, rects, weights);
                            int whash = hash(wavelet);
                            if (wavelets.find(whash) == wavelets.end())
                            {
                                wavelets[whash] = wavelet;
                            }
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
void gen3d(cv::Size * const sampleSize, cv::Point * const position, WaveletMap &wavelets)
{
    const int K = 3; //number of dimensions of the generated wavelets

    std::vector<float> weights(K);
    weights[0] = 1;
    weights[1] = -1;
    weights[2] = 1;

    for(int w = MIN_RECT_WIDTH; w <= SAMPLE_SIZE; w++)
    {
        for(int h = MIN_RECT_HEIGHT; h <= SAMPLE_SIZE; h++)
        {
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

                    for(dx[0] = -SAMPLE_SIZE / w; dx[0] < SAMPLE_SIZE / w; dx[0]++)
                    {
                        for(dy[0] = -SAMPLE_SIZE / h; dy[0] < SAMPLE_SIZE / h; dy[0]++)
                        {
                            for(dx[1] = -SAMPLE_SIZE / w; dx[1] < SAMPLE_SIZE / w; dx[1]++)
                            {
                                for(dy[1] = -SAMPLE_SIZE / h; dy[1] < SAMPLE_SIZE / h; dy[1]++)
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

                                    {//avoids rectangles overlapping
                                        for (int i = 0; i < K; i++)
                                        {
                                            for (int j = 0; j < K; j++)
                                            {
                                                if (x[i] == x[j] && y[i] == y[j])
                                                {
                                                    continue;
                                                }
                                            }
                                        }
                                    }

                                    {
                                        bool overflow = false;
                                        for (int i = 1; i < K; i++) //...and all rectangles fit into the sampling window...
                                        {
                                            if (   x[i] < 0
                                                || y[i] < 0
                                                || x[i] >= SAMPLE_SIZE //x and y must be at least 1 pixel away from the window's last pixel
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

                                    HaarWavelet * wavelet = new HaarWavelet(sampleSize, position, rects, weights);
                                    int whash = hash(wavelet);
                                    if (wavelets.find(whash) == wavelets.end())
                                    {
                                        wavelets[whash] = wavelet;
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



/**
 * Generates Haar wavelets with 4 rectangles.
 */
void gen4d(cv::Size * const sampleSize, cv::Point * const position, WaveletMap &wavelets)
{
    const int K = 4; //number of dimensions of the generated wavelets

    std::vector<float> weights(K);
    weights[0] = 1;
    weights[1] = -1;
    weights[2] = 1;

    for(int w = MIN_RECT_WIDTH; w <= SAMPLE_SIZE; w++)
    {
        for(int h = MIN_RECT_HEIGHT; h <= SAMPLE_SIZE; h++)
        {
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

                    for(dx[0] = -SAMPLE_SIZE / w; dx[0] < SAMPLE_SIZE / w; dx[0]++)
                    {
                        for(dy[0] = -SAMPLE_SIZE / h; dy[0] < SAMPLE_SIZE / h; dy[0]++)
                        {
                            if (dx[0] == 0 && dy[0] == 0)
                            {
                                continue;
                            }

                            x[1] = x[0] + dx[0] * w;
                            y[1] = y[0] + dy[0] * h;

                            if (   x[1] < 0
                                || y[1] < 0
                                || x[1] >= SAMPLE_SIZE
                                || y[1] >= SAMPLE_SIZE
                                || x[1] + w > SAMPLE_SIZE
                                || y[1] + h > SAMPLE_SIZE)
                            {
                                continue;
                            }

                            for(dx[1] = -SAMPLE_SIZE / w; dx[1] < SAMPLE_SIZE / w; dx[1]++)
                            {
                                for(dy[1] = -SAMPLE_SIZE / h; dy[1] < SAMPLE_SIZE / h; dy[1]++)
                                {
                                    if (dx[1] == 0 && dy[1] == 0)
                                    {
                                        continue;
                                    }

                                    x[2] = x[1] + dx[1] * w;
                                    y[2] = y[1] + dy[1] * h;

                                    if (   x[2] < 0
                                        || y[2] < 0
                                        || x[2] >= SAMPLE_SIZE
                                        || y[2] >= SAMPLE_SIZE
                                        || x[2] + w > SAMPLE_SIZE
                                        || y[2] + h > SAMPLE_SIZE)
                                    {
                                        continue;
                                    }

                                    for(dx[2] = -SAMPLE_SIZE / w; dx[2] < SAMPLE_SIZE / w; dx[2]++)
                                    {
                                        for(dy[2] = -SAMPLE_SIZE / h; dy[2] < SAMPLE_SIZE / h; dy[2]++)
                                        {
                                            //avoids rectangle overlapping
                                            if ( dx[2] == 0 && dy[2] == 0 )
                                            {
                                                continue;
                                            }

                                            x[3] = x[2] + dx[2] * w;
                                            y[3] = y[2] + dy[2] * h;

                                            {//avoids rectangles overlapping
                                                for (int i = 0; i < K; i++)
                                                {
                                                    for (int j = 0; j < K; j++)
                                                    {
                                                        if (x[i] == x[j] && y[i] == y[j])
                                                        {
                                                            continue;
                                                        }
                                                    }
                                                }
                                            }

                                            if (   x[3] < 0
                                                || y[3] < 0
                                                || x[3] >= SAMPLE_SIZE
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

                                            HaarWavelet * wavelet = new HaarWavelet(sampleSize, position, rects, weights);
                                            int whash = hash(wavelet);
                                            if (wavelets.find(whash) == wavelets.end())
                                            {
                                                wavelets[whash] = wavelet;
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
}



int main(int argc, char * args[])
{
    if (argc != 2) {
        return 1;
    }

    cv::Size sampleSize(SAMPLE_SIZE, SAMPLE_SIZE); //size in pixels of the trainning images
    cv::Point position(0,0); //always like that during SRFS production

    WaveletMap wavelets;

    gen2d(&sampleSize, &position, wavelets);
    const int d2wavelets = wavelets.size();
    std::cout << "Total 2D wavelets generated: " << d2wavelets << std::endl;
    gen3d(&sampleSize, &position, wavelets);
    const int d3wavelets = wavelets.size() - d2wavelets;
    std::cout << "Total 3D wavelets generated: " << d3wavelets << std::endl;
    gen4d(&sampleSize, &position, wavelets);
    const int d4wavelets = wavelets.size() - d3wavelets - d3wavelets;
    std::cout << "Total 3D wavelets generated: " << d4wavelets << std::endl;

    std::cout << "Wavelets generated: " << wavelets.size() << std::endl;
    std::cout << "Writing wavelets to file...";

    writeToFile(args[1], wavelets);

    std::cout << " done." << std::endl;

    return 0;
}
