#include <iostream>
#include <string>

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "lib/haarwavelet.h"

#define SAMPLE_SIZE 20
#define MIN_RECT_SIZE 3
#define MAX_DIMENSIONS 4



int main(int argc, char * args[])
{
    if (argc != 2) {
        return 1;
    }

    cv::Point position(0,0); //always like that during SRFS production
    cv::Size sampleSize(SAMPLE_SIZE, SAMPLE_SIZE); //size in pixels of the trainning images

    cv::FileStorage waveletStorage(args[1], cv::FileStorage::READ);
    cv::FileNode wavelets = waveletStorage.root();
    cv::FileNodeIterator it = wavelets.begin();
    cv::FileNodeIterator end = wavelets.end();

    int waveletsLoaded = 0;

    cv::Mat viewer = cv::Mat::zeros(sampleSize.height * 10, sampleSize.width * 10, CV_8S);
    cv::Scalar color(255);

    for(;it != end; ++it)
    {
        HaarWavelet wavelet(&sampleSize, &position, *it);
        std::vector<cv::Rect>::const_iterator itRects = wavelet.rects_begin();
        const std::vector<cv::Rect>::const_iterator endRects = wavelet.rects_end();

        for (; itRects != endRects; ++itRects)
        {
            //draw rect on canvas

            cv::Rect r = *itRects;
            cv::Point p1(r.x + 20, r.y + 20);
            cv::Point p2(r.x + r.width + 20, r.y + r.height + 20);

            cv::rectangle(viewer, p1, p2, color);
        }

        waveletsLoaded++;
    }

    std::cout << "Wavelets loaded: " << waveletsLoaded << std::endl;
    waveletStorage.release();

    cv::namedWindow("Wavelets");
    cv::imshow("Wavelets", viewer);
    cv::waitKey(0);

    return 0;
}
