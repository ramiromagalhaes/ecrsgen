#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "lib/haarwavelet.h"



void printIntegral(cv::Mat & integralSum) {
    std::cout << std::endl;
    for (int i = 0; i < integralSum.rows; ++i) {
        for (int j = 0; j < integralSum.cols; ++j) {
            std::cout << integralSum.at<int>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void printImage(cv::Mat & image) {
    std::cout << std::endl;
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            std::cout << (int)image.at<unsigned char>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    cv::Point p(0, 0);  //fixed during SRFS production
    cv::Size s(24, 24); //that's the size of the trainning images

    std::vector<cv::Rect> rects(2);
    rects[0] = cv::Rect(12, 0, 12, 12);
    rects[1] = cv::Rect(0, 12, 12, 12);

    std::vector<float> weights(2);
    weights[0] = 1;
    weights[1] = -1;

    cv::Mat image = cv::imread("/home/ramiro/workspace/ecrsgen/exemplo.bmp", CV_8U); //I must inform the flags
    cv::Mat integralSum(image.rows, image.cols, CV_32S);
    cv::Mat integralSquare(image.rows, image.cols, CV_32S);
    cv::integral(image, integralSum, integralSquare);

    std::vector<double> srfs_vector;

    HaarWavelet w(&s, &p, rects, weights);
    w.setIntegralImages(&integralSum, &integralSquare);
    w.srfs(srfs_vector);

    std::cout << w.dimensions() << std::endl;
    std::cout << w.value() << std::endl;
    std::cout << "[" << srfs_vector[0] << ", " << srfs_vector[1] << "]" << std::endl;

    return 0;
}
