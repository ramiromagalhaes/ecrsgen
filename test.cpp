#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "lib/haarwavelet.h"



inline void printIntegral(cv::Mat & integralSum) {
    std::cout << std::endl;
    for (int i = 0; i < integralSum.rows; ++i) {
        for (int j = 0; j < integralSum.cols; ++j) {
            std::cout << integralSum.at<int>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

inline void printImage(cv::Mat & image) {
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
    cv::Mat integralSum(image.rows + 1, image.cols + 1, CV_32S);
    cv::Mat integralSquare(image.rows + 1, image.cols + 1, CV_32S);
    cv::integral(image, integralSum, integralSquare);

    { //scope for first test
        std::vector<double> srfs_vector;

        HaarWavelet w(&s, &p, rects, weights);
        w.setIntegralImages(&integralSum, &integralSquare);
        w.srfs(srfs_vector);

        std::cout << w.dimensions() << std::endl;
        std::cout << w.value() << std::endl;
        std::cout << "[" << srfs_vector[0] << ", " << srfs_vector[1] << "]" << std::endl;
    }

    { //scope for second test
        std::vector<double> srfs_vector;

        cv::FileStorage fs("/home/ramiro/workspace/ecrsgen/haar.xml", cv::FileStorage::READ);
        cv::FileNode wavelets = fs.root();
        cv::FileNodeIterator it = wavelets.begin();

        HaarWavelet w(&s, &p, *it);//TODO test me
        w.setIntegralImages(&integralSum, &integralSquare);
        w.srfs(srfs_vector);

        std::cout << w.dimensions() << std::endl;
        std::cout << w.value() << std::endl;
        std::cout << "[" << srfs_vector[0] << ", " << srfs_vector[1] << "]" << std::endl;
    }

    return 0;
}
