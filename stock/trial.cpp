#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "haarevaluator.h"



int main(int argc, char* argv[])
{
    if (argc != 3) {
        return 1;
    }

    {
        cv::FileStorage fs(argv[1], cv::FileStorage::WRITE);

        cv::Rect rect1(0, 0, 5, 5);
        cv::Rect rect2(5, 0, 5, 5);
        cv::Rect rect3(0, 5, 5, 5);
        cv::Rect rect4(5, 5, 5, 5);

        fs << CC_RECTS << "[";
            fs << "{:" << "x" << rect1.x << "y" << rect1.y << "width" << rect1.width << "height" << rect1.height << "weight" << 1 << "}";
            fs << "{:" << "x" << rect2.x << "y" << rect2.y << "width" << rect2.width << "height" << rect2.height << "weight" << -1 << "}";
            fs << "{:" << "x" << rect3.x << "y" << rect3.y << "width" << rect3.width << "height" << rect3.height << "weight" << -1 << "}";
            fs << "{:" << "x" << rect4.x << "y" << rect4.y << "width" << rect4.width << "height" << rect4.height << "weight" << 1 << "}";
        fs << "]";

        fs.release();
    }

    HaarEvaluator haar;
    {
        cv::FileStorage fs(argv[1], cv::FileStorage::READ);
        cv::FileNode features = fs[CC_RECTS];

        if (!haar.read(features)) {
            return 2;
        }

        fs.release();
    }

    std::stringstream ss;

    cv::Mat image = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    for (int i = 0; i < image.cols; ++i) {
        for (int j = 0; j < image.rows; ++j) {
            ss << i * image.cols + j << ": " << (int)image.at<uchar>(j, i) << std::endl;
        }
    }

    //TODO Onde diabos estão armazenadas as características???
    ss << haar.features->at(0).rect[0].r.x
        << haar.features->at(1).rect[0].r.x
        << haar.features->at(2).rect[0].r.x
        << haar.features->at(3).rect[0].r.x
        << std::endl;

    haar.setImage(image, cv::Size(10, 10));
    haar.setWindow(cv::Point(0,0));

    ss << "Wavelet" << std::endl;

    ss << haar.calcOrd(0) << std::endl;
    ss << haar.calcOrd(1) << std::endl;
    ss << haar.calcOrd(2) << std::endl;
    ss << haar.calcOrd(3) << std::endl;
    std::cout << ss.str();

    return 0;
}
