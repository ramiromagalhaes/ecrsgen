#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "myhaarevaluator.h"



int main()
{
    std::vector<float> weights(2);
    weights[0] = 1;
    weights[1] = -1;

    std::vector<cv::Rect> rects(2);
    rects[0] = cv::Rect(0, 0, 5, 5);
    rects[1] = cv::Rect(5, 0, 5, 5);

    return 0;
}
