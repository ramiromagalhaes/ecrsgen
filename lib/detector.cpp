#include "detector.h"



Detector::Detector()
{
}



bool Detector::setSubwindow(const int left, const int top)
{
    if( left < 0 || top < 0 ||
        left + defaultSize.width >= image.cols ||
        top + defaultSize.height >= image.rows )
    {
        return false;
    }

    currentRoi.x = left;
    currentRoi.y = top;

    {
        const cv::Rect integralsRoi(currentRoi.x, currentRoi.y, currentRoi.width + 1, currentRoi.height + 1);
        currentIntegralImage = integralImage(integralsRoi);
        currentSquareIntegralImage = squareIntegralImage(integralsRoi);
    }

    const double area = currentRoi.area();
    subwindowMean = (currentIntegralImage.at<int>(currentRoi.width, currentRoi.height) - currentIntegralImage.at<int>(0, 0)) / area;
    subwindowStdDeviation = sqrt(
        subwindowMean * subwindowMean - (currentSquareIntegralImage.at<int>(currentRoi.width, currentRoi.height) - currentSquareIntegralImage.at<int>(0, 0)) / area
    );

//    const double normalized = (pixel - subwindowMean)/sqrt(subwindowVariance);
//    const double normalized_on_gray = (CHAR_MAX/2) + normalized;
}



bool Detector::setSubwindow(const cv::Point topLeftPosition)
{
    setSubwindow(topLeftPosition.x, topLeftPosition.y);
}
