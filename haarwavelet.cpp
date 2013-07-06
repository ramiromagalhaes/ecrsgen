#include "haarwavelet.h"
#include <assert.h>
#include <climits>



HaarWavelet::HaarWavelet() : scale(1), detectorSize(0), detectorPosition(0), detector(0)
{
}

//TODO need to verify if all rects_ are under the detector size boundaries...
HaarWavelet::HaarWavelet(cv::Size * const detectorSize_,
                         cv::Point * const detectorPosition_,
                         std::vector<cv::Rect> rects_,
                         std::vector<float> weights_) : scale(1),
                                                        detectorSize(detectorSize_),
                                                        detectorPosition(detectorPosition_),
                                                        detector(0)
{
    assert(rects.size() == weights.size()); //TODO convert into exception?

    rects = rects_;
    weights = weights_;
}

int HaarWavelet::dimensions() const
{
    return (int)rects.size();
}

bool HaarWavelet::setIntegralImages(cv::Mat * const sum_, cv::Mat * const squareSum_)
{
    if ( !sum_
            || !squareSum_
            || sum_->cols < detectorSize->width
            || sum_->rows < detectorSize->height)
    {
        return false;
    }

    sum = sum_;
    squareSum = squareSum_;

    return true;
}

//TODO still need normalization
double HaarWavelet::value() const
{
    assert(sum && squareSum && detectorPosition); //TODO convert into exception?

    double returnValue = 0;

    const int dim = dimensions();
    for (int i = 0; i < dim; ++i)
    {
        double rectValue = singleRectangleValue(rects[i], *detectorPosition, *sum);
        returnValue += (weights[i] * rectValue);
    }

    return returnValue;
}

void HaarWavelet::srfs(std::vector<double> & srfsVector) const
{
    //TODO convert into exception
    assert(sum && squareSum && detectorPosition);

    const int dim = dimensions();
    srfsVector.resize(dim);

    for (int i = 0; i < dim; ++i)
    {
        srfsVector[i] = singleRectangleValue(rects[i], *detectorPosition, *sum);

        //SRFS works with normalized means (Pavani et al., 2010, section 2.3).
        srfsVector[i] /= (rects[i].size().height * rects[i].size().width / INT_MAX);
    }
}

inline double HaarWavelet::singleRectangleValue(const cv::Rect &rect, const cv::Point &position, const cv::Mat &s) const
{
    double rectVal = 0;

    //As per Lienhart, Maydt, 2002, section 2.2
    const int x = position.x + rect.x;
    const int y = position.y + rect.y;
    const int x_w = x + rect.width;
    const int y_h = y + rect.height;

    //TODO is there a faster and more flexible way to implement this and avoid invoking s.at() functions?
    rectVal += s.at<int>(y, x);     // (x, y)
    rectVal -= s.at<int>(y, x_w);   // (x + w, y)
    rectVal -= s.at<int>(y_h, x);   // (x, y + h)
    rectVal += s.at<int>(y_h, x_w); // (x + w, y + h)

    return rectVal;
}
