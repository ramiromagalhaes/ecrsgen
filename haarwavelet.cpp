#include "haarwavelet.h"
#include <assert.h>
#include <climits>



#define CV_SUM_PTRS( p0, p1, p2, p3, sum, rect, step )                    \
    /* (x, y) */                                                          \
    (p0) = sum + (rect).x + (step) * (rect).y,                            \
    /* (x + w, y) */                                                      \
    (p1) = sum + (rect).x + (rect).width + (step) * (rect).y,             \
    /* (x + w, y) */                                                      \
    (p2) = sum + (rect).x + (step) * ((rect).y + (rect).height),          \
    /* (x + w, y + h) */                                                  \
    (p3) = sum + (rect).x + (rect).width + (step) * ((rect).y + (rect).height)



HaarWavelet::HaarWavelet()
{
    scale = 1;
}

HaarWavelet::HaarWavelet(cv::Size *detectorSize_,
                         cv::Point *detectorPosition_,
                         std::vector<cv::Rect> rects_,
                         std::vector<float> weights_)
{
    assert(rects.size() == weights.size());

    detectorSize = detectorSize_;
    detectorPosition = detectorPosition_;
    rects = rects_;
    weights = weights_;
}

int HaarWavelet::dimensions() const
{
    return (int)rects.size();
}

bool HaarWavelet::setIntegralImages(cv::Mat * const sum_, cv::Mat * const squareSum_)
{
    assert(sum_ && squareSum_);

    if (sum_->cols < detectorSize->width || sum_->rows < detectorSize->height)
    {
        return false;
    }


    sum = sum_;
    squareSum = squareSum_;

    return true;
}

double HaarWavelet::value() const
{
    assert(sum && squareSum && detectorPosition);

    double returnValue = 0;

    const int dim = dimensions();
    for (int i = 0; i < dim; ++i)
    {
        double rectValue = 0;

        const cv::Rect * rect = &rect[i];
        const int x = detectorPosition->x + rect->x;
        const int y = detectorPosition->y + rect->y;
        const int x_w = x + rect->width;
        const int y_h = y + rect->height;

        rectValue += sum->at<int>(y, x);     // (x, y)
        rectValue -= sum->at<int>(y, x_w);   // (x + w, y)
        rectValue -= sum->at<int>(y_h, x);   // (x, y + h)
        rectValue += sum->at<int>(y_h, x_w); // (x + w, y + h)

        returnValue += (weights[i] * rectValue);
    }

    return returnValue;
}

void HaarWavelet::srfs(std::vector<float> & srfsVector) const
{ //TODO normalization
    assert(sum && squareSum && detectorPosition);

    const int dim = dimensions();
    srfsVector.resize(dim);

    for (int i = 0; i < dim; ++i)
    {
        float * const v = &srfsVector[i];

        const cv::Rect * rect = &rect[i];
        const int x = detectorPosition->x + rect->x;
        const int y = detectorPosition->y + rect->y;
        const int x_w = x + rect->width;
        const int y_h = y + rect->height;

        *v += sum->at<int>(y, x);     // (x, y)
        *v -= sum->at<int>(y, x_w);   // (x + w, y)
        *v -= sum->at<int>(y_h, x);   // (x, y + h)
        *v += sum->at<int>(y_h, x_w); // (x + w, y + h)

        //SRFS works with means...
        *v /= (rect->size().height * rect->size().width);
        //...which are normalized (Pavani et al., 2010, section 2.3).
        *v /= INT_MAX;
    }
}
