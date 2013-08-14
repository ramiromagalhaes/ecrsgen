#include "haarwavelet.h"
#include <assert.h>
#include <climits>



HaarWavelet::HaarWavelet() : scale(1),
                             detectorSize(0),
                             detector(0)
{
}

//TODO need to verify if all rects_ are under the detector size boundaries...
HaarWavelet::HaarWavelet(cv::Size * const detectorSize_,
                         std::vector<cv::Rect> rects_,
                         std::vector<float> weights_) : scale(1),
                                                        detectorSize(detectorSize_),
                                                        detector(0)
{
    assert(rects.size() == weights.size()); //TODO convert into exception

    rects = rects_;
    weights = weights_;
}

/*
 * Sample wavelet data in XML.
 *
 * <?xml version="1.0"?>
 * <opencv_storage>
 * <wavelet>
 *   <rects>2</rects> <!--Amount of rectangles are encoded in the rect entity.-->
 *   <rect>12 0 12 12 1 0 12 12 12 -1</rect> <!-- The rect parameters. -->
 * </wavelet>
 * </opencv_storage>
 */
HaarWavelet::HaarWavelet(cv::Size * const detectorSize_,
                         const cv::FileNode &node) : scale(1),
                                                     detectorSize(detectorSize_),
                                                     detector(0)
{
    const int rectangles = (int)node["rects"];

    const cv::FileNode rectNode = node["rect"];
    cv::FileNodeIterator rectParamsIt = rectNode.begin();
    for(int i = 0; i < rectangles; i++)
    {
        float weight;
        cv::Rect rect;

        rectParamsIt >> rect.x
                     >> rect.y
                     >> rect.width
                     >> rect.height;
        rectParamsIt >> weight;

        rects.push_back(rect);
        weights.push_back(weight);
    }
}

/*
 * Sample wavelet data
 * rects x1 y1 w1 h1 w1 x2 y2 w2 h2 w2...
 *
 */
HaarWavelet::HaarWavelet(cv::Size * const detectorSize_,
                         std::istream &input) : scale(1),
                                                detectorSize(detectorSize_),
                                                detector(0)
{
    int rectangles;
    input >> rectangles;

    for (int i = 0; i < rectangles; i++)
    {
        float weight;
        cv::Rect rect;

        input >> rect.x
              >> rect.y
              >> rect.width
              >> rect.height
              >> weight;

        rects.push_back(rect);
        weights.push_back(weight);
    }

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

//TODO Should the normalization happen here?
double HaarWavelet::value() const
{
    assert(sum && squareSum); //TODO convert into exception?

    double returnValue = 0;

    const int dim = dimensions();
    for (int i = 0; i < dim; ++i)
    {
        double rectValue = singleRectangleValue(rects[i], *sum);
        returnValue += (weights[i] * rectValue);
    }

    return returnValue;
}

void HaarWavelet::srfs(std::vector<float> &srfsVector) const
{
    //TODO convert into exception
    assert(sum && squareSum);

    const int dim = dimensions();
    srfsVector.resize(dim);

    for (int i = 0; i < dim; ++i)
    {
        srfsVector[i] = singleRectangleValue(rects[i], *sum);

        //SRFS works with normalized means (Pavani et al., 2010, section 2.3).
        srfsVector[i] /= (rects[i].size().height * rects[i].size().width * UCHAR_MAX);
    }
}


/**
 * See also constructor that takes a cv::FileNode.
 */
bool HaarWavelet::write(cv::FileStorage &fs) const
{
    if (dimensions() == 0) //won't store a meaningless wavelet
    {
        return false;
    }

    fs << "{";
    fs << "rects" << dimensions();

    fs << "rect" << "[";
    for (int i = 0; i < dimensions(); ++i)
    {
        fs << rects[i].x
           << rects[i].y
           << rects[i].width
           << rects[i].height
           << weights[i];
    }
    fs << "]";

    fs << "}";

    return true;
}

/**
 * See also constructor that takes a std::istream.
 */
bool HaarWavelet::write(std::ostream &output) const
{
    if (dimensions() == 0) //won't store a meaningless wavelet
    {
        return false;
    }

    output << dimensions() << " ";

    bool first = true;
    for (int i = 0; i < dimensions(); i++)
    {
        if (first)
        {
            first = false;
        }
        else
        {
            output << " ";
        }
        output << rects[i].x << " "
               << rects[i].y << " "
               << rects[i].width << " "
               << rects[i].height << " "
               << weights[i];
    }

    return true;
}

std::vector<cv::Rect>::const_iterator HaarWavelet::rects_begin() const
{
    return rects.begin();
}

const std::vector<cv::Rect>::const_iterator HaarWavelet::rects_end() const
{
    return rects.end();
}

std::vector<float>::const_iterator HaarWavelet::weights_begin() const
{
    return weights.begin();
}

const std::vector<float>::const_iterator HaarWavelet::weights_end() const
{
    return weights.end();
}

float HaarWavelet::weight(const int index) const
{
    return weights[index];
}

void HaarWavelet::weight(const int index, const float new_value)
{
    weights[index] = new_value;
}

inline double HaarWavelet::singleRectangleValue(const cv::Rect &rect, const cv::Mat &s) const
{
    double rectVal = 0;

    //As per Lienhart, Maydt, 2002, section 2.2
    const int x_w = rect.x + rect.width;
    const int y_h = rect.y + rect.height;

    //TODO is there a faster implementation that avoids invoking s.at() functions? Maybe a pointer to the data?
    rectVal += s.at<int>(rect.y, rect.x); // (x,     y)
    rectVal -= s.at<int>(rect.y, x_w);    // (x + w, y)
    rectVal -= s.at<int>(y_h, rect.x);    // (x,     y + h)
    rectVal += s.at<int>(y_h, x_w);       // (x + w, y + h)

    return rectVal;
}
