#ifndef HAARWAVELET_H
#define HAARWAVELET_H

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

class Detector;

/**
 * @brief The HaarWavelet class represents a single Haar wavelet.
  */
class HaarWavelet
{
public:
    HaarWavelet(cv::Size * detectorSize_, cv::Point * detectorPosition_, std::vector<cv::Rect> rects_, std::vector<float> weights_);
    //HaarWavelet(const cv::FileNode &node);

    //amount of rectangles this Haar wavelet has
    int dimensions() const;

    //void setDetector(Detector * d_);
    bool setIntegralImages(cv::Mat * const sum_, cv::Mat * const squareSum_/*, cv::Mat * tilted*/);
    //bool setPosition(cv::Point * pt);

    //Returns the value of this Haar wavelet when applied to an image in a certain position
    double value() const;

    //Sets the values of the single rectangle feature space
    void srfs(std::vector<float> &srfsVector) const;

private:
    HaarWavelet();

    //Each rectangle and its associated weight of this Haar wavelet
    std::vector<cv::Rect> rects;
    std::vector<float> weights;

    //If scale > 1, the Haar wavelet streaches right and down
    float scale;

    //Integral images: both simple sum and squareSum.
    cv::Mat * sum,
            * squareSum;

    //The top x,y position of the detector window
    cv::Point * detectorPosition;

    //Size of the detector window
    cv::Size * detectorSize;

    //Holds a refernce for the detector that owns this Haar wavelet.
    Detector * detector;
};

#endif // HAARWAVELET_H
