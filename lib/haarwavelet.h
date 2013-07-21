#ifndef HAARWAVELET_H
#define HAARWAVELET_H

#include <vector>
#include <iostream>


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

    /**
     * "Raw" constructor for a Haar wavelet.
     */
    HaarWavelet(cv::Size * const detectorSize_, cv::Point * const detectorPosition_, std::vector<cv::Rect> rects_, std::vector<float> weights_);

    /**
     * Constructs a Haar wavelet from a given cv::FileNode.
     */
    HaarWavelet(cv::Size * const detectorSize_, cv::Point * const detectorPosition_, const cv::FileNode & node);

    /**
     * Constructs a Haar wavelet by reading a input stream.
     */
    HaarWavelet(cv::Size * const detectorSize_, cv::Point * const detectorPosition_, std::istream &input);

    /**
     * Amount of rectangles this Haar wavelet has
     */
    int dimensions() const;

    //void setDetector(Detector * d_);
    //bool setPosition(cv::Point * pt);
    bool setIntegralImages(cv::Mat * const sum_, cv::Mat * const squareSum_/*, cv::Mat * tilted*/);

    /**
     * Returns the value of this Haar wavelet when applied to an image in a certain position
     */
    double value() const;

    /**
     * Sets the values of the single rectangle feature space.
     */
    void srfs(std::vector<float> &srfsVector) const;

    /**
     * Writes this Haar wavelet into the given cv::FileStorage.
     */
    bool write(cv::FileStorage& fs) const;

    /**
     * Writes this Haar wavelet into the given std::ostream.
     */
    bool write(std::ostream &output) const;

    /**
     * Returns a constant iterator to the start of a collection of rectangles that compose this wavelet.
     */
    std::vector<cv::Rect>::const_iterator rects_begin() const;

    /**
     * Returns a constant iterator to the end of the collection of rectangles that compose this wavelet.
     */
    const std::vector<cv::Rect>::const_iterator rects_end() const;

    /**
     * Returns a constant iterator...
     */
    std::vector<float>::const_iterator weights_begin() const;

    /**
     * Returns a constant iterator...
     */
    const std::vector<float>::const_iterator weights_end() const;

private:

    /**
     * Constructs an "empty" instance of this object.
     */
    HaarWavelet();

    /**
     * Calculates the sum of pixels inside a rectangular area of the image.
     */
    inline double singleRectangleValue(const cv::Rect &rect, const cv::Point &position, const cv::Mat &s) const;

    /**
     * Each rectangle and its associated weight of this Haar wavelet
     */
    std::vector<cv::Rect> rects;
    std::vector<float> weights;

    /**
     *If scale > 1, the Haar wavelet streaches right and down
     */
    float scale;

    /**
     *Integral images: both simple sum and squareSum.
     */
    cv::Mat * sum,
            * squareSum;

    /**
     *Size of the detector window
     */
    cv::Size * const detectorSize;

    /**
     *The top x,y position of the detector window
     */
    cv::Point * const detectorPosition;

    /**
     *Holds a refernce for the detector that owns this Haar wavelet.
     */
    Detector * const detector;
};

#endif // HAARWAVELET_H
