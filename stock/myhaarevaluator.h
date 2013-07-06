#ifndef MYHAAREVALUATOR_H
#define MYHAAREVALUATOR_H


#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

class MyHaarEvaluator : public cv::FeatureEvaluator
{
public:
    MyHaarEvaluator(std::vector<cv::Rect> rects, std::vector<float> weights);

    virtual ~MyHaarEvaluator();

    virtual bool read(const cv::FileNode& node);
    virtual cv::Ptr<cv::FeatureEvaluator> clone() const;
    virtual int getFeatureType() const;

    virtual bool setImage(const cv::Mat &image, cv::Size origWinSize);
    virtual bool setWindow(cv::Point p);

    virtual void srfs(std::vector<float> &srfs_vector) const;
    virtual double calcOrd(int featureIdx) const;
    virtual int calcCat(int featureIdx) const;

    //static cv::Ptr<cv::FeatureEvaluator> create(int type);

private:
    MyHaarEvaluator();

    struct Feature {
        Feature(cv::Point position, cv::Size size, float weight_);
        Feature(cv::Rect rect_, float weight_);
        Feature(const cv::FileNode &node);
        float sum() const;
        float mean() const;

        cv::Rect rect;
        float weight;
    };

    //The rectangles that form a single haar wavelet
    std::vector<Feature> features;

    //where on the image the detector currently is
    cv::Point detectorPosition;

    //the height and width of the detector
    cv::Size detectorSize;

    //a reference to the whole image
    cv::Mat image;

    static const int TYPE = -1;
};

#endif // MYHAARWAVELET_H
