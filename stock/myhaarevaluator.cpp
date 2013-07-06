#include "myhaarevaluator.h"



MyHaarEvaluator::Feature::Feature(cv::Point position, cv::Size size, float weight_)
{
    Feature(cv::Rect(position, size), weight_);
}

MyHaarEvaluator::Feature::Feature(cv::Rect rect_, float weight_)
{
    rect = rect_;
    weight = weight_;
}

MyHaarEvaluator::Feature::Feature(const cv::FileNode &node)
{
    /*
    cv::FileNode rectangleNode = node["rect"];
    cv::FileNodeIterator it  = rectangleNode.begin(),
                         end = rectangleNode.end();

    for(; it != end; ++it)
    {
        it >> rect.x
           >> rect.y
           >> rect.width
           >> rect.height
           >> weight;
    }
    */
}

float MyHaarEvaluator::Feature::sum() const
{
    //TODO
    return 0;
}

float MyHaarEvaluator::Feature::mean() const
{
    //TODO
    return 0;
}



MyHaarEvaluator::MyHaarEvaluator()
{
}

MyHaarEvaluator::MyHaarEvaluator(std::vector<cv::Rect> rects, std::vector<float> weights)
{
    //TODO assert rects.size() == weights.size()
    const int rectangles = rects.size();
    features.resize(rectangles, Feature(cv::Rect(), 0));

    for(int i = 0; i < rectangles; ++i)
    {
        features[i] = Feature(rects[i], weights[i]);
    }
}

MyHaarEvaluator::~MyHaarEvaluator()
{
}

bool MyHaarEvaluator::read(const cv::FileNode &node)
{
    features.resize(node.size(), Feature(cv::Rect(), 0));
    cv::FileNodeIterator it  = node.begin(),
                         end = node.end();

    for(int i = 0; it != end; ++it, i++)
    {
        features[i] = Feature(*it);
    }

    return true;
}

cv::Ptr<cv::FeatureEvaluator> MyHaarEvaluator::clone() const
{
    MyHaarEvaluator * newEval = new MyHaarEvaluator();
    newEval->features = features;
    return newEval;
}

int MyHaarEvaluator::getFeatureType() const
{
    return TYPE;
}

bool MyHaarEvaluator::setImage(const cv::Mat &image, cv::Size origWinSize)
{
    if (image.cols < origWinSize.width || image.rows < origWinSize.height)
    {
        return false;
    }

    detectorSize = origWinSize;
}

bool MyHaarEvaluator::setWindow(cv::Point pt)
{
    //TODO atualiza a posição da janela
    if( pt.x < 0 || pt.y < 0 ||
        pt.x + detectorSize.width >= image.cols ||
        pt.y + detectorSize.height >= image.rows )
    {
        return false;
    }


}

void MyHaarEvaluator::srfs(std::vector<float> & srfs_vector) const
{
    srfs_vector.resize(features.size());
    //TODO lots of stuff
}

double MyHaarEvaluator::calcOrd(int featureIdx) const
{
}

int MyHaarEvaluator::calcCat(int featureIdx) const
{
    //TODO is this method supposed to work???
    return 0;
}
