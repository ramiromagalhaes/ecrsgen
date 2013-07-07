#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "myhaarevaluator.h"



int main(int argc, char* argv[])
{
    const cv::Size sampleSize(sampleSummedAreaTable.cols, sampleSummedAreaTable.rows);

    //carrego uma lista de parâmetros de haar wavelets
    //carrego uma lista de arquivos com amostras positivas
    //carrego uma lista de arquivos com amostras negativas
    //crio/carrego uma lista de arquivos de saída

    //para cada amostra (positiva ou negativa)
    for(;;)
    {
        std::string image;
        cv::Mat sample = imread(image);
        cv::Mat sampleSummedAreaTable;
        cv::Mat sampleSquaredSummedAreaTable;
        cv::integral(sample, sampleSummedAreaTable, sampleSquaredSummedAreaTable, CV_32S);

        //para cada parâmetro do haar wavelet
        for(;;)
        {
            std::vector<cv::Rect> rects;
            std::vector<float> weights;

            MyHaarEvaluator haar(rects, weights);

            std::vector<float> srfs_vector(rects.size());

            haar.setImage(sampleSummedAreaTable, sampleSize);
            haar.setWindow(cv::Point(0, 0));
            haar.srfs(srfs_vector);

            //salva srfs_vector adequadamente
        }
    }

    return 0;
}
