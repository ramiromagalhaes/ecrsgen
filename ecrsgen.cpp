#include <iostream>
#include <string>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem.hpp>

#include "lib/haarwavelet.h"

#define SAMPLE_SIZE 20

inline void loadWavelets(const std::string & waveletsFile, cv::Point & position, cv::Size & sampleSize, std::vector<HaarWavelet*> & wavelets)
{
    boost::filesystem::path archive( waveletsFile );

    boost::filesystem::ifstream ifs( archive );
    boost::archive::text_iarchive ia(ifs);

    std::vector<float> weights;
    std::vector<cv::Rect> rects;

    HaarWavelet * wavelet = new HaarWavelet(&sampleSize, &position, rects, weights);
    wavelets.push_back(wavelet);

    ifs.close();
}

inline void loadSamples(const std::string & samplesDir, std::vector<std::string> & samples)
{
    boost::filesystem::path dir(samplesDir);
    const boost::filesystem::directory_iterator end_iter;

    if ( boost::filesystem::exists(dir) && boost::filesystem::is_directory(dir))
    {
        for( boost::filesystem::directory_iterator dir_iter(dir) ; dir_iter != end_iter ; ++dir_iter)
        {
            if ( boost::filesystem::is_regular_file(dir_iter->status()) )
            {
                samples.push_back(dir_iter->path().string());
            }
        }
    }
}

inline void append_srfs(std::vector<double> & srfs_vector, std::string dir, std::string filename)
{
    boost::filesystem::path archive( dir );
    archive = archive / filename;

    boost::filesystem::ofstream ofs( archive );
    boost::archive::text_oarchive oa(ofs);

    bool not_first = false;
    std::vector<double>::iterator srfsIt = srfs_vector.begin();
    const std::vector<double>::iterator srfsEnd = srfs_vector.end();
    for( ;srfsIt != srfsEnd; ++srfsIt)
    {
        if (not_first)
        {
            oa << " ";
        }
        else
        {
            not_first = true;
        }
        oa << *srfsIt;
    }
    std::cout << std::endl;

    ofs.close();
}


int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        return 1;
    }

    const std::string waveletsFile = argv[1]; //load Haar wavelets from here
    const std::string samplesDir = argv[2]; //load samples from here
    const std::string outputDir = argv[3]; //write output here

    cv::Point position(0,0); //always like that during SRFS production
    cv::Size sampleSize(SAMPLE_SIZE, SAMPLE_SIZE); //size in pixels of the trainning images



    std::vector<HaarWavelet*> wavelets; //list of haar wavelets
    loadWavelets(waveletsFile, position, sampleSize, wavelets);

    std::vector<std::string> samples; //samples files from here
    loadSamples(samplesDir, samples);

    //TODO crio/carrego uma lista de arquivos de saída
    std::vector<std::string> outputs; //where I'll write the files



    std::vector<std::string>::iterator samplesIt = samples.begin();
    const std::vector<std::string>::iterator samplesEnd = samples.end();
    for( ;samplesIt != samplesEnd; ++samplesIt)
    {
        cv::Mat sample = cv::imread(*samplesIt);

        cv::Mat integralSum(sample.rows + 1, sample.cols + 1, CV_32S);
        cv::Mat integralSquare(sample.rows + 1, sample.cols + 1, CV_32S);
        cv::integral(sample, integralSum, integralSquare, CV_32S);

        //para cada haar wavelet
        std::vector<HaarWavelet*>::iterator waveletIt = wavelets.begin();
        const std::vector<HaarWavelet*>::iterator waveletsEnd = wavelets.end();
        for(int i = 0 ;waveletIt != waveletsEnd; ++waveletIt, ++i)
        {
            std::vector<double> srfs_vector;

            HaarWavelet * wavelet = *waveletIt;
            wavelet->setIntegralImages(&integralSum, &integralSquare);
            wavelet->srfs(srfs_vector);

            std::stringstream filename;
            filename << "srfs-haar-" << i << ".txt";
            append_srfs(srfs_vector, samplesDir, filename.str());
        }
    }

    return 0;
}
