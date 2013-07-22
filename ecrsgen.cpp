#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/filesystem.hpp>

#include "lib/haarwavelet.h"
#include "lib/haarwaveletutilities.h"



#define SAMPLE_SIZE 20



std::string srfsOutputFileName(HaarWavelet * wavelet)
{
    std::stringstream filename;

    std::vector<cv::Rect>::const_iterator itRects = wavelet->rects_begin();
    const std::vector<cv::Rect>::const_iterator endRects = wavelet->rects_end();
    std::vector<float>::const_iterator itWeights = wavelet->weights_begin();
    //const std::vector<float>::const_iterator endWeights = wavelet.weights_end();

    wavelet->write(filename);
    filename << ".txt";

    return filename.str();
}



int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cout << "Usage " << argv[0] << " " << " WAVELETS_FILE SAMPLES_DIR OUTPUT_DIR" << std::endl;
        return 1;
    }

    const std::string waveletsFileName = argv[1]; //load Haar wavelets from here
    const std::string samplesDirName = argv[2]; //load samples from here
    const std::string outputDirName = argv[3]; //write output here

    cv::Size sampleSize(SAMPLE_SIZE, SAMPLE_SIZE); //size in pixels of the trainning images
    cv::Point position(0,0); //always like that during SRFS production



    //Load a list of Haar wavelets
    std::cout << "Loading wavelets..." << std::endl;
    std::vector<HaarWavelet*> wavelets;
    if (!loadHaarWavelets(&sampleSize, &position, waveletsFileName, wavelets))
    {
        std::cout << "Unable to load Haar wavelets from file " << waveletsFileName << std::endl;
        return 2;
    }
    std::cout << "Loading wavelets done." << std::endl;

    //Check if the samples directory exist and is a directory
    boost::filesystem::path samplesDir(samplesDirName);
    if (!boost::filesystem::exists(samplesDir) || !boost::filesystem::is_directory(samplesDir))
    {
        std::cout << "Sample directory " << samplesDir << " does not exist or is not a directory." << std::endl;
        return 3;
    }

    //Check if the output directory exist and is a directory
    boost::filesystem::path outputDir(outputDirName);
    if (!boost::filesystem::exists(samplesDir) || !boost::filesystem::is_directory(samplesDir))
    {
        std::cout << "Output directory " << samplesDir << " does not exist or is not a directory." << std::endl;
        return 4;
    }



    std::cout << "Generating SRFS..." << std::endl;

    std::vector<HaarWavelet*>::iterator waveletIt = wavelets.begin();
    const std::vector<HaarWavelet*>::iterator waveletsEnd = wavelets.end();
    for(; waveletIt != waveletsEnd; ++waveletIt)
    {
        HaarWavelet * wavelet = *waveletIt;

        //Open/create an output file
        std::string outputFileName = srfsOutputFileName(wavelet);
        boost::filesystem::path outputPath = outputDir / outputFileName;
        outputFileName = outputPath.string();
        std::ofstream output(outputFileName.c_str(), std::ios::out | std::ios::app /*| std::ios::binary*/);
        if (!output.is_open())
        {
            std::cout << "Could not open file " << outputFileName << " to append data." << std::endl;
            return 5;
        }



        //write the amount of rectangles first
        output << wavelet->dimensions() << std::endl;

        //For each sample image, produce the SRFS
        const boost::filesystem::directory_iterator end_iter;
        for( boost::filesystem::directory_iterator dir_iter(samplesDir) ; dir_iter != end_iter ; ++dir_iter)
        {
            if ( !boost::filesystem::is_regular_file(dir_iter->status()) )
            {
                continue;
            }

            //load the sample image...
            const std::string samplename = dir_iter->path().string();
            cv::Mat sample = cv::imread(samplename, CV_LOAD_IMAGE_GRAYSCALE); //TODO check if the image was loaded.
            if (!sample.data)
            {
                std::cerr << "Failed to open file sample file " << samplename;
                continue;
            }
            //...then set it up...
            cv::Mat integralSum(sample.rows + 1, sample.cols + 1, CV_32S);
            cv::Mat integralSquare(sample.rows + 1, sample.cols + 1, CV_32S);
            cv::integral(sample, integralSum, integralSquare, CV_32S);
            //...then produce the SRFS...
            std::vector<float> srfs_vector(wavelet->dimensions());
            wavelet->setIntegralImages(&integralSum, &integralSquare);
            wavelet->srfs(srfs_vector);
            //...and write the it to a file.
            std::vector<float>::const_iterator itsrfs = srfs_vector.begin();
            const std::vector<float>::const_iterator endsrfs = srfs_vector.end();
            for(; itsrfs != endsrfs; ++itsrfs)
            {
                const float f = *itsrfs;
                output << f << " ";
            }
            output << std::endl;
        }
    }

    return 0;
}
