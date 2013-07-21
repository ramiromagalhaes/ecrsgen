#include <string>
#include <iostream>
#include <fstream>

#include "haarwavelet.h"

bool loadHaarWavelets(cv::Size * const sampleSize, cv::Point * const position, const std::string &filename, std::vector<HaarWavelet *> &wavelets)
{
    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ifstream::in);

    if ( !ifs.is_open() )
    {
        return false;
    }

    while(!ifs.eof())
    {
        HaarWavelet * wavelet = new HaarWavelet(sampleSize, position, ifs);
        wavelets.push_back(wavelet);
    }

    ifs.close();

    return true;
}
