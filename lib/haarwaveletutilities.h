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

    while( !ifs.eof() )
    {
        HaarWavelet * wavelet = new HaarWavelet(sampleSize, position, ifs);
        wavelets.push_back(wavelet);
    }

    ifs.close();

    return true;
}



bool writeHaarWavelets(const char * filename, const std::vector<HaarWavelet * > &wavelets)
{
    std::ofstream ofs;
    ofs.open(filename, std::ofstream::out | std::ofstream::trunc);

    if (!ofs.is_open())
    {
        return false;
    }

    std::vector<HaarWavelet * >::const_iterator it = wavelets.begin();
    const std::vector<HaarWavelet * >::const_iterator end = wavelets.end();
    for(; it != end; ++it)
    {
        (*it)->write(ofs);
        ofs << '\n';
    }
    ofs.close();

    return true;
}
