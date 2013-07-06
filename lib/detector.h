#ifndef DETECTOR_H
#define DETECTOR_H

#include "haarwavelet.h"

class Detector
{
public:
    Detector();

private:
    std::vector<HaarWavelet> wavelets;
};

#endif // DETECTOR_H
