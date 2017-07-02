#ifndef LBPPIXEL_H
#define LBPPIXEL_H

#include "adaptivehistogram.h"
#include <vector>

class LBPPixel
{
    public:
        LBPPixel(int, int, int, int);
        virtual ~LBPPixel();

        unsigned char getColor(bool);
        unsigned char getDescriptor();
        bool isBackground(const std::vector<unsigned int>&);
        void setLowestWeightHistogram(std::vector<unsigned int>);
        void setDescriptor(unsigned char);
        void sortHistograms();
        void updateAdaptiveHistograms(const std::vector<unsigned int>&);

        void printHistogramWeights();
        void printPosition();

        void setHistogramNeighbours(const std::vector<LBPPixel*>&);
        std::vector<LBPPixel*> getHistogramNeighbours();

        int getRow();
        int getCol();
    protected:

    private:
        unsigned int color, descriptor;
        static const unsigned char FOREGROUND_COLOR, BACKGROUND_COLOR;
        /**
         * Threshold weight for considering a pixel background
         */
        static const float BACKGROUND_WEIGHT;

        int row, col;

        std::vector<AdaptiveHistogram*> histograms, backgroundHistograms;
        /**
         *  Neighbours in rectangular area from which histogram is calculated from
         */
        std::vector<LBPPixel*> histogramNeighbours;
        void setColor(unsigned char);

        static bool compareWeight(AdaptiveHistogram*, AdaptiveHistogram*);
        void updateBackgroundHistograms();
        void updateHistogramWeights(const std::vector<unsigned int>&, int, float);
        void getBestProximityMatch(const std::vector<unsigned int>&, int&, float&);

};

#endif // LBPPIXEL_H
