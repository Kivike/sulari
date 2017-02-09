#ifndef LBPPIXEL_H
#define LBPPIXEL_H

#include "AdaptiveHistogram.h"
#include <vector>

using namespace std;

class LBPPixel
{
    public:
        LBPPixel(int, int, int, int);
        virtual ~LBPPixel();

        unsigned char getColor(bool);
        unsigned char getDescriptor();
        bool isBackground(const vector<unsigned int>&);
        void setLowestWeightHistogram(vector<unsigned int>);
        void setDescriptor(unsigned char);
        void sortHistograms();
        void updateAdaptiveHistograms(const vector<unsigned int>&);

        void printHistogramWeights();
        void printPosition();

        void setHistogramNeighbours(const vector<LBPPixel*>&);
        vector<LBPPixel*> getHistogramNeighbours();

        int getRow();
        int getCol();
    protected:

    private:
        unsigned char color, descriptor;
        int row, col, histStartRow, histEndRow, histStartCol, histEndCol;
        vector<AdaptiveHistogram*> histograms, backgroundHistograms;
        vector<LBPPixel*> histogramNeighbours;

        void setColor(unsigned char);

        static bool compareWeight(AdaptiveHistogram*, AdaptiveHistogram*);
        void updateBackgroundHistograms();
        void updateHistogramWeights(const vector<unsigned int>&, int, float);
        void getBestProximityMatch(const vector<unsigned int>&, int&, float&);

};

#endif // LBPPIXEL_H
