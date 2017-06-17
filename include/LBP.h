#ifndef LBP_H
#define LBP_H

#include "opencv2/core/core.hpp"

#include <vector>
#include "LBPPixel.h"

class LBP {
public:
    static const float HISTOGRAM_PROXIMITY_THRESHOLD;
    static const unsigned int DESCRIPTOR_RADIUS;
    static const int PIXEL_VALUE_TOLERANCE;
    static const int HISTOGRAM_REGION_SIZE;
    static const unsigned int NEIGHBOUR_COUNT;
    static const unsigned int BIN_COUNT;

    LBP();
    LBP(int, int);
    virtual ~LBP();
    static float getHistogramProximity(const vector<unsigned int>&, const vector<unsigned int>&);
    static void printHistogram(const vector<unsigned int>&);
    static void calculateFeatureDescriptors(cv::Mat*, cv::Mat&);
    static void calculateFeatureDescriptors(cv::Mat*, cv::Mat&, int, int);
    vector<unsigned int> calculateHistogram(LBPPixel*);
    vector<unsigned int> calculateHistogram(int, int, int);
    int getBinCount();
    int getHistogramRegionSize();
protected:

private:

    // Lookup array
    // [pattern] = class/bin
    vector<unsigned int> uniformPatterns;
    long frameCount;
    int neighbourCount, binCount, histogramRegionSize;

    void genUniformPatternClasses(vector<unsigned int>&, unsigned int);

    cv::Mat* getDescriptorMat();
    cv::Mat* pixels;

    void setHistogramNeighbours(LBPPixel*);
};

#endif
