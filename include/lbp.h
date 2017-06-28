#ifndef LBP_H
#define LBP_H

#include "opencv2/core/core.hpp"

#include <vector>
<<<<<<< HEAD:include/LBP.h
#include "LBPPixel.h"
=======

#include "lbppixel.h"
>>>>>>> dev:include/lbp.h

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
<<<<<<< HEAD:include/LBP.h
    static float getHistogramProximity(const vector<unsigned int>&, const vector<unsigned int>&);
    static void printHistogram(const vector<unsigned int>&);
    static void calculateFeatureDescriptors(cv::Mat*, cv::Mat&);
    static void calculateFeatureDescriptors(cv::Mat*, cv::Mat&, int, int);
    vector<unsigned int> calculateHistogram(LBPPixel*);
    vector<unsigned int> calculateHistogram(int, int, int);
    int getBinCount();
    int getHistogramRegionSize();
=======
    int testWithVideo();
    int testWithVideo(const cv::String&);
    static float getHistogramProximity(const std::vector<unsigned int>&, const std::vector<unsigned int>&);
    static void printHistogram(const std::vector<unsigned int>&);
    void handleNewFrame(cv::Mat&);
    vector<unsigned int> calculateHistogram(LBPPixel*);
>>>>>>> dev:include/lbp.h
protected:

private:

<<<<<<< HEAD:include/LBP.h
=======
    std::vector<LBPPixel*> backgroundPixels;
>>>>>>> dev:include/lbp.h
    // Lookup array
    // [pattern] = class/bin
    std::vector<unsigned int> uniformPatterns;
    long frameCount;
    int neighbourCount, binCount, histogramRegionSize;

<<<<<<< HEAD:include/LBP.h
    void genUniformPatternClasses(vector<unsigned int>&, unsigned int);

    cv::Mat* getDescriptorMat();
    cv::Mat* pixels;

    void setHistogramNeighbours(LBPPixel*);
=======
    void genUniformPatternClasses(std::vector<unsigned int>&, unsigned int);

    void calculateFeatureDescriptors(cv::Mat&);
    void calculateFeatureDescriptors(cv::Mat&, int, int);

    cv::Mat* combineFrames(cv::Mat&, cv::Mat&);
    cv::Mat* getDescriptorMat();
    cv::Mat* createMovementMatrix();
    cv::Mat* pixels;
    void initLBPPixels(int, int, int);
    void setHistogramNeighbours(LBPPixel*);
    void showOutputVideo(cv::Mat&, bool);
>>>>>>> dev:include/lbp.h
};

#endif
