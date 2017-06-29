#ifndef LBP_H
#define LBP_H

#include "opencv2/core/core.hpp"
#include <vector>
#include "lbppixel.h"

class LBP {
public:
    static const float HISTOGRAM_PROXIMITY_THRESHOLD;
    static const unsigned int DESCRIPTOR_RADIUS;
    static const int PIXEL_VALUE_TOLERANCE;
    static const int HISTOGRAM_REGION_SIZE;
    static const unsigned int NEIGHBOUR_COUNT;
    static const unsigned int BIN_COUNT;
    static const bool INTERLACE;
    static const bool PRINT_FRAMERATE;
    static const bool COMBINE_FRAMES;

    LBP();
    LBP(int, int);
    virtual ~LBP();
    int getBinCount();
    int getHistogramRegionSize();
    int testWithVideo();
    int testWithVideo(const cv::String&);
    static void calculateFeatureDescriptors(cv::Mat*, cv::Mat&, int, int);
    static float getHistogramProximity(const std::vector<unsigned int>&, const std::vector<unsigned int>&);
    static void printHistogram(const std::vector<unsigned int>&);
    void handleNewFrame(cv::Mat&);
    vector<unsigned int> calculateHistogram(LBPPixel*);
protected:

private:
    // Lookup array
    // [pattern] = class/bin
    std::vector<unsigned int> uniformPatterns;
    long frameCount;
    int neighbourCount, binCount, histogramRegionSize;
    void genUniformPatternClasses(std::vector<unsigned int>&, unsigned int);

    void calculateFeatureDescriptors(cv::Mat*, cv::Mat&);


    cv::Mat* combineFrames(cv::Mat&, cv::Mat&);
    cv::Mat* getDescriptorMat();
    cv::Mat* createMovementMatrix();
    cv::Mat* pixels;
    void initLBPPixels(int, int, int);
    void setHistogramNeighbours(LBPPixel*);
    void showOutputVideo(cv::Mat&, bool);
};

#endif
