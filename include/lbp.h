#ifndef LBP_H
#define LBP_H

#include <vector>
#include <mutex>

#include "opencv2/core/core.hpp"

#include "lbppixel.h"

class LBP
{
public:
    static const int HISTOGRAM_REGION_SIZE;
    static const unsigned int BIN_COUNT;
    static const unsigned int DESCRIPTOR_RADIUS;

    int testWithVideo();
    int testWithVideo(const std::string&);
    static float getHistogramProximity(const std::vector<unsigned int>&, const std::vector<unsigned int>&);
    static void printHistogram(const std::vector<unsigned int>&);
    void handleNewFrame(cv::Mat&);
    std::vector<unsigned int> calculateHistogram(LBPPixel*);
    void calculateFeatureDescriptors(const cv::Mat&);
    void calculateFeatureDescriptors(cv::Mat*, const cv::Mat&);

    LBP();
    ~LBP() {
        for(auto p : backgroundPixels) delete p;
    }
protected:
private:
    static const int PIXEL_VALUE_TOLERANCE;
    static const unsigned int NEIGHBOUR_COUNT;
    static const unsigned int LBP_BIN_COUNT;

    std::vector<LBPPixel*> backgroundPixels;
    cv::Mat* pixels;

    // Lookup array: [pattern] = class/bin
    static std::vector<unsigned int> *uniformPatterns;

    std::vector<unsigned int>* genUniformPatternClasses(const unsigned int);
    cv::Mat combineFrames(const cv::Mat&, const cv::Mat&);
    cv::Mat createMovementMatrix();

    void initLBPPixels(const int, const int, const int);
    void setHistogramNeighbours(LBPPixel*);
    static std::mutex mtx;
};

#endif
