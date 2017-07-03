#ifndef LBP_H
#define LBP_H

#include "opencv2/core/core.hpp"
#include <vector>
#include "lbppixel.h"

class LBP {
public:
    static const int PIXEL_VALUE_TOLERANCE;
    static const int HISTOGRAM_REGION_SIZE;
    static const unsigned int NEIGHBOUR_COUNT;
    static const unsigned int BIN_COUNT;
    static const bool COMBINE_FRAMES;
    static const bool INTERLACE;
    static const bool PRINT_FRAMERATE;
    static const float HISTOGRAM_PROXIMITY_THRESHOLD;
    static const unsigned int DESCRIPTOR_RADIUS;
    LBP();
    virtual ~LBP();
    int testWithVideo();
    int testWithVideo(const std::string&);
    static float getHistogramProximity(const std::vector<unsigned int>&, const std::vector<unsigned int>&);
    static void printHistogram(const std::vector<unsigned int>&);
    void handleNewFrame(cv::Mat&);
    std::vector<unsigned int> calculateHistogram(LBPPixel*);
    void calculateFeatureDescriptors(cv::Mat&);
    void calculateFeatureDescriptors(cv::Mat*, cv::Mat&, int, int);

protected:

private:
    std::vector<LBPPixel*> backgroundPixels;
    long frameCount;
    cv::Mat* pixels;

    // Lookup array: [pattern] = class/bin
    std::vector<unsigned int> uniformPatterns;

    void genUniformPatternClasses(unsigned int);

    cv::Mat* combineFrames(cv::Mat&, cv::Mat&);
    cv::Mat* getDescriptorMat();
    cv::Mat* createMovementMatrix();

    void initLBPPixels(int, int, int);
    void setHistogramNeighbours(LBPPixel*);
    void showOutputVideo(cv::Mat&, bool);
};

#endif
