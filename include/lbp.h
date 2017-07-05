#ifndef LBP_H
#define LBP_H

#include "opencv2/core/core.hpp"
#include <vector>
#include "lbppixel.h"

class LBP {
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
    void calculateFeatureDescriptors(cv::Mat&);
    void calculateFeatureDescriptors(cv::Mat*, cv::Mat&);

    LBP();
    ~LBP() {
        for(auto p : backgroundPixels) delete p;
    }
protected:
private:
    static const int PIXEL_VALUE_TOLERANCE;
    static const unsigned int NEIGHBOUR_COUNT;

    std::vector<LBPPixel*> backgroundPixels;
    long frameCount;
    cv::Mat* pixels;

    // Lookup array: [pattern] = class/bin
    static std::vector<unsigned int> uniformPatterns;

    cv::Mat* combineFrames(cv::Mat&, cv::Mat&);
    cv::Mat* getDescriptorMat();
    cv::Mat* createMovementMatrix();

    void initLBPPixels(int, int, int);
    void setHistogramNeighbours(LBPPixel*);
    void showOutputVideo(cv::Mat&, bool);
};

#endif
