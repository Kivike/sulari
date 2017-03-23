#ifndef LBP_H
#define LBP_H

#include "opencv2/core/core.hpp"
using namespace cv;

#include <vector>

#include "LBPPixel.h"

using namespace std;

class LBP {
public:
    static const float HISTOGRAM_PROXIMITY_THRESHOLD;
    static const unsigned int DESCRIPTOR_RADIUS;
    LBP();
    virtual ~LBP();
    int testWithVideo();
    int testWithVideo(const String&);
    static float getHistogramProximity(const vector<unsigned int>&, const vector<unsigned int>&);
    static void printHistogram(const vector<unsigned int>&);
    vector<unsigned int> calculateHistogram(LBPPixel*);
    void handleNewFrame(Mat&);
protected:

private:
    static const int PIXEL_VALUE_TOLERANCE;
    static const int HISTOGRAM_REGION_SIZE;
    static const unsigned int NEIGHBOUR_COUNT;
    static const unsigned int BIN_COUNT;
    static const bool COMBINE_FRAMES;
    static const bool INTERLACE;
    static const bool PRINT_FRAMERATE;

    vector<LBPPixel*> backgroundPixels;
    // Lookup array
    // [pattern] = class/bin
    vector<unsigned int> uniformPatterns;
    long frameCount;

    void genUniformPatternClasses(vector<unsigned int>&, unsigned int);

    void calculateFeatureDescriptors(Mat&);
    void calculateFeatureDescriptors(Mat&, int, int);
    vector<unsigned int> calculateHistogram(int, int, int);

    Mat* combineFrames(Mat&, Mat&);
    Mat* getDescriptorMat();
    Mat* createMovementMatrix();
    Mat* pixels;


    void initLBPPixels(int, int, int);
    void setHistogramNeighbours(LBPPixel*);
    void showOutputVideo(Mat&, bool);
};

#endif
