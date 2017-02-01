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

    LBP();
    virtual ~LBP();
    int testWithVideo(const String&);
    static float getHistogramProximity(const vector<unsigned int>&, const vector<unsigned int>&);
    static void printHistogram(const vector<unsigned int>&);


protected:

private:
    Mat* pixels;
    vector<LBPPixel*> backgroundPixels;
    // Lookup array
    // [pattern] = class/bin
    unsigned int* uniformPatterns;
    long frameCount;

    void calculateFeatureDescriptors(Mat&);
    vector<unsigned int> calculateHistogram(int, int, int);

    Mat* combineFrames(Mat&, Mat&);
    Mat* getDescriptorMat();
    Mat* createMovementMatrix();
    void handleNewFrame(Mat&);

    vector<unsigned int> calculateHistogram(LBPPixel*);
    unsigned int getUniformPatternClass(vector<vector<unsigned int>>, unsigned int);

    void initLBPPixels(int, int, int);
    void setHistogramNeighbours(LBPPixel*);
    void showOutputVideo(Mat&, bool);

};

#endif
