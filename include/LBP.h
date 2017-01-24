#ifndef LBP_H
#define LBP_H

#include "opencv2/core/core.hpp"
using namespace cv;

class LBP {
public:
    LBP(float, int);
    virtual ~LBP();
    int testWithVideo();

protected:

private:
    Mat* calculateFeatureDescriptors(Mat&);
    Mat* calculateHistograms(Mat&, int);
    Mat* getHistogramProximity(Mat&, Mat&);
    Mat* combineFrames(Mat&, Mat&);
    unsigned int* calculateHistogram(Mat&, int, int, int);
    unsigned int getUniformPatternClass(unsigned int**, unsigned int);
    void printHistogram(unsigned int[]);

};

#endif
