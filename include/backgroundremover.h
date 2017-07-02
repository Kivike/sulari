#ifndef BACKGROUNDREMOVER_H
#define BACKGROUNDREMOVER_H

#include "lbppixel.h"
#include "lbp.h"
#include "opencv2/core/core.hpp"

class BackgroundRemover {
public:
    BackgroundRemover();

    int testWithVideo();
    int testWithVideo(const std::string&);
    void initLBPPixels(int, int, int);
    void setHistogramNeighbours(LBPPixel*);

    cv::Mat* combineFrames(cv::Mat&, cv::Mat&);
    cv::Mat* createMovementMatrix();

    void showOutputVideo(cv::Mat&, bool);

    static void handleFrameRow(LBP*, int, cv::Mat*);
    void onNewFrame(cv::Mat& frame);
    ~BackgroundRemover();
protected:
private:
    static const bool COMBINE_FRAMES;
    static const bool INTERLACE;
    static const bool PRINT_FRAMERATE;
    bool useThreading;
    LBP *lbp;
    cv::Mat *pixels;
    int frameCount;
};


#endif
