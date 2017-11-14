#ifndef BACKGROUNDREMOVER_H
#define BACKGROUNDREMOVER_H

#include <mutex>
#include <vector>

#include "opencv2/core/core.hpp"

#include "lbppixel.h"
#include "lbp.h"

struct BoundingBox {
    unsigned int startx;
    unsigned int endx;
    unsigned int starty;
    unsigned int endy;
};

class BackgroundRemover {
public:
    std::vector<LBPPixel*> fgPixels;
    LBP *lbp;
    BackgroundRemover();
    BoundingBox *fgBoundingBox;

    void initLBPPixels(const int, const int, const int);
    void setHistogramNeighbours(LBPPixel*);

    cv::Mat combineFrames(cv::Mat&, cv::Mat&);
    cv::Mat cropBackground(cv::Mat&, cv::Rect*);
    cv::Mat createMovementMatrix();

    cv::Rect* getForegroundBoundingBox(int, int, int, int);

    static void handleFrameRows(BackgroundRemover*, cv::Mat*, const unsigned int, const unsigned int, const unsigned int);
    void onNewFrame(const cv::Mat& frame);
    ~BackgroundRemover() {
        delete lbp;
        delete fgBoundingBox;
    }
protected:
private:
    bool useThreading;
    cv::Mat *pixels;
    int frameCount;
};


#endif
