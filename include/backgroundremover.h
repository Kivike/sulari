#ifndef BACKGROUNDREMOVER_H
#define BACKGROUNDREMOVER_H

#include "lbppixel.h"
#include "lbp.h"
#include "opencv2/core/core.hpp"
#include <mutex>
#include <vector>

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

    void initLBPPixels(int, int, int);
    void setHistogramNeighbours(LBPPixel*);

    cv::Mat* combineFrames(cv::Mat&, cv::Mat&);
    cv::Mat* createMovementMatrix();

    cv::Rect* getForegroundBoundingBox(unsigned int, unsigned int);
    void showOutputVideo(cv::Mat&, bool);

    static void handleFrameRows(BackgroundRemover*, unsigned int, unsigned int, cv::Mat*);
    void onNewFrame(cv::Mat& frame);
    ~BackgroundRemover() {
        delete lbp;
        delete fgBoundingBox;
    }
protected:
private:
    static const bool COMBINE_FRAMES;
    static const bool INTERLACE;
    static const unsigned int BOUNDING_BOX_PADDING;
    bool useThreading;
    cv::Mat *pixels;
    int frameCount;
};


#endif
