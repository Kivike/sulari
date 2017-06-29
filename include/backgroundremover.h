#ifndef BACKGROUNDREMOVER_H
#define BACKGROUNDREMOVER_H

#include "opencv2/core/core.hpp"
#include "lbppixel.h"
#include "lbp.h"

class BackgroundRemover
{
    public:
        BackgroundRemover();
        virtual ~BackgroundRemover();
        void onNewFrame(cv::Mat& frame);
        int testWithVideo();
        int testWithVideo(const cv::String&);
    protected:

    private:
        static const bool COMBINE_FRAMES;
        static const bool INTERLACE;
        static const bool PRINT_FRAMERATE;

        LBP *lbp;
        bool useThreading;
        vector<LBPPixel*> backgroundPixels;
        int frameCount;
        cv::Mat* pixels;
        cv::Mat* combineFrames(cv::Mat&, cv::Mat&);
        void showOutputVideo(cv::Mat&, bool);
        void initLBPPixels(int, int, int);
        void setHistogramNeighbours(LBPPixel*);
        cv::Mat* createMovementMatrix();
        static void handleFrameRow(LBP*, int, cv::Mat*);
};

#endif // BACKGROUNDREMOVER_H
