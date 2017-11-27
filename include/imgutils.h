#ifndef IMGUTILS_H
#define IMGUTILS_H

#include "opencv2/core.hpp"

class ImgUtils
{
public:
    static cv::Mat clampFrameSize(cv::Mat*, const cv::Size, const cv::Size);
    static cv::Mat* frameMin(cv::Mat&, cv::Mat&);
};

#endif
