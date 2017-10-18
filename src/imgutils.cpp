#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "imgutils.h"

using namespace cv;

/**
 * Scales frame size if needed to match the given limits
 * @param  frame   Frame to scale
 * @param  minSize Minimum rows & columns
 * @param  maxSize Maximum rows & columns
 * @return         Returns scaled frame
 */
Mat ImgUtils::clampFrameSize(Mat *frame, const Size minSize, const Size maxSize) {
    float multiplier = 1;

    if(frame->cols < minSize.width || frame->rows < minSize.height){
        multiplier = minSize.width / (float)frame->cols;

        if(multiplier * frame->rows < minSize.height) {
            multiplier = minSize.height / (float)frame->rows;
        }
    } else if(frame->cols > maxSize.width || frame->rows > maxSize.height) {
        multiplier = maxSize.width / (float)frame->cols;

        if(multiplier * frame->rows > maxSize.height) {
            multiplier = maxSize.height / (float)frame->rows;
        }
    }
    Mat newFrame;
    resize(*frame, newFrame, Size(frame->cols*multiplier, frame->rows*multiplier));
    return newFrame;
}

/**
 * Get frame that takes minimum grayscale value for each pixel
 * @param  img_a     [description]
 * @param  img_b [description]
 * @return         [description]
 */
Mat* ImgUtils::frameMin(Mat& img_a, Mat& img_b) {
    if(img_a.rows != img_b.rows || img_a.cols != img_b.cols) {
        return nullptr;
    }

    Mat *output = new Mat(img_a.rows, img_a.cols, CV_8UC1);

    for(int i = 0; i < img_a.rows; i++) {
        for(int j = 0; j < img_a.cols; j++) {
            output->at<unsigned char>(i, j) = min(img_a.at<unsigned char>(i, j),
                                                  img_b.at<unsigned char>(i, j));
        }
    }
    return output;
}
