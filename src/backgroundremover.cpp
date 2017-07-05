/**
* Removes background from video using adaptive LBP
* Works by comparing area around pixels between frames
* Once area has stayed same for long enough, pixels is considered background
*/

#include <thread>
#include <mutex>
#include <vector>
#include <cmath>
#include <iostream>

#include <execinfo.h>
#include <sys/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "backgroundremover.h"
#include "lbp.h"
#include "lbppixel.h"

using namespace cv;
using namespace std;

const bool BackgroundRemover::COMBINE_FRAMES = true;
const bool BackgroundRemover::INTERLACE = false;

const unsigned int BackgroundRemover::BOUNDING_BOX_PADDING = 10;

mutex mtx;

BackgroundRemover::BackgroundRemover(): pixels(nullptr) {
    lbp = new LBP();
}

//
/**
 * Create pixels and connect histogram neighbours
 * @param rows      Rows in frame
 *  @param cols     Columns in frame
 * @param histCount How many adaptive histograms does a pixel have
 */
void BackgroundRemover::initLBPPixels(int rows, int cols, int histCount) {
    pixels = new Mat(rows, cols, DataType<LBPPixel*>::type);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            pixels->at<LBPPixel*>(i, j) = new LBPPixel(histCount, LBP::BIN_COUNT, i, j);
        }
    }

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            setHistogramNeighbours(pixels->at<LBPPixel*>(i, j));
        }
    }
}

// Set pixels that will be used for calculating histogram
// pixel: pixel of which to set neighbours for

void BackgroundRemover::setHistogramNeighbours(LBPPixel* pixel) {
    int halfRegionSize = LBP::HISTOGRAM_REGION_SIZE/2;

    int startRow = max(1, pixel->getRow() - halfRegionSize);
    int endRow = min(pixels->rows - 1, pixel->getRow() + halfRegionSize);
    int startCol = max(1, pixel->getCol() - halfRegionSize);
    int endCol = min(pixels->cols - 1, pixel->getCol() + halfRegionSize);

    vector<LBPPixel*> neighbours = {};

    for(int i = startRow; i < endRow; i++) {
        for(int j = startCol; j < endCol; j++) {
            LBPPixel* nb = pixels->at<LBPPixel*>(i, j);
            neighbours.push_back(nb);
        }
    }

    pixel->setHistogramNeighbours(neighbours);
}

// Combines image and movement matrix
Mat* BackgroundRemover::combineFrames(Mat& img, Mat& mMatrix) {
    if(img.rows != mMatrix.rows || img.cols != mMatrix.cols) {
        return &img;
    }

    Mat *output = new Mat(img.rows, img.cols, CV_8UC1);

    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
            output->at<unsigned char>(i, j) = min(img.at<unsigned char>(i, j),
                                                  mMatrix.at<unsigned char>(i, j));
        }
    }
    return output;
}

// Create 2-color frame of foreground and background pixels
Mat* BackgroundRemover::createMovementMatrix() {
    Mat* result = new Mat(pixels->rows, pixels->cols, CV_8UC1);

    for(int i = 0; i < result->rows; i++) {
        for(int j = 0; j < result->cols; j++) {
            LBPPixel *pixel = pixels->at<LBPPixel*>(i, j);

            int col = pixel->getColor(false);
            result->at<unsigned char>(i, j) = col;
        }
    }

    return result;
}

void BackgroundRemover::showOutputVideo(Mat &frame, bool combine) {
    Mat movementMatrix = *createMovementMatrix();

    if(combine) {
        // Show original (grey scale) video frame and movement detection frame in same image
        Mat combinedFrame = *combineFrames(frame, movementMatrix);

        imshow("Video", combinedFrame);
    } else {
        imshow("Video", frame);
        imshow("Movement", movementMatrix);
    }
}

Rect* BackgroundRemover::getForegroundBoundingBox(unsigned int max_x, unsigned int max_y) {
    // Apply padding and check that box won't go beyond frame

    unsigned int x = max((int)fgBoundingBox->startx - (int)BOUNDING_BOX_PADDING, 0);
    unsigned int y = max((int)fgBoundingBox->starty - (int)BOUNDING_BOX_PADDING, 0);
    unsigned int width = fgBoundingBox->endx - x + BOUNDING_BOX_PADDING;
    unsigned int height = fgBoundingBox->endy - y + BOUNDING_BOX_PADDING;

    if(x + width > max_x) {
        width -= (x + width - max_x);
    }

    if(y + height > max_y) {
        height -= (y + height - max_y);
    }
    //printf("BB x%d y%d w%d h%d\n", x, y, width, height);
    return new Rect(x, y, width, height);
}

void BackgroundRemover::onNewFrame(Mat& frame) {
    if(pixels == nullptr) {
        initLBPPixels(frame.rows, frame.cols, 3);
    }

    lbp->calculateFeatureDescriptors(pixels, frame);
    int startRow = LBP::DESCRIPTOR_RADIUS;
    int endRow = frame.rows - LBP::DESCRIPTOR_RADIUS;
    int rowInc = 1;

    if(INTERLACE) {
        // Handle every second row
        startRow += frameCount % 2;
        rowInc++;
    }

    //vector<thread> threads = {};
    unsigned int threadCount = thread::hardware_concurrency();
    int rowsPerThread = (endRow - startRow) / threadCount;

    thread threads[threadCount];

    fgBoundingBox = new BoundingBox();
    fgBoundingBox->startx = frame.cols - LBP::DESCRIPTOR_RADIUS;
    fgBoundingBox->endx = 0;
    fgBoundingBox->starty = endRow;
    fgBoundingBox->endy = 0;

    for(unsigned int i = 0; i < threadCount; i++) {
        unsigned int tStartRow = startRow + (i * rowsPerThread);
        unsigned int tEndRow = tStartRow + rowsPerThread;

        threads[i] = thread(handleFrameRows, this, tStartRow, tEndRow, pixels);
    }

    for(unsigned int i = 0; i < threadCount; i++) {
        threads[i].join();
    }
}

/*
 * Handle rows from startRow to endRow
 */
void BackgroundRemover::handleFrameRows(BackgroundRemover *bgr,  unsigned int startRow, unsigned int endRow, Mat* pixels) {
    unsigned int endCol = pixels->cols - LBP::DESCRIPTOR_RADIUS;
    BoundingBox *bbox = bgr->fgBoundingBox;

    for(unsigned int i = startRow; i < endRow; i++) {
        for(unsigned int j = LBP::DESCRIPTOR_RADIUS; j < endCol; j++) {
            LBPPixel *pixel = pixels->at<LBPPixel*>(i, j);
            vector<unsigned int> newHist = bgr->lbp->calculateHistogram(pixel);
            if(!pixel->isBackground(newHist)) {
                mtx.lock();
                // Update foreground bounding box
                if(j < bbox->startx) {
                    bbox->startx = j;
                } else if(j > bbox->endx) {
                    bbox->endx = j;
                }
                if(i < bbox->starty) {
                    bbox->starty = i;
                } else if(i > bbox->endy) {
                    bbox->endy = i;
                }
                mtx.unlock();
            }
            pixel->updateAdaptiveHistograms(newHist);
        }
    }
}
