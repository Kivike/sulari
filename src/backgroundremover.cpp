/**
* Removes background from video using adaptive LBP
*/

#include "backgroundremover.h"
#include "lbp.h"
#include "lbppixel.h"
#include <thread>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>
#include <iostream>
#include <execinfo.h>
#include <sys/time.h>

using namespace cv;
using namespace std;

const bool BackgroundRemover::COMBINE_FRAMES = true;
const bool BackgroundRemover::INTERLACE = false;
const bool BackgroundRemover::PRINT_FRAMERATE = false;

BackgroundRemover::BackgroundRemover()
{
    if(thread::hardware_concurrency() == 1) {
        // Disable threading if only 1 core is used
        useThreading = false;
    }
    //<vector<vector<unsigned int>> bins(BIN_COUNT);
    pixels = nullptr;   // LBPPixel* Mat will be initialized on first frame
    lbp = new LBP(6, 14);
}

int BackgroundRemover::testWithVideo() {
    return this->testWithVideo("");
}

int BackgroundRemover::testWithVideo(const String &filename) {
    VideoCapture cap;

    if(filename.empty()) {
        // Capture from webcam if no video is given
        cout << "Testing LBP with webcam" << endl;
        cap = VideoCapture(-1);
    } else {
        cout << "Testing LBP with file " + filename << endl;
        cap = VideoCapture(filename);
    }

	if (!cap.isOpened()) {
        cout << "Failed to setup camera/video" << endl;
        return -1;
	}

    Mat frame, greyFrame, movementMatrix, combinedFrame;

    for(;;) {
        // And display it:
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;

        cap >> frame;

        if(!frame.data) {
            cap.set(CV_CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        frameCount++;

        // Convert frame to grayscale
        cvtColor(frame, greyFrame, CV_BGR2GRAY);

        // Timing structs for performance monitoring
        struct timeval startT, endT;
        gettimeofday(&startT, NULL);
        onNewFrame(greyFrame);  // ACTUAL ALGORITHM
        gettimeofday(&endT, NULL);

        if(PRINT_FRAMERATE && frameCount % 3 == 0) {
            float seconds = (endT.tv_usec - startT.tv_usec) / 1000000.0f;
            cout << 1/seconds << "fps" << endl;
        }

        showOutputVideo(greyFrame, BackgroundRemover::COMBINE_FRAMES);
    }
    return 0;
}

// Create pixels and connect histogram neighbours
void BackgroundRemover::initLBPPixels(int rows, int cols, int histCount) {
    pixels = new Mat(rows, cols, DataType<LBPPixel*>::type);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            pixels->at<LBPPixel*>(i, j) = new LBPPixel(histCount, lbp->getBinCount(), i, j);
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
    int halfRegionSize = lbp->getHistogramRegionSize()/2;

    int startRow = max(1, pixel->getRow() - halfRegionSize);
    int endRow = min(pixels->rows - 1, pixel->getRow() + halfRegionSize);
    int startCol = max(1, pixel->getCol() - halfRegionSize);
    int endCol = min(pixels->cols - 1, pixel->getCol() + halfRegionSize);

    vector<LBPPixel*> neighbours;

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

void BackgroundRemover::handleFrameRow(LBP *lbp, int row, Mat* pixels) {
    int cols = pixels->cols;

    for(unsigned int i = LBP::DESCRIPTOR_RADIUS; i < cols - LBP::DESCRIPTOR_RADIUS; i++) {
        LBPPixel *pixel = pixels->at<LBPPixel*>(row, i);

        vector<unsigned int> newHist = lbp->calculateHistogram(pixel);
        pixel->isBackground(newHist);
        pixel->updateAdaptiveHistograms(newHist);
    }
}

void BackgroundRemover::onNewFrame(Mat& frame) {
    if(pixels == nullptr) {
        initLBPPixels(frame.rows, frame.cols, 3);
    }


    LBP::calculateFeatureDescriptors(pixels, frame, LBP::DESCRIPTOR_RADIUS, LBP::NEIGHBOUR_COUNT);
    int startRow = LBP::DESCRIPTOR_RADIUS;
    int rowInc = 1;

    if(INTERLACE) {
        startRow += frameCount % 2;
        rowInc++;
    }

    vector<thread> threads;
    cout << "1" << flush;
    // Handle every row in a seperate thread
    for(unsigned int i = startRow; i < frame.rows - LBP::DESCRIPTOR_RADIUS; i+=rowInc) {
        threads.push_back(thread(handleFrameRow, lbp, i, pixels));
    }
    cout << "2" << flush;
    for(auto& th: threads) th.join();
}

BackgroundRemover::~BackgroundRemover()
{
    //dtor
}
