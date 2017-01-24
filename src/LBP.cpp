#include "LBP.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>
#include <iostream>
#include <execinfo.h>
#include <string>
using namespace std;
using namespace cv;

// Lookup array
// [pattern] = class/bin
unsigned int* uniformPatterns;

const int UNIFORM_BIN_COUNT = 8;
const int BIN_COUNT = UNIFORM_BIN_COUNT + 1;
const int NON_UNIFORM_BIN_INDEX = 0;

// How close must histograms be to eachother
float HISTOGRAM_PROXIMITY_THRESHOLD;

// How close to eachother can pixel grayscale values be
// while still considering them the same
int PIXEL_VALUE_TOLERANCE;

// Currently the region is a X*X square
const int HISTOGRAM_REGION_SIZE = 8;

// Show output in seperate frames or in a combined one
const bool COMBINE_FRAMES = false;

LBP::LBP(float proximityThreshold, int pixelTolerance)
{
    HISTOGRAM_PROXIMITY_THRESHOLD = proximityThreshold;
    PIXEL_VALUE_TOLERANCE = pixelTolerance;

    unsigned int** bins = new unsigned int*[BIN_COUNT];

    // bins[0] will be filled with all the rest (non-uniform patterns)
    bins[1] = new unsigned int[8] { 1, 2, 4, 8, 16, 32, 64, 128 };
    bins[2] = new unsigned int[8] { 20, 6, 3, 129, 136, 40, 96, 80 };
    bins[3] = new unsigned int[8] { 22, 7, 131, 137, 168, 104, 112, 84 };
    bins[4] = new unsigned int[8] { 23, 135, 139, 169, 232, 120, 116, 86 };
    bins[5] = new unsigned int[8] { 151, 143, 171, 233, 248, 124, 118, 87 };
    bins[6] = new unsigned int[8] { 159, 175, 235, 249, 252, 126, 119, 215 };
    bins[7] = new unsigned int[8] { 191, 239, 251, 253, 254, 127, 247, 223 };
    bins[8] = new unsigned int[2] { 256, 0};

    uniformPatterns = new unsigned int[256];

    // Fill uniformPattern lookup array
    for(int i = 0; i < 256; i++) {
        unsigned int pattern = getUniformPatternClass(bins, i);
        uniformPatterns[i] = pattern;
    }

    delete[] bins;
}

LBP::~LBP()
{
    //dtor
}

int LBP::testWithVideo() {
    VideoCapture cap("/home/roope/Repot/sulari/videos/eli_walk.avi");

	if (!cap.isOpened()) {
        cout << "Failed to setup camera/video" << endl;
        return -1;
	}

    Mat frame, greyFrame;

    Mat *lastHists = nullptr;
    Mat *newHists = nullptr;
    Mat *descriptors = nullptr;

    cout << "Testing LBP" << endl;

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

        // Convert frame to greyscale
        cvtColor(frame, greyFrame, CV_BGR2GRAY);

        // Calculate feature descriptor for each pixel
        descriptors = calculateFeatureDescriptors(greyFrame);

        // Calculate histogram for each pixel
        newHists = calculateHistograms(*descriptors, HISTOGRAM_REGION_SIZE);

        Mat *proximityMatrix = nullptr;

        if(lastHists != nullptr) {
            // Compare new histogram to previous frame's histogram
            movementMatrix = getHistogramProximity(*newHists, *lastHists);

            //imshow("Proximity", *proximityMatrix);

            for(int i = 0; i < lastHists->rows; i++) {
                for(int j = 0;j < lastHists->cols; j++) {
                    delete[] lastHists->at<unsigned int*>(i, j);
                }
            }

            delete lastHists;
        }

        delete descriptors;

        lastHists = newHists;


        if(movementMatrix != nullptr) {
            if(COMBINE_FRAMES) {
                // Show original video frame and movement detection frame in same image
                Mat* finalFrame = combineFrames(greyFrame, *proximityMatrix);

                imshow("Video", *finalFrame);
                delete finalFrame;
            } else {
                imshow("Video", frame);
                imshow("Movement", *movementMatrix);
            }
        }
        delete movementMatrix;
    }
}

// Combines image and movement matrix
Mat* LBP::combineFrames(Mat& img, Mat& mMatrix) {
    if(img.rows != mMatrix.rows || img.cols != mMatrix.cols) {
        return &img;
    }

    Mat *output = new Mat(img.rows, img.cols, CV_8UC1);

    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
            output->at<unsigned char>(i, j) = min(img.at<unsigned char>(i, j), mMatrix.at<unsigned char>(i, j));
        }
    }
    return output;
}

Mat* LBP::calculateFeatureDescriptors(Mat &src) {
    Mat *descriptors = new Mat(src.rows, src.cols, CV_8UC1);

    for(int i = 1; i < src.rows - 1; i++) {
        for(int j = 1; j < src.cols - 1; j++) {
            unsigned int centerValue = (unsigned int)src.at<unsigned char>(i, j);
            unsigned int thresholdValue = centerValue + PIXEL_VALUE_TOLERANCE;

            unsigned char binaryCode = 0;

            binaryCode |= (src.at<unsigned char>(i - 1, j - 1) > thresholdValue) << 7;
            binaryCode |= (src.at<unsigned char>(i - 1, j    ) > thresholdValue) << 6;
            binaryCode |= (src.at<unsigned char>(i - 1, j + 1) > thresholdValue) << 5;
            binaryCode |= (src.at<unsigned char>(i    , j - 1) > thresholdValue) << 4;
            binaryCode |= (src.at<unsigned char>(i    , j + 1) > thresholdValue) << 3;
            binaryCode |= (src.at<unsigned char>(i + 1, j - 1) > thresholdValue) << 2;
            binaryCode |= (src.at<unsigned char>(i + 1, j    ) > thresholdValue) << 1;
            binaryCode |= (src.at<unsigned char>(i + 1, j + 1) > thresholdValue) << 0;

            descriptors->at<unsigned char>(i, j) = binaryCode;
        }
    }
    return descriptors;
}

Mat* LBP::calculateHistograms(Mat &descriptors, int blockSize) {
    Mat *histograms = new Mat(descriptors.rows, descriptors.cols, DataType<unsigned int*>::type);

    for(int i = 0; i < descriptors.rows; i++) {
        for(int j = 0; j < descriptors.cols; j++) {
            unsigned int* hist = calculateHistogram(descriptors, i, j, blockSize);
            histograms->at<unsigned int*>(i, j) = hist;
        }
    }
    return histograms;
}

unsigned int* LBP::calculateHistogram(Mat &descriptors, int row, int col, int blockSize) {
    int startRow = max(0, row - blockSize/2);
    int endRow = min(descriptors.rows, row + blockSize/2);
    int startCol = max(0, col - blockSize/2);
    int endCol = min(descriptors.cols, col + blockSize/2);

    // Histogram that shows the amount of patterns for each bin
    unsigned int *histogram = new unsigned int [BIN_COUNT]();

    for(int i = startRow; i < endRow; i++) {
        for(int j = startCol; j < endCol; j++) {
            unsigned int pattern = descriptors.at<unsigned char>(i, j);
            unsigned int uniformClass = uniformPatterns[pattern];

            histogram[uniformClass]++;
        }
    }
    return histogram;
}

// Get pattern class based on the binary pattern
unsigned int LBP::getUniformPatternClass(unsigned int** bins, unsigned int pattern) {
    for(unsigned int i = 0; i < sizeof(bins); i++) {
        for(unsigned int j = 0; j < sizeof(bins[i]); j++) {
            if(bins[i][j] == pattern) {
                return i;
            }
        }
    }
    // Returns -1 if pattern doesn't belong to any uniform pattern
    return NON_UNIFORM_BIN_INDEX;
}

// Get two-color matrix with different color depending on how close to eachother histograms are
Mat* LBP::getHistogramProximity(Mat &hists1, Mat &hists2) {
    if(&hists1 == nullptr || &hists2 == nullptr || hists1.rows != hists2.rows || hists1.cols != hists2.cols) return nullptr;

    Mat *intersectionHist = new Mat(hists1.rows, hists1.cols, CV_8U);

    int matches = 0;

    for(int i = 0; i < hists1.rows; i++) {
        for(int j = 0; j < hists1.cols; j++) {
            unsigned int intersection = 0;

            unsigned int* hist1 = hists1.at<unsigned int*>(i, j);
            unsigned int* hist2 = hists2.at<unsigned int*>(i, j);

            for(int k = 0; k < BIN_COUNT; k++) {
                unsigned int minAmount = min(hist1[k], hist2[k]);
                intersection += minAmount;
            }

            if(intersection > HISTOGRAM_PROXIMITY_THRESHOLD * HISTOGRAM_REGION_SIZE * HISTOGRAM_REGION_SIZE) {
                // Background
                intersectionHist->at<unsigned char>(i, j) = 0;
            } else {
                // Moving object
                intersectionHist->at<unsigned char>(i, j) = 240;
            }
        }
    }

    return intersectionHist;
}

// couts all values in the histogram on a single line
void LBP::printHistogram(unsigned int hist[]) {
    string print = "";

    for(int i = 0; i < sizeof(hist); i++) {
        print += to_string(hist[i]) + " ";
    }

    cout << "Hist: " << print << endl;
}


