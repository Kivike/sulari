#include "LBP.h"
#include "LBPPixel.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>
#include <iostream>
#include <execinfo.h>
#include <sys/time.h>

using namespace std;
using namespace cv;

const int UNIFORM_BIN_COUNT = 8;
const int BIN_COUNT = UNIFORM_BIN_COUNT + 1;
const int NON_UNIFORM_BIN_INDEX = 0;

// How close must histograms be to each other for them to be considered
// similar
const float LBP::HISTOGRAM_PROXIMITY_THRESHOLD = 0.9f;

// How close to each other can pixel gray-scale values be
// while still considering them the same
const int PIXEL_VALUE_TOLERANCE = 15;

// Currently the region is a X*X square
const int HISTOGRAM_REGION_SIZE = 14;

// Show output in seperate frames or in a combined one
const bool COMBINE_FRAMES = true;
// If set to true, only every other half of rows are handled on each frame
const bool INTERLACE = true;
// Output fps to console
const bool PRINT_FRAMERATE = true;

const int NEIGHBOURS = 6;

LBP::LBP()
{
    vector<vector<unsigned int>> bins(BIN_COUNT);
    pixels = nullptr;   // LBPPixel* Mat will be initialized on first frame

    // bins[0] will be filled with all the rest (non-uniform patterns)
    bins.at(1) = { 1, 2, 4, 8, 16, 32, 64, 128 };
    bins.at(2) = { 20, 6, 3, 129, 136, 40, 96, 80 };
    bins.at(3) = { 22, 7, 131, 137, 168, 104, 112, 84 };
    bins.at(4) = { 23, 135, 139, 169, 232, 120, 116, 86 };
    bins.at(5) = { 151, 143, 171, 233, 248, 124, 118, 87 };
    bins.at(6) = { 159, 175, 235, 249, 252, 126, 119, 215 };
    bins.at(7) = { 191, 239, 251, 253, 254, 127, 247, 223 };
    bins.at(8) = { 255, 0 };

    uniformPatterns = new unsigned int[256];

    // Fill uniformPattern lookup array
    for(int i = 0; i < 256; i++) {
        unsigned int pattern = getUniformPatternClass(bins, i);
        uniformPatterns[i] = pattern;
    }
}

LBP::~LBP()
{
    //dtor
}

int LBP::testWithVideo(const String &filename) {
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
        handleNewFrame(greyFrame);  // ACTUAL ALGORITHM
        gettimeofday(&endT, NULL);

        if(PRINT_FRAMERATE && frameCount % 3 == 0) {
            float seconds = (endT.tv_usec - startT.tv_usec) / 1000000.0f;
            cout << 1/seconds << "fps" << endl;
        }

        showOutputVideo(greyFrame, COMBINE_FRAMES);
    }
    return 0;
}

// Handle new video/capture frame
void LBP::handleNewFrame(Mat& frame) {
    if(pixels == nullptr) {
        initLBPPixels(frame.rows, frame.cols, 3);
    }

    calculateFeatureDescriptors(frame);

    int startRow = 1;
    int rowInc = 1;

    if(INTERLACE) {
        startRow += frameCount % 2;
        rowInc++;
    }

    for(int i = startRow; i < frame.rows - 1; i+=rowInc) {
        for(int j = 1; j < frame.cols - 1; j++) {
            LBPPixel *pixel = pixels->at<LBPPixel*>(i, j);

            vector<unsigned int> newHist = calculateHistogram(pixel);
            pixel->isBackground(newHist);
            pixel->updateAdaptiveHistograms(newHist);
        }
    }
}

void LBP::initLBPPixels(int rows, int cols, int histCount) {
    pixels = new Mat(rows, cols, DataType<LBPPixel*>::type);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            pixels->at<LBPPixel*>(i, j) = new LBPPixel(histCount, BIN_COUNT, i, j);
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

void LBP::setHistogramNeighbours(LBPPixel* pixel) {
    int startRow = max(1, pixel->getRow() - HISTOGRAM_REGION_SIZE/2);
    int endRow = min(pixels->rows - 1, pixel->getRow() + HISTOGRAM_REGION_SIZE/2);
    int startCol = max(1, pixel->getCol() - HISTOGRAM_REGION_SIZE/2);
    int endCol = min(pixels->cols - 1, pixel->getCol() + HISTOGRAM_REGION_SIZE/2);

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
Mat* LBP::combineFrames(Mat& img, Mat& mMatrix) {
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

void LBP::showOutputVideo(Mat &frame, bool combine) {
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

// Get Mat/video frame of LBP descriptors
Mat* LBP::getDescriptorMat() {
    Mat* descMat = new Mat(pixels->rows, pixels->cols, CV_8UC1);

    for(int i = 0; i < descMat->rows; i++) {
        for(int j = 0; j < descMat->cols; j++) {
            descMat->at<unsigned char>(i, j) = pixels->at<LBPPixel*>(i, j)->getDescriptor();
        }
    }

    return descMat;
}

// Create 2-color frame of foreground and background pixels
Mat* LBP::createMovementMatrix() {
    Mat* result = new Mat(pixels->rows, pixels->cols, CV_8UC1);

    for(int i = 0; i < result->rows; i++) {
        for(int j = 0; j < result->cols; j++) {
            LBPPixel *pixel = pixels->at<LBPPixel*>(i, j);
            int col = pixel->getColor();

            result->at<unsigned char>(i, j) = col;
            result->at<unsigned char>(i, j+1) = col;
        }
    }

    return result;
}

void LBP::calculateFeatureDescriptors(Mat &src) {
    unsigned int threshold;
    unsigned char binaryCode;

    for(int i = 1; i < src.rows - 1; i++) {
        for(int j = 1; j < src.cols - 1; j++) {
            threshold = src.at<unsigned char>(i, j);
            threshold += PIXEL_VALUE_TOLERANCE;

            binaryCode = 0;
            binaryCode |= (src.at<unsigned char>(i - 1, j - 1) >= threshold) << 7;
            binaryCode |= (src.at<unsigned char>(i - 1, j    ) >= threshold) << 6;
            binaryCode |= (src.at<unsigned char>(i - 1, j + 1) >= threshold) << 5;
            binaryCode |= (src.at<unsigned char>(i    , j - 1) >= threshold) << 4;
            binaryCode |= (src.at<unsigned char>(i    , j + 1) >= threshold) << 3;
            binaryCode |= (src.at<unsigned char>(i + 1, j - 1) >= threshold) << 2;
            binaryCode |= (src.at<unsigned char>(i + 1, j    ) >= threshold) << 1;
            binaryCode |= (src.at<unsigned char>(i + 1, j + 1) >= threshold) << 0;

            pixels->at<LBPPixel*>(i, j)->setDescriptor(binaryCode);
        }
    }
}

vector<unsigned int> LBP::calculateHistogram(LBPPixel *pixel) {
    vector<unsigned int> histogram(BIN_COUNT);

    vector<LBPPixel*> neighbours = pixel->getHistogramNeighbours();
    unsigned int n_size = neighbours.size();

    for(size_t i = 0; i < n_size; i++) {
        unsigned int uniformClass = uniformPatterns[neighbours.at(i)->getDescriptor()];
        histogram.at(uniformClass)++;
    }

    return histogram;
}

// Get pattern class based on the binary pattern
unsigned int LBP::getUniformPatternClass(vector<vector<unsigned int>> bins, unsigned int pattern) {
    for(size_t i = 0; i < bins.size(); i++) {
        for(size_t j = 0; j < bins.at(i).size(); j++) {
            if(bins.at(i).at(j) == pattern) {
                return i;
            }
        }
    }
    // Returns -1 if pattern doesn't belong to any uniform pattern
    return NON_UNIFORM_BIN_INDEX;
}

// Calculates how close to eachother histograms are and return float between 0 and 1
float LBP::getHistogramProximity(const vector<unsigned int> &hist1, const vector<unsigned int> &hist2) {
    unsigned int totalCommon = 0;
    unsigned int totalMax = 0;
    for(size_t i = 0; i < hist1.size(); i++) {
        unsigned int minAmount = min(hist1.at(i), hist2.at(i));
        totalMax += hist1.at(i);

        totalCommon += minAmount;
    }

    return (float)totalCommon / (float)(totalMax); // divide by 64
}

// couts all values in the histogram on a single line
void LBP::printHistogram(const vector<unsigned int> &hist) {
    String print = "";

    for(size_t i = 0; i < hist.size(); i++) {
        print += to_string(hist.at(i)) + " ";
    }

    cout << "Hist: " << print << endl;
}


