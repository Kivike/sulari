#include "LBP.h"
#include "LBPPixel.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>
#include <iostream>
#include <execinfo.h>
#include <sys/time.h>
#include <thread>

using namespace std;
using namespace cv;

// How close must histograms be to each other for them to be considered
// similar
const float LBP::HISTOGRAM_PROXIMITY_THRESHOLD = 0.9f;

//!ALGORITHM SETTINGS
// How close to each other can pixel gray-scale values be
// while still considering them the same
const int PIXEL_VALUE_TOLERANCE = 15;
// Currently the region is a X*X square
const int HISTOGRAM_REGION_SIZE = 14;
// If set to true, only every other half of rows are handled on each frame
const bool INTERLACE = false;
const unsigned int NEIGHBOUR_COUNT = 6;
const unsigned int BIN_COUNT = NEIGHBOUR_COUNT + 1;
const unsigned int DESCRIPTOR_RADIUS = 2;
///


// Show output in seperate frames or in a combined one
const bool COMBINE_FRAMES = true;
// Output fps to console
const bool PRINT_FRAMERATE = false;
// Use multiple threads?
bool useThreading = true;
int threadCount;

static LBP* lbp;

LBP::LBP()
{
    if(thread::hardware_concurrency() == 1) {
        // Disable threading if only 1 core is used
        useThreading = false;
    }
    cout << "Using " << threadCount << " threads" << endl;
    //<vector<vector<unsigned int>> bins(BIN_COUNT);
    pixels = nullptr;   // LBPPixel* Mat will be initialized on first frame
    lbp = this;

    genUniformPatternClasses(uniformPatterns, NEIGHBOUR_COUNT);
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

int LBP::testWithVideo() {
    return this->testWithVideo("");
}

void LBP::genUniformPatternClasses(vector<unsigned int> &patterns, unsigned int neighbours) {
    int totalPatterns = pow(2, neighbours);
    patterns = vector<unsigned int>(totalPatterns);
    cout << "Generate pattern classes" << endl;
    cout << "Using " << neighbours << " neighbours -> " << totalPatterns << " patterns" << endl;

    for(unsigned int startPos = 0; startPos < neighbours; startPos++) {
        for(unsigned int bit_count = 1; bit_count < neighbours; bit_count++) {
            unsigned int pattern = 0;
            unsigned int curPos = startPos;
            unsigned int bitsLeft = bit_count;

            while(bitsLeft > 0) {
                bitsLeft--;
                if(curPos == neighbours - 1) {
                    curPos = 0;
                } else {
                    curPos++;
                }
                pattern |= 1 << curPos;
            }
            patterns.at(pattern) = bit_count;
        }
    }

    patterns.at(totalPatterns - 1) = neighbours;
    patterns.at(0) = neighbours;
}

static void handleFrameRow(int row, Mat* pixels) {
    int cols = pixels->cols;

    for(unsigned int i = DESCRIPTOR_RADIUS; i < cols - DESCRIPTOR_RADIUS; i++) {
        LBPPixel *pixel = pixels->at<LBPPixel*>(row, i);

        vector<unsigned int> newHist = lbp->calculateHistogram(pixel);
        pixel->isBackground(newHist);
        pixel->updateAdaptiveHistograms(newHist);
    }
}

// Handle new video/capture frame
void LBP::handleNewFrame(Mat& frame) {
    if(pixels == nullptr) {
        initLBPPixels(frame.rows, frame.cols, 3);
    }

    calculateFeatureDescriptors(frame, DESCRIPTOR_RADIUS, NEIGHBOUR_COUNT);

    int startRow = DESCRIPTOR_RADIUS;
    int rowInc = 1;

    if(INTERLACE) {
        startRow += frameCount % 2;
        rowInc++;
    }

    vector<thread> threads;

    // Handle every row in a seperate thread
    for(unsigned int i = startRow; i < frame.rows - DESCRIPTOR_RADIUS; i+=rowInc) {
        threads.push_back(thread(handleFrameRow, i, pixels));
    }

    for(auto& th: threads) th.join();
}

// Create pixels and connect histogram neighbours
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

            int col = pixel->getColor(false);
            result->at<unsigned char>(i, j) = col;
        }
    }

    return result;
}

// The original LBP descriptor with 8 nearest neighbour pixels
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

// SOURCE: www.bytefish.de/blog/local_binary_patterns/
// Uses wanted radius and neighbours in circular pattern using interpolation
void LBP::calculateFeatureDescriptors(Mat &src, int radius, int neighbours) {
    Mat dst = Mat::zeros(src.rows - 2*radius, src.cols - 2*radius, CV_8UC1);

    for(int n = 0; n < neighbours; n++) {
        float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbours));
        float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbours));

        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));

        float ty = y - fy;
        float tx = x -fx;

        float w1 = (1 - tx) * (1 - ty);
        float w2 = tx * (1 - ty);
        float w3 = (1 - tx) * ty;
        float w4 = tx * ty;

        for(int i = radius; i < src.rows - radius; i++) {
            for(int j = radius; j < src.cols - radius; j++) {
                float t = w1*src.at<unsigned char>(i + fy, j + fx)
                    + w2*src.at<unsigned char>(i + fy, j + cx)
                    + w3*src.at<unsigned char>(i + cy, j + fx)
                    + w4*src.at<unsigned char>(i + cy, j + cx);

                dst.at<unsigned char>(i - radius, j - radius)
                    += ((t > src.at<unsigned char>(i, j))
                        && ((abs(t - src.at<unsigned char>(i, j)) > PIXEL_VALUE_TOLERANCE))) << n;
            }
        }
    }

    for(int i = radius; i < dst.rows - radius; i++) {
        for(int j = radius; j < dst.cols - radius; j++) {
            unsigned char desc = dst.at<unsigned char>(i - radius, j - radius);
            pixels->at<LBPPixel*>(i, j)->setDescriptor(desc);
        }
    }
}

vector<unsigned int> LBP::calculateHistogram(LBPPixel *pixel) {
    vector<unsigned int> histogram(BIN_COUNT);

    vector<LBPPixel*> neighbours = pixel->getHistogramNeighbours();
    unsigned int n_size = neighbours.size();

    for(size_t i = 0; i < n_size; i++) {
        unsigned int desc = neighbours.at(i)->getDescriptor();
        unsigned int uniformClass = uniformPatterns.at(desc);
        histogram.at(uniformClass)++;
    }

    return histogram;
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

    return (float)totalCommon / (float)(totalMax);
}

// couts all values in the histogram on a single line
void LBP::printHistogram(const vector<unsigned int> &hist) {
    String print = "";

    for(size_t i = 0; i < hist.size(); i++) {
        print += to_string(hist.at(i)) + " ";
    }

    cout << "Hist: " << print << endl;
}


