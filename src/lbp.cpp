/*
 * Local Binary patterns
 * Calculates a value by comparing grayscale values of pixels around a pixel
 *
 * Basic idea:
 * 5 6 5    1 1 1
 * 1 4 8 -> 0   1 -> 11101011 (=235)
 * 2 9 8    0 1 1
 */

#include <iostream>
#include <sys/time.h>
#include <cmath>
#include <thread>
#include <mutex>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "lbp.h"
#include "lbppixel.h"

using namespace std;
using namespace cv;

mutex LBP::mtx;

/*
 * ALGORITHM SETTINGS
 */
// How close to each other can pixel gray-scale values be
// while still considering them the same
const int LBP::PIXEL_VALUE_TOLERANCE = 8;
// Currently the region is a X*X square
const int LBP::HISTOGRAM_REGION_SIZE = 12;
// If set to true, only every other half of rows are handled on each frame
const unsigned int LBP::NEIGHBOUR_COUNT = 6;
const unsigned int LBP::DESCRIPTOR_RADIUS = 3;

const unsigned int LBP::BIN_COUNT = LBP::NEIGHBOUR_COUNT + 1;

LBP::LBP(): pixels(nullptr) {}

/**
 * local binary pattern
 * @param neighbours How many neighbours are used for calculating the pattern
 */
vector<unsigned int> genUniformPatternClasses(unsigned int neighbours) {
    int totalPatterns = pow(2, neighbours);
    vector<unsigned int>uniformPatterns = vector<unsigned int>(totalPatterns);
    //cout << "Using " << neighbours << " neighbours -> " << totalPatterns << " patterns" << endl;

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
            uniformPatterns.at(pattern) = bit_count;
        }
    }

    uniformPatterns.at(totalPatterns - 1) = neighbours;
    uniformPatterns.at(0) = neighbours;
    return uniformPatterns;
}

vector<unsigned int> LBP::uniformPatterns = genUniformPatternClasses(LBP::NEIGHBOUR_COUNT);

/**
 * Create pixels and connect histogram neighbours
 * @param rows      [description]
 * @param cols      [description]
 * @param histCount [description]
 */
void LBP::initLBPPixels(int rows, int cols, int histCount) {
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
void LBP::setHistogramNeighbours(LBPPixel* pixel) {
    int startRow = max(1, pixel->getRow() - LBP::HISTOGRAM_REGION_SIZE/2);
    int endRow = min(pixels->rows - 1, pixel->getRow() + LBP::HISTOGRAM_REGION_SIZE/2);
    int startCol = max(1, pixel->getCol() - LBP::HISTOGRAM_REGION_SIZE/2);
    int endCol = min(pixels->cols - 1, pixel->getCol() + LBP::HISTOGRAM_REGION_SIZE/2);

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

/**
 * Get Mat (representing the frame) of LBP descriptors
 * @return Matrix of unsigned integers
 */
Mat* LBP::getDescriptorMat() {
    Mat* descMat = new Mat(pixels->rows, pixels->cols, CV_8UC1);

    for(int i = 0; i < descMat->rows; i++) {
        for(int j = 0; j < descMat->cols; j++) {
            descMat->at<unsigned char>(i, j) = pixels->at<LBPPixel*>(i, j)->getDescriptor();
        }
    }

    return descMat;
}

/**
 * Create 2-color frame of foreground and background pixels
 * @return Frame with black and white pixels
 */
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

/*
 * The original LBP descriptor with 8 nearest neighbour pixels
 */
void LBP::calculateFeatureDescriptors(Mat &src) {
    unsigned int threshold;
    unsigned char binaryCode;

    for(int i = 1; i < src.rows - 1; i++) {
        for(int j = 1; j < src.cols - 1; j++) {
            threshold = src.at<unsigned char>(i, j);
            threshold += LBP::PIXEL_VALUE_TOLERANCE;

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

/*
 * SOURCE: www.bytefish.de/blog/local_binary_patterns/
 * Uses wanted radius and neighbours in circular pattern using interpolation
 * This allows for example neighbour count of 6 which is easier to calculate than the original with 8
 */
void LBP::calculateFeatureDescriptors(Mat *pixels, Mat &src) {
    Mat dst = Mat::zeros(src.rows - 2*DESCRIPTOR_RADIUS, src.cols - 2*DESCRIPTOR_RADIUS, CV_8UC1);

    for(unsigned int n = 0; n < NEIGHBOUR_COUNT; n++) {
        float x = static_cast<float>(DESCRIPTOR_RADIUS) * cos(2.0*M_PI*n/static_cast<float>(NEIGHBOUR_COUNT));
        float y = static_cast<float>(DESCRIPTOR_RADIUS) * -sin(2.0*M_PI*n/static_cast<float>(NEIGHBOUR_COUNT));

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

        for(unsigned int i = DESCRIPTOR_RADIUS; i < src.rows - DESCRIPTOR_RADIUS; i++) {
            for(unsigned int j = DESCRIPTOR_RADIUS; j < src.cols - DESCRIPTOR_RADIUS; j++) {
                float t = w1*src.at<unsigned char>(i + fy, j + fx)
                    + w2*src.at<unsigned char>(i + fy, j + cx)
                    + w3*src.at<unsigned char>(i + cy, j + fx)
                    + w4*src.at<unsigned char>(i + cy, j + cx);

                dst.at<unsigned char>(i - DESCRIPTOR_RADIUS, j - DESCRIPTOR_RADIUS)
                    += ((t > src.at<unsigned char>(i, j))
                        && ((abs(t - src.at<unsigned char>(i, j)) > PIXEL_VALUE_TOLERANCE))) << n;
            }
        }
    }

    for(unsigned i = DESCRIPTOR_RADIUS; i < dst.rows - DESCRIPTOR_RADIUS; i++) {
        for(unsigned int j = DESCRIPTOR_RADIUS; j < dst.cols - DESCRIPTOR_RADIUS; j++) {
            unsigned char desc = dst.at<unsigned char>(i - DESCRIPTOR_RADIUS, j - DESCRIPTOR_RADIUS);
            pixels->at<LBPPixel*>(i, j)->setDescriptor(desc);
        }
    }
}

/**
 * Calculate pixel's histogram based on its neighbours
 * @param pixel
 */
vector<unsigned int> LBP::calculateHistogram(LBPPixel *pixel) {
    vector<unsigned int> histogram(BIN_COUNT);
    vector<LBPPixel*> neighbours = pixel->getHistogramNeighbours();
    unsigned int n_size = neighbours.size();
    unsigned int uniformClass, desc;

    for(size_t i = 0; i < n_size; i++) {
        desc = neighbours.at(i)->getDescriptor();
        uniformClass = LBP::uniformPatterns.at(desc);

        histogram.at(uniformClass)++;
    }

    return histogram;
}

/**
 * Calculates how close to eachother histograms are and return float between 0 and 1
 * @param  hist1
 * @param  hist2
 * @return       Returns float between 0.0 (not matching at all) and 1.0 (exactly same)
 */
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

/**
 * Print all values in the histogram
 */
void LBP::printHistogram(const vector<unsigned int> &hist) {
    String print = "";

    for(size_t i = 0; i < hist.size(); i++) {
        print += to_string(hist.at(i)) + " ";
    }

    cout << "Hist: " << print << endl;
}
