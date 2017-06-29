#include "lbp.h"
#include "lbppixel.h"

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
const int LBP::PIXEL_VALUE_TOLERANCE = 15;
// Currently the region is a X*X square
const int LBP::HISTOGRAM_REGION_SIZE = 14;
// If set to true, only every other half of rows are handled on each frame

const unsigned int LBP::NEIGHBOUR_COUNT = 6;
const unsigned int LBP::BIN_COUNT = LBP::NEIGHBOUR_COUNT + 1;
const unsigned int LBP::DESCRIPTOR_RADIUS = 2;

LBP::LBP() {
    LBP(6, 12);
}

LBP::LBP(int neighbourCount, int histogramRegionSize)
{
    this->neighbourCount = neighbourCount;
    this->binCount = neighbourCount + 1;
    this->histogramRegionSize = histogramRegionSize;

    genUniformPatternClasses(uniformPatterns, LBP::NEIGHBOUR_COUNT);
}

LBP::~LBP()
{
    //dtor
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

// The original LBP descriptor with 8 nearest neighbour pixels
void LBP::calculateFeatureDescriptors(Mat* pixels, Mat &src) {
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

// SOURCE: www.bytefish.de/blog/local_binary_patterns/
// Uses wanted radius and neighbours in circular pattern using interpolation
void LBP::calculateFeatureDescriptors(Mat* pixels, Mat &src, int radius, int neighbours) {
    //cout << pixels->rows << " " << pixels->cols << endl;
    //cout << src.rows << " " << src.cols << endl;
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

int LBP::getBinCount() {
    return this->binCount;
}

int LBP::getHistogramRegionSize() {
    return this->histogramRegionSize;
}

// couts all values in the histogram on a single line
void LBP::printHistogram(const vector<unsigned int> &hist) {
    String print = "";

    for(size_t i = 0; i < hist.size(); i++) {
        print += to_string(hist.at(i)) + " ";
    }

    cout << "Hist: " << print << endl;
}
