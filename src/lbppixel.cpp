/*
 * Model for one video pixel
 * Used in background removal
 *
 * Pixel has multiple adaptive histograms which have
 * their weights updated on every frame
 */

#include <vector>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "lbp.h"
#include "lbppixel.h"

/*
 * ALGORITHM SETTINGS
 */
const float LBPPixel::BACKGROUND_WEIGHT = 0.33;

// How close must histograms be to each other for them to be considered
// similar
const float LBPPixel::HISTOGRAM_PROXIMITY_THRESHOLD = 0.8f; // 0.9f

const unsigned char LBPPixel::FOREGROUND_COLOR = 240;
const unsigned char LBPPixel::BACKGROUND_COLOR = 0;

using namespace std;
using namespace cv;

LBPPixel::LBPPixel(int histogramCount, int binCount, int row, int col)
    : row(row), col(col), descriptor(0) {
    for(int i = 0; i < histogramCount; i++) {
        histograms.push_back(new AdaptiveHistogram(binCount));
    }
}

int LBPPixel::getRow() const {
    return row;
}

int LBPPixel::getCol() const {
    return col;
}

void LBPPixel::setDescriptor(unsigned char descriptor) {
    this->descriptor = descriptor;
}

unsigned char LBPPixel::getDescriptor() const {
    return descriptor;
}

void LBPPixel::setHistogramNeighbours(const vector<LBPPixel*> &neighbourPixels) {
    this->histogramNeighbours = neighbourPixels;
}

vector<LBPPixel*> LBPPixel::getHistogramNeighbours() const {
    return this->histogramNeighbours;
}

/**
 * Compare weights of two adaptive histograms
 * @param  h1
 * @param  h2
 * @return bool
 */
bool LBPPixel::compareWeight(AdaptiveHistogram *h1, AdaptiveHistogram *h2) {
    return h1->getWeight() > h2->getWeight();
}

/**
 * Update adaptive histogram weights based on newly calculated LBP histogram
 * @param newHist            Histogram from new frame
 * @param bestMatchIndex     Which adaptive histogram matched new histogram most
 * @param bestMatchProximity How close was the new histogram to the best match
 */
void LBPPixel::updateHistogramWeights(const vector<unsigned int> &newHist,
    int bestMatchIndex, float bestMatchProximity) {
    for(size_t i = 0; i < histograms.size(); i++) {
        if(i == (unsigned int)bestMatchIndex) {
            histograms.at(i)->updateWithNewData(newHist);
            histograms.at(i)->updateWeight(true);
        } else {
            histograms.at(i)->updateWeight(false);
        }
    }
}

/**
 * Check which histogram matches given histogram the best
 * Match must be >HISTOGRAM_PROXIMITY_THRESHOLD to be considered to match at all
 *
 * @param histogram     Histogram to check against
 * @param bestHistIndex Index of best match
 * @param bestProximity How close the best match is (0.0 - 1.0)
 */
void LBPPixel::getBestProximityMatch(const vector<unsigned int> &histogram, int &bestHistIndex, float &bestProximity) {
    bestHistIndex = -1;
    bestProximity = -1.0f;

    for(size_t i = 0; i < histograms.size(); i++) {
        float proximity = LBP::getHistogramProximity(histograms.at(i)->getBins(), histogram);

        if(proximity > HISTOGRAM_PROXIMITY_THRESHOLD && proximity > bestProximity) {
            bestHistIndex = i;
            bestProximity = proximity;
        }
    }
}

/**
 * Update histograms based on the new histogram
 */
void LBPPixel::updateAdaptiveHistograms(const vector<unsigned int> &histogram) {
    int bestHistIndex;
    float bestProximity;

    getBestProximityMatch(histogram, bestHistIndex, bestProximity);

    if(bestHistIndex >= 0) {
        updateHistogramWeights(histogram, bestHistIndex, bestProximity);
        updateBackgroundHistograms();
    } else {
        setLowestWeightHistogram(histogram);
    }

}

/**
 * Sort histograms by weight
 */
void LBPPixel::sortHistograms() {
    sort(histograms.begin(), histograms.end(), compareWeight);
}

/**
 * Set new bins for the histogram with lowest weight
 * @param hist [description]
 */
void LBPPixel::setLowestWeightHistogram(vector<unsigned int> hist) {
    histograms.back()->setBins(hist);
}

/**
 * Update vector of background histograms
 */
void LBPPixel::updateBackgroundHistograms() {
    backgroundHistograms.clear();

    sortHistograms();

    for(size_t i = 0; i < histograms.size(); i++) {
        float weight = histograms.at(i)->getWeight();

        if(weight > BACKGROUND_WEIGHT) {
            backgroundHistograms.push_back(histograms.at(i));
        } else {
            break;
        }
    }
}

/**
 * Check if histogram matches any background histogram
 * @param  newHist [description]
 * @return         [description]
 */
bool LBPPixel::isBackground(const vector<unsigned int> &newHist) {
    for(size_t i = 0; i < backgroundHistograms.size(); i++) {
        float proximity = LBP::getHistogramProximity(backgroundHistograms.at(i)->getBins(), newHist);

        if(proximity > HISTOGRAM_PROXIMITY_THRESHOLD) {
            setColor(BACKGROUND_COLOR);
            return true;
        }
    }

    setColor(FOREGROUND_COLOR);
    return false;
}

unsigned char LBPPixel::getColor(bool weightGrayValue) {
    if(weightGrayValue) {
        return (1 - histograms.at(0)->getWeight()) * 255;
    } else {
        return color;
    }
}

void LBPPixel::setColor(unsigned char color) {
    this->color = color;
}

void LBPPixel::printHistogramWeights() {
    for(size_t i = 0; i < histograms.size(); i++) {
        cout << i << ": " << histograms.at(i)->getWeight() << " ";
    }
    cout << histograms.at(0)->getBins().size();
    cout << endl;
}

void LBPPixel::printPosition() {
    cout << this << " Pos: " << this->row << " " << this->col << endl;
}
