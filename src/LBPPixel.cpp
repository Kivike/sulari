#include "LBPPixel.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include "LBP.h"

#include <iostream>

const float BACKGROUND_WEIGHT = 0.55;

const unsigned char FOREGROUND_COLOR = 240;
const unsigned char BACKGROUND_COLOR = 0;

using namespace std;
using namespace cv;

LBPPixel::LBPPixel(int histogramCount, int binCount, int row, int col) {
    for(int i = 0; i < histogramCount; i++) {
        histograms.push_back(new AdaptiveHistogram(binCount));
    }

    this->row = row;
    this->col = col;
}

LBPPixel::~LBPPixel()
{
    //dtor
}

int LBPPixel::getRow() {
    return row;
}

int LBPPixel::getCol() {
    return col;
}

void LBPPixel::setDescriptor(unsigned char descriptor) {
    this->descriptor = descriptor;
}

unsigned char LBPPixel::getDescriptor() {
    return descriptor;
}

void LBPPixel::setHistogramNeighbours(const vector<LBPPixel*> &neighbourPixels) {
    this->histogramNeighbours = neighbourPixels;

    for(size_t i = 0; i < neighbourPixels.size(); i++) {
        if(neighbourPixels.at(i) == nullptr) {
            cout << "Setting null as neighbour" << endl;
        }
    }
}

vector<LBPPixel*> LBPPixel::getHistogramNeighbours() {
    return this->histogramNeighbours;
}

bool LBPPixel::compareWeight(AdaptiveHistogram *h1, AdaptiveHistogram *h2) {
    return h1->getWeight() > h2->getWeight();
}

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

// Check which histogram (if any) matches given histogram the best
void LBPPixel::getBestProximityMatch(const vector<unsigned int> &histogram, int &bestHistIndex, float &bestProximity) {
    bestHistIndex = -1;
    bestProximity = -1.0f;

    for(size_t i = 0; i < histograms.size(); i++) {
        float proximity = LBP::getHistogramProximity(histograms.at(i)->getBins(), histogram);

        if(proximity > LBP::HISTOGRAM_PROXIMITY_THRESHOLD && proximity > bestProximity) {
            bestHistIndex = i;
            bestProximity = proximity;
        }
    }
}

// Update histograms based on the new histogram
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

// Sort histograms by weight
void LBPPixel::sortHistograms() {
    sort(histograms.begin(), histograms.end(), compareWeight);
}

// Set new bins for the histogram with lowest weight
void LBPPixel::setLowestWeightHistogram(vector<unsigned int> hist) {
    histograms.back()->setBins(hist);
}

// Update vector of background histograms
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

bool LBPPixel::isBackground(const vector<unsigned int> &newHist) {
    for(size_t i = 0; i < backgroundHistograms.size(); i++) {
        float proximity = LBP::getHistogramProximity(backgroundHistograms.at(i)->getBins(), newHist);

        if(proximity > LBP::HISTOGRAM_PROXIMITY_THRESHOLD) {
            setColor(BACKGROUND_COLOR);
            return true;
        }
    }

    setColor(FOREGROUND_COLOR);
    return false;
}

unsigned char LBPPixel::getColor() {
    return this->color;
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
