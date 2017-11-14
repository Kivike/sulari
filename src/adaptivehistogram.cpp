#include <iostream>
#include <vector>

#include "adaptivehistogram.h"

using namespace std;

const float BIN_LEARN_RATE = 0.01f;
const float WEIGHT_LEARN_RATE = 0.01f;
const float INITIAL_WEIGHT = 0.5f;

AdaptiveHistogram::AdaptiveHistogram(int binCount) {
    bins.resize(binCount);
    weight = 1.0f / binCount;
}

void AdaptiveHistogram::setBins(vector<unsigned int> bins) {
    this->bins = bins;
    weight = INITIAL_WEIGHT;
}

void AdaptiveHistogram::updateWithNewData(const vector<unsigned int> &newBins) {
    for (size_t i = 0; i < bins.size(); i++) {
        bins.at(i) = (BIN_LEARN_RATE * newBins.at(i)) + (1 - BIN_LEARN_RATE) * (bins.at(i));
    }
}

void AdaptiveHistogram::updateWeight(bool match) {
    if (match) {
        // Increase weight
        weight = WEIGHT_LEARN_RATE + (1 - WEIGHT_LEARN_RATE) * weight;
    } else {
        // Decrease weight
        weight = (1 - WEIGHT_LEARN_RATE) * weight;
    }
}

float AdaptiveHistogram::getWeight() const {
    return weight;
}

vector<unsigned int> AdaptiveHistogram::getBins() const {
    return bins;
}

AdaptiveHistogram::~AdaptiveHistogram() {}
