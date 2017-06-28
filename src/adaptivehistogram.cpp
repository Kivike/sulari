#include "adaptivehistogram.h"

#include <iostream>
#include <vector>

using namespace std;

const float BIN_LEARN_RATE = 0.01f;
const float WEIGHT_LEARN_RATE = 0.01f;
const float INITIAL_WEIGHT = 0.01f;

AdaptiveHistogram::AdaptiveHistogram(int binCount)
{
    this->bins.resize(binCount);
    this->weight = 1.0f / binCount;
}

AdaptiveHistogram::~AdaptiveHistogram()
{
    //dtor
}

void AdaptiveHistogram::setBins(vector<unsigned int> bins) {
    this->bins = bins;
    this->weight = INITIAL_WEIGHT;
}

void AdaptiveHistogram::updateWithNewData(const vector<unsigned int> &newBins) {
    for (size_t i = 0; i < bins.size(); i++) {
        this->bins.at(i) = (BIN_LEARN_RATE * newBins.at(i)) + (1 - BIN_LEARN_RATE) * (this->bins.at(i));
    }
}

void AdaptiveHistogram::updateWeight(bool match) {
    if (match) {
        weight = WEIGHT_LEARN_RATE + (1 - WEIGHT_LEARN_RATE) * weight;
    }
    else {
        weight = (1 - WEIGHT_LEARN_RATE) * weight;
    }
}

float AdaptiveHistogram::getWeight() {
    return weight;
}

vector<unsigned int> AdaptiveHistogram::getBins() {
    return bins;
}
