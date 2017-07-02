#pragma once
#ifndef ADAPTIVEHISTOGRAM_H
#define ADAPTIVEHISTOGRAM_H

#include <vector>

class AdaptiveHistogram
{
    public:
        AdaptiveHistogram(int);
        void setBins(std::vector<unsigned int>);
        virtual ~AdaptiveHistogram();

        void updateWithNewData(const std::vector<unsigned int>&);
        void updateWeight(bool);
        float getWeight();
        std::vector<unsigned int> getBins();
    protected:

    private:
        std::vector<unsigned int> bins;
        float weight;
};

#endif // ADAPTIVEHISTOGRAM_H
