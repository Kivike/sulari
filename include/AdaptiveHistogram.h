#ifndef ADAPTIVEHISTOGRAM_H
#define ADAPTIVEHISTOGRAM_H

#include <vector>

using namespace std;

class AdaptiveHistogram
{
    public:
        AdaptiveHistogram(int);
        void setBins(vector<unsigned int>);
        virtual ~AdaptiveHistogram();

        void updateWithNewData(const vector<unsigned int>&);
        void updateWeight(bool);
        float getWeight();
        vector<unsigned int> getBins();
    protected:

    private:
        vector<unsigned int> bins;
        float weight;
};

#endif // ADAPTIVEHISTOGRAM_H
