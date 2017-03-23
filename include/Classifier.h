#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "opencv2/core/core.hpp"

class Classifier {
public:
    virtual bool detectFromFrame(cv::Mat) = 0;
protected:
private:
};

#endif
