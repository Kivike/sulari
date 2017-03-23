#ifndef LBPCLASSIFIER_H
#define LBPCLASSIFIER_H

#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "Classifier.h"

using namespace std;
using namespace cv;

class LBPClassifier : public Classifier
{
    public:
        LBPClassifier(string&);
        ~LBPClassifier();
        bool detectFromFrame(Mat frame);
    protected:

    private:
        CascadeClassifier humanClassifier;
};

#endif // LBPCLASSIFIER_H
