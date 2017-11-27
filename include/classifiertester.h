#ifndef CASCADECLASSIFIERTESTER_H
#define CASCADECLASSIFIERTESTER_H

#include <vector>
#include <string>
#include <memory>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include "backgroundremover.h"
#include "keyframes.h"
#include "testfile.h"

struct TestResult
{
    TestFile *testFile;
    float detectionRate;
    float falsePositiveRate;
    float averageFps;

    TestResult() { }
};

class CascadeClassifierTester
{
    public:
        CascadeClassifierTester();
        ~CascadeClassifierTester() {
            delete backgroundRemover;
        }

        void setCascade(const std::string&, const int, const int);
        void disableBgRemoval();
        void enableBgRemoval();
        TestResult* testVideoFile(TestFile);
    protected:
    private:
        static const bool PRINT_FRAMERATE;
        int windowWidth, windowHeight;
        bool bgRemovalEnabled;
        BackgroundRemover *backgroundRemover;

        std::vector<cv::Rect> handleFrame(const cv::Mat&, int&, int&, int&);
        cv::CascadeClassifier classifier;
        std::vector<std::string> testMaterial;
        std::vector<cv::Rect> filterFound(std::vector<cv::Rect>&);
        TestResult resultAverage(std::vector<TestResult*>);
        void preprocessFrame(cv::Mat&, cv::Mat&);
        void showOutputFrame(std::vector<cv::Rect>&, cv::Mat&);
};

#endif // CASCADECLASSIFIERTESTER_H
