#ifndef CASCADECLASSIFIERTESTER_H
#define CASCADECLASSIFIERTESTER_H

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include "backgroundremover.h"
#include <vector>
#include <string>

struct TestFile {
    std::string path;
    uint peopleCount;
};

struct TestSet {
    std::vector<struct TestFile> files;
    std::string name;

    TestSet(std::string name) {
        this->name = name;
    }
};

struct TestResult {
    TestFile testFile;
    float detectionRate;
    float falseNegativeRate;
};

class CascadeClassifierTester
{
    public:
        CascadeClassifierTester();
        virtual ~CascadeClassifierTester();

        void setTestMaterialFiles(TestSet[]);
        void setCascade(const std::string&, int, int);
        void disableBgRemoval();
        void enableBgRemoval();
        void runTest(struct TestSet*);
        TestResult* testVideoFile(struct TestFile);
    protected:

    private:
        int windowWidth, windowHeight;
        bool bgRemovalEnabled;
        BackgroundRemover *backgroundRemover;
        std::vector<cv::Rect> handleFrame(cv::Mat&, int&, int&, int&);
        cv::CascadeClassifier classifier;
        std::vector<std::string> testMaterial;
        cv::Mat clampFrameSize(cv::Mat*, cv::Size, cv::Size);
        TestResult resultAverage(std::vector<struct TestResult*>);
        void preprocessFrame(cv::Mat&, cv::Mat&);
        void showOutputFrame(std::vector<cv::Rect>&, cv::Mat&);
};

#endif // CASCADECLASSIFIERTESTER_H
