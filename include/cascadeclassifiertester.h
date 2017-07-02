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
    int peopleCount;
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
        void setCascade(const string&, int, int);
        void setBackgroundRemover(BackgroundRemover*);
        void runTest(struct TestSet*);
        TestResult* testVideoFile(struct TestFile);
    protected:

    private:
        int windowWidth, windowHeight;
<<<<<<< Updated upstream
        bool removeBackground;
        cv::CascadeClassifier classifier;
        std::vector<std::string> testMaterial;
        cv::Mat clampFrameSize(cv::Mat*, cv::Size, cv::Size);
        TestResult* testVideoFile(struct TestFile);
=======
        BackgroundRemover *backgroundRemover;
        vector<cv::Rect> handleFrame(cv::Mat&, int&, int&, int&);
        cv::CascadeClassifier classifier;
        std::vector<std::string> testMaterial;
        cv::Mat clampFrameSize(cv::Mat*, cv::Size, cv::Size);
        TestResult resultAverage(std::vector<struct TestResult*>);
        void preprocessFrame(cv::Mat&, cv::Mat&);
>>>>>>> Stashed changes
};

#endif // CASCADECLASSIFIERTESTER_H
