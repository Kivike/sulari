#ifndef CASCADECLASSIFIERTESTER_H
#define CASCADECLASSIFIERTESTER_H

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include <vector>
#include <string>

class CascadeClassifierTester
{
    public:
        struct TestFile {
            std::string path;
            int peopleCount;
        };

        struct TestSet {
            std::vector<struct TestFile> files;
            std::string name;
        };

        struct TestResult {
            TestFile testFile;
            float detectionRate;
            float falseNegativeRate;
        };

        CascadeClassifierTester();
        virtual ~CascadeClassifierTester();

        void setTestMaterialFiles(TestSet[]);
        void setCascade(std::string&, int, int);
        void enableBgRemoval();
        void disableBgRemoval();
        void runTest(struct TestSet);
    protected:

    private:
        int windowWidth, windowHeight;
        bool removeBackground;
        cv::CascadeClassifier classifier;
        std::vector<std::string> testMaterial;
        cv::Mat clampFrameSize(cv::Mat*, cv::Size, cv::Size);
        TestResult testVideoFile(struct TestFile);
};

#endif // CASCADECLASSIFIERTESTER_H
