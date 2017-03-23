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
        CascadeClassifierTester();
        virtual ~CascadeClassifierTester();

        void setTestMaterialFiles(std::vector<std::string>);
        void setCascade(std::string&, int, int);
        void enableBgRemoval();
        void disableBgRemoval();
        void startTests();
    protected:

    private:
        int windowWidth, windowHeight;
        bool removeBackground;
        cv::CascadeClassifier classifier;
        std::vector<std::string> testMaterial;
        cv::Mat clampFrameSize(cv::Mat*, cv::Size, cv::Size);
};

#endif // CASCADECLASSIFIERTESTER_H
