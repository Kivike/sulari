/**
* Class for testing cascade classifiers to detect humans from video sequences
*/
#include "backgroundremover.h"
#include "cascadeclassifiertester.h"
#include "lbp.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <dirent.h>
#include <sys/time.h>
#include <vector>

using namespace std;
using namespace cv;

CascadeClassifierTester::CascadeClassifierTester()
{
    //ctor
}

void CascadeClassifierTester::setCascade(string& file, int width, int height) {
    classifier.load(file);
    this->windowWidth = width;
    this->windowHeight = height;
}

void CascadeClassifierTester::enableBgRemoval() {
    removeBackground = true;
}

void CascadeClassifierTester::disableBgRemoval() {
    removeBackground = false;
}

void CascadeClassifierTester::runTest(struct TestSet testSet) {
    int materialSize = sizeof(testSet)/sizeof(testSet.files.at(0));
    vector<struct TestResult*> results;

    for(int i = 0; i < materialSize; i++) {
        struct TestResult *r = testVideoFile(testSet.files.at(i));
        results.push_back(r);
    }
    TestResult setResult = resultAverage(results);

    printf("%s result: detection rate %f, fp rate %f\n",
           testSet.name.c_str(), setResult.detectionRate, setResult.falseNegativeRate);
}

TestResult* CascadeClassifierTester::testVideoFile(struct TestFile file) {
    VideoCapture cap = VideoCapture(file.path);

    if (!cap.isOpened()) {
        cout << "Failed to open file " << file.path << endl;
        return 0;
    }

    cout << "Testing - " << file.path << endl;

    if(removeBackground) {
        /// Crashes after first video
        //backgroundRemover = new LBP();
    }

    int positives = 0;
    int falseNegatives = 0;
    int misses = 0;
    Mat frame, resizedFrame;
    double totalTime = 0.0;

    // Loop through every frame
    for(;;) {
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
        cap >> frame;

        if(!frame.data) {
            break;
        }

        frame = clampFrameSize(&frame, Size(96, 96), Size(256, 256));
        struct timeval startT, endT;
        gettimeofday(&startT, NULL);
        vector<Rect> found = handleFrame(frame, positives, misses, falseNegatives);
        gettimeofday(&endT, NULL);

        double dur = (double)endT.tv_usec - (double)startT.tv_usec;
        dur /= 1000000.0;

        if(dur > 0) {
            totalTime += dur;
        }

        if(found.size() > 0) {
            if(found.size() >= file.peopleCount) {
                positives += file.peopleCount;

                if(found.size() > file.peopleCount) {
                    falseNegatives++;
                }
            } else {
                positives += found.size();
                misses += file.peopleCount - found.size();
            }

            for(size_t i = 0; i < found.size(); i++) {
                Rect r = found.at(i);

                rectangle(frame, r.tl(), r.br(), Scalar(0, 255, 0), 1);
            }
        } else {
            misses += file.peopleCount;
        }
        //printf("Found %d, postiives %d\n", found.size(), positives);
        imshow("Test", frame);
    }

    int frameCount = cap.get(CV_CAP_PROP_FRAME_COUNT);
    int peopleCount = frameCount * file.peopleCount;
    float detectionRate = positives / (float)peopleCount;
    float falseNegativeRate = falseNegatives / (float)frameCount;
    //float avgCalcDuration = totalTime / frameCount;

    TestResult *result = new TestResult;
    result->detectionRate = detectionRate;
    result->testFile = file;
    result->falseNegativeRate = falseNegativeRate;

    return result;
}

vector<Rect> CascadeClassifierTester::handleFrame(Mat &frame, int &positives, int &misses, int &falseNegatives) {
    vector<Rect> found;

    /*if(bgr != nullptr) {
        bgr->onNewFrame(frame);
    }*/

    classifier.detectMultiScale(frame, found, 1.1, 3, 0|CASCADE_SCALE_IMAGE,
                                     Size(windowWidth, windowHeight));
    return found;
 }

TestResult CascadeClassifierTester::resultAverage(vector<struct TestResult*> results) {
    TestResult avg;

    for(size_t i = 0; i < results.size(); i++) {
        avg.detectionRate += results.at(i)->detectionRate;
        avg.falseNegativeRate += results.at(i)->falseNegativeRate;
    }

    avg.detectionRate /= results.size();
    avg.falseNegativeRate /= results.size();
    avg.testFile = TestFile { "", 0 };

    return avg;
};

// Resize frame to given limits
Mat CascadeClassifierTester::clampFrameSize(Mat* frame, Size minSize, Size maxSize) {
    float multiplier = 1;

    if(frame->cols < minSize.width || frame->rows < minSize.height){
        multiplier = minSize.width / (float)frame->cols;

        if(multiplier * frame->rows < minSize.height) {
            multiplier = minSize.height / (float)frame->rows;
        }
    } else if(frame->cols > maxSize.width || frame->rows > maxSize.height) {
        multiplier = maxSize.width / (float)frame->cols;

        if(multiplier * frame->rows > maxSize.height) {
            multiplier = maxSize.height / (float)frame->rows;
        }
    }
    Mat newFrame;
    resize(*frame, newFrame, Size(frame->cols*multiplier, frame->rows*multiplier));
    return newFrame;
}

vector<string> getFilesInFolder(const char* dirName) {
    DIR *dir;
    struct dirent *ent;
    vector<string> files;
    dir = opendir(dirName);

    if (dir == NULL) {
        cout << "Directory not found" << endl;
        return vector<string>();
    }

    while ((ent = readdir(dir)) != NULL) {
        printf("%s\n", ent->d_name);
    }

    closedir(dir);
    return files;
}

CascadeClassifierTester::~CascadeClassifierTester()
{
    //dtor
}
