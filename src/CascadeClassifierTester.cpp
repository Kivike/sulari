/**
* Class for testing cascade classifiers to detect humans from video sequences
*/

#include "CascadeClassifierTester.h"

#include "BackgroundRemover.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <dirent.h>
#include <sys/time.h>

using namespace std;
using namespace cv;
using namespace Testing;

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
    int materialSize = testSet.files.size();
    printf("Starting test set %s (%d files)\n", testSet.name.c_str(), materialSize);
    vector<TestResult> results;

    for(int i = 0; i < materialSize; i++) {
        struct TestResult r = testVideoFile(testSet.files.at(i));

        results.push_back(r);
    }
    TestResult setResult = resultAverage(results);

    printf("%s result: detection rate %f, fp rate %f, time avg %f\n",
           testSet.name.c_str(), setResult.detectionRate, setResult.falseNegativeRate, setResult.avgCalcDuration);
}

TestResult CascadeClassifierTester::testVideoFile(struct TestFile file) {
    VideoCapture cap = VideoCapture(file.path);

    if (!cap.isOpened()) {
        cout << "Failed to open file " << file.path << endl;
        return TestResult { file, -1, -1 };
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
    BackgroundRemover *backgroundRemover;

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
        vector<Rect> found = handleFrame(frame, backgroundRemover, positives, misses, falseNegatives);
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
    float avgCalcDuration = totalTime / frameCount;

    //printf("Positives %d, total %d\n", positives, peopleCount);
    //printf("DR %f FNR %f\n", detectionRate, falseNegativeRate);
    return TestResult { file, detectionRate, falseNegativeRate, avgCalcDuration };
}

vector<Rect> CascadeClassifierTester::handleFrame(Mat &frame, BackgroundRemover *bgr, int &positives, int &misses, int &falseNegatives) {
    vector<Rect> found;

    if(bgr != nullptr) {
        bgr->onNewFrame(frame);
    }

    // Equalize histogram to make detection easier
    //equalizeHist(frame, frame);

    classifier.detectMultiScale(frame, found, 1.1, 3, 0|CASCADE_SCALE_IMAGE,
                                     Size(windowWidth, windowHeight));
    return found;
}

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

TestResult CascadeClassifierTester::resultAverage(vector<struct TestResult> results) {
    TestResult avg;

    for(size_t i = 0; i < results.size(); i++) {
        avg.detectionRate += results.at(i).detectionRate;
        avg.falseNegativeRate += results.at(i).falseNegativeRate;
    }

    avg.detectionRate /= results.size();
    avg.falseNegativeRate /= results.size();
    avg.testFile = TestFile { "", 0 };

    return avg;
};

vector<string> getFilesInFolder(const char* dirName) {
    DIR *dir;
    struct dirent *ent;

    dir = opendir(dirName);

    if(dir != NULL) {
        ent = readdir(dir);

        while(ent != NULL) {
            printf("%s\n", ent->d_name);
            ent = readdir(dir);
        }
        closedir(dir);
    } else {
        cout << "Dir not found" << endl;
    }

    return vector<string> {};
}

CascadeClassifierTester::~CascadeClassifierTester()
{
    //dtor
}
