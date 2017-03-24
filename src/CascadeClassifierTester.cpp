/**
* Class for testing cascade classifiers to detect humans from video sequences
*/

#include "CascadeClassifierTester.h"

#include "LBP.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <dirent.h>

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
    int materialSize = sizeof(testSet)/sizeof(testSet[0]);
    vector<struct TestResult> results;

    for(int i = 0; i < materialSize; i++) {
        struct TestResult r = testVideoFile(testSet.at(i));

        results.push_back(testVideoFile(materialSize.at(i)));
    }
}

TestResult* CascadeClassifierTester::testVideoFile(struct TestFile file) {
    VideoCapture cap = VideoCapture(file.pathpath);

    if (!cap.isOpened()) {
        cout << "Failed to open file " << filePath << endl;
        continue;
    }

    cout << "Testing - " << filePath << endl;

    if(removeBackground) {
        /// Crashes after first video
        //backgroundRemover = new LBP();
    }

    int positives, falseNegatives, misses;
    Mat frame, resizedFrame;
    LBP *backgroundRemover;

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

        vector<Rect> found;

        if(removeBackground && backgroundRemover != nullptr) {
            backgroundRemover->handleNewFrame(frame);
        }

        // Equalize histogram to make detection easier
        //equalizeHist(frame, frame);

        classifier.detectMultiScale(frame, found, 1.1, 3, 0|CASCADE_SCALE_IMAGE,
                                         Size(windowWidth, windowHeight));

        if(found.size() > 0) {
            if(found.size >= file.peopleCount) {
                positives += file.peopleCount;
                falseNegatives++;
            } else {
                positives += found.size;
                misses += file.peopleCount - found.size;
            }

            for(int i = 0; i < found.size(); i++) {
                Rect r = found.at(i);

                rectangle(frame, r.tl(), r.br(), Scalar(0, 255, 0), 1);
            }
        } else {
            misses += file.peopleCount;
        }

        imshow("Test", frame);
    }

    float detectionRate = positives / (float)(cap.get(CV_CAP_PROP_FRAME_COUNT) * file.peopleCount);
    float falseNegativeRate = falseNegatives / (float)cap.get(CV_CAP_PROP_FRAME_COUNT);

    return struct TestResult { testFiles, detectionRate, falseNegativeRate };
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

vector<string> getFilesInFolder(string& dirName) {
    DIR *dir;
    struct dirent *ent;

    dir = opendir(dirName);

    if(dir != NULL) {
        while(ent = readdir(dir) != NULL) {
            printf("%s\n", ent->d_name);
        }
        closedir(dir);
    } else {
        cout << "Dir not found" << endl;
    }
}

CascadeClassifierTester::~CascadeClassifierTester()
{
    //dtor
}
