/**
* Class for testing cascade classifiers to detect humans from video sequences
*/

#include "cascadeclassifiertester.h"
#include "lbp.h"
#include "backgroundremover.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <dirent.h>
#include <vector>
#include <sys/time.h>

using namespace std;
using namespace cv;

CascadeClassifierTester::CascadeClassifierTester() {}

void CascadeClassifierTester::setCascade(const string& file, int width, int height) {
    classifier.load(file);

    this->windowWidth = width;
    this->windowHeight = height;

    printf("Loaded classifier %s (w:%d h:%d)\n", file.c_str(), width, height);
}

/**
 * Test cascade classifier with video
 * Uses background removal before classifier if bgRemovalEnabled==true
 *
 * @param  file Video file to test
 */
TestResult* CascadeClassifierTester::testVideoFile(struct TestFile file) {
    VideoCapture cap = VideoCapture(file.path);

    if (!cap.isOpened()) {
        cout << "Failed to open file " << file.path << endl;
        return 0;
    }

    printf("TEST [BGR:%d] %s\n", bgRemovalEnabled, file.path.c_str());

    int positives = 0;
    int falseNegatives = 0;
    int misses = 0;
    Mat frame, resizedFrame, ppFrame;
    double totalTime = 0.0;

    if(bgRemovalEnabled) {
        backgroundRemover = new BackgroundRemover();
    } else {
        backgroundRemover = nullptr;
    }

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

        preprocessFrame(frame, ppFrame);

        struct timeval startT, endT;
        gettimeofday(&startT, NULL);
        vector<Rect> found = handleFrame(ppFrame, positives, misses, falseNegatives);
        gettimeofday(&endT, NULL);

        if(found.size() > 0) {
            if(found.size() >= file.peopleCount) {
                positives += file.peopleCount;
                falseNegatives++;
            } else {
                positives += found.size();
                misses += file.peopleCount - found.size();
            }
        } else {
            misses += file.peopleCount;
        }
        //printf("Found %d, postiives %d\n", found.size(), positives);
        showOutputFrame(found, ppFrame);
    }

    float detectionRate = positives / (float)(cap.get(CV_CAP_PROP_FRAME_COUNT) * file.peopleCount);
    float falseNegativeRate = falseNegatives / (float)cap.get(CV_CAP_PROP_FRAME_COUNT);

    TestResult *result = new TestResult();
    result->detectionRate = detectionRate;
    result->testFile = file;
    result->falseNegativeRate = falseNegativeRate;

    return result;
}

void CascadeClassifierTester::enableBgRemoval() {
    bgRemovalEnabled = true;
}

void CascadeClassifierTester::disableBgRemoval() {
    bgRemovalEnabled = false;
}

void CascadeClassifierTester::showOutputFrame(vector<Rect> &found, Mat& frame) {
    Mat bgrFrame;
    Rect fgBBox;

    if(backgroundRemover) {
        fgBBox = *backgroundRemover->getForegroundBoundingBox(frame.cols, frame.rows);
        Mat* movementMatrix = backgroundRemover->createMovementMatrix();
        frame = *backgroundRemover->combineFrames(frame, *movementMatrix);
    }

    cvtColor(frame, bgrFrame, CV_GRAY2BGR);

    if(backgroundRemover) {
        rectangle(bgrFrame, fgBBox.tl(), fgBBox.br(), Scalar(255, 0, 0), 1);
    }

    for(size_t i = 0; i < found.size(); i++) {
        Rect r = found.at(i);

        Point_<int> topLeft = r.tl();
        Point_<int> bottomRight = r.br();

        if(backgroundRemover) {
            // found matches were calculated from foreground frame
            // -> apply offset to match the coordinates in whole frame
            topLeft.x += fgBBox.tl().x;
            topLeft.y += fgBBox.tl().y;
            bottomRight.x += fgBBox.tl().x;
            bottomRight.y += fgBBox.tl().y;
        }

        rectangle(bgrFrame, topLeft, bottomRight, Scalar(0, 255, 0), 1);
    }
    imshow("Test", bgrFrame);
}

void CascadeClassifierTester::preprocessFrame(Mat &frame, Mat &output) {
    Mat clampedFrame = clampFrameSize(&frame, Size(96, 96), Size(256, 256));
    // Convert frame to grayscale
    cvtColor(clampedFrame, output, CV_BGR2GRAY);
}

vector<Rect> CascadeClassifierTester::handleFrame(Mat &frame, int &positives, int &misses, int &falseNegatives) {
    vector<Rect> found = vector<Rect>();

    Mat detectionFrame = frame;

    if(backgroundRemover) {
        backgroundRemover->onNewFrame(frame);
        Rect fgBBox = *(backgroundRemover->getForegroundBoundingBox(frame.cols, frame.rows));

        // Only detect in area marked as foreground by background remover
        detectionFrame = frame(fgBBox);
    }
    classifier.detectMultiScale(detectionFrame, found, 1.1, 3, 0|CASCADE_SCALE_IMAGE,
                                     Size(windowWidth, windowHeight));
    return found;
 }

TestResult CascadeClassifierTester::resultAverage(vector<struct TestResult*> results) {
    TestResult avg = {};

    for(size_t i = 0; i < results.size(); i++) {
        avg.detectionRate += results.at(i)->detectionRate;
        avg.falseNegativeRate += results.at(i)->falseNegativeRate;
    }
    avg.detectionRate /= results.size();
    avg.falseNegativeRate /= results.size();
    avg.testFile = TestFile { "", 0 };

    return avg;
};
/**
 * Scales frame size if needed to match the given limits
 * @param  frame   Frame to scale
 * @param  minSize Minimum rows & columns
 * @param  maxSize Maximum rows & columns
 * @return         Returns scaled frame
 */
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
