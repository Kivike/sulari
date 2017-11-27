/**
* Class for testing cascade classifiers to detect humans from video sequences
*/

#include "classifiertester.h"
#include "lbp.h"
#include "backgroundremover.h"
#include "imgutils.h"
#include "config.h"
#include "testfile.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <dirent.h>
#include <vector>
#include <sys/time.h>
#include <execinfo.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

const bool CascadeClassifierTester::PRINT_FRAMERATE = true;

CascadeClassifierTester::CascadeClassifierTester() {}

void CascadeClassifierTester::setCascade(const string& file, const int width, const int height) {
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
TestResult* CascadeClassifierTester::testVideoFile(struct TestFile *file) {
    VideoCapture cap = VideoCapture(file->getFilePath());

    if (!cap.isOpened()) {
        std::cout << "Failed to open file " << file->getFilePath() << endl;
        return 0;
    }

    long frameCount = 0;
    int positives = 0;
    int falsePositives = 0;
    int misses = 0;
    Mat frame, resizedFrame, ppFrame;
    double totalTime = 0.0;
    backgroundRemover = nullptr;
    unsigned int videoPeopleCount = file->getPeopleCount();
    bool humanInFrame = false;
    int framesWithHuman = 0;

    if(bgRemovalEnabled) {
        backgroundRemover = new BackgroundRemover();
    }

    try {
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

            if (file->isKeyframe(frameCount)) {
                humanInFrame = !humanInFrame;
                /*if (humanInFrame) {
                    std::cout << "ENTER " << frameCount << endl;
                } else {
                    std::cout << "EXIT " << frameCount << endl;
                }*/
            }

            if (humanInFrame) {
                framesWithHuman++;
            }

            frameCount++;
            preprocessFrame(frame, ppFrame);

            unsigned long t_start, t_frame;

            t_start = std::chrono::system_clock::now().time_since_epoch()
                / std::chrono::nanoseconds(1);
            vector<Rect> found = handleFrame(ppFrame, positives, misses, falsePositives);
            t_frame = (std::chrono::system_clock::now().time_since_epoch()
                / std::chrono::nanoseconds(1)) - t_start;

            totalTime += t_frame;

            vector<Rect> foundFiltered = filterFound(found);
            size_t found_count = foundFiltered.size();

            if(found_count > 0) {
                if (humanInFrame) {
                    if(found_count  >= videoPeopleCount) {
                        positives += videoPeopleCount;
                        falsePositives += found_count - videoPeopleCount;
                    } else {
                        positives += found_count;
                        misses += videoPeopleCount - found_count;
                    }
                } else {
                    falsePositives += found_count;
                }
            } else {
                if (humanInFrame) {
                    misses += videoPeopleCount;
                }
            }
            showOutputFrame(foundFiltered, ppFrame);
        }
    } catch (exception &e) {
        std::cout << "ERROR" << endl;
        std::cout << e.what() << endl;
        exit(5);
    }

    float detectionRate = positives / (float)(framesWithHuman);
    float falsePositiveRate = falsePositives / (float)cap.get(CV_CAP_PROP_FRAME_COUNT);

    TestResult *result = new TestResult();
    result->averageFps = frameCount / (totalTime / 1000000000);
    result->detectionRate = detectionRate;
    result->testFile = file;
    result->falsePositiveRate = falsePositiveRate;

    return result;
}

vector<Rect> CascadeClassifierTester::filterFound(vector<Rect> &found) {
    vector<Rect> found_filtered;

    for(size_t i = 0; i < found.size(); i++) {
        Rect r = found.at(i);

        size_t j;

        for(j = 0; j < found.size(); j++) {
            if(i != j && (r & found.at(j)) == r)
                break;
        }

        if(j == found.size())
            found_filtered.push_back(r);
    }
    return found_filtered;
}

void CascadeClassifierTester::enableBgRemoval() {
    bgRemovalEnabled = true;
}

void CascadeClassifierTester::disableBgRemoval() {
    bgRemovalEnabled = false;
}

void CascadeClassifierTester::showOutputFrame(vector<Rect> &found, Mat& frame) {
    Mat bgrFrame;
    Rect *fgBBox = nullptr;

    if(backgroundRemover) {
        fgBBox = backgroundRemover->getForegroundBoundingBox(
            frame.cols, frame.rows, this->windowWidth, this->windowHeight);

        if (fgBBox != nullptr && fgBBox->tl().x >= 0 && fgBBox->tl().y >= 0 &&
            fgBBox->br().x <= frame.cols && fgBBox->br().y <= frame.rows) {
                frame = backgroundRemover->cropBackground(frame, fgBBox);
            }

        //Mat movementMatrix = backgroundRemover->createMovementMatrix();
        //frame = *ImgUtils::frameMin(frame, movementMatrix);
    }

    cvtColor(frame, bgrFrame, CV_GRAY2BGR);

    if(backgroundRemover && fgBBox != nullptr) {
        rectangle(bgrFrame, fgBBox->tl(), fgBBox->br(), Scalar(255, 0, 0), 1);
    }

    for(size_t i = 0; i < found.size(); i++) {
        Rect r = found.at(i);

        Point_<int> topLeft = r.tl();
        Point_<int> bottomRight = r.br();

        if(backgroundRemover && fgBBox != nullptr) {
            // found matches were calculated from foreground frame
            // -> apply offset to match the coordinates in whole frame
            topLeft.x += fgBBox->tl().x;
            topLeft.y += fgBBox->tl().y;
            bottomRight.x += fgBBox->tl().x;
            bottomRight.y += fgBBox->tl().y;
        }

        rectangle(bgrFrame, topLeft, bottomRight, Scalar(0, 255, 0), 1);
    }
    imshow("Test", bgrFrame);
}

void CascadeClassifierTester::preprocessFrame(Mat &frame, Mat &output) {
    Mat clampedFrame = ImgUtils::clampFrameSize(&frame, Size(96, 96), Size(256, 256));
    // Convert frame to grayscale
    cvtColor(clampedFrame, output, CV_BGR2GRAY);
}

vector<Rect> CascadeClassifierTester::handleFrame(const Mat &frame, int &positives, int &misses, int &falsePositives) {
    vector<Rect> found = vector<Rect>();

    Mat detectionFrame = frame;

    if(backgroundRemover) {
        backgroundRemover->onNewFrame(frame);
        Rect *fgBBox = backgroundRemover->getForegroundBoundingBox(
            frame.cols, frame.rows, this->windowWidth, this->windowHeight);

        // Only detect in area marked as foreground by background remover
        try {
            if(fgBBox != nullptr)
                detectionFrame = frame(*fgBBox);
        } catch(exception e) {
            printf("Failed to crop the frame (%d %d %d %d)\n",
                fgBBox->x, fgBBox->y, fgBBox->width, fgBBox->height);
        }
    }

    try {
        classifier.detectMultiScale(detectionFrame, found, 1.1, 3, 0|CASCADE_SCALE_IMAGE,
                                         Size(windowWidth, windowHeight));
    } catch (std::exception e) {
        printf("Failure when detecting from frame %d %d\n", detectionFrame.cols, detectionFrame.rows);
    }

    return found;
 }

TestResult CascadeClassifierTester::resultAverage(vector<struct TestResult*> results) {
    TestResult avg = {};

    for(size_t i = 0; i < results.size(); i++) {
        avg.detectionRate += results.at(i)->detectionRate;
        avg.falsePositiveRate += results.at(i)->falsePositiveRate;
    }
    avg.detectionRate /= results.size();
    avg.falsePositiveRate /= results.size();
    avg.testFile = new TestFile();

    return avg;
};
