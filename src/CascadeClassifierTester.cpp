#include "CascadeClassifierTester.h"

#include "LBP.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
using namespace std;
using namespace cv;

CascadeClassifierTester::CascadeClassifierTester()
{
    //ctor
}

void CascadeClassifierTester::setTestMaterialFiles(vector<string> files) {
    this->testMaterial = files;
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

void CascadeClassifierTester::startTests() {
    VideoCapture cap;
    Mat frame, resizedFrame;
    LBP *backgroundRemover;

    cout << "Testing with " << testMaterial.size() << " files" << endl;

    for(int i = 0; i < testMaterial.size(); i++) {
        cap = VideoCapture(testMaterial.at(i));

        if (!cap.isOpened()) {
            cout << "Failed to open file " << testMaterial.at(i) << endl;
            continue;
        }

        cout << "Testing - " << testMaterial.at(i) << endl;

        if(removeBackground) {
            //backgroundRemover = new LBP();
        }

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
                cout << "FOUND " << found.size() << endl;

                for(int i = 0; i < found.size(); i++) {
                    Rect r = found.at(i);

                    rectangle(frame, r.tl(), r.br(), Scalar(0, 255, 0), 1);
                }
            }

            //rectangle(original, face_i, Scalar(0, 255,0), 1);

            imshow("Test", frame);
        }
    }
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

CascadeClassifierTester::~CascadeClassifierTester()
{
    //dtor
}
