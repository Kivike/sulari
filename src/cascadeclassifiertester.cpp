/**
* Class for testing cascade classifiers to detect humans from video sequences
*/

#include "cascadeclassifiertester.h"
#include "lbp.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <dirent.h>
#include <vector>

using namespace std;
using namespace cv;

CascadeClassifierTester::CascadeClassifierTester()
{
    //ctor
}

void CascadeClassifierTester::setCascade(const string& file, int width, int height) {
    classifier.load(file);

    this->windowWidth = width;
    this->windowHeight = height;

    printf("Loaded classifier %s (w:%d h:%d)", file.c_str(), width, height);
}

void CascadeClassifierTester::setBackgroundRemover(BackgroundRemover *bgr) {
    this->backgroundRemover = bgr;

    if(bgr == nullptr) {
        cout << "Disabled background removal" << endl;
    } else {
        cout << "Enabled background removal" << endl;
    }
}

void CascadeClassifierTester::runTest(struct TestSet* testSet) {
    int materialSize = testSet->files.size();

    vector<struct TestResult*> results;
    cout << materialSize << " files" << endl;

    for(int i = 0; i < materialSize; i++) {
        struct TestResult *r = testVideoFile(testSet->files.at(i));
        results.push_back(r);
    }
    TestResult setResult = resultAverage(results);

    printf("%s result: detection rate %f, fp rate %f\n",
           testSet->name.c_str(), setResult.detectionRate, setResult.falseNegativeRate);
}

TestResult* CascadeClassifierTester::testVideoFile(struct TestFile file) {
    VideoCapture cap = VideoCapture(file.path);

    if (!cap.isOpened()) {
        cout << "Failed to open file " << file.path << endl;
        return 0;
    }

    cout << "Testing - " << file.path << endl;
    if(this->backgroundRemover) {
        cout << "Using background removal" << endl;
    }
    this->backgroundRemover = new BackgroundRemover();

    int positives = 0;
    int falseNegatives = 0;
    int misses = 0;
    Mat frame, resizedFrame, ppFrame;
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

        cout << "F";

        preprocessFrame(frame, ppFrame);

        struct timeval startT, endT;
        gettimeofday(&startT, NULL);
        vector<Rect> found = handleFrame(ppFrame, positives, misses, falseNegatives);
        cout << '.';
        gettimeofday(&endT, NULL);

        vector<Rect> found;

        if(removeBackground && backgroundRemover) {
            backgroundRemover->handleNewFrame(frame);
        }

        // Equalize histogram to make detection easier
        //equalizeHist(frame, frame);

        classifier.detectMultiScale(frame, found, 1.1, 3, 0|CASCADE_SCALE_IMAGE,
                                         Size(windowWidth, windowHeight));

        if(found.size() > 0) {
            if(found.size() >= file.peopleCount) {
                positives += file.peopleCount;
                falseNegatives++;
            } else {
                positives += found.size();
                misses += file.peopleCount - found.size();
            }

            for(int i = 0; i < found.size(); i++) {
                Rect r = found.at(i);

                rectangle(ppFrame, r.tl(), r.br(), Scalar(0, 255, 0), 1);
            }
        } else {
            misses += file.peopleCount;
        }
        //printf("Found %d, postiives %d\n", found.size(), positives);
        imshow("Test", ppFrame);
    }

    float detectionRate = positives / (float)(cap.get(CV_CAP_PROP_FRAME_COUNT) * file.peopleCount);
    float falseNegativeRate = falseNegatives / (float)cap.get(CV_CAP_PROP_FRAME_COUNT);

    TestResult *result = new TestResult();
    result->detectionRate = detectionRate;
    result->testFile = file;
    result->falseNegativeRate = falseNegativeRate;

    return result;
}

void CascadeClassifierTester::preprocessFrame(Mat &frame, Mat &output) {
    Mat clampedFrame = clampFrameSize(&frame, Size(96, 96), Size(256, 256));
    // Convert frame to grayscale
    cvtColor(clampedFrame, output, CV_BGR2GRAY);
}

vector<Rect> CascadeClassifierTester::handleFrame(Mat &frame, int &positives, int &misses, int &falseNegatives) {
    vector<Rect> found;

    if(this->backgroundRemover) {
        cout << 'r';
        backgroundRemover->onNewFrame(frame);
    }

    cout << 'd' << flush;
    classifier.detectMultiScale(frame, found, 1.1, 3, 0|CASCADE_SCALE_IMAGE,
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

    dir = opendir(dirName);

    if (dir == NULL) {
        cout << "Directory not found" << endl;
        return vector<string>();
    }

    while ((ent = readdir(dir)) != NULL) {
        printf("%s\n", ent->d_name);
    }

    closedir(dir);
}

CascadeClassifierTester::~CascadeClassifierTester()
{
    //dtor
}
