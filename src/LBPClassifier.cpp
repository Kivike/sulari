#include "LBPClassifier.h"

#include <iostream>
#include <stdio.h>

LBPClassifier::LBPClassifier(string& cascadeFile)
{
    if(!humanClassifier.load(cascadeFile)) {
        cout << "Failed to load cascade file" << endl;
    }
}

bool LBPClassifier::detectFromFrame(Mat frame) {
    std::vector<Rect> persons;
    Mat frame_gray;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    humanClassifier.detectMultiScale(frame_gray, persons, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

    for(size_t i = 0; i < persons.size(); i++) {

    }

    return persons.size() > 0;
}

LBPClassifier::~LBPClassifier()
{
    //dtor
}
