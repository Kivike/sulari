/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include "backgroundremover.h"
#include "peopledetector.h"
#include "lbp.h"
#include "cascadeclassifiertester.h"
#include "tests.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;


static void runTests() {
    cout << "Run tests" << endl;
    CascadeClassifierTester *cct = new CascadeClassifierTester();

    std::string cascade = "cascade/cascade_lbp4.xml";
    cct->setCascade(cascade, 32, 64);
    cct->enableBgRemoval();

    Tests* tests = new Tests();
    tests->setTester(cct)->run();
}

int main(int argc, const char *argv[]) {
    int exitCode = 0;

    if(argc == 1) {
        // NO ARGUMENTS
        BackgroundRemover *bg = new BackgroundRemover();

        exitCode = bg->testWithVideo("videos/kth/lena_walk2.avi");
    } else if(argc == 2) {
        cout << argv[1] << endl;
        if(strcmp(argv[1],"-test") == 0) {
            runTests();
        }
    } else {
        cout << "Invalid args" << endl;
    }

    exit(exitCode);
}

/*static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}*/
