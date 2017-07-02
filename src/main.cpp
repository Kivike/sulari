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

#include "cascadeclassifiertester.h"
#include "tests.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;

bool parseArgs(int, const char *[], string&, bool&, bool&, string&);
int runWithParams(const string&, const string&, bool);
int runTests();

int main(int argc, const char *argv[]) {
    int exitCode = 0;

    string filename = "";
    string classifier = "cascade/cascade_lbp4.xml";
    bool removeBg = false;
    bool webcam = true;

    if(argc == 1) {
        // Run with default values
        exitCode = runWithParams(filename, classifier, removeBg);
    } else {
        if(parseArgs(argc, argv, filename, webcam, removeBg, classifier)) {
            // -test
            exitCode = runTests();
        } else {
            if(classifier == "" && !removeBg) {
                cout << "Nothing to do; -c [classifier] to set classifier";
                cout << " and/or -bg to use background removal" << endl;
            } else {
                // Run with given parameters
                exitCode = runWithParams(filename, classifier, removeBg);
            }
        }
    }
    return exitCode;
}

//
// Return true if -test arg is set (using tests.cpp)
bool parseArgs(int argc, const char *argv[], string &filename, bool& webcam, bool& removeBg, string& classifier) {
    for(int i = 0; i < argc; i++) {
        if(argv[i][0] == '-') {
            if(!strcmp(argv[i], "-test")) {
                return true;
            } else if(!strcmp(argv[i], "-f")) {
                filename = *argv[i+1];
                continue;
            } else if(!strcmp(argv[i], "-c")) {
                classifier = *argv[i+1];
                continue;
            } else if(!strcmp(argv[i], "-bg")) {
                removeBg = true;
                continue;
            } else {
                printf("Unknown argument %s\n", argv[i]);
            }
        }
    }
    return false;
}

int runTests() {
    cout << "Run tests" << endl;
    try {
        Tests* tests = new Tests();
        tests->run();
    } catch(const std::out_of_range& e) {
        cout << "OUT OF RANGE ERROR" << endl;
    }

    return 0;
}

int runWithParams(const string &filename, const string &classifier, bool removeBg) {
    CascadeClassifierTester *cct = new CascadeClassifierTester();

    cout << "Running with given parameters" << endl;

    cct->setCascade(classifier, 32, 64);

    if(removeBg) {
        cct->enableBgRemoval();
    } else {
        cct->disableBgRemoval();
    }

    if(filename.length() == 0) {
        //cct->
    }
    return 0;
}
