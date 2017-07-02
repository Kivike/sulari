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

bool parseArgs(const char *[], string&, bool&, bool&, string&);
int runWithParams(const string&, const string&, bool);
int runTests();

int main(int argc, const char *argv[]) {
    runTests();

    return 0;

    string filename = "";
    string classifier = "cascade/cascade_lbp4.xml";
    bool removeBg = false;
    bool webcam = true;

    if(argc > 1) {
        if(parseArgs(argv, filename, webcam, removeBg, classifier)) {
            exitCode = runTests();
        } else {
            if(classifier == "" && !removeBg) {
                cout << "Nothing to do; -c [classifier] to set classifier";
                cout << " and/or -bg to use background removal" << endl;
            } else {
                exitCode = runWithParams(filename, classifier, removeBg);
            }
        }
    }

    if(argc == 1) {
        // NO ARGUMENTS
        LBP *lbp = new LBP();
        int exitCode = lbp->testWithVideo("videos/lena_walk2.avi");
        exit(exitCode);
    } else if(argc == 2) {
        cout << argv[1] << endl;
        if(strcmp(argv[1],"--tests") == 0) {
            exitCode = runTests();
        }
    } else {
        // Check for valid command line arguments, print usage
        // if no arguments were given.
        if (argc != 4) {
            cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/csv.ext> </path/to/device id>" << endl;
            cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection." << endl;
            cout << "\t </path/to/csv.ext> -- Path to the CSV file with the face database." << endl;
            cout << "\t <device id> -- The webcam device id to grab frames from." << endl;
            exit(1);
        }
    }

    // Get the path to your CSV:
    //string fn_haar = string(argv[1]);
    //string fn_csv = string(argv[2]);
    //int deviceId = atoi(argv[3]);
    //// These vectors hold the images and corresponding labels:
    //vector<Mat> images;
    //vector<int> labels;
    //// Read in the data (fails if no valid input filename is given, but you'll get an error message):
    //try {
    //    read_csv(fn_csv, images, labels);
    //} catch (cv::Exception& e) {
    //    cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
    //    // nothing more we can do
    //    exit(1);
    //}

//
// Return true if -test arg is set (using tests.cpp)
bool parseArgs(const char *argv[], string &filename, bool& webcam, bool& removeBg, string& classifier) {
    for(size_t i = 0; i < sizeof(*argv)/sizeof(*argv[0]); i++) {
        if(argv[i][0] == '-') {
            if(!strcmp(argv[i], "-test")) {
                return true;
            }
            /*switch(argv[i]) {
                case '-test':
                    return true;
                case '-f':
                    filename = *argv[i+1];
                    break;
                case '-classifier':
                case '-c':
                    classifier = *argv[i+1];
                    break;
                case '-bg':
                    removeBg = true;
                    break;
            }*/
        }
    }
    return false;
}

int runTests() {
    cout << "Run tests" << endl;
    Tests* tests = new Tests();
    tests->run();
    return 0;
}

int runWithParams(const string &filename, const string &classifier, bool removeBg) {
    CascadeClassifierTester *cct = new CascadeClassifierTester();

    if(classifier != nullptr)
        cct->setCascade(classifier, 32, 64);

    if(removeBg)
        cct->setBackgroundRemover(new BackgroundRemover());

    Tests().setTester(cct)->run();
    return 0;
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
