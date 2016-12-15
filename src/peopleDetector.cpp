#include <iostream>
#include "opencv2/opencv.hpp"
#include <chrono>
#include "peopleDetector.h"
#include <typeinfo>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::ml;

PeopleDetector::PeopleDetector(void) {

}

int PeopleDetector::testPeopleDetection() {
	VideoCapture cap(CV_CAP_ANY);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	if (!cap.isOpened())
		return -1;

	Mat img;
	HOGDescriptor hog;

    //hog.winSize = Size(640, 480);

	//vector<float> descriptor;
	//string descriptorFile = "/home/roope/Dev/sulari/hog/genfiles/descriptorvector.dat";
    //descriptor = loadDescriptorFromFile(descriptorFile);
    //cout << descriptor.size()<< endl;
	//hog.setSVMDetector(descriptor);
	//cout << HOGDescriptor::getDefaultPeopleDetector().size() << endl;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	namedWindow("video capture", CV_WINDOW_AUTOSIZE);

    int fps = 60;

    chrono::milliseconds startingTime = chrono::duration_cast< chrono::milliseconds >( chrono::system_clock::now().time_since_epoch());

	for(;;){
        chrono::milliseconds currentTime = chrono::duration_cast< chrono::milliseconds >( chrono::system_clock::now().time_since_epoch());
        if((currentTime - startingTime).count() > 1000 / fps){
            startingTime = chrono::duration_cast< chrono::milliseconds >( chrono::system_clock::now().time_since_epoch());

            cap >> img;
            if (!img.data)
                continue;

            vector<Rect> found, found_filtered;
            hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);

            cout << "AAA";

            size_t i, j;
            for (i = 0; i < found.size(); i++){
                Rect r = found[i];
                for (j = 0; j < found.size(); j++)
                    if (j != i && (r & found[j]) == r)
                        break;
                if (j == found.size())
                    found_filtered.push_back(r);
            }
            for (i = 0; i < found_filtered.size(); i++){
                Rect r = found_filtered[i];
                r.x += cvRound(r.width*0.1);
                r.width = cvRound(r.width*0.8);
                r.y += cvRound(r.height*0.06);
                r.height = cvRound(r.height*0.9);
                rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
            }
            imshow("video capture", img);
        }
		if (waitKey(20) >= 0)
			break;
	}
	return 0;
}

vector<float> PeopleDetector::loadDescriptorFromFile(string &fileName) {
    ifstream is(fileName);
    istream_iterator<float> start(is), _end;
    vector<float> numbers(start, _end);
    cout << numbers.size() << endl;
    //Ptr<SVM> svm;
    //svm = StatModel::load<SVM>("/home/roope/Dev/sulari/hog/genfiles/cvHOGClassifier.yaml");
    //vector<float> hogDetector;
    //get_svm_detector(svm, hogDetector);

    return numbers;
}

chrono::milliseconds PeopleDetector::getCurrentMillis() {
	chrono::milliseconds ms = chrono::duration_cast<chrono::milliseconds>(
		chrono::system_clock::now().time_since_epoch());

	return ms;
}

void PeopleDetector::get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector )
{
    // get the support vectors
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);

    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );
    hog_detector.clear();

    hog_detector.resize(sv.cols + 1);
    memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
    hog_detector[sv.cols] = (float)-rho;
}

