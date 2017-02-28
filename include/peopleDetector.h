#ifndef PEOPLE_DETECT_H
#define PEOPLE_DETECT_H

#include <chrono>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

class PeopleDetector {
	public:
	    PeopleDetector();
		PeopleDetector(const string&);
		int testPeopleDetection();
	private:
		chrono::milliseconds getCurrentMillis();
		vector<float> loadDescriptorFromFile(string&);
		void get_svm_detector(const Ptr<SVM>&, vector<float>&);
		int fps;
		string videoFile;
};

#endif
