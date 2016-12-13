#include <iostream>
#include "opencv2/opencv.hpp"
#include <chrono>
#include "peopleDetector.h"

using namespace std;
using namespace cv;

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
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	namedWindow("video capture", CV_WINDOW_AUTOSIZE);

	double fps = cap.get(CAP_PROP_FPS);
	double fps2 = cap.get(CV_CAP_PROP_FPS);

	chrono::milliseconds lastMillis = getCurrentMillis();

	//img = imread("./pictures/pedestrian.jpg");
	while (true)
	{
		chrono::milliseconds curMillis = getCurrentMillis();

		if (curMillis - lastMillis > chrono::milliseconds(1000)) {
			lastMillis = curMillis;
		} else {
			continue;
		}

		cap >> img;
		if (!img.data)
			continue;

		vector<Rect> found, found_filtered;
		hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);

		size_t i, j;
		for (i = 0; i < found.size(); i++)
		{
			Rect r = found[i];
			for (j = 0; j < found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
				found_filtered.push_back(r);
		}
		for (i = 0; i < found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.06);
			r.height = cvRound(r.height*0.9);
			rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
		}
		imshow("video capture", img);
		if (waitKey(20) >= 0)
			break;
	}
	return 0;
}

chrono::milliseconds PeopleDetector::getCurrentMillis() {
	chrono::milliseconds ms = chrono::duration_cast<chrono::milliseconds>(
		chrono::system_clock::now().time_since_epoch());

	return ms;
}
