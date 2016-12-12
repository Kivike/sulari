#ifndef PEOPLE_DETECT_H
#define PEOPLE_DETECT_H

#include <chrono>

using namespace std;

class PeopleDetector {
	public:
		PeopleDetector();
		int testPeopleDetection();
	private:
		chrono::milliseconds getCurrentMillis();
};

#endif