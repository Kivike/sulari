#ifndef TESTS_H
#define TESTS_H

#include "cascadeclassifiertester.h"

class Tests {
public:
    Tests();
    Tests* setTester(CascadeClassifierTester*);
    Tests* run();
private:
    CascadeClassifierTester* tester;
    std::vector<TestSet*> getTestSets();
    void runSetAll(TestSet*);
    void runSet(TestSet*);
};

#endif
