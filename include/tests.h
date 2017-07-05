#ifndef TESTS_H
#define TESTS_H

#include "cascadeclassifiertester.h"

class Tests {
public:
    Tests* setTester(CascadeClassifierTester*);
    Tests* run();
    Tests();
    ~Tests() {
        delete tester;
    }
private:
    CascadeClassifierTester* tester;
    std::vector<TestSet*> getTestSets();
    void runSetAll(TestSet*);
    void runSet(TestSet*);
};

#endif
