#ifndef TESTS_H
#define TESTS_H

#include "classifiertester.h"

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
    std::vector<TestSet*> sets;

    std::vector<TestSet*> getTestSets();
    void runSetAll(TestSet*);
    void runSet(TestSet*);
    void testCascade(TestSet*, const std::string&, int, int);
    void printResult(TestResult&);
};

#endif
