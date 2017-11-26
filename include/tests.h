#ifndef TESTS_H
#define TESTS_H

#include "classifiertester.h"
#include "testset.h"

#include <fstream>
#include <string>

class Tests {
public:
    Tests* setTester(CascadeClassifierTester*);
    Tests* run();
    Tests();
    ~Tests() {
        delete this->tester;
    }
protected:
private:
    CascadeClassifierTester* tester;
    std::vector<TestSet*> sets;

    std::vector<TestSet*> getTestSets();
    void runSetAll(TestSet*);
    void runSet(TestSet*);
    void testCascade(TestSet*, const std::string&, int, int);
    void printResult(TestResult&);

    //std::ofstream testOutputCsv
};
#endif