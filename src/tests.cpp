#include "tests.h"
#include "classifiertester.h"
#include "keyframes.h"
#include <vector>
#include <iostream>
#include <memory>
#include <fstream>

using namespace std;
using namespace cv;

Tests::Tests() {}

Tests* Tests::run() {
    sets = getTestSets();

    cout << "Running " << sets.size() << " test sets" << endl;

    for(size_t i = 0; i < sets.size(); i++) {
        runSetAll(sets.at(i));
    }
    return this;
}

//
// Run all test sets with both classifier types and both with and without background removal
//
void Tests::runSetAll(TestSet *set) {
    tester = new CascadeClassifierTester();

    testCascade(set, "cascade/lbp_09.xml", 32, 64);
    testCascade(set, "cascade/lbp_10.xml", 32, 64);
    testCascade(set, "cascade/lbp_11.xml", 32, 64);
    testCascade(set, "cascade/haar_01.xml", 20, 40);
    testCascade(set, "cascade/haar_02.xml", 20, 40);
    testCascade(set, "cascade/haar_04.xml", 20, 40);
    testCascade(set, "cascade/haar_06.xml", 20, 40);
    testCascade(set, "cascade/lbp_01.xml", 20, 40);
    testCascade(set, "cascade/lbp_02.xml", 20, 40);
    testCascade(set, "cascade/lbp_03.xml", 20, 40);
    testCascade(set, "cascade/lbp_04.xml", 20, 40);
    testCascade(set, "cascade/lbp_05.xml", 20, 40);
    testCascade(set, "cascade/lbp_06.xml", 20, 40);
    testCascade(set, "cascade/lbp_07.xml", 20, 40);
    testCascade(set, "cascade/lbp_08.xml", 20, 40);

}

void Tests::testCascade(TestSet *set, const string& file, int width, int height) {
    tester = new CascadeClassifierTester();
    tester->setCascade(file, width, height);
    tester->enableBgRemoval();
    runSet(set);
    tester->disableBgRemoval();
    runSet(set);
}

void Tests::runSet(TestSet *set) {
    cout << "INIT SET" << endl;
    set->init();

    std::vector<TestFile*> files = set->getFiles();

    for (size_t i = 0; i < files.size(); i++) {
        TestResult *result = tester->testVideoFile(*(files.at(i)));

        if(result != nullptr) {
            printResult(*result);
            delete result;
        }
    }
}

vector<TestSet*> Tests::getTestSets() {
    vector<TestSet*> sets = vector<TestSet*>();
    sets.push_back(new TestSet("KTH running", "videos/kth/running", "videos/kth/kth_running_keyframes.csv"));
    sets.push_back(new TestSet("KTH walking", "videos/kth/walking", "videos/kth/kth_walking_keyframes.csv"));
    return sets;
}

void Tests::printResult(TestResult &result) {
    printf(" [det: %.2f][falsep: %.2f][fps: %.2f]\n",
        result.detectionRate,
        result.falsePositiveRate,
        result.averageFps
    );
}
