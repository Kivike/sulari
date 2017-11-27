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

    // 10 stages
    testCascade(set, "cascade/haar_10.xml", 23, 46);
    // 12 stages
    testCascade(set, "cascade/haar_11.xml", 23, 46);
    // 14 stages
    testCascade(set, "cascade/haar_12.xml", 23, 46);
    // 16 stages
    testCascade(set, "cascade/haar_13.xml", 23, 46);

    // 10 stages
    testCascade(set, "cascade/lbp_14.xml", 48, 96);
    // 12 stages
    testCascade(set, "cascade/lbp_16.xml", 48, 96);
    // 14 stages
    testCascade(set, "cascade/lbp_17.xml", 48, 96);
    // 16 stages
    testCascade(set, "cascade/lbp_19.xml", 48, 96);
}

void Tests::testCascade(TestSet *set, const string& file, int width, int height)
{
    tester = new CascadeClassifierTester();
    tester->setCascade(file, width, height);
    tester->enableBgRemoval();
    runSet(set);
    tester->disableBgRemoval();
    runSet(set);
}

void Tests::runSet(TestSet *set)
{
    set->init();

    std::vector<TestFile*> files = set->getFiles();

    for (size_t i = 0; i < files.size(); i++) {
        std::cout << files.at(i)->getFilename() << std::flush;
        TestResult *result = tester->testVideoFile(files.at(i));

        if(result != nullptr) {
            printResult(*result);
            delete result;
        }
    }
}

vector<TestSet*> Tests::getTestSets()
{
    vector<TestSet*> sets = vector<TestSet*>();
    sets.push_back(new TestSet("KTH running", "videos/kth/running", "videos/kth/kth_running_keyframes.csv"));
    sets.push_back(new TestSet("KTH walking", "videos/kth/walking", "videos/kth/kth_walking_keyframes.csv"));
    return sets;
}

void Tests::printResult(TestResult &result)
{
    printf(" [det: %.2f][falsep: %.2f][fps: %.2f]\n",
        result.testFile->getFilePath().c_str(),
        result.detectionRate,
        result.falsePositiveRate,
        result.averageFps
    );
}
