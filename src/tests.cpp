#include "tests.h"
#include "classifiertester.h"
#include <vector>
#include <iostream>

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

    testCascade(set, "cascade/haar_01.xml", 20, 40);
    testCascade(set, "cascade/haar_02.xml", 20, 40);
    testCascade(set, "cascade/haar_03.xml", 20, 40);
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
    cout << "Run set " << set->name << endl;
    for (size_t i = 0; i < set->files.size(); i++) {
        TestResult *result = tester->testVideoFile(set->files.at(i));
        printResult(*result);
        delete result;
    }
}

vector<TestSet*> Tests::getTestSets() {
    vector<TestSet*> sets = vector<TestSet*>();

    TestSet* weizmann = new TestSet("Weizmann");
    weizmann->files = vector<TestFile> {
        TestFile{ "videos/weizmann/daria_walk.avi", 1 },
        TestFile{ "videos/weizmann/ira_walk.avi", 1 },
        TestFile{ "videos/weizmann/lena_walk2.avi", 1 }
    };

    TestSet* ut_interaction = new TestSet("ut_interaction");
    ut_interaction->files = vector<TestFile> {
        TestFile{ "videos/ut-interaction/seq5.avi", 2 },
        TestFile{ "videos/ut-interaction/seq1.avi", 2 },
        TestFile{ "videos/ut-interaction/seq2.avi", 2 },
        TestFile{ "videos/ut-interaction/seq3.avi", 2 },
        TestFile{ "videos/ut-interaction/seq4.avi", 2 },
        TestFile{ "videos/ut-interaction/seq6.avi", 2 },
        TestFile{ "videos/ut-interaction/seq7.avi", 2 },
        TestFile{ "videos/ut-interaction/seq8.avi", 2 },
        TestFile{ "videos/ut-interaction/seq9.avi", 2 },
        TestFile{ "videos/ut-interaction/seq10.avi",  2 }
    };

    sets.push_back(weizmann);
    //sets.push_back(ut_interaction);
    return sets;
}

void Tests::printResult(TestResult &result) {
    printf(" [det: %.2f][falsep: %.2f][fps: %.2f]\n",
        result.detectionRate,
        result.falsePositiveRate,
        result.averageFps
    );
}
