#include "tests.h"
#include "cascadeclassifiertester.h"
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

Tests::Tests() {}

Tests* Tests::run() {
    vector<TestSet*> sets = getTestSets();

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

    tester->setCascade("cascade/cascade_lbp4.xml", 32, 64);
    tester->enableBgRemoval();
    runSet(set);
    tester->disableBgRemoval();
    runSet(set);

    tester->setCascade("cascade/cascade_haar.xml", 32, 64);
    tester->disableBgRemoval();
    runSet(set);
    tester->enableBgRemoval();
    runSet(set);
}

void Tests::runSet(TestSet *set) {
    for (size_t i = 0; i < set->files.size(); i++) {
        TestResult *result = tester->testVideoFile(set->files.at(i));
        printResult(*result);
        delete result;
    }
}

vector<TestSet*> Tests::getTestSets() {
    vector<TestSet*> sets = vector<TestSet*>();

    TestSet* kth = new TestSet("KTH");
    kth->files = vector<TestFile> {
        TestFile{ "videos/kth/daria_walk.avi", 1 },
        TestFile{ "videos/kth/ira_walk.avi", 1 },
        TestFile{ "videos/kth/lena_walk2.avi", 1 }
    };

    TestSet* ut_interaction = new TestSet("ut_interaction");
    ut_interaction->files = vector<TestFile> {
        TestFile{ "videos/ut-interaction/seq1.avi", 2 },
        TestFile{ "videos/ut-interaction/seq2.avi", 2 },
        TestFile{ "videos/ut-interaction/seq3.avi", 2 },
        TestFile{ "videos/ut-interaction/seq4.avi", 2 },
        TestFile{ "videos/ut-interaction/seq5.avi", 2 },
        TestFile{ "videos/ut-interaction/seq6.avi", 2 },
        TestFile{ "videos/ut-interaction/seq7.avi", 2 },
        TestFile{ "videos/ut-interaction/seq8.avi", 2 },
        TestFile{ "videos/ut-interaction/seq9.avi", 2 },
        TestFile{ "videos/ut-interaction/seq10.avi",  2 }
    };

    sets.push_back(kth);
    sets.push_back(ut_interaction);
    return sets;
}

void Tests::printResult(TestResult &result) {
    printf(" [det: %.2f][falsep: %.2f][fps: %.2f]\n",
        result.detectionRate,
        result.falsePositiveRate,
        result.averageFps
    );
}
