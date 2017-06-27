#include "tests.h"
#include "cascadeclassifiertester.h"
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

Tests::Tests() {

}

Tests* Tests::setTester(CascadeClassifierTester* tester) {
    this->tester = tester;
    return this;
}

Tests* Tests::run() {
    if (!this->tester) {
        cout << "Tester not set" << endl;
        return nullptr;
    }

    TestSet kth = {
        vector<TestFile>{
        TestFile{ "videos/kth/daria_walk.avi", 1 },
            TestFile{ "videos/kth/daria_walk2.avi", 1 },
            TestFile{ "videos/kth/lena_walk2.avi", 1 }
    },
        "KTH"
    };
    TestSet ut_interaction = {
        vector<TestFile>{
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
    },
        "ut_interaction"
    };

    tester->runTest(kth);
    tester->runTest(ut_interaction);
    return this;
}
