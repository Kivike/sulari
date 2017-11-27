#ifndef TESTSET_H
#define TESTSET_H

#include <vector>

#include "testfile.h"

class TestSet
{
public:
    TestSet(std::string, std::string, std::string);
    void init();
    std::vector<TestFile*> getFiles();
    ~TestSet()
    {
    	for (auto p : files) {
    		delete p;
    	}
    }
protected:
private:
    std::string name, infoFilePath, videoDir;
    std::vector<TestFile*> files;

    std::vector<TestFile*> readCsv();
};
#endif