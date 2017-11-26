#ifndef TESTFILE_H
#define TESTFILE_H

#include <string>
#include <vector>

class TestFile
{
public:
    TestFile(std::string, std::vector<int>);
    TestFile();
    bool isKeyframe(int frame);
    unsigned int getPeopleCount() const;
    std::string getFilePath();
protected:
private:
    std::vector<int> keyframes;
    unsigned int peopleCount;
    std::string filePath;
};
#endif