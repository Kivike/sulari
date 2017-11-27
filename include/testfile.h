#ifndef TESTFILE_H
#define TESTFILE_H

#include <string>
#include <vector>

class TestFile
{
public:
    TestFile(std::string, std::string, std::vector<int>);
    TestFile();
    bool isKeyframe(int frame);
    unsigned int getPeopleCount() const;
    std::string getFilePath();
    std::string getFilename();
protected:
private:
    std::vector<int> keyframes;
    unsigned int peopleCount;
    std::string filePath, filename;
};
#endif