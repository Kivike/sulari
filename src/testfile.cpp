#include "testfile.h"

TestFile::TestFile(std::string filePath, std::vector<int> keyframes)
{
	this->filePath = filePath;
	this->keyframes = keyframes;
	this->peopleCount = 1;
}

TestFile::TestFile()
{
	this->filePath = "";
	this->peopleCount = 0;
}

bool TestFile::isKeyframe(int frame)
{
	for (size_t i = 0; i < this->keyframes.size(); i++) {
		if (this->keyframes.at(i) == frame) {
			return true;
		}
	}
}

unsigned int TestFile::getPeopleCount() const
{
	return this->peopleCount;
}

std::string TestFile::getFilePath()
{
	return this->filePath;
}