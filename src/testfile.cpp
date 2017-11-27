#include "testfile.h"
#include <iostream>

TestFile::TestFile(std::string filePath, std::string filename, std::vector<int> keyframes)
{
	this->filePath = filePath;
	this->filename = filename;
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
	return false;
}

unsigned int TestFile::getPeopleCount() const
{
	return this->peopleCount;
}

std::string TestFile::getFilePath()
{
	return this->filePath;
}

std::string TestFile::getFilename()
{
	return this->filename;
}