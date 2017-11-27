#include <vector>
#include <iostream>

#include "testset.h"
#include "testfile.h"
#include "keyframes.h"

TestSet::TestSet(std::string name, std::string videoDir, std::string infoFilePath)
{
	this->name = name;
	this->videoDir = videoDir;
	this->infoFilePath = infoFilePath;
}

void TestSet::init()
{
	this->files = this->readCsv();
}

std::vector<TestFile*> TestSet::getFiles()
{
	return this->files;
}

std::vector<TestFile*> TestSet::readCsv()
{
	KeyframeCsv csv = KeyframeCsv(this->infoFilePath);
	std::vector<std::string> filenames;
	std::vector<std::vector<int>> keyframes;
 
	csv.read(filenames, keyframes);

	std::cout << "File contained " << filenames.size() << " videos" << std::endl;
	std::vector<TestFile*> files;

	for(size_t i = 0; i < filenames.size(); i++) {
		std::string filePath = this->videoDir + '/' + filenames.at(i);

		files.push_back(new TestFile(filePath, keyframes.at(i)));
	}
	return files;
}