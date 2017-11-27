#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#include "keyframes.h"

KeyframeCsv::KeyframeCsv(std::string filename)
{
	this->filename = filename;
	this->delimiter = ',';
}

void KeyframeCsv::read(std::vector<std::string>& filenames, std::vector<std::vector<int>>& keyframes)
{
	std::stringstream buffer;
	std::ifstream file(this->filename);

	if (file) {
		std::stringstream buffer;
		buffer << file.rdbuf();

		std::string line;

		while (std::getline(buffer, line)) {
			std::vector<int> videoKeyframes;
			std::string cell;
			std::stringstream lineBuffer;

			lineBuffer << line;

			int column = 0;

			while (std::getline(lineBuffer, cell, this->delimiter)) {
				if (column == 0) {
					filenames.push_back(cell);
				} else {
					try {
						int frame = std::stoi(cell);
						videoKeyframes.push_back(frame);	
					} catch (const std::invalid_argument &ia) {}
				}
				column++;
			}
			keyframes.push_back(videoKeyframes);
		}
	}
}