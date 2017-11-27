#ifndef KEYFRAMES_H
#define KEYFRAMES_H

#include <vector>

class KeyframeCsv
{
	public:
		KeyframeCsv(std::string);
		void read(std::vector<std::string>&, std::vector<std::vector<int>>&);
	protected:
	private:
		std::string filename;
		char delimiter;
};

#endif
