#pragma once

#include <vector>

class Example
{
public:
	Example() = default;
	explicit Example(int classIndex) : classIndex(classIndex)
	{

	}

	std::vector<float> features;
	int classIndex;
};
