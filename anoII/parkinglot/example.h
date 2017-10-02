#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

class Example
{
public:
	Example() = default;
	explicit Example(int classIndex, cv::Mat image) : classIndex(classIndex), image(image)
	{

	}

	std::vector<float> features;
	int classIndex;
	cv::Mat image;
};
