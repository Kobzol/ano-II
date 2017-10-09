#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>

#include "extractor/extractor.h"

class Example
{
public:
	static Example create(const std::vector<std::unique_ptr<Extractor>>& extractors, const cv::Mat& place, int classIndex);

	Example() = default;
	explicit Example(int classIndex, cv::Mat image);

	std::vector<float> features;
	int classIndex;
	cv::Mat image;
};
