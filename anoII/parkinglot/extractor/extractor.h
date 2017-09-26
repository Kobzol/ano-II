#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

class Extractor
{
public:
	Extractor() = default;
	virtual ~Extractor() = default;

	virtual std::vector<float> extract(cv::Mat image) = 0;
};
