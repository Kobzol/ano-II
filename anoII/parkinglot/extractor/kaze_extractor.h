#pragma once

#include "extractor.h"

#include <opencv2/features2d.hpp>

class KazeExtractor : public Extractor
{
public:
	virtual std::vector<float> extract(cv::Mat image) override
	{
		cv::resize(image, image, cv::Size(64, 64));

		auto kaze = cv::KAZE::create();
		std::vector<cv::KeyPoint> keypoints;
		std::vector<float> features;

		kaze->detectAndCompute(image, {}, keypoints, features);

		features.resize(400);
		return features;
	}

	virtual bool needsCopy() const override
	{
		return true;
	}
};
#pragma once
