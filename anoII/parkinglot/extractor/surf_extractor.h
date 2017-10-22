#pragma once

#include "extractor.h"

#include <opencv2/xfeatures2d.hpp>

class SurfExtractor : public Extractor
{
public:
	virtual std::vector<float> extract(cv::Mat image) override
	{
		cv::resize(image, image, cv::Size(64, 64));

		auto surf = cv::xfeatures2d::SURF::create();
		std::vector<cv::KeyPoint> keypoints;
		std::vector<float> features;

		surf->detectAndCompute(image, {}, keypoints, features);

		features.resize(500);
		return features;
	}

	virtual bool needsCopy() const override
	{
		return true;
	}
};
