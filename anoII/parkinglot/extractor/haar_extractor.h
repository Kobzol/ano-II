#pragma once

#include "extractor.h"

#include <memory>

class HaarExtractor : public Extractor
{
public:
	HaarExtractor(std::unique_ptr<cv::CascadeClassifier> classifier) : classifier(std::move(classifier))
	{

	}

	virtual std::vector<float> extract(cv::Mat image) override
	{
		cv::resize(image, image, cv::Size(64, 64));

		std::vector<cv::Rect> cars;
		this->classifier->detectMultiScale(image,
			cars,
			2,
			0,
			0,
			image.size() / 2,
			image.size()
		);
		
		return { (float) cars.size() };
	}

	virtual bool needsCopy() const override
	{
		return true;
	}

private:
	std::unique_ptr<cv::CascadeClassifier> classifier;
};
