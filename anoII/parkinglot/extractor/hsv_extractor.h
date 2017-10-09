#pragma once

#include "extractor.h"

class HSVExtractor : public Extractor
{
public:
	virtual std::vector<float> extract(cv::Mat image) override
	{
		cv::Mat hsv;
		cv::cvtColor(image, hsv, CV_BGR2HSV);
		cv::medianBlur(hsv, hsv, 3);

		float avgHue = 0.0f;
		for (int x = 0; x < hsv.rows; x++)
		{
			for (int y = 0; y < hsv.cols; y++)
			{
				avgHue += hsv.at<cv::Vec3b>(x, y)[0];
			}
		}

		avgHue /= (hsv.rows * hsv.cols);

		float deviance = 0;
		for (int x = 0; x < hsv.rows; x++)
		{
			for (int y = 0; y < hsv.cols; y++)
			{
				deviance += std::pow(std::abs(avgHue - hsv.at<cv::Vec3b>(x, y)[0]), 2);
			}
		}

		return { deviance / (hsv.rows * hsv.cols) };
	}

	virtual bool needsCopy() const override
	{
		return true;
	}
};
