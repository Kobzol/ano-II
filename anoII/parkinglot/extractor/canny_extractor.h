#pragma once

#include "extractor.h"

class CannyExtractor : public Extractor
{
public:
	explicit CannyExtractor(uchar threshold) : threshold(threshold)
	{

	}

	std::vector<float> extract(cv::Mat image) override
	{
		const int size = 80;

		cv::Mat resized;
		cv::cvtColor(image, image, CV_BGR2GRAY);
		cv::resize(image, resized, cv::Size(size, size));
		cv::medianBlur(resized, resized, 3);

		cv::Mat sobel = this->sobel(resized);

		std::vector<float> features;
		features.push_back(this->extractPixels(sobel));

		return features;
	}

private:
	cv::Mat sobel(const cv::Mat& image)
	{
		cv::Mat sobel = image.clone();
		cv::Canny(image, sobel, this->threshold, this->threshold * 3);
		return sobel;
	}

	float extractPixels(const cv::Mat& image)
	{
		float count = 0.0f;
		for (int i = 0; i < image.rows; i++)
		{
			for (int j = 0; j < image.cols; j++)
			{
				if (image.at<uchar>(i, j) == 255)
				{
					count++;
				}
			}
		}

		return count;
	}

	uchar threshold;
};
