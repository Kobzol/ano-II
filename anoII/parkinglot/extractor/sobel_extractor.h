#pragma once

#include "extractor.h"

// morphological gradient
class SobelExtractor : public Extractor
{
public:
	explicit SobelExtractor(uchar threshold): threshold(threshold)
	{

	}

	std::vector<float> extract(cv::Mat image) override
	{
		const int size = 80;

		cv::cvtColor(image, image, CV_BGR2GRAY);

		cv::Mat resized = image.clone();
		cv::resize(resized, resized, cv::Size(size, size));
		cv::medianBlur(resized, resized, 3);

		cv::Mat sobel = this->sobel(resized);
		
		std::vector<float> features;
		features.push_back(this->extractPixels(sobel, this->threshold));

		int divisions = 10;
		int subWidth = sobel.cols / divisions;
		int subHeight = sobel.rows / divisions;

		for (int x = 0; x < sobel.rows; x += subHeight)
		{
			for (int y = 0; y < sobel.cols; y += subWidth)
			{
				if (x + subHeight >= sobel.rows || y + subWidth >= sobel.cols) continue;
				features.push_back(this->extractPixels(sobel(cv::Rect(y, x, subWidth, subHeight)), this->threshold));
			}
		}

		return features;
	}

private:
	cv::Mat sobel(const cv::Mat& image)
	{
		cv::Mat gradX;
		cv::Sobel(image, gradX, CV_16S, 1, 0);
		cv::convertScaleAbs(gradX, gradX);

		cv::Mat gradY;
		cv::Sobel(image, gradY, CV_16S, 0, 1);
		cv::convertScaleAbs(gradY, gradY);

		cv::Mat sobel;
		cv::addWeighted(gradX, 0.5, gradY, 0.5, 0.0, sobel);

		return sobel;
	}

	float extractPixels(const cv::Mat& image, uchar threshold)
	{
		float count = 0.0f;
		for (int i = 0; i < image.rows; i++)
		{
			for (int j = 0; j < image.cols; j++)
			{
				if (image.at<uchar>(i, j) >= threshold)
				{
					count++;
				}
			}
		}

		return count;
	}

	uchar threshold;
};
