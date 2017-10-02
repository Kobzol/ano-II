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

		cv::Mat resized;
		cv::cvtColor(image, image, CV_BGR2GRAY);
		cv::resize(image, resized, cv::Size(size, size));
		cv::medianBlur(resized, resized, 3);

		cv::Mat sobel = this->sobel(resized);
		cv::medianBlur(sobel, sobel, 3);
		
		std::vector<float> features;
		features.push_back(this->extractPixels(sobel, this->threshold));

		int divisions = 10;
		for (int x = 0; x < resized.rows; x += (resized.rows / divisions))
		{
			for (int y = 0; y < resized.cols; y += (resized.cols / divisions))
			{
				features.push_back(this->extractPixels(resized, this->threshold, x, y, size / divisions));
			}
		}

		return features;
	}

private:
	cv::Mat sobel(cv::Mat image)
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

	float extractPixels(cv::Mat image, uchar threshold)
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
	float extractPixels(cv::Mat image, uchar threshold, int x, int y, int size)
	{
		float count = 0.0f;
		for (int i = x; i < x + size; i++)
		{
			for (int j = y; j < y + size; j++)
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
